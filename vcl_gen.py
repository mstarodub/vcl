import torch
import torch.nn as nn
import wandb
from tqdm.auto import trange
from torchvision.utils import make_grid
from itertools import chain

import util
from util import torch_device
from bayesian_layer import BayesianLinear
import generative
from generative import Generative, elbo


class Dgm(Generative):
  def __init__(
    self,
    in_dim,
    hidden_dim,
    latent_dim,
    architecture,
    ntasks,
    multihead,
    batch_size,
    learning_rate,
    classifier,
    layer_init_logstd_mean,
    layer_init_logstd_std,
  ):
    super().__init__(
      in_dim=in_dim,
      hidden_dim=hidden_dim,
      latent_dim=latent_dim,
      architecture=architecture,
      ntasks=ntasks,
      multihead=multihead,
      batch_size=batch_size,
      learning_rate=learning_rate,
      classifier=classifier,
    )

    self.decoder_shared = nn.Sequential(
      BayesianLinear(
        *self.dec_shared_l1,
        layer_init_logstd_mean,
        layer_init_logstd_std,
      ),
      nn.ReLU(),
      BayesianLinear(
        *self.dec_shared_l2,
        layer_init_logstd_mean,
        layer_init_logstd_std,
      ),
    )
    # no activation in between, this is accounted for in generative.decode
    self.decoder_heads = nn.ModuleList(
      nn.Sequential(
        BayesianLinear(
          *self.dec_heads_l1,
          layer_init_logstd_mean,
          layer_init_logstd_std,
        ),
        nn.ReLU(),
        BayesianLinear(
          *self.dec_heads_l2,
          layer_init_logstd_mean,
          layer_init_logstd_std,
        ),
      )
      for _ in range(ntasks if self.multihead else 1)
    )

  @property
  def bayesian_layers(self):
    return [
      layer
      for layer in list(self.decoder_shared)
      # list of ModuleLists
      + list(chain.from_iterable(self.decoder_heads))
      if isinstance(layer, BayesianLinear)
    ]

  @property
  def shared_bayesian_layers(self):
    return [
      layer
      for layer in list(self.decoder_shared)
      + (list(chain.from_iterable(self.decoder_heads)) if not self.multihead else [])
      if isinstance(layer, BayesianLinear)
    ]

  def compute_kl_loss(self):
    return sum(layer.kl_layer() for layer in self.shared_bayesian_layers)

  def update_prior(self):
    for layer in self.bayesian_layers:
      layer.update_prior_layer()

  def train_epoch(self, loader, opt, task, epoch):
    self.train()
    device = torch_device()
    for batch, batch_data in enumerate(loader):
      orig, ta = batch_data
      orig, ta = orig.to(device), ta.to(device)
      self.zero_grad()
      gen, mu, log_sigma = self(orig, ta)
      loss = -elbo(gen, mu, log_sigma, orig) + self.compute_kl_loss() / len(
        loader.dataset
      )
      loss.backward()
      opt.step()
      if batch % self.logging_every == 0 and orig.shape[0] == self.batch_size:
        metrics = {
          'task': task,
          'epoch': epoch,
          'train/train_loss': loss,
        }
        self.wandb_log(metrics)

  def train_test_run(self, tasks, num_epochs):
    self.train()
    # wandb.watch(self, log_freq=100)
    cumulative_img_samples = []
    for task, (train_loader, test_loaders) in enumerate(tasks):
      opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      for epoch in trange(num_epochs, desc=f'task {task}'):
        self.train_epoch(train_loader, opt, task, epoch)
      self.test_run(test_loaders, task)
      self.update_prior()
      img_samples = util.samples(self, upto_task=task, multihead=True)
      img_recons = util.reconstructions(
        self, test_loaders, upto_task=task, multihead=True
      )
      self.wandb_log_images_collect(
        task, img_samples, img_recons, cumulative_img_samples
      )
    final_grid = make_grid(cumulative_img_samples, nrow=1, padding=0)
    wandb.log({'task': task, 'cumulative_samples': wandb.Image(final_grid)})

  def wandb_log(self, metrics):
    sli = 0
    for sl in self.decoder_shared:
      if isinstance(sl, BayesianLinear):
        metrics[f'sigma/s_{sli}_sigma_w'] = sl.log_sigma_w.exp().mean().detach().item()
        sli += 1
    for hi, head in enumerate(self.decoder_heads):
      hli = 0
      for hl in head:
        if isinstance(hl, BayesianLinear):
          metrics[f'sigma/h_{hi}_{hli}_sigma_w'] = (
            hl.log_sigma_w.exp().mean().detach().item()
          )
          hli += 1
    wandb.log(metrics)


def generative_model_pipeline(params):
  loaders, classifier = generative.get_loaders_classifier(params)

  model = Dgm(
    in_dim=params.in_dim,
    hidden_dim=params.hidden_dim,
    latent_dim=params.latent_dim,
    architecture=params.architecture,
    ntasks=params.ntasks,
    multihead=params.multihead,
    batch_size=params.batch_size,
    layer_init_logstd_mean=params.layer_init_logstd_mean,
    layer_init_logstd_std=params.layer_init_logstd_std,
    learning_rate=params.learning_rate,
    classifier=classifier,
  ).to(torch_device())

  model.train_test_run(loaders, num_epochs=params.epochs)
  return model
