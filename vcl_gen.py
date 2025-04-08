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
from generative import Generative


class Dgm(Generative):
  def __init__(
    self,
    in_dim,
    hidden_dim,
    latent_dim,
    ntasks,
    batch_size,
    learning_rate,
    classifier,
    layer_init_logstd_mean,
    layer_init_logstd_std,
    bayesian_train_samples,
    bayesian_test_samples,
  ):
    super().__init__(
      latent_dim=latent_dim,
      ntasks=ntasks,
      batch_size=batch_size,
      learning_rate=learning_rate,
      classifier=classifier,
    )

    self.bayesian_train_samples = bayesian_train_samples
    self.bayesian_test_samples = bayesian_test_samples

    self.encoders = nn.ModuleList(
      nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2 * latent_dim),
      )
      for _ in range(ntasks)
    )

    self.decoder_shared = nn.Sequential(
      BayesianLinear(
        latent_dim,
        hidden_dim,
        layer_init_logstd_mean,
        layer_init_logstd_std,
      ),
      nn.ReLU(),
      BayesianLinear(
        hidden_dim,
        hidden_dim,
        layer_init_logstd_mean,
        layer_init_logstd_std,
      ),
      nn.ReLU(),
    )

    self.decoder_heads = nn.ModuleList(
      nn.Sequential(
        BayesianLinear(
          hidden_dim,
          hidden_dim,
          layer_init_logstd_mean,
          layer_init_logstd_std,
        ),
        nn.ReLU(),
        BayesianLinear(
          hidden_dim,
          in_dim,
          layer_init_logstd_mean,
          layer_init_logstd_std,
        ),
        nn.Sigmoid(),
      )
      for _ in range(ntasks)
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
      layer for layer in list(self.decoder_shared) if isinstance(layer, BayesianLinear)
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
      gen_mu_log_sigmas = [self(orig, ta) for _ in range(self.bayesian_train_samples)]
      uncerts = torch.stack(
        [
          self.classifier.classifier_uncertainty(gmls[0], ta)
          for gmls in gen_mu_log_sigmas
        ]
      )
      elbos = torch.stack(
        [self.elbo(gmls[0], gmls[1], gmls[2], orig) for gmls in gen_mu_log_sigmas]
      )
      loss = -elbos.mean(0) + self.compute_kl_loss() / len(loader.dataset)
      loss.backward()
      opt.step()
      if batch % self.logging_every == 0 and orig.shape[0] == self.batch_size:
        metrics = {
          'task': task,
          'epoch': epoch,
          'train/train_loss': loss,
          'train/train_uncert': uncerts.mean(0),
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
        metrics[f'sigma/s_{sli}_sigma_w'] = (
          torch.std(torch.exp(sl.log_sigma_w)).detach().item()
        )
        sli += 1
    for hi, head in enumerate(self.decoder_heads):
      hli = 0
      for hl in head:
        if isinstance(hl, BayesianLinear):
          metrics[f'sigma/h_{hi}_{hli}_sigma_w'] = (
            torch.std(torch.exp(hl.log_sigma_w)).detach().item()
          )
          hli += 1
    wandb.log(metrics)


def generative_model_pipeline(params):
  loaders, classifier = generative.get_loaders_classifier(params)

  model = Dgm(
    in_dim=params.in_dim,
    hidden_dim=params.hidden_dim,
    latent_dim=params.latent_dim,
    ntasks=params.ntasks,
    batch_size=params.batch_size,
    layer_init_logstd_mean=params.layer_init_logstd_mean,
    layer_init_logstd_std=params.layer_init_logstd_std,
    bayesian_train_samples=params.bayesian_train_samples,
    bayesian_test_samples=params.bayesian_test_samples,
    learning_rate=params.learning_rate,
    classifier=classifier,
  ).to(torch_device())

  model.train_test_run(loaders, num_epochs=params.epochs)
  return model
