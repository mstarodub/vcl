import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm.auto import tqdm, trange
from torchvision.utils import make_grid
from itertools import chain

import dataloaders
import accuracy
import util
from util import torch_device
from bayesian_layer import BayesianLinear


class Dgm(nn.Module):
  def __init__(
    self,
    in_dim,
    hidden_dim,
    latent_dim,
    ntasks,
    batch_size,
    layer_init_std,
    learning_rate,
    classifier,
    bayesian_train_samples,
    bayesian_test_samples,
    logging_every=10,
  ):
    super().__init__()
    self.logging_every = logging_every
    self.ntasks = ntasks
    self.batch_size = batch_size
    self.classifier = classifier
    self.learning_rate = learning_rate
    self.bayesian_train_samples = bayesian_train_samples
    self.bayesian_test_samples = bayesian_test_samples

    self.latent_dim = latent_dim
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
      BayesianLinear(latent_dim, hidden_dim, layer_init_std),
      nn.ReLU(),
      BayesianLinear(hidden_dim, hidden_dim, layer_init_std),
      nn.ReLU(),
    )
    self.decoder_heads = nn.ModuleList(
      nn.Sequential(
        BayesianLinear(hidden_dim, hidden_dim, layer_init_std),
        nn.ReLU(),
        BayesianLinear(hidden_dim, in_dim, layer_init_std),
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

  def encode(self, x, task):
    curr_batch_size = x.shape[0]
    scatter_encoders = torch.zeros(
      curr_batch_size,
      self.encoders[0][-1].out_features,
      device=x.device,
      dtype=x.dtype,
    )
    for enc_idx, enc in enumerate(self.encoders):
      mask = task == enc_idx
      scatter_encoders += enc(x) * mask.unsqueeze(1)
    mu = scatter_encoders[:, : self.latent_dim]
    log_sigma = scatter_encoders[:, self.latent_dim :]
    eps = torch.randn_like(mu, device=x.device)
    z = mu + torch.exp(log_sigma) * eps
    return z, mu, log_sigma

  def decode(self, z, task):
    curr_batch_size = z.shape[0]
    h = self.decoder_shared(z)
    scatter_heads = torch.zeros(
      curr_batch_size,
      self.decoder_heads[0][-2].out_dim,
      device=z.device,
      dtype=z.dtype,
    )
    for head_idx, head in enumerate(self.decoder_heads):
      mask = task == head_idx
      scatter_heads += head(h) * mask.unsqueeze(1)
    return scatter_heads

  @torch.no_grad()
  def sample(self, digit):
    self.eval()
    z = torch.randn(1, self.latent_dim, device=torch_device())
    return self.decode(z, digit)

  def forward(self, x, task):
    z, mu, log_sigma = self.encode(x, task)
    return self.decode(z, task), mu, log_sigma

  # we don't scale this by the batch_size, instead one should change learning_rate
  def elbo(self, gen, mu, log_sigma, orig):
    reconstr_likelihood = -F.binary_cross_entropy(gen, orig, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 - mu**2 + (2 * log_sigma) - torch.exp(2 * log_sigma))
    return reconstr_likelihood - kl_loss

  def compute_kl(self):
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
      loss = -elbos.mean(0) + self.compute_kl() / len(loader.dataset)
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
    wandb.watch(self, log_freq=100)
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

  @torch.no_grad()
  def test_run(self, loaders, task):
    self.eval()
    device = torch_device()
    avg_uncertainties = []
    for test_task, loader in tqdm(enumerate(loaders), desc=f'task {task} phase t'):
      task_uncertainties = []
      for batch, batch_data in enumerate(loader):
        orig, ta = batch_data[0], batch_data[1]
        orig, ta = orig.to(device), ta.to(device)
        # TODO: no bayesian sampling for testing right now
        gen, mu, log_sigma = self(orig, ta)
        uncert = self.classifier.classifier_uncertainty(gen, ta)
        task_uncertainties.append(uncert.item())
      task_uncertainty = np.mean(task_uncertainties)
      wandb.log({'task': task, f'test/test_uncert_task_{test_task}': task_uncertainty})
      avg_uncertainties.append(task_uncertainty)
    wandb.log({'task': task, 'test/test_uncert': np.mean(avg_uncertainties)})

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

  def wandb_log_images_collect(
    self, task, img_samples, img_recons, cumulative_img_samples
  ):
    metrics = {
      'task': task,
      'samples': wandb.Image(img_samples, caption=f'task {task}'),
      'recons': wandb.Image(img_recons, caption=f'task {task}'),
    }
    wandb.log(metrics)
    cumulative_img_samples.append(
      torch.cat(
        [
          img_samples,
          torch.zeros(img_samples.shape[0], 28, (self.ntasks - task - 1) * 28),
        ],
        # (C, H, W)
        dim=2,
      )
    )


def generative_model_pipeline(params):
  loaders = None
  if params.problem == 'mnist':
    loaders = dataloaders.mnist_cont_task_loaders(params.batch_size)
  if params.problem == 'nmnist':
    loaders = dataloaders.nmnist_cont_task_loaders(params.batch_size)
  classifier = accuracy.init_classifier(params.problem)

  model = Dgm(
    in_dim=params.in_dim,
    hidden_dim=params.hidden_dim,
    latent_dim=params.latent_dim,
    ntasks=params.ntasks,
    batch_size=params.batch_size,
    layer_init_std=params.layer_init_std,
    bayesian_train_samples=params.bayesian_train_samples,
    bayesian_test_samples=params.bayesian_test_samples,
    learning_rate=params.learning_rate,
    classifier=classifier,
  ).to(torch_device())

  model.train_test_run(loaders, num_epochs=params.epochs)
  return model
