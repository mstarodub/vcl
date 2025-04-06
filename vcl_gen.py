import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm.auto import tqdm, trange
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
    logging_every=10,
  ):
    super().__init__()
    self.logging_every = logging_every
    self.batch_size = batch_size
    self.classifier = classifier
    self.learning_rate = learning_rate

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
      # list of modulelists
      + list(chain.from_iterable(self.decoder_heads))
      if isinstance(layer, BayesianLinear)
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

  def elbo(self, mu, log_sigma, gen, orig):
    reconstr_likelihood = -F.mse_loss(gen, orig, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 - mu**2 + (2 * log_sigma) - torch.exp(2 * log_sigma))
    return reconstr_likelihood - kl_loss

  def compute_kl(self):
    return sum(layer.kl_layer() for layer in self.bayesian_layers)

  def train_epoch(self, loader, opt, task, epoch):
    self.train()
    device = torch_device()
    for batch, batch_data in enumerate(loader):
      orig, ta = batch_data
      orig, ta = orig.to(device), ta.to(device)
      self.zero_grad()
      # TODO: no bayesian sampling for now
      gen, mu, log_sigma = self(orig, ta)
      uncert = self.classifier.classifier_uncertainty(gen, ta)
      loss = -self.elbo(mu, log_sigma, gen, orig) + self.compute_kl() / len(
        loader.dataset
      )
      loss.backward()
      opt.step()
      if batch % self.logging_every == 0 and orig.shape[0] == self.batch_size:
        metrics = {
          'task': task,
          'epoch': epoch,
          'train_loss': loss,
          'train_uncert': uncert,
        }
        self.wandb_log(metrics)

  def train_test_run(self, tasks, num_epochs):
    self.train()
    wandb.watch(self, log_freq=100)
    for task, (train_loader, test_loaders) in enumerate(tasks):
      opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      for epoch in trange(num_epochs, desc=f'task {task}'):
        self.train_epoch(train_loader, opt, task, epoch)
      self.test_run(test_loaders, task)

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
        # TODO: no bayesian sampling for now
        gen, mu, log_sigma = self(orig, ta)
        uncert = self.classifier.classifier_uncertainty(gen, ta)
        task_uncertainties.append(uncert.item())
      task_uncertainty = np.mean(task_uncertainties)
      wandb.log({'task': task, f'test_uncert_task_{test_task}': task_uncertainty})
      avg_uncertainties.append(task_uncertainty)
    wandb.log({'task': task, 'test_uncert': np.mean(avg_uncertainties)})

  def wandb_log(self, metrics):
    for bli, bl in enumerate(self.decoder_shared):
      if isinstance(bl, BayesianLinear):
        metrics[f's_{bli}_sigma_w'] = (
          torch.std(torch.exp(bl.log_sigma_w)).detach().item()
        )
    for hi, head in enumerate(self.decoder_heads):
      for hli, hl in enumerate(head):
        if isinstance(bl, BayesianLinear):
          metrics[f'h_{hi}_{hli}_sigma_w'] = (
            torch.std(torch.exp(hl.log_sigma_w)).detach().item()
          )
    wandb.log(metrics)


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
    learning_rate=params.learning_rate,
    classifier=classifier,
  ).to(torch_device())

  model.train_test_run(loaders, num_epochs=params.epochs)
  mixed_test_loader = dataloaders.mnist_vanilla_task_loaders(batch_size=128)[1]
  util.plot_reconstructions(model, mixed_test_loader, multihead=True)
  util.plot_samples(model, multihead=True)
  return model
