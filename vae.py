import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import trange

import util
from util import torch_device
import dataloaders
import accuracy


class Vae(nn.Module):
  def __init__(
    self,
    in_dim,
    hidden_dim,
    latent_dim,
    learning_rate,
    classifier,
  ):
    super().__init__()
    self.learning_rate = learning_rate
    self.classifier = classifier
    self.latent_dim = latent_dim
    self.encoder = nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, 2 * latent_dim),
    )
    self.decoder = nn.Sequential(
      nn.Linear(latent_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, in_dim),
      nn.Sigmoid(),
    )

  def forward(self, x):
    out = self.encoder(x)
    mu, log_sigma = out[:, : self.latent_dim], out[:, self.latent_dim :]
    eps = torch.randn_like(mu, device=x.device)
    z = mu + torch.exp(log_sigma) * eps
    return self.decoder(z), mu, log_sigma

  def elbo(self, mu, log_sigma, gen, orig):
    reconstr_likelihood = -F.binary_cross_entropy(gen, orig, reduction='sum')
    # reconstr_likelihood = -F.mse_loss(gen, orig, reduction='mean')
    # kl_div_gaussians, but (mu_2, sigma_2) == (0, 1)
    kl_loss = -0.5 * torch.sum(1 - mu**2 + (2 * log_sigma) - torch.exp(2 * log_sigma))
    return reconstr_likelihood - kl_loss

  def train_epoch(self, loader, opt):
    device = torch_device()
    self.train()
    losses, uncertainties = [], []
    for batch_data in loader:
      data, target = batch_data[0], batch_data[1]
      data, target = data.to(device), target.to(device)
      opt.zero_grad()
      gen, mu, log_sigma = self(data)
      loss = -self.elbo(mu, log_sigma, gen, data)
      uncert = self.classifier.classifier_uncertainty(gen, target)
      loss.backward()
      opt.step()
      losses.append(loss.item())
      uncertainties.append(uncert.item())
    return np.mean(losses), np.mean(uncertainties)

  @torch.no_grad()
  def test_run(self, loader):
    device = torch_device()
    self.eval()
    losses, uncertainties = [], []
    for batch_data in loader:
      data, target = batch_data[0], batch_data[1]
      data, target = data.to(device), target.to(device)
      gen, mu, log_sigma = self(data)
      loss = -self.elbo(mu, log_sigma, gen, data)
      uncert = self.classifier.classifier_uncertainty(gen, target)
      losses.append(loss.item())
      uncertainties.append(uncert.item())
    return np.mean(losses), np.mean(uncertainties)

  def train_run(self, train_loader, test_loader, num_epochs):
    opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    for epoch in (pbar := trange(num_epochs)):
      train_loss, train_uncert = self.train_epoch(train_loader, opt)
      test_loss, test_uncert = self.test_run(test_loader)
      pbar.set_description(
        f'epoch {epoch}: train loss {train_loss:.4f} test loss {test_loss:.4f} train uncert {train_uncert:.4f} test uncert {test_uncert:.4f}'
      )

  @torch.no_grad()
  def sample(self):
    self.eval()
    z = torch.randn(1, self.latent_dim, device=torch_device())
    return self.decoder(z)


def baseline_generative_model(num_epochs, problem):
  loaders = None
  if problem == 'mnist':
    loaders = dataloaders.mnist_vanilla_task_loaders(batch_size=128)
  if problem == 'nmnist':
    loaders = dataloaders.nmnist_vanilla_task_loaders(batch_size=128)
  classifier = accuracy.init_classifier(problem)

  model = Vae(
    in_dim=28 * 28,
    hidden_dim=500,
    latent_dim=50,
    learning_rate=1e-3,
    classifier=classifier,
  ).to(torch_device())

  train_loader, test_loader = loaders
  model.train_run(train_loader, test_loader, num_epochs=num_epochs)
  util.show_imgs(util.samples(model, upto_task=9, multihead=False))
  return model
