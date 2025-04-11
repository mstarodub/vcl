import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import trange

import util
from util import torch_device
import dataloaders
import accuracy
from generative import elbo


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

  def train_epoch(self, loader, opt):
    device = torch_device()
    self.train()
    losses = []
    for batch_data in loader:
      data, target = batch_data[0], batch_data[1]
      data, target = data.to(device), target.to(device)
      opt.zero_grad()
      gen, mu, log_sigma = self(data)
      loss = -elbo(mu, log_sigma, gen, data)
      loss.backward()
      opt.step()
      losses.append(loss.item())
    return np.mean(losses)

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
      train_loss = self.train_epoch(train_loader, opt)
      util.show_imgs(util.samples(self, upto_task=9, multihead=False))
      pbar.set_description(f'epoch {epoch}: train loss {train_loss:.4f}')
    test_loss, test_uncert = self.test_run(test_loader)

  @torch.no_grad()
  def sample(self):
    self.eval()
    z = torch.randn(1, self.latent_dim, device=torch_device())
    return self.decoder(z)


def baseline_generative_model(num_epochs, problem):
  loaders = None
  if problem == 'mnist':
    loaders = dataloaders.mnist_vanilla_task_loaders(batch_size=256)
  if problem == 'nmnist':
    loaders = dataloaders.nmnist_vanilla_task_loaders(batch_size=256)
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
  return model
