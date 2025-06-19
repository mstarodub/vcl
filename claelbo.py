import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from bayesian_layer import BayesianLinear


class RNVPFlow(nn.Module):
  def __init__(
    self,
    dim_z,
    n_flows,
    n_hidden,
    dim_hidden,
  ):
    super().__init__()
    self.nets = nn.ModuleList()
    self.mu_heads = nn.ModuleList()
    self.sigma_heads = nn.ModuleList()

    activation = nn.ReLU()

    for _ in range(n_flows):
      layers = [nn.Linear(dim_z, dim_hidden)]
      for _ in range(n_hidden):
        layers.extend([activation, nn.Linear(dim_hidden, dim_hidden)])

      self.nets.append(nn.Sequential(*layers, activation))
      self.mu_heads.append(nn.Linear(dim_hidden, dim_z))
      self.sigma_heads.append(nn.Linear(dim_hidden, dim_z))

  def forward(self, z):
    log_det = 0
    for i in range(self.n_flows):
      mask = torch.bernoulli(0.5 * torch.ones_like(z)).to(z.device)
      f = self.nets[i]
      g = self.mu_heads[i]
      k = self.sigma_heads[i]
      z_masked = mask * z
      h = f(z_masked)
      mu = g(h)
      sigma = F.sigmoid(k(h))
      z = z_masked + (1 - mask) * (z * sigma + (1 - sigma) * mu)
      # TODO: unsure
      log_det += (1 - mask) * sigma.log().sum(dim=1)
    return z, log_det


class DdmFlowed(nn.Module):
  def __init__(
    self,
    in_dim,
    hidden_dim,
    out_dim,
    hidden_layers,
    _ntasks,
    batch_size,
    learning_rate,
    layer_init_logstd_mean,
    layer_init_logstd_std,
  ):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.layers = nn.Sequential()
    self.layers.append(
      BayesianLinear(in_dim, hidden_dim, layer_init_logstd_mean, layer_init_logstd_std)
    )
    self.layers.append(nn.ReLU())
    for _ in range(hidden_layers):
      self.layers.append(
        BayesianLinear(
          hidden_dim, hidden_dim, layer_init_logstd_mean, layer_init_logstd_std
        )
      )
      self.layers.append(nn.ReLU())
    self.layers.append(
      BayesianLinear(hidden_dim, out_dim, layer_init_logstd_mean, layer_init_logstd_std)
    )

  def forward(self, x):
    for layer in self.layers:
      if isinstance(layer, BayesianLinear):
        x = layer(x, deterministic=False)
      # ReLU
      else:
        x = layer(x)
    return x

  @property
  def bayesian_layers(self):
    return [layer for layer in self.layers if isinstance(layer, BayesianLinear)]

  def criterion(self, pred, target):
    pass

  def update_prior(self):
    for layer in self.layers:
      layer.update_prior_layer()

  def train_epoch(self, loader, opt, task, epoch):
    for batch, (data, target) in enumerate(loader):
      self.zero_grad()
      pred = self(data)
      loss = -self.criterion(pred, target)
      loss.backward()
      opt.step()
      # TODO: log

  def train_test_run(self, tasks, num_epochs):
    self.train()
    for task, (train_loaders, test_loaders) in enumerate(tasks):
      train_loader = train_loaders[-1]
      opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      for epoch in range(num_epochs):
        self.train_epoch(train_loader, opt, task, epoch)
      self.update_prior()
    self.test_run(test_loaders, task)

  @torch.no_grad()
  def test_run(self, loaders, task):
    self.eval()
    # TODO
