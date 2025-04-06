import torch
import torch.nn as nn
import torch.nn.functional as F


import dataloaders
from accuracy import accuracy
from util import torch_device


class SILayer(nn.Linear):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    # theta tilde
    self.weight_old_task = self.weight.clone().detach()
    self.bias_old_task = self.bias.clone().detach()

    # to compute theta' * dt in the integral
    self.weight_old = self.weight.clone().detach()
    self.bias_old = self.bias.clone().detach()

    # the w_k
    self.relevance_weight = torch.zeros_like(self.weight)
    self.relevance_bias = torch.zeros_like(self.bias)

    # the Omega_k
    self.omega_weight = torch.zeros_like(self.weight)
    self.omega_bias = torch.zeros_like(self.bias)


class Dsi(nn.Module):
  def __init__(
    self,
    in_dim,
    hidden_dim,
    out_dim,
    hidden_layers,
    ntasks,
    batch_size,
    learning_rate,
    per_task_opt=True,
    multihead=False,
    logging_every=10,
  ):
    super().__init__()
    self.logging_every = logging_every
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.per_task_opt = per_task_opt
    self.multihead = multihead

    self.c = 1
    self.xi = 0.1

    self.shared = nn.Sequential()
    self.shared.append(nn.SILayer(in_dim, hidden_dim))
    self.shared.append(nn.ReLU())
    for _ in range(hidden_layers - 1):
      self.shared.append(nn.SILayer(hidden_dim, hidden_dim))
      self.shared.append(nn.ReLU())

    self.heads = nn.ModuleList(
      [nn.SILayer(hidden_dim, out_dim) for _ in range(ntasks if self.multihead else 1)]
    )

    # TODO modularize, same as in vcl
    @property
    def linear_layers(self):
      return [layer for layer in self.shared if isinstance(layer, nn.SILayer)] + list(
        self.heads
      )

    def forward(self, x, task=None):
      x = self.shared(x)
      if self.multihead:
        curr_batch_size, out_dim = x.shape[0], self.heads[0].out_features
        out = torch.zeros(curr_batch_size, out_dim, device=x.device, dtype=x.dtype)
        for head_idx, head in enumerate(self.heads):
          mask = task == head_idx
          out += head(x) * mask.unsqueeze(1)
        return out
      else:
        return self.heads[0](x)

    def train_epoch(self, loader, opt, task, epoch):
      device = torch_device()
      for batch, batch_data in enumerate(loader):
        if self.multihead:
          data, target, t = batch_data
        else:
          (data, target), t = batch_data, None
        data, target = data.to(device), target.to(device)
        self.zero_grad()
        pred = self(data, task=t)
        acc = accuracy(pred, target)

        loss_likelihood = F.cross_entropy(pred, target)

        # create sum
        surrogate_loss = 0
        for layer in self.linear_layers:
          surrogate_loss += torch.sum(
            layer.omega_weight * (layer.weight_old_task - layer.weight) ** 2
          )
          surrogate_loss += torch.sum(
            layer.omega_bias * (layer.bias_old_task - layer.bias) ** 2
          )
        surrogate_loss *= self.c

        loss = loss_likelihood + surrogate_loss
        loss.backward()
        opt.step()

        if batch % self.logging_every == 0 and batch_data.shape[0] == self.batch_size:
          metrics = {'task': task, 'epoch': epoch, 'train_loss': loss, 'train_acc': acc}
          self.wandb_log(metrics)

        # update self.w_k
        for layer in self.linear_layers:
          layer.relevance_weight += -layer.weight.grad * (
            layer.weight - layer.weight_old
          )
          layer.relevance_bias += -layer.bias.grad * (layer.bias - layer.bias_old)

          layer.weight_old = layer.weight.clone().detach()
          layer.bias_old = layer.bias.clone().detach()

  def train_test_run(self, tasks, num_epochs):
    self.train()
    opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    wandb.watch(self, log_freq=100)
    old_coreset_idx = []
    for task, (train_loaders, test_loaders) in enumerate(tasks):
      if self.per_task_opt:
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

      for epoch in trange(num_epochs, desc=f'task {task} phase 1'):
        self.train_epoch(train_loaders[-1], opt, task, epoch)

      # update Delta_k (by updating layer.weight_old_task)
      # update Omega_k
      for layer in self.linear_layers:
        delta_weight = layer.weight - layer.weight_old_task
        layer.weight_old_task = layer.weight.clone().detach()

        delta_bias = layer.bias - layer.bias_old_task
        layer.bias_old_task = layer.bias.clone().detach()

        layer.omega_weight += layer.relevance_weight / (delta_weight**2 + layer.xi)
        layer.omega_bias += layer.relevance_bias / (delta_bias**2 + layer.xi)

      self.test_run(test_loaders, task)
