import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm.auto import trange, tqdm

import dataloaders
from accuracy import accuracy
from util import torch_device
from si_layer import SILayer


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
    c,
    xi,
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
    self.c = c
    self.xi = xi

    self.shared = nn.Sequential()
    self.shared.append(SILayer(in_dim, hidden_dim))
    self.shared.append(nn.ReLU())
    for _ in range(hidden_layers - 1):
      self.shared.append(SILayer(hidden_dim, hidden_dim))
      self.shared.append(nn.ReLU())

    self.heads = nn.ModuleList(
      [SILayer(hidden_dim, out_dim) for _ in range(ntasks if self.multihead else 1)]
    )

  @property
  def linear_layers(self):
    return [layer for layer in self.shared if isinstance(layer, SILayer)] + list(
      self.heads
    )

  @property
  def shared_linear_layers(self):
    return [layer for layer in self.shared if isinstance(layer, SILayer)] + (
      list(self.heads) if not self.multihead else []
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

  def compute_surrogate_loss(self):
    return self.c * sum(layer.surrogate_layer() for layer in self.shared_linear_layers)

  def train_epoch(self, loader, opt, task, epoch):
    device = torch_device()
    for batch, batch_data in enumerate(loader):
      if self.multihead:
        data, target, t = batch_data
        t = t.to(device)
      else:
        (data, target), t = batch_data, None
      data, target = data.to(device), target.to(device)
      self.zero_grad()
      pred = self(data, task=t)
      acc = accuracy(pred, target)
      loss = F.cross_entropy(pred, target) + self.compute_surrogate_loss()
      loss.backward()
      opt.step()
      if batch % self.logging_every == 0 and data.shape[0] == self.batch_size:
        metrics = {
          'task': task,
          'epoch': epoch,
          'train/train_loss': loss,
          'train/train_acc': acc,
        }
        wandb.log(metrics)

      for layer in self.linear_layers:
        layer.update_importance()

  def train_test_run(self, tasks, num_epochs):
    self.train()
    opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    # wandb.watch(self, log_freq=100)
    for task, (train_loaders, test_loaders) in enumerate(tasks):
      if self.per_task_opt:
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      for epoch in trange(num_epochs, desc=f'task {task}'):
        self.train_epoch(train_loaders[-1], opt, task, epoch)

      for layer in self.linear_layers:
        layer.update_omega(self.xi)

      self.test_run(test_loaders, task)

  @torch.no_grad()
  def test_run(self, loaders, task):
    self.eval()
    device = torch_device()
    avg_accuracies = []
    for test_task, loader in tqdm(enumerate(loaders), desc=f'task {task} test'):
      task_accuracies = []
      for batch, batch_data in enumerate(loader):
        if self.multihead:
          data, target, t = batch_data
          t = t.to(device)
        else:
          (data, target), t = batch_data, None
        data, target = data.to(device), target.to(device)
        pred = self(data, task=t)
        acc = accuracy(pred, target)
        task_accuracies.append(acc.item())
      task_accuracy = np.mean(task_accuracies)
      wandb.log({'task': task, f'test/test_acc_task_{test_task}': task_accuracy})
      avg_accuracies.append(task_accuracy)
    wandb.log({'task': task, 'test/test_acc': np.mean(avg_accuracies)})


def discriminative_model_pipeline(params):
  loaders = None
  if params.problem == 'pmnist':
    loaders = dataloaders.pmnist_task_loaders(params.batch_size)
  if params.problem == 'smnist' or params.problem == 'smnist-singlehead':
    loaders = dataloaders.splitmnist_task_loaders(
      params.batch_size,
      multihead=params.multihead,
    )
  if params.problem == 'nmnist':
    loaders = dataloaders.notmnist_task_loaders(params.batch_size)

  model = Dsi(
    in_dim=params.in_dim,
    hidden_dim=params.hidden_dim,
    out_dim=params.out_dim,
    hidden_layers=params.hidden_layers,
    ntasks=params.ntasks,
    batch_size=params.batch_size,
    learning_rate=params.learning_rate,
    c=params.c,
    xi=params.xi,
    per_task_opt=params.per_task_opt,
    multihead=params.multihead,
  ).to(torch_device())

  model.train_test_run(loaders, num_epochs=params.epochs)
  return model
