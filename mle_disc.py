import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import trange
import os

from accuracy import accuracy
from util import torch_device


class DNet(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim, hidden_layers, learning_rate):
    super().__init__()
    self.linear = nn.Sequential()
    # first hidden layer
    self.linear.append(nn.Linear(in_dim, hidden_dim))
    self.linear.append(nn.ReLU())
    # remaining hidden layers
    for _ in range(hidden_layers - 1):
      self.linear.append(nn.Linear(hidden_dim, hidden_dim))
      self.linear.append(nn.ReLU())
    # output layer
    self.linear.append(nn.Linear(hidden_dim, out_dim))

    self.learning_rate = learning_rate

  def forward(self, x):
    return self.linear(x)

  def train_epoch(self, loader, loss_fn, opt):
    device = torch_device()
    self.train()
    losses, accuracies = [], []
    for batch_data in loader:
      data, target = batch_data[0], batch_data[1]
      data, target = data.to(device), target.to(device)
      opt.zero_grad()
      pred = self(data)
      loss = loss_fn(pred, target)
      acc = accuracy(pred, target)
      loss.backward()
      opt.step()
      losses.append(loss.item())
      accuracies.append(acc.item())
    return np.mean(losses), np.mean(accuracies)

  def train_run(self, train_loader, test_loader, num_epochs):
    loss_fn = F.cross_entropy
    opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    for epoch in (pbar := trange(num_epochs)):
      train_loss, train_acc = self.train_epoch(train_loader, loss_fn, opt)
      test_loss, test_acc = self.test_run(test_loader, loss_fn)
      pbar.set_description(
        f'pretrain epoch {epoch}: train loss {train_loss:.4f} train acc {train_acc:.4f} test loss {test_loss:.4f} test acc {test_acc:.4f}'
      )

  @torch.no_grad()
  def test_run(self, loader, loss_fn):
    device = torch_device()
    self.eval()
    losses, accuracies = [], []
    for batch_data in loader:
      data, target = batch_data[0], batch_data[1]
      data, target = data.to(device), target.to(device)
      pred = self(data)
      loss = loss_fn(pred, target)
      acc = accuracy(pred, target)
      losses.append(loss.item())
      accuracies.append(acc.item())
    return np.mean(losses), np.mean(accuracies)


def pretrain_mle(params, train_loader, test_loader):
  device = torch_device()
  mle = DNet(
    in_dim=params.in_dim,
    hidden_dim=params.hidden_dim,
    out_dim=params.out_dim,
    hidden_layers=params.hidden_layers,
    learning_rate=params.learning_rate,
  )

  mle_path = f'pretrained/mle_{params.pretrain_epochs}.pt'
  if os.path.exists(mle_path):
    mle.load_state_dict(torch.load(mle_path, map_location=device))
    mle.to(device)
  else:
    mle.to(device)
    mle.train_run(
      train_loader,
      test_loader,
      num_epochs=params.pretrain_epochs,
    )
    torch.save(mle.state_dict(), mle_path)

  return mle
