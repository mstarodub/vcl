import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import trange
import os

import dataloaders
from util import torch_device


class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      # (1, 28, 28)
      nn.Conv2d(1, 16, kernel_size=3, padding=1),
      # (16, 28, 28)
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.Conv2d(16, 16, kernel_size=3, padding=1),
      # (16, 28, 28)
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
      # (16, 14, 14)
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.Conv2d(16, 32, kernel_size=3, padding=1),
      # (32, 14, 14)
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.Flatten(),
      # 32 * 14 * 14 = 6272
      nn.Dropout(0.4),
      nn.Linear(6272, 10),
    )
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(16)
    self.conv3 = nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2)
    self.bn3 = nn.BatchNorm2d(16)
    self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.bn4 = nn.BatchNorm2d(32)
    self.flatten = nn.Flatten()
    self.dropout = nn.Dropout(0.4)
    self.fc = nn.Linear(6272, 10)

  def forward(self, x):
    # -1 infers batch size
    x = x.reshape(-1, 1, 28, 28)
    return self.model(x)


class CNNEnsembleClassifier(nn.Module):
  def __init__(self, n_ensemble=3):
    super().__init__()
    self.device = torch_device()
    self.models = nn.ModuleList([CNN().to(self.device) for _ in range(n_ensemble)])
    self.opts = [torch.optim.AdamW(model.parameters()) for model in self.models]

  def forward(self, x):
    return torch.stack([model(x) for model in self.models]).mean(dim=0)

  def train_epoch(self, loader):
    self.train()
    losses, accuracies = [], []
    for data, label in loader:
      data, label = data.to(self.device), label.to(self.device)
      for model, opt in zip(self.models, self.opts):
        opt.zero_grad()
        pred = model(data)
        loss = F.cross_entropy(pred, label)
        acc = accuracy(pred, label)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        accuracies.append(acc.item())
    return np.mean(losses), np.mean(accuracies)

  @torch.no_grad()
  def test_run(self, loader):
    self.eval()
    losses, accuracies = [], []
    for data, label in loader:
      data, label = data.to(self.device), label.to(self.device)
      avg_out = self(data)
      loss = F.cross_entropy(avg_out, label)
      acc = accuracy(avg_out, label)
      losses.append(loss.item())
      accuracies.append(acc.item())
    return np.mean(losses), np.mean(accuracies)

  def train_run(self, train_loader, test_loader, num_epochs):
    for epoch in (pbar := trange(num_epochs)):
      train_loss, train_acc = self.train_epoch(train_loader)
      test_loss, test_acc = self.test_run(test_loader)
      pbar.set_description(
        f'classifier: epoch {epoch}: train loss {train_loss:.4f} test loss {test_loss:.4f} train acc {train_acc:.4f} test acc {test_acc:.4f}'
      )

  @torch.no_grad()
  def classifier_uncertainty(self, gen, target):
    logits = self(gen)
    eps = 1e-8
    probs = F.softmax(logits, dim=1).clamp(min=eps)
    one_hot = F.one_hot(target, num_classes=10).float()
    # D_KL(one-hot || predicted)
    return F.kl_div(probs.log(), one_hot, reduction='batchmean')


def init_classifier(problem, num_epochs=30, batch_size=256):
  device = torch_device()
  csf = CNNEnsembleClassifier()

  csf_path, csf_loaders = None, None
  if problem == 'mnist':
    csf_path = 'classifier_mnist.pt'
    csf_loaders = dataloaders.mnist_vanilla_task_loaders(batch_size)
  if problem == 'nmnist':
    csf_path = 'classifier_nmnist.pt'
    csf_loaders = dataloaders.nmnist_vanilla_task_loaders(batch_size)

  if os.path.exists(csf_path):
    # .to(device) is inplace for nn.Module (but not for tensors!)
    csf.load_state_dict(torch.load(csf_path, map_location=device))
    csf.to(device)
  else:
    csf.to(device)
    csf.train_run(*csf_loaders, num_epochs=num_epochs)
    torch.save(csf.state_dict(), csf_path)

  return csf


def accuracy(pred, target):
  # undo one-hot encoding, if applicable
  target_idx = target.argmax(dim=1) if target.ndim == pred.ndim else target
  return (pred.argmax(dim=1) == target_idx).float().mean()
