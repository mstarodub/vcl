import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm, trange
import wandb
from typing import Optional, List, Set, Any

import mle_disc
import dataloaders
from accuracy import accuracy
from util import torch_device
from bayesian_layer import BayesianLinear


class Ddm(nn.Module):
  def __init__(
    self,
    in_dim,
    hidden_dim,
    out_dim,
    hidden_layers,
    ntasks,
    batch_size,
    layer_init_logstd_mean,
    layer_init_logstd_std,
    learning_rate,
    per_task_opt=True,
    coreset_size=0,
    mle=None,
    multihead=False,
    gaussian=False,
    heteroscedastic=False,
    logging_every=10,
  ):
    super().__init__()
    self.logging_every = logging_every
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.per_task_opt = per_task_opt
    self.coreset_size = coreset_size
    self.multihead = multihead
    self.target_out_dim = out_dim
    self.gaussian = gaussian
    self.heteroscedastic = heteroscedastic

    self.shared = nn.Sequential()
    self.shared.append(
      BayesianLinear(in_dim, hidden_dim, layer_init_logstd_mean, layer_init_logstd_std)
    )
    self.shared.append(nn.ReLU())
    for _ in range(hidden_layers - 1):
      self.shared.append(
        BayesianLinear(
          hidden_dim, hidden_dim, layer_init_logstd_mean, layer_init_logstd_std
        )
      )
      self.shared.append(nn.ReLU())
    self.shared.forward = self.shared_forward

    head_out_features = 2 * out_dim if self.heteroscedastic else out_dim
    self.heads = nn.ModuleList(
      [
        BayesianLinear(
          hidden_dim, head_out_features, layer_init_logstd_mean, layer_init_logstd_std
        )
        for _ in range(ntasks if self.multihead else 1)
      ]
    )

    if mle and not self.gaussian:
      self.init_from_mle(mle)

  def shared_forward(self, x, deterministic=False):
    for layer in self.shared:
      if isinstance(layer, BayesianLinear):
        x = layer(x, deterministic=deterministic)
      # ReLU
      else:
        x = layer(x)
    return x

  def init_from_mle(self, mle):
    for i, layer in enumerate(self.shared):
      if isinstance(layer, BayesianLinear):
        layer.mu_w.data = mle.linear[i].weight.clone().detach()
        layer.mu_b.data = mle.linear[i].bias.clone().detach()
    if not self.multihead:
      self.heads[0].mu_w.data = mle.linear[-1].weight.clone().detach()
      self.heads[0].mu_b.data = mle.linear[-1].bias.clone().detach()

  @property
  def bayesian_layers(self):
    return [layer for layer in self.shared if isinstance(layer, BayesianLinear)] + list(
      self.heads
    )

  @property
  def shared_bayesian_layers(self):
    return (
      [layer for layer in self.shared if isinstance(layer, BayesianLinear)]
      # the single (shared) head is stored in self.heads for single-head
      + (list(self.heads) if not self.multihead else [])
    )

  def update_prior(self):
    for layer in self.bayesian_layers:
      layer.update_prior_layer()

  def restore_from_prior(self):
    for layer in self.bayesian_layers:
      layer.restore_from_prior_layer()

  def compute_kl(self):
    # notice that the gradients of the unused heads are zeroed and hence the KL terms are zero too
    return sum(layer.kl_layer() for layer in self.shared_bayesian_layers)

  def forward(self, x, task=None, deterministic=False):
    x = self.shared(x, deterministic=deterministic)
    if self.multihead:
      curr_batch_size, out_dim = x.shape[0], self.heads[0].out_dim
      out = torch.zeros(curr_batch_size, out_dim, device=x.device, dtype=x.dtype)
      for head_idx, head in enumerate(self.heads):
        mask = task == head_idx
        # (curr_batch_size, out_dim) * (curr_batch_size, 1)
        out += head(x, deterministic=deterministic) * mask.unsqueeze(1)
      return out
    else:
      return self.heads[0](x, deterministic=deterministic)

  # returns the first component of L_SGVB, without the N factor
  def sgvb_mc(self, pred, target):
    # we classification task, so target is a index to the right class
    # apply softmax:
    # p(target | pred) = exp(pred_target) / sum_{i=0}^len(pred) exp(pred_i)
    # according to (3) in the local reparam paper, we apply the log,
    # and arrive at the l_n from https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
    # we use reduction=mean to get the 1/M factor and the outer sum,
    # and finally arrive at eq (3) without the N factor
    if self.gaussian:
      if self.heteroscedastic:
        mean = pred[:, : self.target_out_dim]
        log_var = pred[:, self.target_out_dim :]
        # we need to clamp this here, otherwise exp(log_var) becomes NaN
        # (overflows?) after the transition to task 1
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        sq_err = (target - mean) ** 2
        return -(log_var + sq_err / log_var.exp()).mean()
      else:
        return -F.mse_loss(pred, target, reduction='mean')
    else:
      return -F.cross_entropy(pred, target, reduction='mean')

  def wandb_log(self, metrics):
    bli = 0
    for bl in self.shared:
      if isinstance(bl, BayesianLinear):
        metrics[f'sigma/s_{bli}_sigma_w'] = bl.log_sigma_w.exp().mean().detach().item()
        bli += 1
    for hi, hl in enumerate(self.heads):
      metrics[f'sigma/h_{hi}_sigma_w'] = hl.log_sigma_w.exp().mean().detach().item()
    wandb.log(metrics)

  # sample from (D_t) \cup C_{t-1}
  def select_coreset(
    self, task_size, old_coreset: Optional[List[Set[int]]]
  ) -> List[Set[int]]:
    assert self.coreset_size <= task_size
    if not old_coreset:
      return [set(np.random.permutation(np.arange(0, task_size))[: self.coreset_size])]
    covered_tasks = len(old_coreset)
    strat_size = len(old_coreset[0])
    for task in old_coreset:
      assert len(task) == strat_size
    assert self.coreset_size // covered_tasks == strat_size
    new_strat_size = self.coreset_size // (covered_tasks + 1)
    new_coreset = []
    for task in old_coreset:
      new_coreset.append(
        set(np.random.choice(list(task), new_strat_size, replace=False))
      )
    new_coreset.append(
      set(np.random.choice(range(0, task_size), new_strat_size, replace=False))
    )
    return new_coreset

  # returns (D_t \cup C_{t-1}) - C_t = (D_t - C_t) \cup (C_{t-1} - C_t)
  @staticmethod
  def select_augmented_complement(
    task_size, old_coreset: Optional[List[Set[int]]], new_coreset: List[Set[int]]
  ) -> List[Set[int]]:
    complement: List[Set[int]] = []
    if not old_coreset:
      assert len(new_coreset) == 1
      return [set(range(0, task_size)) - new_coreset[0]]
    for i in range(len(new_coreset)):
      if i == len(new_coreset) - 1:
        strat = set(range(0, task_size)) - new_coreset[i]
      else:
        strat = set(old_coreset[i]) - new_coreset[i]
      complement.append(strat)
    return complement

  # all_data is a list of dataloaders
  def create_dataloader(self, indexes: List[Set[int]], all_data: List[Any]):
    assert len(all_data) >= len(indexes)
    if self.multihead:
      data = [
        (all_data[i].dataset[j][0], all_data[i].dataset[j][1], i)
        for i, idx_set in enumerate(indexes)
        for j in idx_set
      ]
      dataset = torch.utils.data.TensorDataset(
        torch.stack([p[0] for p in data]) if data else torch.empty(0),
        torch.stack([p[1] for p in data]) if data else torch.empty(0),
        torch.tensor([p[2] for p in data]) if data else torch.empty(0),
      )
    else:
      data = [
        all_data[i].dataset[j] for i, idx_set in enumerate(indexes) for j in idx_set
      ]
      dataset = torch.utils.data.TensorDataset(
        torch.stack([p[0] for p in data]) if data else torch.empty(0),
        torch.stack([p[1] for p in data]) if data else torch.empty(0),
      )
    return torch.utils.data.DataLoader(
      dataset,
      batch_size=self.batch_size if self.batch_size else max(1, len(dataset)),
      shuffle=True if data else False,
      num_workers=12 if torch.cuda.is_available() else 0,
    )

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
      loss = -self.sgvb_mc(pred, target) + self.compute_kl() / len(loader.dataset)
      loss.backward()
      opt.step()
      if self.heteroscedastic:
        mean_pred = pred[:, : self.target_out_dim]
        avg_std_pred = (0.5 * pred[:, self.target_out_dim :]).exp().mean()
        pred = mean_pred
      acc = accuracy(pred, target)
      if batch % self.logging_every == 0 and data.shape[0] == self.batch_size:
        metrics = {
          'task': task,
          'epoch': epoch,
          'train/train_loss': loss,
          'train/train_acc': acc,
        } | ({'train/train_pred_avg_std': avg_std_pred} if self.heteroscedastic else {})
        self.wandb_log(metrics)

  def train_test_run(self, tasks, num_epochs):
    self.train()
    opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    # wandb.watch(self, log_freq=100)
    old_coreset_idx = []
    for task, (train_loaders, test_loaders) in enumerate(tasks):
      coreset_idx = self.select_coreset(len(train_loaders[-1].dataset), old_coreset_idx)
      complement_idx = self.select_augmented_complement(
        len(train_loaders[-1].dataset), old_coreset_idx, coreset_idx
      )
      coreset_loader = self.create_dataloader(coreset_idx, train_loaders)
      complement_loader = self.create_dataloader(complement_idx, train_loaders)
      old_coreset_idx = coreset_idx

      if self.per_task_opt:
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

      # restore network parameters to \tilde{q}_{t}
      if task > 0:
        self.restore_from_prior()

      # (2)
      # precondition: network prior is \tilde{q}_{t-1}
      # network parameters are whatever
      for epoch in trange(num_epochs, desc=f'task {task} phase 1'):
        self.train_epoch(complement_loader, opt, task, epoch)
      # ==> network parameters are \tilde{q}_t

      self.update_prior()
      # network parameters and network prior are \tilde{q}_t

      # (3)
      # precondition: network prior is \tilde{q}_{t}
      for epoch in trange(num_epochs, desc=f'task {task} phase 2'):
        self.train_epoch(coreset_loader, opt, task, epoch)
      # ==> network parameters are q_t

      self.test_run(test_loaders, task)

  @torch.no_grad()
  def test_run(self, loaders, task):
    self.eval()
    device = torch_device()
    avg_accuracies, avg_rmses = [], []
    for test_task, loader in tqdm(enumerate(loaders), desc=f'task {task} phase t'):
      task_accuracies, task_rmses = [], []
      for batch, batch_data in enumerate(loader):
        if self.multihead:
          data, target, t = batch_data
          t = t.to(device)
        else:
          (data, target), t = batch_data, None
        data, target = data.to(device), target.to(device)
        mean_pred = self(data, task=t, deterministic=True)
        if self.heteroscedastic:
          mean_mean_pred = mean_pred[:, : self.target_out_dim]
          # we could log this maybe
          _avg_std_mean_pred = (0.5 * mean_pred[:, self.target_out_dim :]).exp().mean()
          mean_pred = mean_mean_pred
        acc = accuracy(mean_pred, target)
        task_accuracies.append(acc.item())
        rmse = (
          torch.sqrt(F.mse_loss(mean_pred, target) + 1e-10)
          if self.gaussian
          else torch.tensor(0)
        )
        task_rmses.append(rmse)
      task_accuracy = np.mean(task_accuracies)
      task_rmse = np.mean(task_rmses)
      avg_accuracies.append(task_accuracy)
      avg_rmses.append(task_rmse)
      wandb.log({'task': task, f'test/test_acc_task_{test_task}': task_accuracy})
      if self.gaussian:
        wandb.log({'task': task, f'test/test_rmse_task_{test_task}': task_rmse})
    wandb.log({'task': task, 'test/test_acc': np.mean(avg_accuracies)})
    if self.gaussian:
      wandb.log({'task': task, 'test/test_rmse': np.mean(avg_rmses)})


def discriminative_model_pipeline(params):
  torch.autograd.set_detect_anomaly(True)

  loaders, baseline_loaders = None, None
  if params.problem == 'pmnist':
    baseline_loaders = dataloaders.pmnist_task_loaders(params.batch_size)[0]
    loaders = dataloaders.pmnist_task_loaders(
      params.batch_size, regression=params.gaussian
    )
  if params.problem == 'smnist':
    baseline_loaders = dataloaders.splitmnist_task_loaders(params.batch_size)[0]
    loaders = dataloaders.splitmnist_task_loaders(
      params.batch_size, regression=params.gaussian
    )
  if params.problem == 'nmnist':
    baseline_loaders = dataloaders.notmnist_task_loaders(params.batch_size)[0]
    loaders = dataloaders.notmnist_task_loaders(
      params.batch_size, regression=params.gaussian
    )

  if params.pretrain_epochs > 0:
    mle = mle_disc.pretrain_mle(params, baseline_loaders[0][0], baseline_loaders[1][0])

  else:
    mle = None

  model = Ddm(
    in_dim=params.in_dim,
    hidden_dim=params.hidden_dim,
    out_dim=params.out_dim,
    hidden_layers=params.hidden_layers,
    ntasks=params.ntasks,
    batch_size=params.batch_size,
    layer_init_logstd_mean=params.layer_init_logstd_mean,
    layer_init_logstd_std=params.layer_init_logstd_std,
    per_task_opt=params.per_task_opt,
    coreset_size=params.coreset_size,
    learning_rate=params.learning_rate,
    mle=mle,
    multihead=params.multihead,
    gaussian=params.gaussian,
    heteroscedastic=params.heteroscedastic,
  ).to(torch_device())

  # we have lots of dynamic control flow. not sure about this
  # model = torch.compile(model)

  model.train_test_run(loaders, num_epochs=params.epochs)
  return model
