import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm.auto import trange, tqdm

import util
import experiments
import dataloaders
from accuracy import accuracy
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
    batch_size,
    learning_rate,
    layer_init_logstd_mean,
    layer_init_logstd_std,
    device,
    logging_every=10,
  ):
    super().__init__()
    self.device = device
    self.logging_every = logging_every
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

  def forward(self, x, deterministic=False):
    for layer in self.layers:
      if isinstance(layer, BayesianLinear):
        x = layer(x, deterministic=deterministic)
      # activation
      else:
        x = layer(x)
    return x

  @property
  def bayesian_layers(self):
    return [layer for layer in self.layers if isinstance(layer, BayesianLinear)]

  # TODO
  def criterion(self, pred, target, dataset_sz):
    return (
      F.cross_entropy(pred, target, reduction='mean')
      + sum(layer.kl_layer() for layer in self.bayesian_layers) / dataset_sz
    )

  def update_prior(self):
    for layer in self.bayesian_layers:
      layer.update_prior_layer()

  def train_epoch(self, loader, opt, task, epoch):
    for batch, (data, target) in enumerate(loader):
      data, target = data.to(self.device), target.to(self.device)
      self.zero_grad()
      pred = self(data)
      loss = self.criterion(pred, target, len(loader.dataset))
      loss.backward()
      opt.step()
      acc = accuracy(pred, target)
      if batch % self.logging_every == 0 and data.shape[0] == self.batch_size:
        metrics = {
          'task': task,
          'epoch': epoch,
          'train/train_loss': loss,
          'train/train_acc': acc,
        }
        util.log(metrics)

  @torch.no_grad()
  def test_run(self, loaders, task):
    self.eval()
    avg_accs = []
    for test_task, loader in tqdm(enumerate(loaders), desc=f'task {task} test'):
      task_accs = []
      for batch, (data, target) in enumerate(loader):
        data, target = data.to(self.device), target.to(self.device)
        mean_pred = self(data, deterministic=True)
        acc = accuracy(mean_pred, target)
        task_accs.append(acc.item())
      task_acc = np.mean(task_accs)
      avg_accs.append(task_acc)
      util.log({'task': task, f'test/test_acc_task_{test_task}': task_acc})
    util.log({'task': task, 'test/test_acc': np.mean(avg_accs)})

  def train_test_run(self, tasks, num_epochs):
    self.train()
    for task, (train_loaders, test_loaders) in enumerate(tasks):
      train_loader = train_loaders[-1]
      opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      for epoch in trange(num_epochs, desc=f'task {task} train'):
        self.train_epoch(train_loader, opt, task, epoch)
      self.update_prior()
      self.test_run(test_loaders, task)


def flow_model_pipeline(params):
  torch.autograd.set_detect_anomaly(True)
  loaders = dataloaders.splitmnist_task_loaders(
    params.batch_size,
    multihead=False,
  )
  device = util.torch_device()
  model = DdmFlowed(
    in_dim=params.in_dim,
    hidden_dim=params.hidden_dim,
    out_dim=params.out_dim,
    hidden_layers=params.hidden_layers,
    batch_size=params.batch_size,
    learning_rate=params.learning_rate,
    layer_init_logstd_mean=params.layer_init_logstd_mean,
    layer_init_logstd_std=params.layer_init_logstd_std,
    device=device,
  ).to(device)
  model = torch.compile(model)

  model.train_test_run(loaders, num_epochs=params.epochs)
  return model


def flow_pipeline(params, wandb_log=False):
  wandb_mode = 'online' if wandb_log else 'disabled'
  with wandb.init(project='vcl', config=params, mode=wandb_mode):
    params = wandb.config
    util.wandb_setup_axes()
    model = flow_model_pipeline(params)
  return model


params = experiments.disc_singlehead_smnist | dict(
  model='vcl-mnf',
  epochs=120,
  batch_size=None,
  learning_rate=1e-3,
  layer_init_logstd_mean=-4,
  layer_init_logstd_std=0.01,
)

flow_pipeline(params)
