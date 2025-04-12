import torch
import torch.nn as nn
import wandb
from tqdm.auto import trange
from torchvision.utils import make_grid
from itertools import chain

import util
from util import torch_device
from si_layer import SILayer
import generative
from generative import Generative, elbo


class Gsi(Generative):
  def __init__(
    self,
    in_dim,
    hidden_dim,
    latent_dim,
    architecture,
    ntasks,
    batch_size,
    learning_rate,
    classifier,
    c,
    xi,
  ):
    super().__init__(
      in_dim=in_dim,
      hidden_dim=hidden_dim,
      latent_dim=latent_dim,
      architecture=architecture,
      ntasks=ntasks,
      batch_size=batch_size,
      learning_rate=learning_rate,
      classifier=classifier,
    )

    self.c = c
    self.xi = xi

    self.decoder_shared = nn.Sequential(
      SILayer(*self.dec_shared_l1),
      nn.ReLU(),
      SILayer(*self.dec_shared_l2),
    )

    self.decoder_heads = nn.ModuleList(
      nn.Sequential(
        SILayer(*self.dec_heads_l1),
        nn.ReLU(),
        SILayer(*self.dec_heads_l2),
      )
      for _ in range(ntasks)
    )

  @property
  def linear_layers(self):
    return [
      layer
      for layer in list(self.decoder_shared)
      + list(chain.from_iterable(self.decoder_heads))
      if isinstance(layer, SILayer)
    ]

  @property
  def shared_linear_layers(self):
    return [layer for layer in list(self.decoder_shared) if isinstance(layer, SILayer)]

  def compute_surrogate_loss(self):
    return self.c * sum(layer.surrogate_layer() for layer in self.shared_linear_layers)

  def train_epoch(self, loader, opt, task, epoch):
    self.train()
    device = torch_device()
    for batch, batch_data in enumerate(loader):
      orig, ta = batch_data
      orig, ta = orig.to(device), ta.to(device)
      self.zero_grad()
      gen, mu, log_sigma = self(orig, ta)
      loss = -elbo(gen, mu, log_sigma, orig) + self.compute_surrogate_loss()
      loss.backward()
      opt.step()
      if batch % self.logging_every == 0 and orig.shape[0] == self.batch_size:
        metrics = {
          'task': task,
          'epoch': epoch,
          'train/train_loss': loss,
        }
        wandb.log(metrics)

      for layer in self.linear_layers:
        layer.update_importance()

  def train_test_run(self, tasks, num_epochs):
    self.train()
    # wandb.watch(self, log_freq=100)
    cumulative_img_samples = []
    for task, (train_loader, test_loaders) in enumerate(tasks):
      opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      for epoch in trange(num_epochs, desc=f'task {task}'):
        self.train_epoch(train_loader, opt, task, epoch)

      for layer in self.linear_layers:
        layer.update_omega(self.xi)

      self.test_run(test_loaders, task)
      img_samples = util.samples(self, upto_task=task, multihead=True)
      img_recons = util.reconstructions(
        self, test_loaders, upto_task=task, multihead=True
      )
      self.wandb_log_images_collect(
        task, img_samples, img_recons, cumulative_img_samples
      )
    final_grid = make_grid(cumulative_img_samples, nrow=1, padding=0)
    wandb.log({'task': task, 'cumulative_samples': wandb.Image(final_grid)})


def generative_model_pipeline(params):
  loaders, classifier = generative.get_loaders_classifier(params)

  model = Gsi(
    in_dim=params.in_dim,
    hidden_dim=params.hidden_dim,
    latent_dim=params.latent_dim,
    architecture=params.architecture,
    ntasks=params.ntasks,
    batch_size=params.batch_size,
    learning_rate=params.learning_rate,
    c=params.c,
    xi=params.xi,
    classifier=classifier,
  ).to(torch_device())

  model.train_test_run(loaders, num_epochs=params.epochs)
  return model
