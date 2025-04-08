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
from generative import Generative


class Gsi(Generative):
  def __init__(
    self,
    in_dim,
    hidden_dim,
    latent_dim,
    ntasks,
    batch_size,
    learning_rate,
    classifier,
    c,
    xi,
  ):
    super().__init__(
      latent_dim=latent_dim,
      ntasks=ntasks,
      batch_size=batch_size,
      learning_rate=learning_rate,
      classifier=classifier,
    )

    self.c = c
    self.xi = xi

    self.encoders = nn.ModuleList(
      nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2 * latent_dim),
      )
      for _ in range(ntasks)
    )

    self.decoder_shared = nn.Sequential(
      SILayer(latent_dim, hidden_dim),
      nn.ReLU(),
      SILayer(hidden_dim, hidden_dim),
      nn.ReLU(),
    )

    self.decoder_heads = nn.ModuleList(
      nn.Sequential(
        SILayer(hidden_dim, hidden_dim),
        nn.ReLU(),
        SILayer(hidden_dim, in_dim),
        nn.Sigmoid(),
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

  def compute_surrogate_loss(self):
    return self.c * sum(layer.surrogate_layer() for layer in self.linear_layers)

  def train_epoch(self, loader, opt, task, epoch):
    self.train()
    device = torch_device()
    for batch, batch_data in enumerate(loader):
      orig, ta = batch_data
      orig, ta = orig.to(device), ta.to(device)
      self.zero_grad()
      gen, mu, log_sigma = self(orig, ta)
      uncert = self.classifier.classifier_uncertainty(gen, ta)
      loss = -self.elbo(gen, mu, log_sigma, orig) + self.compute_surrogate_loss()
      loss.backward()
      opt.step()
      if batch % self.logging_every == 0 and orig.shape[0] == self.batch_size:
        metrics = {
          'task': task,
          'epoch': epoch,
          'train/train_loss': loss,
          'train/train_uncert': uncert,
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
    ntasks=params.ntasks,
    batch_size=params.batch_size,
    learning_rate=params.learning_rate,
    c=params.c,
    xi=params.xi,
    classifier=classifier,
  ).to(torch_device())

  model.train_test_run(loaders, num_epochs=params.epochs)
  return model
