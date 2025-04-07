import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm.auto import tqdm

import dataloaders
import accuracy
from util import torch_device


class Generative(nn.Module):
  def __init__(
    self,
    latent_dim,
    ntasks,
    batch_size,
    learning_rate,
    classifier,
    logging_every=10,
  ):
    super().__init__()
    self.latent_dim = latent_dim
    self.ntasks = ntasks
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.classifier = classifier
    self.logging_every = logging_every

  def encode(self, x, task):
    curr_batch_size = x.shape[0]
    scatter_encoders = torch.zeros(
      curr_batch_size,
      self.encoders[0][-1].out_features,
      device=x.device,
      dtype=x.dtype,
    )
    for enc_idx, enc in enumerate(self.encoders):
      mask = task == enc_idx
      scatter_encoders += enc(x) * mask.unsqueeze(1)
    mu = scatter_encoders[:, : self.latent_dim]
    log_sigma = scatter_encoders[:, self.latent_dim :]
    eps = torch.randn_like(mu, device=x.device)
    z = mu + torch.exp(log_sigma) * eps
    return z, mu, log_sigma

  def decode(self, z, task):
    curr_batch_size = z.shape[0]
    h = self.decoder_shared(z)
    scatter_heads = torch.zeros(
      curr_batch_size,
      self.decoder_heads[0][-2].out_dim,
      device=z.device,
      dtype=z.dtype,
    )
    for head_idx, head in enumerate(self.decoder_heads):
      mask = task == head_idx
      scatter_heads += head(h) * mask.unsqueeze(1)
    return scatter_heads

  @torch.no_grad()
  def sample(self, digit):
    self.eval()
    z = torch.randn(1, self.latent_dim, device=torch_device())
    return self.decode(z, digit)

  def forward(self, x, task):
    z, mu, log_sigma = self.encode(x, task)
    return self.decode(z, task), mu, log_sigma

  # we don't scale this by the batch_size, instead one should change learning_rate
  def elbo(self, gen, mu, log_sigma, orig):
    reconstr_likelihood = -F.binary_cross_entropy(gen, orig, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 - mu**2 + (2 * log_sigma) - torch.exp(2 * log_sigma))
    return reconstr_likelihood - kl_loss

  @torch.no_grad()
  def test_run(self, loaders, task):
    self.eval()
    device = torch_device()
    avg_uncertainties = []
    for test_task, loader in tqdm(enumerate(loaders), desc=f'task {task} phase t'):
      task_uncertainties = []
      for batch, batch_data in enumerate(loader):
        orig, ta = batch_data[0], batch_data[1]
        orig, ta = orig.to(device), ta.to(device)
        gen, mu, log_sigma = self(orig, ta)
        uncert = self.classifier.classifier_uncertainty(gen, ta)
        task_uncertainties.append(uncert.item())
      task_uncertainty = np.mean(task_uncertainties)
      wandb.log({'task': task, f'test/test_uncert_task_{test_task}': task_uncertainty})
      avg_uncertainties.append(task_uncertainty)
    wandb.log({'task': task, 'test/test_uncert': np.mean(avg_uncertainties)})

  def wandb_log_images_collect(
    self, task, img_samples, img_recons, cumulative_img_samples
  ):
    metrics = {
      'task': task,
      'samples': wandb.Image(img_samples, caption=f'task {task}'),
      'recons': wandb.Image(img_recons, caption=f'task {task}'),
    }
    wandb.log(metrics)
    cumulative_img_samples.append(
      torch.cat(
        [
          img_samples,
          torch.zeros(img_samples.shape[0], 28, (self.ntasks - task - 1) * 28),
        ],
        # (C, H, W)
        dim=2,
      )
    )


def get_loaders_classifier(params):
  loaders = None
  if params.problem == 'mnist':
    loaders = dataloaders.mnist_cont_task_loaders(params.batch_size)
  if params.problem == 'nmnist':
    loaders = dataloaders.nmnist_cont_task_loaders(params.batch_size)
  classifier = accuracy.init_classifier(params.problem)
  return loaders, classifier
