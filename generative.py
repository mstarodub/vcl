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
    in_dim,
    hidden_dim,
    latent_dim,
    architecture,
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
    self.architecture = architecture

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

    latent_to_hidden = (latent_dim, hidden_dim)
    hidden_to_hidden = (hidden_dim, hidden_dim)
    hidden_to_in = (hidden_dim, in_dim)
    if architecture == 1:
      self.dec_heads_l1 = latent_to_hidden
      self.dec_heads_l2 = hidden_to_hidden
      self.dec_shared_l1 = hidden_to_hidden
      self.dec_shared_l2 = hidden_to_in
    if architecture == 2:
      self.dec_shared_l1 = latent_to_hidden
      self.dec_shared_l2 = hidden_to_hidden
      self.dec_heads_l1 = hidden_to_hidden
      self.dec_heads_l2 = hidden_to_in

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
    curr_batch_size, x_prime = z.shape[0], None
    if self.architecture == 2:
      # shared
      h = self.decoder_shared(z)
      h = F.relu(h)
      # followed by heads
      x_prime = torch.zeros(
        curr_batch_size,
        self.dec_heads_l2[1],
        device=z.device,
        dtype=z.dtype,
      )
      for head_idx, head in enumerate(self.decoder_heads):
        mask = task == head_idx
        x_prime += head(h) * mask.unsqueeze(1)
    if self.architecture == 1:
      # heads
      h = torch.zeros(
        curr_batch_size,
        self.dec_heads_l2[1],
        device=z.device,
        dtype=z.dtype,
      )
      for head_idx, head in enumerate(self.decoder_heads):
        mask = task == head_idx
        h += head(z) * mask.unsqueeze(1)
      h = F.relu(h)
      # followed by shared
      x_prime = self.decoder_shared(h)
    return F.sigmoid(x_prime)

  @torch.no_grad()
  def sample(self, digit):
    self.eval()
    z = torch.randn(1, self.latent_dim, device=torch_device())
    return self.decode(z, digit)

  def forward(self, x, task):
    z, mu, log_sigma = self.encode(x, task)
    return self.decode(z, task), mu, log_sigma

  def compute_test_ll(self, orig, ta, mu, log_sigma):
    # num_samples = 5_000: compute_test_ll took 197.9113 seconds
    num_samples = 10
    batch_size = orig.shape[0]
    # q(z|x)
    q_z_dist = torch.distributions.Normal(mu, torch.exp(log_sigma))
    # p(z)
    prior_dist = torch.distributions.Normal(
      torch.zeros_like(mu), torch.ones_like(log_sigma)
    )
    z_samples = q_z_dist.sample((num_samples,))
    z_samples_flat = z_samples.reshape(-1, self.latent_dim)
    gen_flat = self.decode(z_samples_flat, ta.repeat(num_samples))
    gen = gen_flat.reshape(num_samples, batch_size, -1)
    # p(x|z, θ) - pseudo likelihood
    log_p_x_z = -F.binary_cross_entropy(
      gen, orig.expand(num_samples, -1, -1), reduction='none'
    ).sum(dim=-1)
    log_p_z = prior_dist.log_prob(z_samples).sum(dim=-1)
    log_q_z_x = q_z_dist.log_prob(z_samples).sum(dim=-1)
    log_weights = log_p_x_z + log_p_z - log_q_z_x
    log_mean_weights = torch.logsumexp(log_weights, dim=0) - np.log(num_samples)
    return log_mean_weights.mean()

  @torch.no_grad()
  def test_run(self, loaders, task):
    self.eval()
    device = torch_device()
    avg_uncertainties, avg_testlls = [], []
    for test_task, loader in tqdm(enumerate(loaders), desc=f'task {task} phase t'):
      task_uncertainties, task_testlls = [], []
      for batch, batch_data in enumerate(loader):
        orig, ta = batch_data[0], batch_data[1]
        orig, ta = orig.to(device), ta.to(device)
        gen, mu, log_sigma = self(orig, ta)
        uncert = self.classifier.classifier_uncertainty(gen, ta)
        test_ll = self.compute_test_ll(orig, ta, mu, log_sigma)
        task_testlls.append(test_ll.item())
        task_uncertainties.append(uncert.item())
      task_uncertainty, task_testll = np.mean(task_uncertainties), np.mean(task_testlls)
      wandb.log(
        {
          'task': task,
          f'test/test_uncert_task_{test_task}': task_uncertainty,
          f'test/test_ll_task_{test_task}': task_testll,
        }
      )
      avg_uncertainties.append(task_uncertainty)
      avg_testlls.append(task_testll)
    wandb.log(
      {
        'task': task,
        'test/test_uncert': np.mean(avg_uncertainties),
        'test/test_ll': np.mean(avg_testlls),
      }
    )

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


def elbo(gen, mu, log_sigma, orig):
  batch_size = orig.shape[0]
  reconstr_likelihood = -F.binary_cross_entropy(gen, orig, reduction='sum')
  # kl_div_gaussians, but (mu_2, sigma_2) == (0, 1)
  kl_loss = -0.5 * torch.sum(1 - mu**2 + (2 * log_sigma) - torch.exp(2 * log_sigma))
  return (reconstr_likelihood - kl_loss) / batch_size


def get_loaders_classifier(params):
  loaders = None
  if params.problem == 'mnist':
    loaders = dataloaders.mnist_cont_task_loaders(params.batch_size)
  if params.problem == 'nmnist':
    loaders = dataloaders.nmnist_cont_task_loaders(params.batch_size)
  classifier = accuracy.init_classifier(params.problem)
  return loaders, classifier
