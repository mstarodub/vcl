import torch
import torch.nn as nn

import dataloaders
from util import torch_device
from accuracy import classifier_certainty
from bayesian_layer import BayesianLinear


class Dgm(nn.Module):
  def __init__(
    self,
    in_dim,
    hidden_dim,
    latent_dim,
    ntasks,
    layer_init_std,
    learning_rate,
    classifier,
    logging_every=10,
  ):
    super().__init__()
    self.logging_every = logging_every
    self.classifier = classifier
    self.learning_rate = learning_rate

    self.latent_dim = latent_dim
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
      BayesianLinear(latent_dim, hidden_dim, layer_init_std),
      nn.ReLU(),
      BayesianLinear(hidden_dim, hidden_dim, layer_init_std),
      nn.ReLU(),
    )
    self.decoder_heads = nn.ModuleList(
      nn.Sequential(
        BayesianLinear(hidden_dim, hidden_dim, layer_init_std),
        nn.ReLU(),
        BayesianLinear(hidden_dim, in_dim, layer_init_std),
        nn.Sigmoid(),
      )
      for _ in range(ntasks)
    )

    def encode(self, x, task):
      curr_batch_size = x.shape[0]
      scatter_encoders = torch.zeros(
        curr_batch_size,
        2 * self.encoders[0][-1].out_dim,
        device=x.device,
        dtype=x.dtype,
      )
      for enc_idx, enc in enumerate(self.encoder):
        mask = task == enc_idx
        scatter_encoders += enc(x) * mask.unsqueeze(1)
      mu = scatter_encoders[:, : self.latent_dim]
      log_sigma = scatter_encoders[:, self.latent_dim :]
      eps = torch.randn_like(mu, device=x.device)
      return mu + torch.exp(log_sigma) * eps

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

    def forward(self, x, task):
      z = self.encode(x, task)
      return decode(z, task)

    def train_epoch(self, loader, opt, task, epoch):
      self.train()
      device = torch_device()
      for batch, batch_data in enumerate(loader):
        orig, ta = batch_data
        orig, ta = orig.to(device), ta.to(device)
        self.zero_grad()
        # TODO: no bayesian sampling for now
        gen = self(orig, ta)
        acc = classifier_certainty(self.classifier, gen, ta)
        loss = -self.elbo(...) + self.compute_kl() / len
        # XXX
        (loader.dataset)
      pass

    @torch.no_grad()
    def test_run(self, loaders, task):
      self.eval()
      device = torch_device()
      pass

    def train_test_run(self, tasks, num_epochs):
      self.train()
      pass


def generative_model_pipeline(params):
  loaders, classifier = None, None
  if params.problem == 'mnist':
    loaders = dataloaders.mnist_cont_task_loaders(params.batch_size)
    classifier = None
  if params.problem == 'nmnist':
    loaders = dataloaders.nmnist_cont_task_loaders(params.batch_size)
    classifier = None

  model = Dgm(
    in_dim=params.in_dim,
    hidden_dim=params.hidden_dim,
    latent_dim=params.latent_dim,
    ntasks=params.ntasks,
    layer_init_std=params.layer_init_std,
    learning_rate=params.learning_rate,
    classifier=classifier,
  ).to(torch_device())
  model.train_test_run(loaders, num_epochs=params.epochs)
  return model
