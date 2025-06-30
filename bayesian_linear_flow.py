import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class BayesianLinear(nn.Module):
  def __init__(self, in_dim, out_dim, log_sigma_mean, log_sigma_std):
    super().__init__()
    device = util.torch_device()
    self.in_dim = in_dim
    self.out_dim = out_dim

    init_mu_w = torch.empty(out_dim, in_dim, device=device)
    torch.nn.init.kaiming_normal_(init_mu_w, nonlinearity='relu')
    self.mu_w = nn.Parameter(init_mu_w)

    init_mu_b = torch.zeros(out_dim, device=device)
    self.mu_b = nn.Parameter(init_mu_b)

    init_log_sigma_w = torch.empty(out_dim, in_dim)
    torch.nn.init.normal_(init_log_sigma_w, mean=log_sigma_mean, std=log_sigma_std)
    self.log_sigma_w = nn.Parameter(init_log_sigma_w)

    init_log_sigma_b = torch.empty(out_dim)
    torch.nn.init.normal_(init_log_sigma_b, mean=log_sigma_mean, std=log_sigma_std)
    self.log_sigma_b = nn.Parameter(init_log_sigma_b)

    self.prior_mu_w = torch.zeros_like(self.mu_w, device=device)
    self.prior_sigma_w = torch.ones_like(self.log_sigma_w, device=device)
    self.prior_mu_b = torch.zeros_like(self.mu_b, device=device)
    self.prior_sigma_b = torch.ones_like(self.log_sigma_b, device=device)

  def forward(self, x, flow, deterministic=False):
    z0 = torch.randn(flow.dim_z, device=x.device)
    zt, _ = flow(z0)
    # TODO: not sure about biases, do we want to flow them?
    mu_out = F.linear(x * zt, self.mu_w, self.mu_b)
    sigma_out = torch.sqrt(
      F.linear(x**2, torch.exp(2 * self.log_sigma_w), torch.exp(2 * self.log_sigma_b))
    )
    eps = torch.randn_like(mu_out, device=x.device)
    return mu_out + sigma_out * eps

  def kl_layer(self):
    pass

  def update_prior_layer(self):
    self.prior_mu_w = self.mu_w.clone().detach()
    self.prior_sigma_w = torch.exp(self.log_sigma_w.clone().detach())
    self.prior_mu_b = self.mu_b.clone().detach()
    self.prior_sigma_b = torch.exp(self.log_sigma_b.clone().detach())
