import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class BayesianLinear(nn.Module):
  def __init__(self, in_dim, out_dim, log_sigma_w_mean, log_sigma_w_std):
    super().__init__()
    device = util.torch_device()
    self.in_dim = in_dim
    self.out_dim = out_dim

    # posteriors, mu - possibly overwritten with mle
    init_mu_w = torch.empty(out_dim, in_dim)
    # approx. equivalent to mean=0, std=0.1 => most values are in [-0.1, 0.1]
    torch.nn.init.kaiming_normal_(init_mu_w, nonlinearity='relu')
    self.mu_w = nn.Parameter(init_mu_w)
    # kaiming already gives us reasonable variance
    init_mu_b = torch.zeros(out_dim)
    self.mu_b = nn.Parameter(init_mu_b)

    # posteriors, sigma
    # for different std values we will get 99.7% of the distribution ...
    # 1e-1: +-30% below/above
    # 1e-2: +-3% below/above
    # 1e-3: +. 0.3% below/above
    # whereas log mean ranges from 0 to -27 => mean from 1 to 1e-12
    # gives us a lognormal distribution over sigma_w
    # (which is ok since it needs to be >0)
    init_log_sigma_w = torch.empty(out_dim, in_dim)
    torch.nn.init.normal_(
      init_log_sigma_w,
      mean=log_sigma_w_mean,
      std=log_sigma_w_std,
    )
    self.log_sigma_w = nn.Parameter(init_log_sigma_w)
    # we want sigma_b to be close to 0, similar to classic NN bias init
    init_log_sigma_b = torch.log(1e-10 * torch.ones(out_dim))
    self.log_sigma_b = nn.Parameter(init_log_sigma_b)

    # priors
    self.prior_mu_w = torch.zeros_like(self.mu_w, device=device)
    self.prior_sigma_w = torch.ones_like(self.log_sigma_w, device=device)
    self.prior_mu_b = torch.zeros_like(self.mu_b, device=device)
    self.prior_sigma_b = torch.ones_like(self.log_sigma_b, device=device)

  def forward(self, x):
    mu_out = F.linear(x, self.mu_w, self.mu_b)
    sigma_out = torch.sqrt(
      F.linear(x**2, torch.exp(2 * self.log_sigma_w), torch.exp(2 * self.log_sigma_b))
    )
    # standard normal
    eps = torch.randn_like(mu_out, device=x.device)
    return mu_out + sigma_out * eps

  def kl_layer(self):
    return util.kl_div_gaussians(
      self.mu_w, torch.exp(self.log_sigma_w), self.prior_mu_w, self.prior_sigma_w
    ) + util.kl_div_gaussians(
      self.mu_b, torch.exp(self.log_sigma_b), self.prior_mu_b, self.prior_sigma_b
    )

  def update_prior_layer(self):
    self.prior_mu_w = self.mu_w.clone().detach()
    self.prior_sigma_w = torch.exp(self.log_sigma_w.clone().detach())
    self.prior_mu_b = self.mu_b.clone().detach()
    self.prior_sigma_b = torch.exp(self.log_sigma_b.clone().detach())

  def restore_from_prior_layer(self):
    self.mu_w.data = self.prior_mu_w.clone().detach()
    self.log_sigma_w.data = torch.log(self.prior_sigma_w.clone().detach())
    self.mu_b.data = self.prior_mu_b.clone().detach()
    self.log_sigma_b.data = torch.log(self.prior_sigma_b.clone().detach())
