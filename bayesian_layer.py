import torch
import torch.nn as nn
import torch.nn.functional as F

from util import torch_device


class BayesianLinear(nn.Module):
  def __init__(self, in_dim, out_dim, init_std):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim

    # posteriors
    init_w = torch.empty(out_dim, in_dim)
    # torch.nn.init.kaiming_normal_(init_w, mode='fan_in', nonlinearity='relu')
    torch.nn.init.normal_(init_w, mean=0, std=0.1)
    self.mu_w = nn.Parameter(init_w)
    # init_b = torch.zeros(out_dim)
    init_b = torch.empty(out_dim)
    torch.nn.init.normal_(init_b, mean=0, std=0.1)
    self.mu_b = nn.Parameter(init_b)
    # self.log_sigma_w = nn.Parameter(torch.log(init_std * torch.ones(out_dim, in_dim)))
    init_sig_w = torch.empty(out_dim, in_dim)
    torch.nn.init.normal_(init_sig_w, mean=-3, std=0.1)
    self.log_sigma_w = nn.Parameter(init_sig_w)
    # self.log_sigma_b = nn.Parameter(torch.log(init_std * torch.ones(out_dim)))
    init_sig_b = torch.empty(out_dim)
    torch.nn.init.normal_(init_sig_b, mean=-3, std=0.1)
    self.log_sigma_b = nn.Parameter(init_sig_b)

    assert init_w.size(dim=0) == out_dim and init_w.size(dim=1) == in_dim
    assert init_b.size(dim=0) == out_dim

    # priors
    self.prior_mu_w = torch.zeros_like(self.mu_w, device=torch_device())
    self.prior_sigma_w = torch.ones_like(self.log_sigma_w, device=torch_device())
    self.prior_mu_b = torch.zeros_like(self.mu_b, device=torch_device())
    self.prior_sigma_b = torch.ones_like(self.log_sigma_b, device=torch_device())

  def forward(self, x):
    mu_out = F.linear(x, self.mu_w, self.mu_b)
    sigma_out = torch.sqrt(
      F.linear(x**2, torch.exp(2 * self.log_sigma_w), torch.exp(2 * self.log_sigma_b))
    )
    # standard normal
    eps = torch.randn_like(mu_out, device=x.device)
    return mu_out + sigma_out * eps
