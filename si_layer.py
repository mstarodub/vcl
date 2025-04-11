import torch
import torch.nn as nn

from util import torch_device


class SILayer(nn.Linear):
  def __init__(self, in_dim, out_dim):
    super().__init__(in_dim, out_dim)
    self.in_dim = in_dim
    self.out_dim = out_dim
    device = torch_device()
    # tilde{\theta}
    self.weight_old_task = self.weight.clone().detach()
    self.bias_old_task = self.bias.clone().detach()

    # needed to compute theta' * dt in the integral, is updated per batch
    self.weight_old = self.weight.clone().detach()
    self.bias_old = self.bias.clone().detach()

    # w_k
    self.importance_weight = torch.zeros_like(self.weight, device=device)
    self.importance_bias = torch.zeros_like(self.bias, device=device)

    # Omega_k
    self.omega_weight = torch.zeros_like(self.weight, device=device)
    self.omega_bias = torch.zeros_like(self.bias, device=device)

  def surrogate_layer(self):
    return torch.sum(
      self.omega_weight * (self.weight_old_task - self.weight) ** 2
    ) + torch.sum(self.omega_bias * (self.bias_old_task - self.bias) ** 2)

  # update w_k
  @torch.no_grad()
  def update_importance(self):
    # unsure about this. we approximate the parameter update by just taking the difference,
    # effectively pushing the 1/step_size factor into the c
    # however, adam step size is not constant which might pose an issue
    self.importance_weight += -self.weight.grad * (self.weight - self.weight_old)
    self.importance_bias += -self.bias.grad * (self.bias - self.bias_old)

    self.weight_old = self.weight.clone().detach()
    self.bias_old = self.bias.clone().detach()

  @torch.no_grad()
  def update_omega(self, xi):
    delta_weight = self.weight - self.weight_old_task
    self.weight_old_task = self.weight.clone().detach()

    delta_bias = self.bias - self.bias_old_task
    self.bias_old_task = self.bias.clone().detach()

    self.omega_weight += self.importance_weight / (delta_weight**2 + xi)
    self.omega_bias += self.importance_bias / (delta_bias**2 + xi)

    # new task, new set of w_ks
    self.importance_weight.zero_()
    self.importance_bias.zero_()
