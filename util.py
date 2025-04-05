import torch
import numpy as np
import matplotlib.pyplot as plt


def torch_version():
  print(
    'using torch version',
    torch.__version__,
    'running on',
    torch.ones(1, device=torch_device()).device,
  )


def torch_device(enable_mps=False, enable_cuda=True):
  if torch.backends.mps.is_available() and enable_mps:
    device = torch.device('mps')
  elif torch.cuda.is_available() and enable_cuda:
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  return device


# reproducibility
def seed(seed=0):
  np.random.seed(seed)
  torch.manual_seed(seed)
  # torch.use_deterministic_algorithms(True)


def kl_div_gaussians(mu_1, sigma_1, mu_2, sigma_2):
  return torch.sum(
    torch.log(sigma_2 / sigma_1)
    + (sigma_1**2 + (mu_1 - mu_2) ** 2) / (2 * sigma_2**2)
    - 1 / 2
  )


def visualize_sample_img(loader):
  images, labels = next(iter(loader))
  first_img = images[0].reshape(28, 28)
  plt.figure(figsize=(1, 1))
  plt.imshow(first_img, cmap='gray')
  plt.title(labels[0].item())
  plt.axis('off')
  plt.show()


def plot_samples(generative_model, multihead):
  device = torch_device()
  axes = plt.subplots(1, 10, figsize=(10, 1))[1]
  for d in range(10):
    if multihead:
      sample = generative_model.sample(torch.tensor([d], device=device)).reshape(28, 28)
    else:
      sample = generative_model.sample().reshape(28, 28)
    axes[d].imshow(sample, cmap='gray')
    axes[d].axis('off')
  plt.show()


@torch.no_grad()
def plot_reconstructions(generative_model, loader, multihead):
  device = torch_device()
  generative_model.eval()
  data, task = next(iter(loader))
  data, task = data[:10], task[:10]
  data, task = data.to(device), task.to(device)
  recon, _, _ = generative_model(data, task) if multihead else generative_model(data)
  axes = plt.subplots(2, 10, figsize=(10, 2))[1]
  for i in range(10):
    axes[0, i].imshow(data[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(recon[i].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
