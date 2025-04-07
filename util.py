import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


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


def show_imgs(imgs):
  # single flattened image (784,)
  if torch.is_tensor(imgs) and imgs.ndim == 1 and imgs.shape[0] == 784:
    grid = [[imgs.reshape(28, 28)]]
  # single non-flattened image (from torchvision.utils.make_grid)
  # need to move channels to last axis for imshow
  elif torch.is_tensor(imgs):
    grid = [[imgs.permute(1, 2, 0)]]
  # list of flattened images
  elif (
    isinstance(imgs, list)
    and torch.is_tensor(imgs[0])
    and imgs[0].ndim == 1
    and imgs[0].shape[0] == 784
  ):
    grid = [[img.reshape(28, 28) for img in imgs]]
  # list containing two batches of flattened images where each batch: (x, 784)
  elif (
    isinstance(imgs, list)
    and torch.is_tensor(imgs[0])
    and imgs[0].ndim == 2
    and imgs[0].shape[1] == 784
  ):
    grid = [[img.reshape(28, 28) for img in batch] for batch in imgs]
  else:
    raise ValueError('unsupported input format')

  rows, cols = len(grid), len(grid[0])
  fig, axes = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)
  for i in range(rows):
    for j in range(cols):
      axes[i][j].imshow(grid[i][j], cmap='gray')
      axes[i][j].axis('off')
  plt.tight_layout()
  plt.show()


def samples(generative_model, upto_task, multihead):
  device = torch_device()
  images = []
  for t in range(upto_task + 1):
    if multihead:
      gen = generative_model.sample(torch.tensor([t], device=device))
    else:
      gen = generative_model.sample()
    images.append(gen.reshape(1, 28, 28))
  img = make_grid(images, nrow=10, padding=0).cpu()
  # show_imgs(img)
  return img


@torch.no_grad()
def reconstructions(generative_model, loaders, upto_task, multihead):
  device = torch_device()
  generative_model.eval()
  origs, recons = [], []
  for t in range(upto_task + 1):
    loader = loaders[t]
    data, task = next(iter(loader))
    data, task = data[0:1].to(device), task[0:1].to(device)
    recon, _, _ = generative_model(data, task) if multihead else generative_model(data)
    origs.append(data)
    recons.append(recon)
  origs = torch.cat(origs, dim=0)
  recons = torch.cat(recons, dim=0)
  img = make_grid(
    torch.cat(
      [
        origs.reshape(upto_task + 1, 1, 28, 28),
        recons.reshape(upto_task + 1, 1, 28, 28),
      ]
    ),
    nrow=upto_task + 1,
    padding=0,
  ).cpu()
  # show_imgs(img)
  return img
