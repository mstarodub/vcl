import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import wandb
from copy import deepcopy


# cursed (and ruff formatting makes it ugly)
class infix_operator:
  def __init__(self, function, left=None, right=None):
    self.function = function
    self.left = left
    self.right = right

  def __call__(self, *args, **kwargs):
    return self.function(*args, **kwargs)

  def __rmul__(self, left):
    if self.right is None:
      if self.left is None:
        return infix_operator(self.function, left=left)
      else:
        raise SyntaxError('Infix operator already has its left argument')
    else:
      return self.function(left, self.right)

  def __mul__(self, right):
    if self.left is None:
      if self.right is None:
        return infix_operator(self.function, right=right)
      else:
        raise SyntaxError('Infix operator already has its right argument')
    else:
      return self.function(self.left, right)


# '|' for dictionaries is not recursive!
# {'x': {'y': 1}} | {'x': {'z': 2}} == {'x': {'z': 2}}
# but
# {'x': {'y': 1}} *dict_merge* {'x': {'z': 2}} == {'x': {'y': 1, 'z': 2}}
@infix_operator
def dict_merge(a: dict, b: dict) -> dict:
  res = deepcopy(a)
  for bk, bv in b.items():
    av = res.get(bk)
    if isinstance(av, dict) and isinstance(bv, dict):
      res[bk] = dict_merge(av, bv)
    else:
      res[bk] = deepcopy(bv)
  return res


def torch_version():
  print(
    'using torch version',
    torch.__version__,
    'running on',
    torch.ones(1, device=torch_device()).device,
  )


def torch_device(enable_cuda=True):
  if torch.cuda.is_available() and enable_cuda:
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


def wandb_setup_axes():
  # custom x axis
  wandb.define_metric('task')

  # general metrics
  for metric in [
    'test/test_acc',
    'test/test_uncert',
    'samples',
    'recons',
    'cumulative_samples',
    # 'epoch',
    # 'train/train_uncert',
    # 'train/train_loss',
    # 'train/train_acc',
  ]:
    wandb.define_metric(metric, step_metric='task')

  # heads and per-task stats
  # vcl_disc: hi = 0,...,ntasks-1
  # vcl_gen: hi: 0,...,ntasks-1, hli: 0,1
  max_ntasks = 10
  for task in range(max_ntasks):
    wandb.define_metric(f'test/test_acc_task_{task}', step_metric='task')
    wandb.define_metric(f'test/test_uncert_task_{task}', step_metric='task')
    # wandb.define_metric(f'sigma/h_{task}_sigma_w', step_metric='task')
    # wandb.define_metric(f'sigma/h_{task}_{0}_sigma_w', step_metric='task')
    # wandb.define_metric(f'sigma/h_{task}_{1}_sigma_w', step_metric='task')

  # shared layers
  # vcl_disc: 0,...,self.hidden_layers-1
  # vcl_gen: 0,1
  max_hidden_layers = 4
  for sli in range(max_hidden_layers):
    # wandb.define_metric(f'sigma/s_{sli}_sigma_w', step_metric='task')
    pass
