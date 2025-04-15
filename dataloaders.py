import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
from copy import copy


def transform_base():
  return transforms.Compose(
    [
      # no-op for mnist, but relevant for ImageFolder datasets
      transforms.ToImage(),
      # the ImageFolder PIL loader converts pngs to rgb. greyscale ones
      # get their channel duplicated into three identical channels
      # transforms.Grayscale(num_output_channels=1) then applies a weighted
      # average because e.g. green is more important in true colour images.
      # so in the end we get the original channel value back
      transforms.Grayscale(),
      transforms.ToDtype(torch.float32, scale=True),
    ]
  )


def transform(mean=None, std=None):
  if mean is None or std is None:
    return transforms.Compose(
      [
        transform_base(),
        transforms.Lambda(torch.flatten),
      ]
    )
  return transforms.Compose(
    [
      transform_base(),
      transforms.Normalize(mean=[mean], std=[std]),
      transforms.Lambda(torch.flatten),
    ]
  )


# we do not use this except for classification.
# v1.ToTensor == v2.ToImage \circ v2.ToDtype(torch.float32, scale=True)
# maps image data to [0,1]. normalizing this would yield negative values
# allowing slightly better convergence in classifiers with MSE at the expense
# of breaking BCE and making VAE MSE code a lot more cumbersome
# summary:
# VAE + BCE: needs [0,1] (probability interpretation from bernoulli)
# VAE + MSE: ideally normalized, but have to rescale decoder output(!) -> [0,1] ok
# classification (F.cross_entropy): ideally normalized
# regression: allows us to reuse [0,1] targets from classification, also interpretable as class probabilites
def mean_std_image_dataset(dataset: torch.utils.data.Dataset):
  dataset.transform = transform_base()
  t_data = torch.stack([d[0] for d in dataset]).squeeze()
  assert len(t_data.shape) == 3
  return t_data.mean(dim=(0, 1, 2)).item(), t_data.std(dim=(0, 1, 2)).item()


def precomp_mnist_stats(verify=False):
  if verify:
    mean_mnist, std_mnist = mean_std_image_dataset(
      datasets.MNIST('./data', train=True, download=True)
    )
    print(mean_mnist, std_mnist)
  return (0.1307, 0.3081)


def precomp_notmnist_stats(verify=False):
  if verify:
    mean_notmnist_small, std_notmnist_small = mean_std_image_dataset(
      datasets.ImageFolder('./data/notMNIST_small')
    )
    print(mean_notmnist_small, std_notmnist_small)
  return (0.4229, 0.4573)


def mnist_vanilla_task_loaders(batch_size):
  data_train = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transform(),
    target_transform=torch.tensor,
  )
  data_test = datasets.MNIST(
    './data',
    train=False,
    download=True,
    transform=transform(),
    target_transform=torch.tensor,
  )
  train_loader = torch.utils.data.DataLoader(
    data_train,
    batch_size=batch_size if batch_size else max(1, len(data_train)),
    shuffle=True,
    num_workers=12 if torch.cuda.is_available() else 0,
  )
  test_loader = torch.utils.data.DataLoader(
    data_test,
    batch_size=batch_size if batch_size else max(1, len(data_test)),
    num_workers=12 if torch.cuda.is_available() else 0,
  )
  return train_loader, test_loader


def nmnist_vanilla_task_loaders(batch_size):
  train_part = 0.8
  notmnist_data = datasets.ImageFolder(
    './data/notMNIST_small',
    transform=transform(),
  )
  train_subset, test_subset = torch.utils.data.random_split(
    notmnist_data, [train_part, 1 - train_part]
  )
  train_idx = torch.tensor(train_subset.indices)
  test_idx = torch.tensor(test_subset.indices)
  train_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(notmnist_data, train_idx),
    batch_size=batch_size if batch_size else max(1, len(train_idx)),
    shuffle=True,
    num_workers=12 if torch.cuda.is_available() else 0,
  )
  test_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(notmnist_data, test_idx),
    batch_size=batch_size if batch_size else max(1, len(test_idx)),
    shuffle=True,
    num_workers=12 if torch.cuda.is_available() else 0,
  )
  return train_loader, test_loader


def mnist_cont_task_loaders(batch_size):
  loaders, cumulative_test_loaders = [], []
  train_ds = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transform(),
    target_transform=torch.tensor,
  )
  test_ds = datasets.MNIST(
    './data',
    train=False,
    download=True,
    transform=transform(),
    target_transform=torch.tensor,
  )

  for task in range(10):
    train_mask = train_ds.targets == task
    test_mask = test_ds.targets == task
    train_idx = torch.nonzero(train_mask).squeeze()
    test_idx = torch.nonzero(test_mask).squeeze()

    train_loader = torch.utils.data.DataLoader(
      torch.utils.data.Subset(train_ds, train_idx),
      batch_size=batch_size if batch_size else max(1, len(train_idx)),
      shuffle=True,
      num_workers=12 if torch.cuda.is_available() else 0,
    )
    cumulative_test_loaders.append(
      torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_ds, test_idx),
        batch_size=batch_size if batch_size else max(1, len(test_idx)),
        shuffle=True,
        num_workers=12 if torch.cuda.is_available() else 0,
      )
    )
    loaders.append((train_loader, cumulative_test_loaders.copy()))

  return loaders


def nmnist_cont_task_loaders(batch_size):
  train_part = 0.8
  loaders, cumulative_test_loaders = [], []
  notmnist_data = datasets.ImageFolder(
    './data/notMNIST_small',
    transform=transform(),
  )
  notmnist_data_targets = torch.tensor(notmnist_data.targets)
  train_subset, test_subset = torch.utils.data.random_split(
    notmnist_data, [train_part, 1 - train_part]
  )
  train_idx_full = torch.tensor(train_subset.indices)
  train_targets = notmnist_data_targets[train_idx_full]
  test_idx_full = torch.tensor(test_subset.indices)
  test_targets = notmnist_data_targets[test_idx_full]
  for task in range(10):
    train_mask = train_targets == task
    test_mask = test_targets == task
    train_idx_sub = torch.nonzero(train_mask).squeeze()
    test_idx_sub = torch.nonzero(test_mask).squeeze()
    train_idx = train_idx_full[train_idx_sub]
    test_idx = test_idx_full[test_idx_sub]

    train_loader = torch.utils.data.DataLoader(
      torch.utils.data.Subset(notmnist_data, train_idx),
      batch_size=batch_size if batch_size else max(1, len(train_idx)),
      shuffle=True,
      num_workers=12 if torch.cuda.is_available() else 0,
    )

    cumulative_test_loaders.append(
      torch.utils.data.DataLoader(
        torch.utils.data.Subset(notmnist_data, test_idx),
        batch_size=batch_size if batch_size else max(1, len(test_idx)),
        shuffle=True,
        num_workers=12 if torch.cuda.is_available() else 0,
      )
    )
    loaders.append((train_loader, cumulative_test_loaders.copy()))

  return loaders


def pmnist_task_loaders(batch_size, regression=False):
  mean_mnist, std_mnist = precomp_mnist_stats()
  num_tasks = 10

  def transform_permute(idx):
    return transforms.Compose(
      [
        transform(mean_mnist, std_mnist),
        # nasty pytorch bug: if we try to use num_workers > 0
        # on macos, pickling this fails, and nothing seems to help
        # things I tried: using dill instead of pickle for the
        # multiprocessing.reduction.ForkingPickler, partial, a class
        transforms.Lambda(lambda x: x[idx]),
      ]
    )

  def transform_target(x):
    x = torch.tensor(x)
    if regression:
      x = F.one_hot(x, num_tasks).float()
    return x

  perms = [torch.randperm(28 * 28) for _ in range(num_tasks)]
  loaders, cumulative_train_loaders, cumulative_test_loaders = [], [], []
  for task in range(num_tasks):
    tf = transform_permute(perms[task])
    data_train = datasets.MNIST(
      './data',
      train=True,
      download=True,
      transform=tf,
      target_transform=transform_target,
    )
    data_test = datasets.MNIST(
      './data',
      train=False,
      download=True,
      transform=tf,
      target_transform=transform_target,
    )
    cumulative_train_loaders.append(
      torch.utils.data.DataLoader(
        data_train,
        batch_size=batch_size if batch_size else max(1, len(data_train)),
        shuffle=True,
        num_workers=12 if torch.cuda.is_available() else 0,
      )
    )
    cumulative_test_loaders.append(
      torch.utils.data.DataLoader(
        data_test,
        batch_size=batch_size if batch_size else max(1, len(data_test)),
        num_workers=12 if torch.cuda.is_available() else 0,
      )
    )
    loaders.append((cumulative_train_loaders.copy(), cumulative_test_loaders.copy()))
  return loaders


def collate_add_task(task):
  def collate_fn(batch):
    data_list, target_list = zip(*batch)
    data_batch = torch.stack(data_list, 0)
    target_batch = torch.stack(target_list, 0)
    task_tensor = torch.full((len(batch),), task)
    return data_batch, target_batch, task_tensor

  return collate_fn


def transform_label(class_a, class_b, onehot):
  if onehot:
    return lambda x: F.one_hot(torch.tensor(0 if x == class_a else 1), 2).float()
  else:
    return lambda x: torch.tensor(0 if x == class_a else 1)


def splitmnist_task_loaders(batch_size, regression=False):
  mean_mnist, std_mnist = precomp_mnist_stats()

  loaders, cumulative_train_loaders, cumulative_test_loaders = [], [], []

  # 5 classification tasks
  for task, (a, b) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]):
    train_ds = datasets.MNIST(
      './data',
      train=True,
      download=True,
      transform=transform(mean_mnist, std_mnist),
      target_transform=transform_label(a, b, regression),
    )
    test_ds = datasets.MNIST(
      './data',
      train=False,
      download=True,
      transform=transform(mean_mnist, std_mnist),
      target_transform=transform_label(a, b, regression),
    )
    # only include the two digits for this task
    train_mask = (train_ds.targets == a) | (train_ds.targets == b)
    test_mask = (test_ds.targets == a) | (test_ds.targets == b)
    train_idx = torch.nonzero(train_mask).squeeze()
    test_idx = torch.nonzero(test_mask).squeeze()
    collate_fn = collate_add_task(task)
    cumulative_train_loaders.append(
      torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_ds, train_idx),
        batch_size=batch_size if batch_size else max(1, len(train_idx)),
        shuffle=True,
        num_workers=12 if torch.cuda.is_available() else 0,
        collate_fn=collate_fn,
      )
    )
    cumulative_test_loaders.append(
      torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_ds, test_idx),
        batch_size=batch_size if batch_size else max(1, len(test_idx)),
        shuffle=True,
        num_workers=12 if torch.cuda.is_available() else 0,
        collate_fn=collate_fn,
      )
    )
    loaders.append((cumulative_train_loaders.copy(), cumulative_test_loaders.copy()))

  return loaders


def notmnist_task_loaders(batch_size, regression=False):
  mean_notmnist_small, std_notmnist_small = precomp_notmnist_stats()

  train_part = 0.8
  loaders, cumulative_train_loaders, cumulative_test_loaders = [], [], []
  notmnist_data = datasets.ImageFolder(
    './data/notMNIST_small',
    transform=transform(mean_notmnist_small, std_notmnist_small),
  )
  notmnist_data_targets = torch.tensor(notmnist_data.targets)
  train_subset, test_subset = torch.utils.data.random_split(
    notmnist_data, [train_part, 1 - train_part]
  )
  train_idx_full = torch.tensor(train_subset.indices)
  train_targets = notmnist_data_targets[train_idx_full]
  test_idx_full = torch.tensor(test_subset.indices)
  test_targets = notmnist_data_targets[test_idx_full]

  # 5 classification tasks:
  # (A,B), (C,D), (E,F), (G,H), (I,J) -> (0,1), (2,3), (4,5), (6,7), (8,9)
  for task, (a, b) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]):
    train_mask = (train_targets == a) | (train_targets == b)
    test_mask = (test_targets == a) | (test_targets == b)
    train_idx_sub = torch.nonzero(train_mask).squeeze()
    test_idx_sub = torch.nonzero(test_mask).squeeze()

    # map indicies back to original
    train_idx = train_idx_full[train_idx_sub]
    test_idx = test_idx_full[test_idx_sub]

    # shallow copy to un-share the target_transform, sharing the underlying data
    train_ds = copy(notmnist_data)
    train_ds.target_transform = transform_label(a, b, regression)
    test_ds = copy(notmnist_data)
    test_ds.target_transform = transform_label(a, b, regression)

    collate_fn = collate_add_task(task)
    cumulative_train_loaders.append(
      torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_ds, train_idx),
        batch_size=batch_size if batch_size else max(1, len(train_idx)),
        shuffle=True,
        num_workers=12 if torch.cuda.is_available() else 0,
        collate_fn=collate_fn,
      )
    )
    cumulative_test_loaders.append(
      torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_ds, test_idx),
        batch_size=batch_size if batch_size else max(1, len(test_idx)),
        shuffle=True,
        num_workers=12 if torch.cuda.is_available() else 0,
        collate_fn=collate_fn,
      )
    )
    loaders.append((cumulative_train_loaders.copy(), cumulative_test_loaders.copy()))

  return loaders
