import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import wandb
from typing import Optional, List, Set, Any

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

def transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(torch.flatten)
    ])


def pmnist_task_loaders():
    def transform_permute(idx):
        return transforms.Compose([
            transform(),
            # nasty pytorch bug: if we try to use num_workers > 0
            # on macos, pickling this fails, and nothing seems to help
            # things I tried: using dill instead of pickle for the
            # multiprocessing.reduction.ForkingPickler, partial, a class
            transforms.Lambda(lambda x: x[idx])
        ])

    num_tasks = 10
    batch_size = 256
    perms = [torch.randperm(28 * 28) for _ in range(num_tasks)]
    loaders, cumulative_train_loaders, cumulative_test_loaders = [], [], []
    for task in range(num_tasks):
        tf = transform_permute(perms[task])
        data_train = datasets.MNIST('./data', train=True, download=True, transform=tf, target_transform=torch.tensor)
        data_test = datasets.MNIST('./data', train=False, download=True, transform=tf, target_transform=torch.tensor)
        cumulative_train_loaders.append(torch.utils.data.DataLoader(
            data_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12 if torch.cuda.is_available() else 0
        ))
        cumulative_test_loaders.append(torch.utils.data.DataLoader(
          data_test,
          batch_size=batch_size,
          num_workers=12 if torch.cuda.is_available() else 0
        ))
        loaders.append((cumulative_train_loaders.copy(), cumulative_test_loaders.copy()))
    return loaders

def visualize_sample_img(loader):
    images, labels = next(iter(loader))
    first_img = images[0].reshape(28, 28)
    plt.figure(figsize=(1, 1))
    plt.imshow(first_img, cmap='gray')
    plt.title(labels[0].item())
    plt.axis('off')
    plt.show()

def splitmnist_task_loaders():
    loaders, cumulative_test_loaders = [], []

    def transform_onehot(class_a, class_b):
        one_hot = {class_a: torch.tensor([1., 0.]), class_b: torch.tensor([0., 1.])}
        return lambda x: one_hot[int(x)]

    def collate_add_task(task_idx):
      def collate_fn(batch):
          data_list, target_list = zip(*batch)
          data_batch = torch.stack(data_list, 0)
          target_batch = torch.stack(target_list, 0)
          task_idx_tensor = torch.full((len(batch),), task_idx)
          return data_batch, target_batch, task_idx_tensor
      return collate_fn

    # 5 classification tasks
    for task, (a, b) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]):
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform(), target_transform=transform_onehot(a, b))
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform(), target_transform=transform_onehot(a, b))
        # only include the two digits for this task
        train_mask = (train_ds.targets == a) | (train_ds.targets == b)
        test_mask = (test_ds.targets == a) | (test_ds.targets == b)
        train_idx = torch.nonzero(train_mask).squeeze()
        test_idx = torch.nonzero(test_mask).squeeze()
        collate_fn = collate_add_task(task)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_ds, train_idx),
            batch_size=len(train_idx),
            shuffle=True,
            num_workers=12 if torch.cuda.is_available() else 0,
            collate_fn=collate_fn
        )
        cumulative_test_loaders.append(torch.utils.data.DataLoader(
          torch.utils.data.Subset(test_ds, test_idx),
          batch_size=len(test_idx),
          shuffle=True,
          num_workers=12 if torch.cuda.is_available() else 0,
          collate_fn=collate_fn
        ))
        loaders.append((train_loader, cumulative_test_loaders.copy()))
    return loaders

class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.linear(x)

    def train_epoch(self, loader, loss_fn, opt):
        device = torch_device()
        self.train()
        losses, accuracies = [], []
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            pred = self(data)
            loss = loss_fn(pred, target)
            acc = accuracy(pred, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            accuracies.append(acc.item())
        return np.mean(losses), np.mean(accuracies)

    def train_run(self, train_loader, test_loader, num_epochs=100):
        loss_fn = F.cross_entropy
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        for i in trange(num_epochs, desc='pretrain'):
            train_loss, train_acc = self.train_epoch(train_loader, loss_fn, opt)
            test_loss, test_acc = self.test_run(test_loader, loss_fn)
        print(f"after epoch {num_epochs}: train loss {train_loss:.4f} train acc {train_acc:.4f} test loss {test_loss:.4f} test acc {test_acc:.4f}")

    @torch.no_grad()
    def test_run(self, loader, loss_fn):
        device = torch_device()
        self.eval()
        losses, accuracies = [], []
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = self(data)
            loss = loss_fn(pred, target)
            acc = accuracy(pred, target)
            losses.append(loss.item())
            accuracies.append(acc.item())
        return np.mean(losses), np.mean(accuracies)

class BayesianLinear(nn.Module):
    def __init__(self, in_dim, out_dim, init_std, init_w=None, init_b=None):
        super().__init__()
        if init_w is None:
            init_w = torch.empty(out_dim, in_dim)
            torch.nn.init.kaiming_normal_(init_w, mode='fan_in', nonlinearity='relu')
        if init_b is None:
            init_b = torch.zeros(out_dim)

        assert init_w.size(dim=0) == out_dim and init_w.size(dim=1) == in_dim
        assert init_b.size(dim=0) == out_dim

        self.mu_w = nn.Parameter(init_w)
        self.log_sigma_w = nn.Parameter(torch.log(init_std * torch.ones(out_dim, in_dim)))
        self.mu_b = nn.Parameter(init_b)
        self.log_sigma_b = nn.Parameter(torch.log(init_std * torch.ones(out_dim)))

        self.prior_mu_w = torch.zeros_like(self.mu_w, device=torch_device())
        self.prior_sigma_w = torch.ones_like(self.log_sigma_w, device=torch_device())
        self.prior_mu_b = torch.zeros_like(self.mu_b, device=torch_device())
        self.prior_sigma_b = torch.ones_like(self.log_sigma_b, device=torch_device())

    def forward(self, x):
        mu_out = F.linear(x, self.mu_w, self.mu_b)
        sigma_out = torch.sqrt(F.linear(x**2, torch.exp(2 * self.log_sigma_w), torch.exp(2 * self.log_sigma_b)))
        # standard normal
        eps = torch.randn_like(mu_out)
        return mu_out + sigma_out * eps

class Ddm(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        batch_size=256,
        layer_init_std=1e-3,
        per_task_opt=True,
        bayesian_test_samples=1,
        bayesian_train_samples=1,
        coreset_size=0,
        mle=None,
        multihead=False,
        logging_every=10
    ):
        super().__init__()
        self.logging_every = logging_every
        self.batch_size = batch_size
        self.per_task_opt = per_task_opt
        self.bayesian_test_samples = bayesian_test_samples
        self.bayesian_train_samples = bayesian_train_samples
        self.coreset_size = coreset_size
        self.multihead = multihead
        self.layers = nn.Sequential(
            BayesianLinear(
                in_dim,
                hidden_dim,
                layer_init_std,
                init_w=mle.linear[0].weight if mle else None,
                init_b=mle.linear[0].bias if mle else None
            ),
            nn.ReLU(),
            BayesianLinear(
                hidden_dim,
                hidden_dim,
                layer_init_std,
                init_w=mle.linear[2].weight if mle else None,
                init_b=mle.linear[2].bias if mle else None
            ),
            nn.ReLU(),
            BayesianLinear(
                hidden_dim,
                out_dim,
                layer_init_std,
                init_w=mle.linear[4].weight if mle else None,
                init_b=mle.linear[4].bias if mle else None
            ),
        )

    def update_prior(self):
        for bl in self.layers:
            if isinstance(bl, BayesianLinear):
                bl.prior_mu_w = bl.mu_w.clone().detach()
                bl.prior_sigma_w = torch.exp(bl.log_sigma_w.clone().detach())
                bl.prior_mu_b = bl.mu_b.clone().detach()
                bl.prior_sigma_b = torch.exp(bl.log_sigma_b.clone().detach())

    def restore_from_prior(self):
        for bl in self.layers:
            if isinstance(bl, BayesianLinear):
                bl.mu_w.data = bl.prior_mu_w.clone().detach()
                bl.log_sigma_w.data = torch.log(bl.prior_sigma_w.clone().detach())
                bl.mu_b.data = bl.prior_mu_b.clone().detach()
                bl.log_sigma_b.data = torch.log(bl.prior_sigma_b.clone().detach())

    @staticmethod
    def kl_div_gaussians(mu_1, sigma_1, mu_2, sigma_2):
        return torch.sum(torch.log(sigma_2 / sigma_1) + (sigma_1**2 + (mu_1 - mu_2)**2)/(2*sigma_2**2) - 1/2)

    def compute_kl(self):
        res = 0
        for bl in self.layers:
            if isinstance(bl, BayesianLinear):
                res += self.kl_div_gaussians(
                    bl.mu_w, torch.exp(bl.log_sigma_w), bl.prior_mu_w, bl.prior_sigma_w
                )
                res += self.kl_div_gaussians(
                    bl.mu_b, torch.exp(bl.log_sigma_b), bl.prior_mu_b, bl.prior_sigma_b
                )
        return res

    def forward(self, x):
        return self.layers(x)

    # returns the first component of L_SGVB, without the N factor
    @staticmethod
    def sgvb_mc(pred, target):
        # we classification task, so target is a index to the right class
        # apply softmax:
        # p(target | pred) = exp(pred_target) / sum_{i=0}^len(pred) exp(pred_i)
        # according to (3) in the local reparam paper, we apply the log,
        # and arrive at the l_n from https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        # we use reduction=mean (the default) to get the 1/M factor and the outer sum,
        # and finally arrive at eq (3) without the N factor
        return -F.cross_entropy(pred, target)

    def train_epoch(self, loader, opt, task, epoch):
        device = torch_device()
        for batch, batch_data in enumerate(loader):
            if self.multihead:
              data, target, t = batch_data
            else:
              data, target = batch_data
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            preds = [self(data) for _ in range(self.bayesian_train_samples)]
            mean_pred = torch.stack(preds).mean(0)
            losses = torch.stack([self.sgvb_mc(pred, target) for pred in preds])
            loss = -losses.mean(0) + self.compute_kl() / len(loader.dataset)
            acc = accuracy(mean_pred, target)
            if batch % self.logging_every == 0:
                wandb.log({'task': task, 'epoch': epoch, 'train_loss': loss, 'train_acc': acc})
                # log tensors
                for bli, bl in enumerate(self.layers):
                  if isinstance(bl, BayesianLinear):
                      wandb.log({
                          f'{bli}_sigma_w': torch.std(torch.exp(bl.log_sigma_w)).detach().item(),
                      })
            loss.backward()
            opt.step()

    # sample from (D_t) \cup C_{t-1}
    def select_coreset(self, task_size, old_coreset: Optional[List[Set[int]]]) -> List[Set[int]]:
        assert self.coreset_size <= task_size
        if not old_coreset:
            return [set(np.random.permutation(np.arange(0, task_size))[:self.coreset_size])]
        covered_tasks = len(old_coreset)
        strat_size = len(old_coreset[0])
        for task in old_coreset:
            assert len(task) == strat_size
        assert self.coreset_size // covered_tasks == strat_size
        new_strat_size = self.coreset_size // (covered_tasks + 1)
        new_coreset = []
        for task in old_coreset:
            new_coreset.append(set(np.random.choice(list(task), new_strat_size, replace=False)))
        new_coreset.append(set(np.random.choice(range(0, task_size), new_strat_size, replace=False)))
        return new_coreset

    # returns (D_t \cup C_{t-1}) - C_t = (D_t - C_t) \cup (C_{t-1} - C_t)
    @staticmethod
    def select_augmented_complement(task_size, old_coreset: Optional[List[Set[int]]], new_coreset: List[Set[int]]) -> List[Set[int]]:
        complement: List[Set[int]] = []
        if not old_coreset:
            assert len(new_coreset) == 1
            return [set(range(0, task_size)) - new_coreset[0]]
        for i in range(len(new_coreset)):
            if i == len(new_coreset) - 1:
                strat = set(range(0, task_size)) - new_coreset[i]
            else:
                strat = set(old_coreset[i]) - new_coreset[i]
            complement.append(strat)
        return complement

    # all_data is a list of dataloaders
    def create_dataloader(self, indexes: List[Set[int]], all_data: List[Any]):
        assert len(all_data) >= len(indexes)
        data = [all_data[i].dataset[j] for i, idx_set in enumerate(indexes) for j in idx_set]
        if self.multihead:
          dataset = torch.utils.data.TensorDataset(
              torch.stack([p[0] for p in data]) if data else torch.empty(0),
              torch.stack([p[1] for p in data]) if data else torch.empty(0),
              torch.stack([p[2] for p in data]) if data else torch.empty(0),
          )
        else:
          dataset = torch.utils.data.TensorDataset(
              torch.stack([p[0] for p in data]) if data else torch.empty(0),
              torch.stack([p[1] for p in data]) if data else torch.empty(0)
          )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size if self.batch_size else max(1, len(dataset)),
            shuffle=True if data else False,
            num_workers=12 if torch.cuda.is_available() else 0
        )

    def train_test_run(self, tasks, num_epochs=100):
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        wandb.watch(self, log_freq=100)
        old_coreset_idx = []
        for task, (train_loaders, test_loaders) in enumerate(tasks):
            coreset_idx = self.select_coreset(len(train_loaders[-1].dataset), old_coreset_idx)
            complement_idx = self.select_augmented_complement(len(train_loaders[-1].dataset), old_coreset_idx, coreset_idx)
            coreset_loader = self.create_dataloader(coreset_idx, train_loaders)
            complement_loader = self.create_dataloader(complement_idx, train_loaders)
            old_coreset_idx = coreset_idx

            if self.per_task_opt:
                opt = torch.optim.Adam(self.parameters(), lr=1e-3)

            # restore network parameters to \tilde{q}_{t}
            if task > 0:
                self.restore_from_prior()

            # (2)
            # precondition: network prior is \tilde{q}_{t-1}
            # network parameters are whatever
            for epoch in trange(num_epochs, desc=f'task {task+1} phase 1'):
                self.train_epoch(complement_loader, opt, task, epoch)
            # ==> network parameters are \tilde{q}_t

            self.update_prior()
            # network parameters and network prior are \tilde{q}_t

            # (3)
            # precondition: network prior is \tilde{q}_{t}
            for epoch in trange(num_epochs, desc=f'task {task+1} phase 2'):
                self.train_epoch(coreset_loader, opt, task, epoch)
            # ==> network parameters are q_t

            self.test_run(test_loaders, task)

    @torch.no_grad()
    def test_run(self, loaders, task):
        self.eval()
        device = torch_device()
        avg_accuracies = []
        for test_task, loader in enumerate(loaders):
          task_accuracies = []
          for batch, batch_data in enumerate(loader):
              if self.multihead:
                data, target, t = batch_data
              else:
                data, target = batch_data
              data, target = data.to(device), target.to(device)
              # E[argmax_y p(y | theta, x)] != argmax_y E[p(y | \theta, x)]
              # lhs: 1 sample; rhs: can get better approximation via MC
              # 1 sample is unbiased for the p(y | \theta, x), but argmax breaks this
              preds = [self(data) for _ in range(self.bayesian_test_samples)]
              mean_pred = torch.stack(preds).mean(0)
              acc = accuracy(mean_pred, target)
              task_accuracies.append(acc.item())
          task_accuracy = np.mean(task_accuracies)
          wandb.log({'task': task, f'test_acc_task_{test_task}': task_accuracy})
          avg_accuracies.append(task_accuracy)
        wandb.log({'task': task, 'test_acc': np.mean(avg_accuracies)})

def accuracy(pred, target):
    # undo one-hot encoding, if applicable
    target_idx = target.argmax(dim=1) if target.ndim == pred.ndim else target
    return (pred.argmax(dim=1) == target_idx).sum() / pred.shape[0]

def model_pipeline(params):
    with wandb.init(project='vcl', config=params):
        params = wandb.config
        if params.problem == 'pmnist':
          baseline_loaders = pmnist_task_loaders()[0]
          loaders = pmnist_task_loaders()
        elif params.problem == 'smnist':
          baseline_loaders = splitmnist_task_loaders()[0]
          loaders = splitmnist_task_loaders()
        else:
          loaders, baseline_loaders = None, None

        if params.pretrain_epochs > 0:
          mle = Net(params.in_dim, params.hidden_dim, params.out_dim).to(torch_device())
          mle = torch.compile(mle)
          mle.train_run(baseline_loaders[0][0], baseline_loaders[1][0], params.pretrain_epochs)
        else:
          mle = None

        model = Ddm(params.in_dim, params.hidden_dim, params.out_dim,
                    batch_size=params.batch_size,
                    layer_init_std=params.layer_init_std,
                    per_task_opt=params.per_task_opt,
                    bayesian_test_samples=params.bayesian_test_samples,
                    bayesian_train_samples=params.bayesian_train_samples,
                    coreset_size=params.coreset_size,
                    multihead=params.multihead,
                    mle=mle
        ).to(torch_device())
        model = torch.compile(model)
        model.train_test_run(loaders, num_epochs=params.epochs)
    return model

if __name__ == '__main__':
  wandb.login()

  print("torch version", torch.__version__, "running on", torch.ones(1, device=torch_device()).device)
  seed(0)

  ddm_pmnist_run = dict(
      in_dim=28*28,
      hidden_dim=100,
      out_dim=10,
      epochs=100,
      batch_size=256,
      pretrain_epochs=10,
      coreset_size=5000,
      per_task_opt=False,
      layer_init_std=1e-8,
      bayesian_test_samples=100,
      bayesian_train_samples=10,
      multihead=False,
      problem='pmnist',
      model='vcl'
  )

  ddm_smnist_run = dict(
    in_dim=28*28,
    hidden_dim=256,
    out_dim=2,
    epochs=120,
    batch_size=None,
    pretrain_epochs=10,
    coreset_size=0,
    per_task_opt=False,
    layer_init_std=1e-6,
    bayesian_test_samples=100,
    bayesian_train_samples=10,
    multihead=True,
    problem='smnist',
    model='vcl'
  )

  model = model_pipeline(ddm_pmnist_run)
  # model = model_pipeline(ddm_smnist_run)
