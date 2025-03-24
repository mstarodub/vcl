import marimo

__generated_with = "0.11.26"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import torchvision.datasets as datasets
    import torchvision.transforms.v2 as transforms
    from tqdm.auto import tqdm, trange
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    return F, dataclass, datasets, nn, np, plt, torch, tqdm, trange, transforms


@app.cell
def _():
    import wandb
    wandb.login()
    return (wandb,)


@app.cell
def _(np, torch):
    def torch_select_backend(enable_mps=False, enable_cuda=True):
        if torch.backends.mps.is_available() and enable_mps:
            device = torch.device('mps')
        elif torch.cuda.is_available() and enable_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        torch.set_default_device(device)
        print("torch version", torch.__version__, "running on", torch.ones(1).device)

    # reproducibility
    def seed(seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)

    torch_select_backend()
    seed(0)
    return seed, torch_select_backend


@app.cell
def _(datasets, plt, torch, transforms):
    def transform():
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])

    def transform_permute(idx):
        return transforms.Compose([
            transform(),
            transforms.Lambda(lambda x: x[idx])
        ])

    def pmnist_task_loaders():    
        num_tasks = 10
        batch_size = 256
        perms = [torch.randperm(28 * 28) for _ in range(num_tasks)]
        loaders, cumulative_test_datasets = [], []
        for task in range(num_tasks):
            tf = transform_permute(perms[task])
            data_train = datasets.MNIST('./data', train=True, download=True, transform=tf, target_transform=torch.tensor)
            data_test = datasets.MNIST('./data', train=False, download=True, transform=tf, target_transform=torch.tensor)
            train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
            cumulative_test_datasets.append(data_test)
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(cumulative_test_datasets),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True
            )
            loaders.append((train_loader, test_loader))
        return loaders

    def visualize_sample_img(loader):
        images, labels = next(iter(loader))
        first_img = images[0].reshape(28, 28)
        plt.figure(figsize=(1, 1))
        plt.imshow(first_img, cmap='gray')
        plt.title(labels[0].item())
        plt.axis('off')
        plt.show()
    return (
        pmnist_task_loaders,
        transform,
        transform_permute,
        visualize_sample_img,
    )


@app.cell
def _(pmnist_task_loaders, torch, visualize_sample_img):
    # pmnist

    # show the permutation
    visualize_sample_img(pmnist_task_loaders()[0][0])
    # should be 10 classes
    example_loader, _ = pmnist_task_loaders()[0]
    print("unique classes:", torch.unique(example_loader.dataset.targets).tolist())
    return (example_loader,)


@app.cell
def _(datasets, torch, transform):
    def splitmnist_task_loaders():
        batch_size=256
        loaders, cumulative_test_datasets = [], []

        def transform_onehot(class_a, class_b):
            one_hot = {class_a: torch.tensor([1, 0]), class_b: torch.tensor([0, 1])}
            return lambda x: one_hot[int(x)]

        # 5 classification tasks
        for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]:
            train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform(), target_transform=transform_onehot(a, b))
            test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform(), target_transform=transform_onehot(a, b))

            # only include the two digits for this task
            train_mask = (train_ds.targets == a) | (train_ds.targets == b)
            test_mask = (test_ds.targets == a) | (test_ds.targets == b)
            train_idx = torch.nonzero(train_mask).squeeze()
            test_idx = torch.nonzero(test_mask).squeeze()
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_ds, train_idx),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True
            )
            cumulative_test_datasets.append(torch.utils.data.Subset(test_ds, test_idx))
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(cumulative_test_datasets),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True
            )
            loaders.append((train_loader, test_loader))
        return loaders
    return (splitmnist_task_loaders,)


@app.cell
def _(splitmnist_task_loaders):
    # split mnist

    # balanced number of classes per task
    from collections import defaultdict
    example_loaders = splitmnist_task_loaders()
    counts = defaultdict(int)
    for _, y in example_loaders[0][0].dataset:
        counts[tuple(y.tolist())] += 1
    print(counts)
    return counts, defaultdict, example_loaders, y


@app.cell
def _(F, nn, np, pmnist_task_loaders, torch, trange, wandb):
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
            self.train()
            losses, accuracies = [], []
            for data, target in loader:
                data, target = data.to(torch.get_default_device()), target.to(torch.get_default_device())
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
            loss_fn = nn.CrossEntropyLoss()
            opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
            for i in trange(num_epochs, desc='pretrain'):
                train_loss, train_acc = self.train_epoch(train_loader, loss_fn, opt)
                test_loss, test_acc = self.test_run(test_loader, loss_fn)
            print(f"after epoch {num_epochs}: train loss {train_loss:.4f} train acc {train_acc:.4f} test loss {test_loss:.4f} test acc {test_acc:.4f}")

        @torch.no_grad()
        def test_run(self, loader, loss_fn):
            self.eval()
            losses, accuracies = [], []
            for data, target in loader:
                data, target = data.to(torch.get_default_device()), target.to(torch.get_default_device())
                pred = self(data)
                loss = loss_fn(pred, target)
                acc = accuracy(pred, target)
                losses.append(loss.item())
                accuracies.append(acc.item())
            return np.mean(losses), np.mean(accuracies)

    class BayesianLinear(nn.Module):
        def __init__(self, in_dim, out_dim, init_w=None, init_b=None):
            super().__init__()

            if init_w is None:
                init_w = torch.randn(out_dim, in_dim)
            if init_b is None:
                init_b = torch.randn(out_dim)

            init_var = 1e-6
            self.mu_w = nn.Parameter(init_w)
            self.log_sigma_w = nn.Parameter(torch.log(init_var * torch.ones(out_dim, in_dim)))
            self.mu_b = nn.Parameter(init_b)
            self.log_sigma_b = nn.Parameter(torch.log(init_var * torch.ones(out_dim)))

            self.prior_mu_w = torch.zeros_like(self.mu_w)
            self.prior_sigma_w = torch.ones_like(self.log_sigma_w)
            self.prior_mu_b = torch.zeros_like(self.mu_b)
            self.prior_sigma_b = torch.ones_like(self.log_sigma_b)

        # TODO: need a way to pass deterministic from the model (nn.Sequential)
        def forward(self, x, deterministic=False):
            mu_out = F.linear(x, self.mu_w, self.mu_b)
            if deterministic:
                return mu_out
            else:
                sigma_out = torch.sqrt(F.linear(x**2, torch.exp(2 * self.log_sigma_w), torch.exp(2 * self.log_sigma_b)))
                eps = torch.randn_like(mu_out)
                return mu_out + sigma_out * eps

    class Ddm(nn.Module):
        # TODO: make sure it works the same as the algorithm without coreset when
        #   coreset_size = 0, and also works when coreset_k_newtask > coreset_size
        def __init__(self, in_dim, hidden_dim, out_dim, coreset_size=0, coreset_k_newtask=200, logging_every=10):
            baseline_mnist = Net(in_dim, hidden_dim, out_dim)
            baseline_mnist.train_run(*pmnist_task_loaders()[0], 5)

            super().__init__()
            self.logging_every = logging_every
            self.coreset_size = coreset_size
            self.coreset_k_newtask = coreset_k_newtask
            self.bayesian_layers = nn.Sequential(
                BayesianLinear(
                    in_dim,
                    hidden_dim,
                    init_w=baseline_mnist.linear[0].weight,
                    init_b=baseline_mnist.linear[0].bias
                ),
                nn.ReLU(),
                BayesianLinear(
                    hidden_dim,
                    hidden_dim,
                    init_w=baseline_mnist.linear[2].weight,
                    init_b=baseline_mnist.linear[2].bias
                ),
                nn.ReLU(),
                BayesianLinear(
                    hidden_dim,
                    out_dim,
                    init_w=baseline_mnist.linear[4].weight,
                    init_b=baseline_mnist.linear[4].bias
                ),
            )

        def update_prior(self):
            for bl in self.bayesian_layers:
                if isinstance(bl, BayesianLinear):
                    bl.prior_mu_w = bl.mu_w.clone().detach()
                    bl.prior_sigma_w = torch.exp(bl.log_sigma_w.clone().detach())
                    bl.prior_mu_b = bl.mu_b.clone().detach()
                    bl.prior_sigma_b = torch.exp(bl.log_sigma_b.clone().detach())

        @staticmethod
        def kl_div_gaussians(mu_1, sigma_1, mu_2, sigma_2):
            return torch.sum(torch.log(sigma_2 / sigma_1) + (sigma_1**2 + (mu_1 - mu_2)**2)/(2*sigma_2**2) - 1/2)

        def compute_kl(self):
            res = 0
            for bl in self.bayesian_layers:
                if isinstance(bl, BayesianLinear):
                    res += self.kl_div_gaussians(
                        bl.mu_w, torch.exp(bl.log_sigma_w), bl.prior_mu_w, bl.prior_sigma_w
                    )
                    res += self.kl_div_gaussians(
                        bl.mu_b, torch.exp(bl.log_sigma_b), bl.prior_mu_b, bl.prior_sigma_b
                    )
            return res

        def train_log(self, task, epoch, loss, acc, log_tensors=False):
            wandb.log({'task': task, 'epoch': epoch, 'train_loss': loss, 'train_acc': acc})
            if log_tensors:
                for bli, bl in enumerate(self.bayesian_layers):
                    if isinstance(bl, BayesianLinear):
                        wandb.log({
                            f'{bli}_sigma_w': torch.exp(bl.log_sigma_w).detach(),
                            f'{bli}_mu_w': bl.mu_w.detach()
                        })

        def forward(self, x):
            return self.bayesian_layers(x)

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
            for batch, (data, target) in enumerate(loader):
                data, target = data.to(torch.get_default_device()), target.to(torch.get_default_device())
                opt.zero_grad()
                pred = self(data)
                loss = -(len(loader.dataset) * self.sgvb_mc(pred, target)) + self.compute_kl()
                acc = accuracy(pred, target)
                if batch % self.logging_every == 0:
                    self.train_log(task, epoch, loss, acc)
                loss.backward()
                opt.step()

        # sample(D_t \cup C_{t-1})
        def select_coreset(self, current_task_dataset, old_coreset):
            n_new_points = min(self.coreset_k_newtask, self.coreset_size)
            idx_new = torch.randperm(len(current_task_dataset))[:n_new_points]
            coreset = [current_task_dataset[i] for i in idx_new]

            if (remaining := self.coreset_size - n_new_points) and old_coreset:
                n_old_points = min(remaining, len(old_coreset))
                idx_old = torch.randperm(len(old_coreset))[:n_old_points]
                coreset += [old_coreset[i] for i in idx_old]

            coreset_dataset = torch.utils.data.TensorDataset(
                torch.stack([p[0] for p in coreset]) if coreset else torch.empty(0),
                torch.stack([p[1] for p in coreset]) if coreset else torch.empty(0)
            )
            return torch.utils.data.DataLoader(coreset_dataset, batch_size=64, pin_memory=True)

        # returns (D_t \cup C_{t-1}) - C_t = (D_t - C_t) \cup (C_{t-1} - C_t)
        @staticmethod
        def select_augmented_complement(current_task_dataset, old_coreset, new_coreset):
            new_coreset_ids = set(id(p[0]) for p in new_coreset)
            coreset_complement = [p for p in old_coreset if id(p[0]) not in new_coreset_ids]
            data_complement_idx = [i for i in range(len(current_task_dataset)) if id(current_task_dataset[i][0]) not in new_coreset_ids]
            data_complement_dataset = torch.utils.data.Subset(current_task_dataset, data_complement_idx)
            coreset_complement_dataset = torch.utils.data.TensorDataset(
                torch.stack([p[0] for p in coreset_complement]) if coreset_complement else torch.empty(0),
                torch.stack([p[1] for p in coreset_complement]) if coreset_complement else torch.empty(0)
            )
            return torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([
                data_complement_dataset,
                coreset_complement_dataset
            ]), batch_size=256, pin_memory=True)
    
        def train_run(self, tasks, num_epochs=100):
            opt = torch.optim.Adam(self.parameters(), lr=1e-3)
            self.train()
            wandb.watch(self, log_freq=100)
            old_coreset = []
            for task, (train_loader, test_loader) in enumerate(tasks):
                coreset_loader = self.select_coreset(train_loader.dataset, old_coreset)
                complement_loader = self.select_augmented_complement(train_loader.dataset, old_coreset, list(coreset_loader.dataset))
            
                # (2)
                # precondition: network prior is \tilde{q}_{t-1}
                # network parameters are whatever
                for epoch in trange(num_epochs, desc=f'task {task} phase 1'):
                    self.train_epoch(complement_loader, opt, task, epoch)
                # ==> network parameters are \tilde{q}_t
    
                self.update_prior()
                # network parameters and network prior are \tilde{q}_t
        
                # (3)
                # precondition: network prior is \tilde{q}_{t}
                for epoch in trange(num_epochs, desc=f'task {task} phase 2'):
                    self.train_epoch(coreset_loader, opt, task, epoch)
                # ==> network parameters are q_t
    
                self.test_run(test_loader, task)

        @torch.no_grad()
        def test_run(self, loader, task):
            self.eval()
            accuracies = []
            for batch, (data, target) in enumerate(loader):
                data, target = data.to(torch.get_default_device()), target.to(torch.get_default_device())
                pred = self(data)
                acc = accuracy(pred, target)
                accuracies.append(acc.item())
            wandb.log({'task': task, 'test_acc': np.mean(accuracies)})

    def accuracy(pred, target):
        return (pred.argmax(dim=1) == target).sum() / pred.shape[0]

    def model_pipeline(params):
        with wandb.init(project='vcl', config=params):
            params = wandb.config
            model = Ddm(28*28, 100, params.classes, coreset_size=params.coreset_size)
            model.train_run(pmnist_task_loaders(), num_epochs=params.epochs)
        return model

    run_1 = dict(
        classes=10,
        epochs=100,
        coreset_size=2500,
        problem='pmnist'
    )
    model_pipeline(run_1)
    return BayesianLinear, Ddm, Net, accuracy, model_pipeline, run_1


@app.cell
def _(mo):
    mo.md(
        r"""
        starting from the left rhs term of (4) in paper and multiplyign by 1/N_t (we could make this precise by using the Expectation over all minibatches):
        1/N_t sum^{N_t}_n E(log(p(y_n, x_n)) == grob == 1/M sum^{M}_m E(log(p(y_m, x_m)))
        also (* N_t)
        sum^{N_t}_n E(log(p(y_n, x_n)) == grob == N_t / M sum^{M}_m E(log(p(y_m, x_m))
        und damit

        ---
        eine folgerung aus der unabhängigkeitsannahme ist dass die tasks die grad nicht trainiert werden deren marginal distribution der gewichte nicht verändert werden

        ---

        bei normaler reparametrization (weights direkt samplen):

        jede weight normal und unabhänig -> eine row (ein datenpunkt) der outputs unter sich unabhängig, aber nicht zueinander, ausser man sampled für jeden datenpunkt im batch die weights neu
        wir könnten zwar drauf scheissen, denn es ist ja ineffizient das zu machen (für jeden datenpunkt im batch eine neue weight matrix samplen) aber dann ist die varianz des monte carlo estimators L_SGVB nicht mit O(1/M) minibatch size skalierend (siehe paper, dann wäre es O(M)), was "in practice" erwünscht ist

        es ist trotzdem ein monte carlo estimator, oder man kann es so bezeichnen (obwohl pro minibatch nur ein mal ein epsilon gesampled wird), weil wir ne varianzreduktion über M haben. (in dem fall nicht über mehr samples).

        ---

        angenommen wir möchten jetzt die gute konvergenz haben, also samplen wir "die w's pro datenpunkt im minibatch".

        aber, wenn wir was konstantes (die daten sind fixiert) auf was gaussches dranmultiplizieren kriegen wir was gaussches.
        das können wir nutzen:
        durch die matrixmultiplikation verschwindet aber die hidden dimension, und wir haben nur noch out_dim viele dinge zu generieren.

        ---

        das naive generiert pro layer alle weights

        das lokale reparam generiert immer nur die zwischenresultate (die werte die in die aktivierungen gefüttert werden), was genau die anzahl der nodes im netzwerk entspricht

        dh., es ist worth it solange im netzwerk params > nodes gilt, was immer der fall ist
        """
    )
    return


@app.cell
def _(accuracy, nn, np, pmnist_task_loaders, torch, trange):
    # SI implementation of a simple MLP for classification.

    class SINet(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, si_c=0.5, epsilon=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
            self.si_c = si_c
            self.epsilon = epsilon
            # Initialise SI bookkeeping for each parameter.
            for name, p in self.named_parameters():
                name = name.replace('.', '_')
                self.register_buffer(f'si_prev_{name}', p.data.clone())
                self.register_buffer(f'si_star_{name}', p.data.clone())
                self.register_buffer(f'si_omega_{name}', torch.zeros_like(p.data))
                self.register_buffer(f'si_importance_{name}', torch.zeros_like(p.data))

        def forward(self, x):
            return self.net(x)

        # Accumulate SI importance after each update.
        def update_si(self):
            for name, p in self.named_parameters():
                clean_name = name.replace('.', '_')
                if p.grad is not None:
                    delta = p.data - getattr(self, f'si_prev_{clean_name}')
                    getattr(self, f'si_omega_{clean_name}').add_((-p.grad * delta).detach())
                    getattr(self, f'si_prev_{clean_name}').copy_(p.data)

        # Compute the SI regularisation loss.
        def si_reg_loss(self):
            reg = 0.
            for name, p in self.named_parameters():
                clean_name = name.replace('.', '_')
                si_importance = getattr(self, f'si_importance_{clean_name}')
                si_star = getattr(self, f'si_star_{clean_name}')
                reg += (si_importance * (p.data - si_star).pow(2)).sum()
            return self.si_c * reg

        # Update consolidated importances at task end.
        def consolidate(self):
            for name, p in self.named_parameters():
                clean_name = name.replace('.', '_')
                si_star = getattr(self, f'si_star_{clean_name}')
                delta_total = p.data - si_star
                si_omega = getattr(self, f'si_omega_{clean_name}')
                si_importance = getattr(self, f'si_importance_{clean_name}')
                importance_update = si_omega / (delta_total.pow(2) + self.epsilon)
                si_importance.add_(importance_update)
                si_omega.zero_()
                si_star.copy_(p.data)

        def train_task(self, train_loader, epochs=10, lr=1e-3):
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()
            for epoch in trange(epochs, desc='Training task'):
                self.train()
                losses, accs = [], []
                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = self(data)
                    loss = loss_fn(output, target) + self.si_reg_loss()
                    loss.backward()
                    optimizer.step()
                    self.update_si()
                    losses.append(loss.item())
                    accs.append(accuracy(output, target))
                print(f"Epoch {epoch}: Loss {np.mean(losses):.4f}, Accuracy {np.mean(accs):.4f}")
            self.consolidate()

        def test_run(self, test_loader):
            self.eval()
            loss_fn = nn.CrossEntropyLoss()
            losses, accs = [], []
            with torch.no_grad():
                for data, target in test_loader:
                    output = self(data)
                    loss = loss_fn(output, target)
                    losses.append(loss.item())
                    accs.append(accuracy(output, target))
            return np.mean(losses), np.mean(accs)

    def si_model_pipeline(params):
        # params should be a dict with keys: 'classes', 'epochs', 'si_c', 'epsilon'
        model = SINet(28*28, 100, params['classes'])
        tasks = pmnist_task_loaders()  # List of (train_loader, test_loader) tuples.
        for task, (train_loader, test_loader) in enumerate(tasks):
            print(f"=== Task {task} ===")
            model.train_task(train_loader, epochs=params['epochs'], lr=1e-3)
            test_loss, test_acc = model.test_run(test_loader)
            print(f"After Task {task}: Test Loss {test_loss:.4f}, Test Accuracy {test_acc:.4f}")
        return model

    # Example pipeline parameters.
    run_params = {
        'classes': 10,
        'epochs': 100,
        'problem': 'pmnist'
    }

    # model = si_model_pipeline(run_params)
    return SINet, run_params, si_model_pipeline


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
