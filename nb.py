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
    return F, datasets, nn, np, plt, torch, tqdm, trange, transforms


@app.cell
def _():
    import wandb
    wandb.login()
    return (wandb,)


@app.cell
def _(np, torch):
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

    print("torch version", torch.__version__, "running on", torch.ones(1, device=torch_device()).device)
    seed(0)
    return seed, torch_device


@app.cell
def _(datasets, plt, torch, transforms):
    def transform():
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(torch.flatten)
        ])

    def transform_permute(idx):
        return transforms.Compose([
            transform(),
            # nasty pytorch bug: if we try to use num_workers > 0
            # on macos, pickling this fails, and nothing seems to help
            # things I tried: using dill instead of pickle for the 
            # multiprocessing.reduction.ForkingPickler, partial, a class
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
            train_loader = torch.utils.data.DataLoader(
                data_train, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=12 if torch.cuda.is_available() else 0
            )
            cumulative_test_datasets.append(data_test)
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(cumulative_test_datasets),
                batch_size=batch_size,
                shuffle=True,
                num_workers=12 if torch.cuda.is_available() else 0
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
def _():
    # pmnist

    # show the permutation
    # visualize_sample_img(pmnist_task_loaders()[0][0])
    # should be 10 classes
    # example_loader, _ = pmnist_task_loaders()[0]
    # print("unique classes:", torch.unique(example_loader.dataset.targets).tolist())
    return


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
                num_workers=12 if torch.cuda.is_available() else 0
            )
            cumulative_test_datasets.append(torch.utils.data.Subset(test_ds, test_idx))
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(cumulative_test_datasets),
                batch_size=batch_size,
                shuffle=True,
                num_workers=12 if torch.cuda.is_available() else 0
            )
            loaders.append((train_loader, test_loader))
        return loaders
    return (splitmnist_task_loaders,)


@app.cell
def _():
    # split mnist

    # balanced number of classes per task
    # from collections import defaultdict
    # example_loaders = splitmnist_task_loaders()
    # counts = defaultdict(int)
    # for _, y in example_loaders[0][0].dataset:
    #     counts[tuple(y.tolist())] += 1
    # print(counts)
    return


@app.cell
def _(F, nn, np, pmnist_task_loaders, torch, torch_device, trange, wandb):
    # TODO: per test-task accuracy breakdown
    # TODO: use He ReLu layer init if pretrain=0

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
            loss_fn = nn.CrossEntropyLoss()
            opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
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
                init_w = torch.randn(out_dim, in_dim)
            if init_b is None:
                init_b = torch.randn(out_dim)

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
            layer_init_std=1e-3,
            per_task_opt=False,
            bayesian_samples=1,
            coreset_k=0, 
            pretrain_epochs=5,
            logging_every=10
        ):

            baseline_mnist = Net(in_dim, hidden_dim, out_dim).to(torch_device())
            baseline_mnist = torch.compile(baseline_mnist)
            baseline_mnist.train_run(*pmnist_task_loaders()[0], pretrain_epochs)

            super().__init__()
            self.logging_every = logging_every
            self.per_task_opt = per_task_opt
            self.bayesian_samples = bayesian_samples
            self.coreset_k = coreset_k
            self.layers = nn.Sequential(
                BayesianLinear(
                    in_dim,
                    hidden_dim,
                    layer_init_std,
                    init_w=baseline_mnist.linear[0].weight,
                    init_b=baseline_mnist.linear[0].bias
                ),
                nn.ReLU(),
                BayesianLinear(
                    hidden_dim,
                    hidden_dim,
                    layer_init_std,
                    init_w=baseline_mnist.linear[2].weight,
                    init_b=baseline_mnist.linear[2].bias
                ),
                nn.ReLU(),
                BayesianLinear(
                    hidden_dim,
                    out_dim,
                    layer_init_std,
                    init_w=baseline_mnist.linear[4].weight,
                    init_b=baseline_mnist.linear[4].bias
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
            for batch, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                pred = self(data)
                loss = -self.sgvb_mc(pred, target) + self.compute_kl() / len(loader.dataset)
                acc = accuracy(pred, target)
                if batch % self.logging_every == 0:
                    wandb.log({'task': task, 'epoch': epoch, 'train_loss': loss, 'train_acc': acc})
                loss.backward()
                opt.step()

        # sample(D_t) \cup C_{t-1}
        def select_coreset(self, current_task_dataset, old_coreset):
            idx_new = torch.randperm(len(current_task_dataset))[:self.coreset_k]
            coreset = [current_task_dataset[i] for i in idx_new] + old_coreset

            coreset_dataset = torch.utils.data.TensorDataset(
                torch.stack([p[0] for p in coreset]) if coreset else torch.empty(0),
                torch.stack([p[1] for p in coreset]) if coreset else torch.empty(0)
            )
            return torch.utils.data.DataLoader(
                coreset_dataset, 
                batch_size=64, 
                shuffle=True if coreset else False,
                num_workers=12 if torch.cuda.is_available() else 0
            )

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
            ]), batch_size=256, shuffle=True, num_workers=12 if torch.cuda.is_available() else 0)

        def train_test_run(self, tasks, num_epochs=100):
            self.train()
            opt = torch.optim.Adam(self.parameters(), lr=1e-3)
            wandb.watch(self, log_freq=100)
            old_coreset = []
            for task, (train_loader, test_loader) in enumerate(tasks):
                coreset_loader = self.select_coreset(train_loader.dataset, old_coreset)
                complement_loader = self.select_augmented_complement(train_loader.dataset, old_coreset, list(coreset_loader.dataset))

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
                # TODO: ONLY USED FOR PREDICITION NOT PROPAGATION
                # precondition: network prior is \tilde{q}_{t}
                for epoch in trange(num_epochs, desc=f'task {task+1} phase 2'):
                    self.train_epoch(coreset_loader, opt, task, epoch)
                # ==> network parameters are q_t

                self.test_run(test_loader, task)

        @torch.no_grad()
        def test_run(self, loader, task):
            self.eval()
            device = torch_device()
            accuracies = []
            for batch, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                # E[argmax_y p(y | theta, x)] != argmax_y E[p(y | \theta, x)]
                # lhs: 1 sample; rhs: can get better approximation via MC
                # 1 sample is unbiased for the p(y | \theta, x), but argmax breaks this
                preds = [self(data) for _ in range(self.bayesian_samples)]
                mean_pred = torch.stack(preds).mean(0)
                acc = accuracy(mean_pred, target)
                accuracies.append(acc.item())
            wandb.log({'task': task, 'test_acc': np.mean(accuracies)})

    def accuracy(pred, target):
        return (pred.argmax(dim=1) == target).sum() / pred.shape[0]

    def model_pipeline(params):
        with wandb.init(project='vcl', config=params):
            params = wandb.config
            model = Ddm(28*28, 100, params.classes,
                        layer_init_std=params.layer_init_std,
                        per_task_opt=params.per_task_opt,
                        bayesian_samples=params.bayesian_samples, 
                        coreset_k=params.coreset_k,
                        pretrain_epochs=params.pretrain_epochs
            ).to(torch_device())
            model = torch.compile(model)
            model.train_test_run(pmnist_task_loaders(), num_epochs=params.epochs)
        return model

    ddm_pmnist_run = dict(
        classes=10,
        epochs=100,
        pretrain_epochs=1,
        coreset_k=0,
        per_task_opt=False,
        layer_init_std=1e-3,
        bayesian_samples=100,
        problem='pmnist',
        model='vcl'
    )
    model = model_pipeline(ddm_pmnist_run)
    return (
        BayesianLinear,
        Ddm,
        Net,
        accuracy,
        ddm_pmnist_run,
        model,
        model_pipeline,
    )


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
