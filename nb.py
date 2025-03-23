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
    def torch_mps(enable_mps):
        if torch.backends.mps.is_available() and enable_mps:
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        torch.set_default_device(device)
        print("torch version", torch.__version__, "running on", torch.ones(1).device)

    # reproducibility
    def seed(seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)

    torch_mps(False)
    seed(0)
    return seed, torch_mps


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

    def pmnist_task(permute=True):    
        perm = torch.randperm(28 * 28)
        tf = transform_permute(perm) if permute else transform()
        data_train = datasets.MNIST('./data', train=True, download=True, transform=tf)
        data_test = datasets.MNIST('./data', train=False, download=True, transform=tf)

        batch_size = 256
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)

        return train_loader, test_loader

    def pmnist_task_loaders():
        return [pmnist_task() for _ in range(10)]

    def visualize_sample_img(loader):
        images, labels = next(iter(loader))
        first_img = images[0].reshape(28, 28)
        plt.figure(figsize=(1, 1))
        plt.imshow(first_img, cmap='gray')
        plt.title(labels[0].item())
        plt.axis('off')
        plt.show()
    return (
        pmnist_task,
        pmnist_task_loaders,
        transform,
        transform_permute,
        visualize_sample_img,
    )


@app.cell
def _(pmnist_task, visualize_sample_img):
    visualize_sample_img(pmnist_task(permute=False)[0])
    visualize_sample_img(pmnist_task(permute=True)[0])
    return


@app.cell
def _(pmnist_task, torch):
    _ , loader = pmnist_task()
    unique_classes = torch.unique(loader.dataset.targets)
    print(unique_classes.tolist())
    type(unique_classes[0].item())
    return loader, unique_classes


@app.cell
def _(datasets, torch, transform):
    def splitmnist_task_loaders():
        batch_size=256
        loaders = []
        # 5 classification tasks
        for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]:
            def transform_onehot(class_a, class_b):
                one_hot = {class_a: torch.tensor([1, 0]), class_b: torch.tensor([0, 1])}
                return lambda x: one_hot[int(x)]

            train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform(), target_transform=transform_onehot(a, b))
            test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform(), target_transform=transform_onehot(a, b))

            # only include the two digits for this task
            train_mask = (train_ds.targets == a) | (train_ds.targets == b)
            test_mask = (test_ds.targets == a) | (test_ds.targets == b)
            train_idx = torch.nonzero(train_mask).squeeze()
            test_idx = torch.nonzero(test_mask).squeeze()
            train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_ds, train_idx), batch_size=batch_size)
            test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_ds, test_idx), batch_size=batch_size)

            loaders.append((train_loader, test_loader))
        return loaders
    return (splitmnist_task_loaders,)


@app.cell
def _(splitmnist_task_loaders):
    from collections import defaultdict
    lds = splitmnist_task_loaders()
    counts = defaultdict(int)
    for _, y in lds[0][0].dataset:
        counts[tuple(y.tolist())] += 1
    print(counts)
    return counts, defaultdict, lds, y


@app.cell
def _(lds):
    data, labels = next(iter(lds[0][0]))
    print("Data shape:", data.shape)
    print("First image label:", labels[0])

    for i in range(10):
        print(labels[i])
    return data, i, labels


@app.cell
def _(Net, pmnist_task_loaders, train):
    baseline_pmnist = Net(28*28, 10)
    for train_loader, test_loader in pmnist_task_loaders():
        print("next permuted task")
        train(baseline_pmnist, train_loader, test_loader, 1)
    return baseline_pmnist, test_loader, train_loader


@app.cell
def _():
    # for a nn.Sequential container, weight setting:
    # mod.linear[0].weight.data = torch.tensor([1. ,2. ,3. ,4. ,5.], requires_grad=True)[:, None]
    #   mod.linear[0].bias.data = torch.zeros((5, ), requires_grad=True)  # bias is not a scalar here
    return


@app.cell
def _(mo):
    mo.md(r"""eine folgerung aus der unabhängigkeitsannahme ist dass die tasks die grad nicht trainiert werden deren marginal distribution der gewichte nicht verändert werden""")
    return


app._unparsable_cell(
    r"""
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
            for i in trange(num_epochs):
                train_loss, train_acc = self.train_epoch(train_loader, loss_fn, opt)
                test_loss, test_acc = self.test_run(test_loader, loss_fn)
                print(f\"epoch {i}: train loss {train_loss:.4f} train acc {train_acc:.4f} test loss {test_loss:.4f} test acc {test_acc:.4f}\")

        @torch.no_grad()
        def test_run(self, loader, loss_fn):
            self.eval()
            losses, accuracies = [], []
            for data, target in loader:
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
                # init_w = torch.randn(out_dim, in_dim)
                init_w = torch.zeros(out_dim, in_dim)
            if init_b is None:
                # init_b = torch.randn(out_dim)
                init_b = torch.zeros(out_dim)

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
        def __init__(self, in_dim, hidden_dim, out_dim):
            baseline_mnist = Net(28*28, 100, 10)
            baseline_mnist.train_run(*pmnist_task(permute=False), 5)

            super().__init__()
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
                    res += kl_div_gaussians(
                        bl.mu_w, torch.exp(bl.log_sigma_w), bl.prior_mu_w, bl.prior_sigma_w
                    )
                    res += kl_div_gaussians(
                        bl.mu_b, torch.exp(bl.log_sigma_b), bl.prior_mu_b, bl.prior_sigma_b
                    )
            return res

        def train_log(self, task, epoch, loss, acc, log_tensors=False):
            wandb.log({'task': task, 'epoch': epoch, 'loss': loss, 'acc': acc})
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

    def accuracy(pred, target):
        return (pred.argmax(dim=1) == target).sum() / pred.shape[0]

    def train_epoch_new(model, train_loader, opt, task, epoch):
        for batch, (data, target) in enumerate(train_loader):
            opt.zero_grad()
            pred = model(data)
            loss = -(len(train_loader.dataset) * sgvb_mc(pred, target)) + model.compute_kl()
            acc = accuracy(pred, target)
            if batch % log_every == 0:
                model.train_log(task, epoch, loss, acc)
            loss.backward()
            opt.step()

    def train_run(model, tasks, num_epochs=100, log_every=10):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        wandb.watch(model, log_freq=100)
    
        for task, (train_loader, test_loader) in enumerate(tasks):

            #update coreset

            #get C_t
            def select_coreset(current_task_dataset, coreset_size, old_coreset):
                pass
                #return sample(D_t \cup C_{t-1})

            #get D_t \cup C_{t-1} - C_t
            def select_augmented_complement(D_t, C_{t_1}, C_t):
                #return D_t \cup C_{t-1} - C_t
                pass

            coreset_loader = select_coreset(...)

            complement_loader = select_augmented_complement(...)

            # (2)
            #annahme: Netzwerk prior im Zustand \tilde{q}_{t-1}
            #egal wie die parameter sind (bis auf locale minima)
            for epoch in trange(num_epochs):
                train_epoch_new(model, complement_loader, opt, task, epoch)
            # ==> Netzwerk parameter im Zustand \tilde{q}_t

            model.update_prior()
            # Netwerk parameter und netwerk prior im Zustand \tilde{q}_t
        
            # (3)
            #annahme: Netzwerk prior im Zustand \tilde{q}_{t}
            for epoch in trange(num_epochs):
                train_epoch_new(model, coreset_loader, opt, task, epoch)
            # ==> Netzwerk parameter im Zustand q_t

            # testen / statistiken
        

    # starting from the left rhs term of (4) in paper and multiplyign by 1/N_t (we could make this precise by using the Expectation over all minibatches):
    # 1/N_t sum^{N_t}_n E(log(p(y_n, x_n)) == grob == 1/M sum^{M}_m E(log(p(y_m, x_m)))
    # also (* N_t)
    # sum^{N_t}_n E(log(p(y_n, x_n)) == grob == N_t / M sum^{M}_m E(log(p(y_m, x_m))
    # und damit

    @torch.no_grad()
    def test_run(model, loader):
        model.eval()
        for data, target in loader:
            pred = model(data)
            loss = loss_fn(pred, target)

    def model_pipeline(params):
        with wandb.init(project='vcl', config=params):
            params = wandb.config
            model = Ddm(28*28, 100, params.classes)
            train_run(model, pmnist_task_loaders(), num_epochs=params.epochs)
            # test(model, pmnist_test_loader())
        return model

    run_1 = dict(
        classes=10,
        epochs=10,
        problem='pmnist'
    )
    model_pipeline(run_1)
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(
        r"""
        bei normaler reparametrization (weights direkt samplen):

        jede weight normal und unabhänig -> eine row (ein datenpunkt) der outputs unter sich unabhängig, aber nicht zueinander, ausser man sampled für jeden datenpunkt im batch die weights neu
        wir könnten zwar drauf scheissen, denn es ist ja ineffizient das zu machen (für jeden datenpunkt im batch eine neue weight matrix samplen) aber dann ist die varianz des monte carlo estimators L_SGVB nicht mit O(1/M) minibatch size skalierend (siehe paper, dann wäre es O(M)), was "in practice" erwünscht ist

        es ist trotzdem ein monte carlo estimator, oder man kann es so bezeichnen (obwohl pro minibatch nur ein mal ein epsilon gesampled wird), weil wir ne varianzreduktion über M haben. (in dem fall nicht über mehr samples).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        angenommen wir möchten jetzt die gute konvergenz haben, also samplen wir "die w's pro datenpunkt im minibatch".

        aber, wenn wir was konstantes (die daten sind fixiert) auf was gaussches dranmultiplizieren kriegen wir was gaussches.
        das können wir nutzen:
        durch die matrixmultiplikation verschwindet aber die hidden dimension, und wir haben nur noch out_dim viele dinge zu generieren.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        das naive generiert pro layer alle weights

        das lokale reparam generiert immer nur die zwischenresultate (die werte die in die aktivierungen gefüttert werden), was genau die anzahl der nodes im netzwerk entspricht

        dh., es ist worth it solange im netzwerk params > nodes gilt, was immer der fall ist
        """
    )
    return


@app.cell
def _(torch):
    torch.randn(3,3)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
