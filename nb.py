import marimo

__generated_with = "0.11.22"
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
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    return F, dataclass, datasets, nn, np, plt, torch, tqdm, transforms


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
def _(nn):
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
    return (Net,)


@app.cell
def _(nn, np, torch, tqdm):
    def accuracy(pred, target):
        return (pred.argmax(dim=1) == target).sum() / pred.shape[0]

    def train_epoch(model, loader, loss_fn, opt):
        model.train()
        losses, accuracies = [], []
        for data, target in loader:
            opt.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, target)
            acc = accuracy(pred, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            accuracies.append(acc.item())
        return np.mean(losses), np.mean(accuracies)

    def train(model, train_loader, test_loader, num_epochs=100):
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for i in tqdm(range(num_epochs)):
            train_loss, train_acc = train_epoch(model, train_loader, loss_fn, opt)
            test_loss, test_acc = infer(model, test_loader, loss_fn)
            print(f"epoch {i}: train loss {train_loss:.4f} train acc {train_acc:.4f} test loss {test_loss:.4f} test acc {test_acc:.4f}")

    @torch.no_grad()
    def infer(model, loader, loss_fn):
        model.eval()
        losses, accuracies = [], []
        for data, target in loader:
            pred = model(data)
            loss = loss_fn(pred, target)
            acc = accuracy(pred, target)
            losses.append(loss.item())
            accuracies.append(acc.item())
        return np.mean(losses), np.mean(accuracies)
    return accuracy, infer, train, train_epoch


@app.cell
def _(Net, pmnist_task, train):
    baseline_mnist = Net(28*28, 100, 10)
    train(baseline_mnist, *pmnist_task(permute=False), 30)
    return (baseline_mnist,)


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


@app.cell
def _(F, criterion, dataset_size, nn, torch):
    class BayesianLinear(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.mu_w = nn.Parameter(torch.randn(out_dim, in_dim))
            self.log_sigma_w = nn.Parameter(torch.randn(out_dim, in_dim))
            self.mu_b = nn.Parameter(torch.randn(out_dim))
            self.log_sigma_b = nn.Parameter(torch.randn(out_dim))

        def forward(self, x):
            mu_out = F.linear(x, self.mu_w, self.mu_b)
            sigma_out = torch.sqrt(F.linear(x**2, torch.exp(2 * self.log_sigma_w), torch.exp(2 * self.log_sigma_b)))
            eps = torch.randn_like(mu_out)
            return mu_out + sigma_out * eps

    def kl_div_gaussians(mu_1, sigma_1, mu_2, sigma_2):
        return torch.log(sigma_2 / sigma_1) + (sigma_1**2 + (sigma_1 - sigma_2)**2)/(2*sigma_2**2) - 1/2

    # @dataclass
    # class KL():
    #    acc_kl = 0

    class Ddm(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            # self.kl = KL()
            self.bayesian_layers = nn.Sequential(
                BayesianLinear(in_dim, hidden_dim),
                nn.ReLU(),
                BayesianLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
                BayesianLinear(hidden_dim, out_dim)
            )

        def forward(self, x):
            return self.bayesian_layers(x)

        # L_VCL: returns a tuple of (L_SGVB, KL) where the first component does not have the N factor
        def criterion(self, pred, target):
            # we classification task, so target is a index to the right class
            # apply softmax:
            # p(target | pred) = exp(pred_target) / sum_{i=0}^len(pred) exp(pred_i)
            # according to (3) in the local reparam paper, we apply the log,
            # and arrive at the l_n from https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
            # we use reduction=mean (the default) to get the 1/M factor and the outer sum,
            # and finally arrive at eq (3) without the N factor
            mc = -F.cross_entropy(pred, target)
            kl = ...
            return mc, kl

    def train():
        ...

        mc, kl = criterion()
        loss = -(dataset_size * mc) + kl

    dd = Ddm(784, 100, 10)
    return BayesianLinear, Ddm, dd, kl_div_gaussians, train


@app.cell
def _(dd, torch):
    dd(torch.randn(784, 5))
    return


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


if __name__ == "__main__":
    app.run()
