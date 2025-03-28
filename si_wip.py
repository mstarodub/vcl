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
