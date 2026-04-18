import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os

# Ensure we can import the dge package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dge.optimizer import DGEOptimizer

# =============================================================================
# CONFIGURACION
# =============================================================================
SEED         = 42
N_TRAIN      = 3_000     
N_TEST       = 600       
BATCH_SIZE   = 256       
TOTAL_EVALS  = 150_000   # Un poco más de tiempo para redes profundas
# Deep Architecture: 5 Hidden Layers (Sufficiently deep to test gradient propagation)
ARCH_DEEP    = (784, 64, 64, 64, 64, 10) 

torch.manual_seed(SEED)
np.random.seed(SEED)

def load_mnist_subset(n_train=N_TRAIN, n_test=N_TEST):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    full_test  = datasets.MNIST('./data', train=False, download=True, transform=transform)

    X_tr_all = full_train.data.float().view(-1, 784) / 255.0
    y_tr_all = full_train.targets
    X_te_all = full_test.data.float().view(-1, 784) / 255.0
    y_te_all = full_test.targets

    # Manual normalization
    X_tr_all = (X_tr_all - 0.1307) / 0.3081
    X_te_all = (X_te_all - 0.1307) / 0.3081

    rng = np.random.default_rng(SEED)
    tr_idx = rng.choice(len(y_tr_all), size=n_train, replace=False)
    te_idx = rng.choice(len(y_te_all), size=n_test,  replace=False)

    return X_tr_all[tr_idx], y_tr_all[tr_idx], X_te_all[te_idx], y_te_all[te_idx]

class SimpleMLP(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def get_params(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()]).numpy()

    def set_params(self, params):
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(torch.from_numpy(params[idx:idx+size]).view(p.shape))
            idx += size

def evaluate_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

def run_experiment(name, arch, total_evals, type="dge", lr=0.5):
    print(f"\n>>> TESTING {name} | Arch: {arch} | Budget: {total_evals} evals")
    X_train, y_train, X_test, y_test = load_mnist_subset()
    model = SimpleMLP(arch)
    params = model.get_params()
    D = len(params)
    print(f"    Params (D): {D:,}")
    
    t0 = time.time()
    evals = 0
    rng_mb = np.random.default_rng(SEED)
    
    criterion = nn.CrossEntropyLoss()

    if type == "dge":
        k = math.ceil(math.log2(D))
        optimizer = DGEOptimizer(
            dim=D, lr=lr, delta=1e-3, 
            total_steps=total_evals // (2 * k),
            clip_norm=0.05, seed=SEED
        )
        
        while evals < total_evals:
            idx = rng_mb.integers(0, len(y_train), size=BATCH_SIZE)
            Xb, yb = X_train[idx], y_train[idx]
            
            def f(p):
                model.set_params(p)
                with torch.no_grad():
                    return criterion(model(Xb), yb).item()

            params, n = optimizer.step(f, params)
            evals += n
            
            if evals % 20_000 < n:
                model.set_params(params)
                acc = evaluate_accuracy(model, X_test, y_test)
                print(f"      DGE Evals: {evals:>7} | Test Acc: {acc:.2%}")

    else: # Analytic
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        while evals < total_evals:
            idx = rng_mb.integers(0, len(y_train), size=BATCH_SIZE)
            Xb, yb = X_train[idx], y_train[idx]
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            evals += 3
            
            if evals % 20_000 < 3:
                acc = evaluate_accuracy(model, X_test, y_test)
                print(f"      Adam Evals: {evals:>7} | Test Acc: {acc:.2%}")

    final_acc = evaluate_accuracy(model, X_test, y_test)
    elapsed = time.time() - t0
    print(f"    FINAL: Acc={final_acc:.2%} | Time={elapsed:.1f}s")
    return final_acc, elapsed

if __name__ == "__main__":
    # Test 1: Shallow
    run_experiment("Shallow", (784, 32, 10), 100_000, type="dge")
    run_experiment("Shallow", (784, 32, 10), 100_000, type="adam")
    
    # Test 2: Deep
    run_experiment("Deep", ARCH_DEEP, 150_000, type="dge")
    run_experiment("Deep", ARCH_DEEP, 150_000, type="adam")
