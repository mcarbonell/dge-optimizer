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
# CONFIGURACION (Sincronizada con examples/train_mnist.py)
# =============================================================================
SEED         = 42
N_TRAIN      = 3_000     
N_TEST       = 600       
BATCH_SIZE   = 256       
TOTAL_EVALS  = 100_000   # Presupuesto total de evaluaciones (Forward passes)
ARCH         = (784, 32, 10)

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

    # Manual normalization to match train_mnist.py
    X_tr_all = (X_tr_all - 0.1307) / 0.3081
    X_te_all = (X_te_all - 0.1307) / 0.3081

    rng = np.random.default_rng(SEED)
    tr_idx = rng.choice(len(y_tr_all), size=n_train, replace=False)
    te_idx = rng.choice(len(y_te_all), size=n_test,  replace=False)

    X_tr = X_tr_all[tr_idx]
    y_tr = y_tr_all[tr_idx]
    X_te = X_te_all[te_idx]
    y_te = y_te_all[te_idx]
    return X_tr, y_tr, X_te, y_te

# =============================================================================
# MODELO PYTORCH (Equivalente al forward_np)
# =============================================================================
class SimpleMLP(nn.Module):
    def __init__(self, arch=ARCH):
        super().__init__()
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        # He initialization to match train_mnist.py
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

# =============================================================================
# WRAPPERS PARA COMPARACION
# =============================================================================

def evaluate_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

def run_dge(X_train, y_train, X_test, y_test, total_evals):
    print("\n--- Running DGE (Zeroth-Order) ---")
    model = SimpleMLP()
    params = model.get_params()
    D = len(params)
    k = math.ceil(math.log2(D))
    
    # We count 2*k evaluations per step
    optimizer = DGEOptimizer(
        dim=D, lr=0.5, delta=1e-3, 
        total_steps=total_evals // (2 * k),
        clip_norm=0.05, seed=SEED
    )
    
    evals = 0
    t0 = time.time()
    rng_mb = np.random.default_rng(SEED)

    while evals < total_evals:
        idx = rng_mb.integers(0, len(y_train), size=BATCH_SIZE)
        Xb, yb = X_train[idx], y_train[idx]
        
        # Loss function for DGE (receives numpy flat array)
        def f(p):
            model.set_params(p)
            with torch.no_grad():
                logits = model(Xb)
                loss = nn.CrossEntropyLoss()(logits, yb)
            return loss.item()

        params, n = optimizer.step(f, params)
        evals += n

        if evals % 10_000 == 0 or evals == n:
            model.set_params(params)
            tr_acc = evaluate_accuracy(model, X_train, y_train)
            te_acc = evaluate_accuracy(model, X_test, y_test)
            print(f"  Evals: {evals:>7} | Train Acc: {tr_acc:.2%} | Test Acc: {te_acc:.2%}")

    return evaluate_accuracy(model, X_test, y_test), time.time() - t0

def run_analytic(X_train, y_train, X_test, y_test, total_evals, opt_name="Adam", lr=0.001):
    print(f"\n--- Running {opt_name} (Analytic Backprop) ---")
    model = SimpleMLP()
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    
    evals = 0
    # In standard benchmarking, 1 backward pass is often counted as 2 forward passes.
    # So 1 complete training step (Forward + Backward) = 3 Evals.
    EVALS_PER_STEP = 3
    
    t0 = time.time()
    rng_mb = np.random.default_rng(SEED)

    while evals < total_evals:
        idx = rng_mb.integers(0, len(y_train), size=BATCH_SIZE)
        Xb, yb = X_train[idx], y_train[idx]
        
        model.train()
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        
        evals += EVALS_PER_STEP

        if (evals // EVALS_PER_STEP) % 100 == 0: # Log every 100 steps
             pass # Too frequent for evals log
             
        if evals >= 10_000 and (evals - EVALS_PER_STEP) < (evals // 10000) * 10000:
             # Just a check to log at similar intervals
             pass

    # Final log
    tr_acc = evaluate_accuracy(model, X_train, y_train)
    te_acc = evaluate_accuracy(model, X_test, y_test)
    print(f"  Evals: {evals:>7}+ | Train Acc: {tr_acc:.2%} | Test Acc: {te_acc:.2%}")

    return te_acc, time.time() - t0

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_subset()
    
    results = {}
    
    # Run DGE
    dge_acc, dge_time = run_dge(X_train, y_train, X_test, y_test, TOTAL_EVALS)
    results["DGE"] = {"Acc": dge_acc, "Time": dge_time}
    
    # Run Adam
    adam_acc, adam_time = run_analytic(X_train, y_train, X_test, y_test, TOTAL_EVALS, "Adam", lr=0.001)
    results["Adam"] = {"Acc": adam_acc, "Time": adam_time}
    
    # Run SGD
    sgd_acc, sgd_time = run_analytic(X_train, y_train, X_test, y_test, TOTAL_EVALS, "SGD", lr=0.01)
    results["SGD"] = {"Acc": sgd_acc, "Time": sgd_time}

    print("\n" + "="*50)
    print(f"{'Algorithm':<15} | {'Test Accuracy':<15} | {'Time (s)':<10}")
    print("-"*50)
    for name, data in results.items():
        print(f"{name:<15} | {data['Acc']:>15.2%} | {data['Time']:>10.1f}s")
    print("="*50)
    print(f"Note: Adam/SGD budget capped at {TOTAL_EVALS} evals (counting 1 step = 3 evals).")
