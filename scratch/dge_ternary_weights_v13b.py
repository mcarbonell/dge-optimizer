import numpy as np
import math
import time
import torch
import torch.nn as nn
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
TOTAL_EVALS  = 150_000   
ARCH         = (784, 128, 10) 

torch.manual_seed(SEED)
np.random.seed(SEED)

def load_mnist_subset(n_train=N_TRAIN, n_test=N_TEST):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=False,  download=True, transform=transform) 
    
    X_all = full_train.data.float().view(-1, 784) / 255.0
    y_all = full_train.targets
    X_all = (X_all - 0.1307) / 0.3081

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_all), size=n_train+n_test, replace=False)
    
    X_tr = X_all[idx[:n_train]]
    y_tr = y_all[idx[:n_train]]
    X_te = X_all[idx[n_train:]]
    y_te = y_all[idx[n_train:]]
    return X_tr, y_tr, X_te, y_te

# =============================================================================
# MODELO TERNARIO v13b (Corrected Init)
# =============================================================================
class TernaryModel(nn.Module):
    def __init__(self, arch=ARCH, threshold=0.5):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(arch[i], arch[i+1], bias=False) for i in range(len(arch)-1)])
        self.threshold = threshold
        self._init_weights()

    def _init_weights(self):
        # UNIFORM INITIALIZATION between -1 and 1 to ensure many weights start active!
        for m in self.layers:
            nn.init.uniform_(m.weight, -1.0, 1.0)

    def forward(self, x, ternary=True):
        h = x
        for i, layer in enumerate(self.layers):
            w = layer.weight
            if ternary:
                w_tern = torch.zeros_like(w)
                w_tern[w > self.threshold] = 1.0
                w_tern[w < -self.threshold] = -1.0
                w = w_tern
            
            h = h @ w.t()
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        return h

    def get_params(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()]).numpy()

    def set_params(self, params):
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(torch.from_numpy(params[idx:idx+size]).view(p.shape))
            idx += size

    def get_sparsity(self):
        total = 0
        zeros = 0
        with torch.no_grad():
            for layer in self.layers:
                w = layer.weight
                w_tern = torch.zeros_like(w)
                w_tern[w > self.threshold] = 1.0
                w_tern[w < -self.threshold] = -1.0
                total += w.numel()
                zeros += (w_tern == 0).sum().item()
        return zeros / total

def evaluate_accuracy(model, X, y, ternary=True):
    model.eval()
    with torch.no_grad():
        logits = model(X, ternary=ternary)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

def run_experiment(name, total_evals, lr=0.5):
    print(f"\n>>> TESTING {name} (TERNARY v13b) | Budget: {total_evals} evals")
    X_train, y_train, X_test, y_test = load_mnist_subset()
    model = TernaryModel(threshold=0.5)
    params = model.get_params()
    D = len(params)
    print(f"    Initial Sparsity: {model.get_sparsity():.1%}")
    
    t0 = time.time()
    evals = 0
    rng_mb = np.random.default_rng(SEED)
    criterion = nn.CrossEntropyLoss()

    k = math.ceil(math.log2(D))
    optimizer = DGEOptimizer(
        dim=D, lr=lr, delta=0.05, # Larger delta for boundary exploration
        total_steps=total_evals // (2 * k),
        clip_norm=0.1, seed=SEED
    )
    
    next_log = 10_000
    while evals < total_evals:
        idx = rng_mb.integers(0, len(y_train), size=BATCH_SIZE)
        Xb, yb = X_train[idx], y_train[idx]
        
        def f(p):
            model.set_params(p)
            with torch.no_grad():
                return criterion(model(Xb, ternary=True), yb).item()

        params, n = optimizer.step(f, params)
        evals += n
        
        if evals >= next_log:
            model.set_params(params)
            acc = evaluate_accuracy(model, X_test, y_test, ternary=True)
            sp = model.get_sparsity()
            print(f"      DGE Evals: {evals:>7} | Ternary Acc: {acc:.2%} | Sparsity: {sp:.1%}")
            next_log += 20_000

    final_acc = evaluate_accuracy(model, X_test, y_test, ternary=True)
    final_sp = model.get_sparsity()
    elapsed = time.time() - t0
    print(f"    FINAL TERNARY v13b: Acc={final_acc:.2%} | Sparsity={final_sp:.1%} | Time={elapsed:.1f}s")
    return final_acc, elapsed

if __name__ == "__main__":
    print("DGE: Revised Ternary Training (Uniform Init + Larger Delta)")
    run_experiment("Ternary DGE v13b", TOTAL_EVALS, lr=0.5)
