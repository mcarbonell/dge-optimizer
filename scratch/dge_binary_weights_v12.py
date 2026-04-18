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
ARCH         = (784, 128, 10) # Un poco más ancha para compensar la pérdida de precisión por binarización

torch.manual_seed(SEED)
np.random.seed(SEED)

def load_mnist_subset(n_train=N_TRAIN, n_test=N_TEST):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=False,  download=True, transform=transform) # Usamos test set para subset rápido
    
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
# MODELO BINARIO (Binary Weights)
# =============================================================================
class BinaryModel(nn.Module):
    def __init__(self, arch=ARCH):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(arch[i], arch[i+1], bias=False) for i in range(len(arch)-1)])
        # No bias for pure binary simplicity, or we could keep it. Let's keep it without bias.

    def forward(self, x, binary=True):
        h = x
        for i, layer in enumerate(self.layers):
            w = layer.weight
            if binary:
                # Use ONLY signs of weights!
                w = torch.sign(w)
            
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

def evaluate_accuracy(model, X, y, binary=True):
    model.eval()
    with torch.no_grad():
        logits = model(X, binary=binary)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

def run_experiment(name, total_evals, lr=0.5):
    print(f"\n>>> TESTING {name} (BINARY WEIGHTS {-1, 1}) | Budget: {total_evals} evals")
    X_train, y_train, X_test, y_test = load_mnist_subset()
    model = BinaryModel()
    params = model.get_params()
    D = len(params)
    print(f"    Parameters: {D:,}")
    
    t0 = time.time()
    evals = 0
    rng_mb = np.random.default_rng(SEED)
    criterion = nn.CrossEntropyLoss()

    k = math.ceil(math.log2(D))
    optimizer = DGEOptimizer(
        dim=D, lr=lr, delta=0.01, # Larger delta helps explore sign flips in FP space
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
                # Evaluamos el loss usando pesos BINARIOS
                return criterion(model(Xb, binary=True), yb).item()

        params, n = optimizer.step(f, params)
        evals += n
        
        if evals >= next_log:
            model.set_params(params)
            acc = evaluate_accuracy(model, X_test, y_test, binary=True)
            print(f"      DGE Evals: {evals:>7} | Binary Test Acc: {acc:.2%}")
            next_log += 20_000

    final_acc = evaluate_accuracy(model, X_test, y_test, binary=True)
    elapsed = time.time() - t0
    print(f"    FINAL BINARY DGE: Acc={final_acc:.2%} | Time={elapsed:.1f}s")
    return final_acc, elapsed

if __name__ == "__main__":
    print("DGE: Training a network where inference uses ONLY signs of weights (Quantized {-1, 1})")
    run_experiment("Binary DGE", TOTAL_EVALS, lr=0.5)
