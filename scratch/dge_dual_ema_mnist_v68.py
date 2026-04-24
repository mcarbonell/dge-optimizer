"""
dge_dual_ema_mnist_v68.py
=========================
Experimento: Dual Sign-EMA (DS-EMA) vs SMA Consistency en MNIST.

Compara la eficiencia de convergencia del nuevo prototipo DS-EMA contra
el DGEOptimizer canónico con ventana de consistencia de 20 pasos.

Configuracion:
  - Dataset: MNIST (3000 train, 600 test)
  - Arquitectura: 784 -> 64 -> 10 (MLP)
  - Budget: 300,000 evaluaciones
  - K_blocks: log2(D)
"""

import math
import os
import sys
import time
import json
from pathlib import Path

import numpy as np

# Permite importar DGEOptimizer del paquete local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dge.optimizer import DGEOptimizer

# ---------------------------------------------------------------------------
# Modelo MLP NumPy
# ---------------------------------------------------------------------------

def n_params(arch):
    return sum(a * b + b for a, b in zip(arch[:-1], arch[1:]))

def forward(X, params, arch):
    h, i = X, 0
    for l_in, l_out in zip(arch[:-1], arch[1:]):
        W = params[i:i + l_in * l_out].reshape(l_in, l_out)
        i += l_in * l_out
        b = params[i:i + l_out]
        i += l_out
        h = h @ W + b
        if l_out != arch[-1]: h = np.maximum(h, 0.0)
    return h

def cross_entropy(X, y, params, arch):
    logits = forward(X, params, arch)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-9)
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9)))

def accuracy(X, y, params, arch):
    return float((forward(X, params, arch).argmax(axis=1) == y).mean())

# ---------------------------------------------------------------------------
# Prototipo DualEMA integrado para MNIST
# ---------------------------------------------------------------------------

class DualEMADGE_MNIST:
    """Prototipo DS-EMA optimizado para el loop de entrenamiento."""
    def __init__(self, dim, k_blocks, lr=0.1, delta=1e-3, total_steps=10000, seed=42):
        self.dim, self.k, self.lr0, self.delta0, self.total_steps = dim, k_blocks, lr, delta, total_steps
        self.rng = np.random.default_rng(seed)
        self.m, self.v, self.t = np.zeros(dim), np.zeros(dim), 0
        self.ema_f, self.ema_s = np.zeros(dim), np.zeros(dim)
        self.alpha_f, self.alpha_s = 0.3, 0.05

    def step(self, f, x):
        self.t += 1
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        lr = self.lr0 * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        delta = self.delta0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        
        grad = np.zeros(self.dim)
        perm = self.rng.permutation(self.dim)
        blocks = np.array_split(perm, self.k)
        for block in blocks:
            if len(block) == 0: continue
            signs = self.rng.choice([-1.0, 1.0], size=len(block))
            pert = np.zeros(self.dim)
            pert[block] = signs * delta
            grad[block] = (f(x + pert) - f(x - pert)) / (2.0 * delta) * signs
            
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * (grad**2)
        mh, vh = self.m / (1.0 - 0.9**self.t), self.v / (1.0 - 0.999**self.t)
        
        s = np.sign(grad)
        self.ema_f = (1-self.alpha_f)*self.ema_f + self.alpha_f*s
        self.ema_s = (1-self.alpha_s)*self.ema_s + self.alpha_s*s
        
        mask = (np.sign(self.ema_f) == np.sign(self.ema_s)) * np.abs(self.ema_s)
        upd = lr * mask * mh / (np.sqrt(vh) + 1e-8)
        return x - upd, 2 * self.k

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_data():
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True, download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = ds_tr.data.float().view(-1, 784).numpy() / 255.0
    y_tr = ds_tr.targets.numpy()
    X_te = ds_te.data.float().view(-1, 784).numpy() / 255.0
    y_te = ds_te.targets.numpy()
    X_tr = (X_tr - 0.1307) / 0.3081
    X_te = (X_te - 0.1307) / 0.3081
    return X_tr[:3000], y_tr[:3000], X_te[:1000], y_te[:1000]

def train(label, opt, params0, X_tr, y_tr, X_te, y_te, arch, budget):
    params = params0.copy()
    evals, best_acc = 0, 0
    rng_mb = np.random.default_rng(42)
    t0 = time.time()
    
    print(f"\nTraining {label}...")
    while evals < budget:
        idx = rng_mb.integers(0, len(y_tr), size=256)
        Xb, yb = X_tr[idx], y_tr[idx]
        def f(p): return cross_entropy(Xb, yb, p, arch)
        params, n = opt.step(f, params)
        evals += n
        if evals % 50000 < n:
            acc = accuracy(X_te, y_te, params, arch)
            best_acc = max(best_acc, acc)
            print(f"  Evals: {evals:>7,} | Test Acc: {acc:>6.2%} | Time: {time.time()-t0:>4.0f}s")
    return best_acc

if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_data()
    ARCH = (784, 64, 10)
    D = n_params(ARCH)
    K = max(1, math.ceil(math.log2(D)))
    BUDGET = 300_000
    STEPS = BUDGET // (2 * K)
    
    rng_init = np.random.default_rng(42)
    params0 = rng_init.normal(0, 0.1, D) # Simple init para el test
    
    # 1. DGEOptimizer (SMA Window=20)
    opt_sma = DGEOptimizer(dim=D, k_blocks=K, lr=0.1, total_steps=STEPS, consistency_window=20, seed=42)
    acc_sma = train("DGEOptimizer (SMA Window=20)", opt_sma, params0, X_tr, y_tr, X_te, y_te, ARCH, BUDGET)
    
    # 2. DualEMADGE (v68)
    opt_ema = DualEMADGE_MNIST(dim=D, k_blocks=K, lr=0.1, total_steps=STEPS, seed=42)
    acc_ema = train("DualEMADGE (v68 DS-EMA)", opt_ema, params0, X_tr, y_tr, X_te, y_te, ARCH, BUDGET)
    
    print(f"\n{'='*40}")
    print(f"SMA Consistency Acc: {acc_sma:.2%}")
    print(f"DS-EMA Consistency Acc: {acc_ema:.2%}")
    print(f"{'='*40}")
    
    # Guardar resultados
    res_path = Path("results/raw/v68_mnist_dual_ema.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w") as f:
        json.dump({"sma_acc": acc_sma, "ema_acc": acc_ema, "budget": BUDGET, "arch": ARCH}, f, indent=2)
