"""
dge_dual_ema_stress_test_v69.py
==============================
Prueba de robustez: SMA vs DS-EMA con diferentes batches y semillas.
"""

import math
import os
import sys
import time
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dge.optimizer import DGEOptimizer

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

class DualEMADGE_MNIST:
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
        self.m, self.v = 0.9 * self.m + 0.1 * grad, 0.999 * self.v + 0.001 * (grad**2)
        mh, vh = self.m / (1.0 - 0.9**self.t), self.v / (1.0 - 0.999**self.t)
        s = np.sign(grad)
        self.ema_f = (1-self.alpha_f)*self.ema_f + self.alpha_f*s
        self.ema_s = (1-self.alpha_s)*self.ema_s + self.alpha_s*s
        mask = (np.sign(self.ema_f) == np.sign(self.ema_s)) * np.abs(self.ema_s)
        upd = lr * mask * mh / (np.sqrt(vh) + 1e-8)
        return x - upd, 2 * self.k

def load_data():
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True, download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr, y_tr = ds_tr.data.float().view(-1, 784).numpy() / 255.0, ds_tr.targets.numpy()
    X_te, y_te = ds_te.data.float().view(-1, 784).numpy() / 255.0, ds_te.targets.numpy()
    X_tr, X_te = (X_tr - 0.1307) / 0.3081, (X_te - 0.1307) / 0.3081
    return X_tr[:3000], y_tr[:3000], X_te[:600], y_te[:600]

def run_test(label, opt_type, params0, X_tr, y_tr, X_te, y_te, arch, budget, batch_size, seed):
    params = params0.copy()
    evals, best_acc = 0, 0
    D = n_params(arch)
    K = max(1, math.ceil(math.log2(D)))
    STEPS = budget // (2 * K)
    
    if opt_type == "SMA":
        opt = DGEOptimizer(dim=D, k_blocks=K, lr=0.005, total_steps=STEPS, consistency_window=20, seed=seed)
    else:
        opt = DualEMADGE_MNIST(dim=D, k_blocks=K, lr=0.005, total_steps=STEPS, seed=seed)
        
    rng_mb = np.random.default_rng(seed + 100)
    while evals < budget:
        idx = rng_mb.integers(0, len(y_tr), size=batch_size)
        def f(p): return cross_entropy(X_tr[idx], y_tr[idx], p, arch)
        params, n = opt.step(f, params)
        evals += n
    return accuracy(X_te, y_te, params, arch)

if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_data()
    ARCH = (784, 32, 10) # Red mas pequeña para velocidad en el stress test
    D = n_params(ARCH)
    BUDGET = 150_000
    BATCHES = [4, 8, 16, 32, 64, 128, 256]
    SEEDS = [42]
    
    results = {"SMA": {}, "DS-EMA": {}}
    
    print(f"STRESS TEST v69: SMA vs DS-EMA | Budget: {BUDGET} | Arch: {ARCH}")
    for bs in BATCHES:
        print(f"\nTesting Batch Size: {bs}")
        for opt_type in ["SMA", "DS-EMA"]:
            accs = []
            for s in SEEDS:
                rng_init = np.random.default_rng(s)
                p0 = rng_init.normal(0, 0.1, D)
                acc = run_test(f"{opt_type}_BS{bs}_S{s}", opt_type, p0, X_tr, y_tr, X_te, y_te, ARCH, BUDGET, bs, s)
                accs.append(acc)
            results[opt_type][bs] = {"mean": np.mean(accs), "std": np.std(accs)}
            print(f"  {opt_type:<6}: {np.mean(accs):>6.2%} +/- {np.std(accs):>5.2%}")

    with open("results/raw/v69_stress_test.json", "w") as f:
        json.dump(results, f, indent=2)
