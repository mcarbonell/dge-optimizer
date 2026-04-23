"""
dge_selective_canine_v47.py
=================================
Experiment: Competitive Stereo Sniffing.
Instead of applying a lateral correction AFTER the DGE step, 
we use the nostrils to choose the BEST starting point for the DGE step.

Mechanism:
1. Generate an orthogonal vector v_perp relative to momentum.
2. Evaluate L_left at (x - eps*v_perp) and L_right at (x + eps*v_perp).
3. Select the nostril with the LOWER loss.
4. Perform the full DGE step starting from THAT nostril.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dge.torch_optimizer import TorchDGEOptimizer

try:
    from torchvision import datasets, transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False

try:
    import torch_directml
    device = torch_directml.device()
    print(f"DirectML: {device}")
except ImportError:
    device = torch.device("cpu")
    print("CPU")

# ---------------------------------------------------------------------------
# Models & Metrics
# ---------------------------------------------------------------------------
class NeuronOrderedMLP:
    def __init__(self, arch):
        self.arch = list(arch)
        self.layer_block_sizes = []
        self.layer_param_counts = []
        for a, b in zip(arch[:-1], arch[1:]):
            block_sz = a + 1
            self.layer_block_sizes.append(block_sz)
            self.layer_param_counts.append(block_sz * b)
        self.dim = sum(self.layer_param_counts)

    def forward(self, X, params_batch):
        P = params_batch.shape[0]
        h = X.unsqueeze(0).expand(P, -1, -1)
        offset = 0
        for i, (l_in, l_out) in enumerate(zip(self.arch[:-1], self.arch[1:])):
            sz = self.layer_param_counts[i]
            block_sz = self.layer_block_sizes[i]
            layer_p = params_batch[:, offset : offset + sz].view(P, l_out, block_sz)
            W = layer_p[:, :, :l_in].transpose(1, 2) 
            b = layer_p[:, :, l_in:].view(P, 1, l_out)
            h = torch.bmm(h, W) + b
            if l_out != self.arch[-1]: h = torch.relu(h)
            offset += sz
        return h

# ---------------------------------------------------------------------------
# Competitive Canine Optimizer
# ---------------------------------------------------------------------------
class CompetitiveCanineOptimizer(TorchDGEOptimizer):
    def step_selective(self, f_batched, x: torch.Tensor, eps_sniff=1e-3):
        n_evals = 0
        
        # 1. Lateral Sniff (Selection)
        m = self.m
        if torch.norm(m) > 1e-9:
            r = torch.randn_like(m)
            v_perp = r - (torch.dot(r, m) / torch.dot(m, m)) * m
            v_perp = v_perp / (torch.norm(v_perp) + 1e-8)
            
            p_nostrils = torch.stack([
                eps_sniff * v_perp,
                -eps_sniff * v_perp
            ])
            X_sniff = x.unsqueeze(0) + p_nostrils
            L = f_batched(X_sniff)
            n_evals += 2
            
            # Choose the best starting point
            if L[0] < L[1]:
                x_start = x + eps_sniff * v_perp
            else:
                x_start = x - eps_sniff * v_perp
        else:
            x_start = x
            
        # 2. Perform DGE Step from the best nostril
        x_new, n_dge = super().step(f_batched, x_start)
        n_evals += n_dge
        
        return x_new, n_evals

def load_full_mnist():
    assert HAS_TV
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr_d = ((ds_tr.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    y_tr_d = ds_tr.targets
    X_te_d = ((ds_te.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    y_te_d = ds_te.targets
    return X_tr_d, y_tr_d, X_te_d, y_te_d

def batched_ce(logits, targets):
    P, B, C = logits.shape
    t = targets.unsqueeze(0).expand(P, -1)
    return F.cross_entropy(logits.reshape(P*B, C), t.reshape(-1),
                           reduction='none').view(P, B).mean(dim=1)

def zo_acc(model, X, y, params, chunk=1000):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), chunk):
            lo = model.forward(X[i:i+chunk], params.unsqueeze(0)).squeeze(0)
            correct += (lo.argmax(1) == y[i:i+chunk]).sum().item()
    return correct / len(y)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ARCH          = (784, 128, 64, 10)
BUDGET_DGE    = 500_000
LR_ZO         = 0.05
DELTA         = 1e-3
BATCH_SIZE    = 256
LOG_INTERVAL  = 25_000

if __name__ == "__main__":
    X_tr_all, y_tr_all, X_te_all, y_te_all = load_full_mnist()
    X_tr_d = X_tr_all.to(device)
    y_tr_d = y_tr_all.to(device)
    X_te_d = X_te_all.to(device)
    y_te_d = y_te_all.to(device)

    model = NeuronOrderedMLP(ARCH)
    layer_sizes = model.layer_param_counts
    k_blocks = [512, 64, 10]

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v47: Competitive Canine Sniffing")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print(f"Mechanism: Selection via Nostrils -> DGE Step from best nostril")
    print(f"{'='*70}")

    seed = 42
    torch.manual_seed(seed)
    params0 = torch.zeros(model.dim, device=device)
    offset = 0
    for l_in, l_out in zip(ARCH[:-1], ARCH[1:]):
        std = math.sqrt(2.0 / l_in)
        sz = l_out * (l_in + 1)
        layer_p = torch.zeros((l_out, l_in + 1), device=device)
        layer_p[:, :l_in] = torch.randn((l_out, l_in), device=device) * std
        params0[offset : offset + sz] = layer_p.flatten()
        offset += sz

    total_k = sum(k_blocks)
    total_steps = BUDGET_DGE // (2 * total_k + 2)
    opt = CompetitiveCanineOptimizer(
        dim=model.dim, layer_sizes=layer_sizes, k_blocks=k_blocks,
        lr=LR_ZO, delta=DELTA, total_steps=total_steps,
        consistency_window=20, seed=seed, device=device, chunk_size=128
    )

    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)
    evals = 0
    best_test = 0.0
    t0 = time.time()
    next_log = LOG_INTERVAL

    print(f"\n  [Selective Canine] evals/step = {2 * total_k + 2}")
    print(f"  {'evals':>10}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}")
    print(f"  {'-'*45}")

    while evals < BUDGET_DGE:
        idx = torch.randperm(60000, generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr_d[idx], y_tr_d[idx]

        def f_batched(p_batch):
            with torch.no_grad():
                logits = model.forward(Xb, p_batch)
                return batched_ce(logits, yb)
        
        # Selective Step
        params, n = opt.step_selective(f_batched, params, eps_sniff=DELTA)
        evals += n

        if evals >= next_log or evals >= BUDGET_DGE:
            te_acc  = zo_acc(model, X_te_d, y_te_d, params)
            best_test = max(best_test, te_acc)
            elapsed = time.time() - t0
            print(f"  {evals:>10,}  {te_acc:>8.2%}  {best_test:>8.2%}  {elapsed:>6.0f}s")
            next_log += LOG_INTERVAL
