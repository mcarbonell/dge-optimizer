"""
dge_bias_blocks_v40.py
=================================
Experiment to test the impact of "Noise-Free Bias Blocks".
We compare:
1. Baseline: Standard DGE with O(sqrt_D) blocks (biases mixed with weights).
2. Bias-Specific: Each bias is its own block (size 1), while weights use O(sqrt_D).

Architecture: (784, 128, 64, 10)
Budget: 500,000 evals
"""

import json
import math
import time
from collections import deque
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
class BatchedMLP:
    def __init__(self, arch):
        self.arch = list(arch)
        # Weights and biases are separate layers for Bias-Specific test
        self.raw_sizes = []
        for a, b in zip(arch[:-1], arch[1:]):
            self.raw_sizes.append(a * b) # Weights
            self.raw_sizes.append(b)     # Bias
        self.dim = sum(self.raw_sizes)

    def forward(self, X, params_batch):
        P = params_batch.shape[0]
        h = X.unsqueeze(0).expand(P, -1, -1)
        i = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            W = params_batch[:, i:i + l_in * l_out].view(P, l_in, l_out)
            i += l_in * l_out
            b = params_batch[:, i:i + l_out].view(P, 1, l_out)
            i += l_out
            h = torch.bmm(h, W) + b
            if l_out != self.arch[-1]:
                h = torch.relu(h)
        return h

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

def load_full_mnist():
    assert HAS_TV
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = ((ds_tr.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    X_te = ((ds_te.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    return X_tr, ds_tr.targets, X_te, ds_te.targets

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------
def run_experiment(name, model, params0, X_tr, y_tr, X_te, y_te, seed, layer_sizes, k_blocks):
    total_k = sum(k_blocks)
    total_steps = BUDGET_DGE // (2 * total_k)
    
    opt = TorchDGEOptimizer(
        dim=model.dim, 
        layer_sizes=layer_sizes, 
        k_blocks=k_blocks,
        lr=LR_ZO, 
        delta=DELTA, 
        total_steps=total_steps,
        consistency_window=20, 
        seed=seed, 
        device=device, 
        chunk_size=128
    )

    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)

    evals = 0
    best_test = 0.0
    curve_evals, curve_acc = [], []
    next_log = LOG_INTERVAL
    
    f_eval_time = 0.0
    t0 = time.time()

    print(f"\n  [{name}] evals/step = {2 * total_k}")
    print(f"  {'evals':>10}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}")
    print(f"  {'-'*45}")

    while evals < BUDGET_DGE:
        idx = torch.randperm(60000, generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr[idx], y_tr[idx]

        def f_batched(p_batch):
            with torch.no_grad():
                logits = model.forward(Xb, p_batch)
                return batched_ce(logits, yb)
        
        step_t0 = time.time()
        params, n = opt.step(f_batched, params)
        f_eval_time += (time.time() - step_t0)
        evals += n

        if evals >= next_log or evals >= BUDGET_DGE:
            te_acc  = zo_acc(model, X_te, y_te, params)
            best_test = max(best_test, te_acc)
            curve_evals.append(evals)
            curve_acc.append(round(te_acc, 4))
            elapsed = time.time() - t0
            print(f"  {evals:>10,}  {te_acc:>8.2%}  {best_test:>8.2%}  {elapsed:>6.0f}s")
            next_log += LOG_INTERVAL

    wall_time = time.time() - t0
    return {
        "method": name, "best_test_acc": best_test,
        "curve_evals": curve_evals, "curve_acc": curve_acc,
        "total_evals": evals, "wall_time": wall_time
    }

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARCH          = (784, 128, 64, 10)
BUDGET_DGE    = 500_000
LR_ZO         = 0.05
DELTA         = 1e-3
BATCH_SIZE    = 256
SEEDS         = [42]
LOG_INTERVAL  = 25_000

if __name__ == "__main__":
    X_tr_all, y_tr_all, X_te_all, y_te_all = load_full_mnist()
    X_tr_d = X_tr_all.to(device)
    y_tr_d = y_tr_all.to(device)
    X_te_d = X_te_all.to(device)
    y_te_d = y_te_all.to(device)

    model = BatchedMLP(ARCH)
    
    # 1. Configuration: Baseline O(sqrt_D)
    # Layers are [W1+b1, W2+b2, W3+b3]
    baseline_layer_sizes = [a * b + b for a, b in zip(ARCH[:-1], ARCH[1:])]
    baseline_k_blocks = [max(1, int(math.sqrt(sz))) for sz in baseline_layer_sizes]

    # 2. Configuration: Bias-Specific Blocks
    # Layers are [W1, b1, W2, b2, W3, b3]
    bias_layer_sizes = []
    bias_k_blocks = []
    for a, b in zip(ARCH[:-1], ARCH[1:]):
        # Weights
        w_sz = a * b
        bias_layer_sizes.append(w_sz)
        bias_k_blocks.append(max(1, int(math.sqrt(w_sz))))
        # Bias
        b_sz = b
        bias_layer_sizes.append(b_sz)
        bias_k_blocks.append(b_sz) # 1 block per bias (size 1)

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v40: Bias-Specific Blocks")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print(f"Baseline K = {sum(baseline_k_blocks)}")
    print(f"Bias-Specific K = {sum(bias_k_blocks)}")
    print(f"{'='*70}")

    summary = {}
    for seed in SEEDS:
        print(f"\n--- SEED {seed} ---")
        torch.manual_seed(seed)
        params0 = torch.zeros(model.dim, device=device)
        off = 0
        for l_in, l_out in zip(ARCH[:-1], ARCH[1:]):
            std = math.sqrt(2.0 / l_in)
            w = l_in * l_out
            params0[off:off+w] = torch.randn(w, device=device) * std
            off += w + l_out # Biases start at 0

        # Run Baseline
        res_base = run_experiment("Baseline_O(sqrt_D)", model, params0, X_tr_d, y_tr_d, X_te_d, y_te_d, seed, baseline_layer_sizes, baseline_k_blocks)
        
        # Run Bias-Specific
        res_bias = run_experiment("Bias_Specific", model, params0, X_tr_d, y_tr_d, X_te_d, y_te_d, seed, bias_layer_sizes, bias_k_blocks)

        summary[seed] = {"baseline": res_base["best_test_acc"], "bias": res_bias["best_test_acc"]}

    print(f"\n{'='*70}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*70}")
    for name in ["baseline", "bias"]:
        accs = [v[name] for v in summary.values()]
        print(f"  {name:<15}: {np.mean(accs):.2%} ± {np.std(accs):.2%}")

    out_path = Path(__file__).parent.parent / "results" / "raw" / "v40_bias_blocks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"experiment": "v40_bias_blocks", "results": summary}, f, indent=2)
    print(f"\nJSON: {out_path}")
