"""
dge_k_scaling_v35.py
=================================
Experiment to test the sensitivity of DGE to the number of blocks (K).
We test different scaling formulas for K relative to the layer size D:
  - O(1): Constant (e.g. K=32 for all layers)
  - O(log D): Logarithmic scaling
  - O(sqrt D): Square root scaling
  - O(D / 100): 1% of the layer size (close to the original baseline)
  - O(D / 10): 10% of the layer size

Budget is fixed at 500,000 evaluations to make the test fast.
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
# Config
# ---------------------------------------------------------------------------
ARCH          = (784, 128, 64, 10)
BUDGET_DGE    = 500_000   # Presupuesto rápido
LR_ZO         = 0.05
DELTA         = 1e-3
BATCH_SIZE    = 256
SEEDS         = [42]      # Solo 1 semilla para iterar super rápido
LOG_INTERVAL  = 25_000

# ---------------------------------------------------------------------------
# K-Scaling Formulas
# ---------------------------------------------------------------------------
def get_k_blocks(sizes, rule):
    if rule == "O(1)":
        return [32 for _ in sizes]
    elif rule == "O(log_D)":
        return [max(1, int(math.log2(D))) for D in sizes]
    elif rule == "O(sqrt_D)":
        return [max(1, int(math.sqrt(D))) for D in sizes]
    elif rule == "O(D/100)":
        return [max(1, int(D * 0.01)) for D in sizes]
    elif rule == "O(D/10)":
        return [max(1, int(D * 0.1)) for D in sizes]
    else:
        raise ValueError(f"Unknown rule: {rule}")

# ---------------------------------------------------------------------------
# Models & Metrics
# ---------------------------------------------------------------------------
class BatchedMLP:
    def __init__(self, arch):
        self.arch = list(arch)
        self.sizes = [a * b + b for a, b in zip(arch[:-1], arch[1:])]
        self.dim = sum(self.sizes)

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

def run_dge(name, k_blocks, model, params0, X_tr, y_tr, X_te, y_te, seed):
    
    total_steps = BUDGET_DGE // (2 * sum(k_blocks))
    
    opt = TorchDGEOptimizer(
        dim=model.dim,
        layer_sizes=model.sizes,
        k_blocks=k_blocks,
        lr=LR_ZO,
        delta=DELTA,
        total_steps=total_steps,
        consistency_window=20, # Usamos consistencia V1 por defecto ya que fue la mejor
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

    hdr = f"  {'evals':>10}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}"
    print(f"\n  [{name}]  K={k_blocks} (sum={sum(k_blocks)})")
    print(hdr)
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
    internal_overhead = wall_time - f_eval_time

    return {
        "method": name, 
        "k_blocks": k_blocks,
        "seed": seed,
        "best_test_acc": round(best_test, 4),
        "curve_evals": curve_evals, 
        "curve_acc": curve_acc,
        "total_evals": evals, 
        "wall_time": round(wall_time, 2),
        "f_eval_time": round(f_eval_time, 2),
        "internal_overhead_time": round(internal_overhead, 2)
    }

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_full_mnist():
    assert HAS_TV
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = ((ds_tr.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    X_te = ((ds_te.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    return X_tr, ds_tr.targets, X_te, ds_te.targets

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_tr_all, y_tr_all, X_te_all, y_te_all = load_full_mnist()
    X_tr_d = X_tr_all.to(device)
    y_tr_d = y_tr_all.to(device)
    X_te_d = X_te_all.to(device)
    y_te_d = y_te_all.to(device)

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v35: K-Blocks Scaling Formulas")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print(f"{'='*70}")

    all_results = []
    
    RULES = ["O(1)", "O(log_D)", "O(sqrt_D)", "O(D/100)", "O(D/10)"]
    summary = {r: [] for r in RULES}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        model = BatchedMLP(ARCH)
        torch.manual_seed(seed)
        params0 = torch.zeros(model.dim, device=device)
        off = 0
        for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
            std = math.sqrt(2.0 / l_in)
            w = l_in * l_out
            params0[off:off+w] = torch.randn(w, device=device) * std
            off += w + l_out

        for rule in RULES:
            k_blocks = get_k_blocks(model.sizes, rule)
            r = run_dge(rule, k_blocks, model, params0, X_tr_d, y_tr_d, X_te_d, y_te_d, seed)
            all_results.append(r)
            summary[rule].append(r["best_test_acc"])

    print(f"\n{'='*70}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*70}")
    for rule in RULES:
        accs = summary[rule]
        mean, std = np.mean(accs), np.std(accs)
        print(f"  {rule:<15}: {mean:.2%} ± {std:.2%}")

    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v35_k_scaling.json"
    
    payload = {
        "experiment": "v35_k_scaling",
        "budget_dge": BUDGET_DGE,
        "seeds": SEEDS,
        "summary": {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v} for m, v in summary.items()},
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON: {out_path}")
