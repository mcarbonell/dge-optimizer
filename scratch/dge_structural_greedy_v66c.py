"""
dge_structural_greedy_v66b.py
=================================
Experiment to test GREEDY ADAPTIVE Node Perturbation (Evolution Strategies).
No EMA (No Adam). No learning rate.
Immediate updates: If a perturbation improves the loss, apply immediately.

TWIST (v66b): CAPPED Adaptive Deltas per block! (1/5th Success Rule)
Same as v66, but we apply a strict maximum bound to the deltas to prevent
"batch overfitting" where lucky blocks grow their deltas to 0.5 and 
destabilize the entire network on the next batch.

- max_delta_add = 0.01 (1% max absolute change)
- max_delta_mul = 0.05 (5% max relative change)
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
# Greedy Adaptive Structural Optimizer (Capped)
# ---------------------------------------------------------------------------

class CappedGreedyAdaptiveOptimizer:
    def __init__(
        self,
        arch: list[int],
        dim: int,
        delta_add_init: float = 1e-3, 
        delta_mul_init: float = 1e-3, 
        success_factor: float = 1.2,
        fail_factor: float = 0.90,
        min_delta: float = 1e-6,
        max_delta_add: float = 0.05, # CAPPED!
        max_delta_mul: float = 0.05, # CAPPED!
        device: torch.device | str = "cpu",
    ):
        self.arch = arch
        self.dim = dim
        self.success_factor = success_factor
        self.fail_factor = fail_factor
        self.min_delta = min_delta
        self.max_delta_add = max_delta_add
        self.max_delta_mul = max_delta_mul
        self.device = torch.device(device) if isinstance(device, str) else device
        self.t = 0
        
        # Build block indices
        self.blocks = []
        offset = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            w_offset = offset
            b_offset = offset + l_in * l_out
            
            # 1. Fan-In blocks
            for j in range(l_out):
                w_indices = w_offset + torch.arange(l_in, device=self.device) * l_out + j
                b_index = torch.tensor([b_offset + j], device=self.device)
                self.blocks.append(torch.cat([w_indices, b_index]))
                
            # 2. Fan-Out blocks
            for i in range(l_in):
                w_indices = w_offset + i * l_out + torch.arange(l_out, device=self.device)
                self.blocks.append(w_indices)
                
            # 3. Bias blocks
            for j in range(l_out):
                self.blocks.append(torch.tensor([b_offset + j], device=self.device))
                
            offset += l_in * l_out + l_out
            
        self.num_blocks = len(self.blocks)
        
        # ADAPTIVE DELTAS PER BLOCK
        self.deltas_add = torch.full((self.num_blocks,), delta_add_init, device=self.device)
        self.deltas_mul = torch.full((self.num_blocks,), delta_mul_init, device=self.device)

    def step_greedy(self, f_batched, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        self.t += 1
        is_additive = (self.t % 2 == 0)
        
        evals = 0
        
        # Evaluate base loss
        L0 = f_batched(x.unsqueeze(0))[0].item()
        evals += 1
        
        # Random permutation of blocks for this sweep
        perm = torch.randperm(self.num_blocks, device="cpu").tolist()
        
        for b_idx in perm:
            indices = self.blocks[b_idx]
            
            # Fetch the adaptive delta for this specific block
            delta = self.deltas_add[b_idx].item() if is_additive else self.deltas_mul[b_idx].item()
            
            x_plus = x.clone()
            x_minus = x.clone()
            
            if is_additive:
                x_plus[indices] += delta
                x_minus[indices] -= delta
            else:
                x_plus[indices] *= (1.0 + delta)
                x_minus[indices] *= (1.0 - delta)
                
            # Evaluate both perturbations
            X_batch = torch.stack([x_plus, x_minus])
            losses = f_batched(X_batch)
            evals += 2
            
            L_plus = losses[0].item()
            L_minus = losses[1].item()
            
            improved = False
            # Greedy acceptance
            if L_plus < L0 and L_plus <= L_minus:
                x = x_plus
                L0 = L_plus
                improved = True
            elif L_minus < L0 and L_minus < L_plus:
                x = x_minus
                L0 = L_minus
                improved = True
                
            # -------------------------------------------------------------
            # EVOLUTION STRATEGY: Update the Delta!
            # -------------------------------------------------------------
            new_delta = delta * self.success_factor if improved else delta * self.fail_factor
            
            # Apply strict ceilings
            if is_additive:
                new_delta = max(self.min_delta, min(self.max_delta_add, new_delta))
                self.deltas_add[b_idx] = new_delta
            else:
                new_delta = max(self.min_delta, min(self.max_delta_mul, new_delta))
                self.deltas_mul[b_idx] = new_delta
                
        return x, evals

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARCH          = (784, 128, 64, 10)
BUDGET_DGE    = 2500_000
DELTA_ADD_0   = 2e-3  
DELTA_MUL_0   = 2e-3  
BATCH_SIZE    = 8192
SEEDS         = [42]
LOG_INTERVAL  = 50_000

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

def run_dge(name, opt_cls, model, params0, X_tr, y_tr, X_te, y_te, seed):
    
    opt = opt_cls(
        arch=ARCH, dim=model.dim, 
        delta_add_init=DELTA_ADD_0, delta_mul_init=DELTA_MUL_0, 
        device=device
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

    hdr = f"  {'evals':>10}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}  {'Mode'}"
    print(f"\n  [{name}]  evals/sweep = {1 + opt.num_blocks * 2}")
    print(hdr)
    print(f"  {'-'*55}")

    while evals < BUDGET_DGE:
        idx = torch.randperm(60000, generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr[idx], y_tr[idx]

        def f_batched(p_batch):
            with torch.no_grad():
                logits = model.forward(Xb, p_batch)
                return batched_ce(logits, yb)
        
        step_t0 = time.time()
        params, n = opt.step_greedy(f_batched, params)
        f_eval_time += (time.time() - step_t0)
        evals += n

        if evals >= next_log or evals >= BUDGET_DGE:
            te_acc  = zo_acc(model, X_te, y_te, params)
            best_test = max(best_test, te_acc)
            curve_evals.append(evals)
            curve_acc.append(round(te_acc, 4))
            elapsed = time.time() - t0
            
            mode_str = "ADD" if (opt.t % 2 == 0) else "MUL"
            print(f"  {evals:>10,}  {te_acc:>8.2%}  {best_test:>8.2%}  {elapsed:>6.0f}s  [{mode_str}]")
            next_log += LOG_INTERVAL

    wall_time = time.time() - t0
    internal_overhead = wall_time - f_eval_time
    
    print(f"  --> Final Median Delta Add: {opt.deltas_add.median().item():.5f} | Max: {opt.deltas_add.max().item():.5f}")
    print(f"  --> Final Median Delta Mul: {opt.deltas_mul.median().item():.5f} | Max: {opt.deltas_mul.max().item():.5f}")

    return {
        "method": name, 
        "seed": seed,
        "best_test_acc": round(best_test, 4),
        "curve_evals": curve_evals, 
        "curve_acc": curve_acc,
        "total_evals": evals, 
        "wall_time": round(wall_time, 2),
        "f_eval_time": round(f_eval_time, 2),
        "internal_overhead_time": round(internal_overhead, 2)
    }

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
    print(f"EXPERIMENTO v66b: CAPPED Greedy ADAPTIVE Structural")
    print(f"NO EMA. Immediate updates. Adaptive Deltas (Max limits enforced!)")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print(f"{'='*70}")

    all_results = []
    
    METHODS = [
        ("DGE_Greedy_Capped", CappedGreedyAdaptiveOptimizer)
    ]
    summary = {m[0]: [] for m in METHODS}

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

        for name, opt_cls in METHODS:
            r = run_dge(name, opt_cls, model, params0, X_tr_d, y_tr_d, X_te_d, y_te_d, seed)
            all_results.append(r)
            summary[name].append(r["best_test_acc"])

    print(f"\n{'='*70}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*70}")
    for name in [x[0] for x in METHODS]:
        accs = summary[name]
        mean, std = np.mean(accs), np.std(accs)
        print(f"  {name:<25}: {mean:.2%} ± {std:.2%}")

    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v66c_greedy_capped_structural.json"
    
    payload = {
        "experiment": "v66c_greedy_capped_structural",
        "budget_dge": BUDGET_DGE,
        "seeds": SEEDS,
        "summary": {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v} for m, v in summary.items()},
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON: {out_path}")
