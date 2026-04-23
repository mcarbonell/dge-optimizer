"""
dge_structural_alternating_v64.py
=================================
Experiment to test ALTERNATING Node Perturbation via Structural DGE.
Combines the logic of v36 (Additive) and v63 (Multiplicative).

- Even steps: ADDITIVE perturbation (+/- delta).
- Odd steps: MULTIPLICATIVE perturbation (* (1 +/- delta)).

The hypothesis is that combining both might allow the network to:
1. Break dead weights (0.0) via additive steps (where multiplicative does nothing).
2. Avoid gradient washing and scale appropriately via multiplicative steps.

Blocks:
1. Fan-In blocks (all weights to a neuron + its bias)
2. Fan-Out blocks (all weights from a neuron)
3. Bias blocks (individual biases)
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
# Structural DGE Optimizer (Alternating)
# ---------------------------------------------------------------------------

class StructuralAlternatingOptimizer:
    def __init__(
        self,
        arch: list[int],
        dim: int,
        lr: float = 0.05,
        delta: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        total_steps: int = 10_000,
        lr_decay: float = 0.01,
        delta_decay: float = 0.1,
        consistency_window: int = 20,
        device: torch.device | str = "cpu",
        clip_norm: float | None = None,
        chunk_size: int | None = None,
    ):
        self.arch = arch
        self.dim = dim
        self.lr0 = lr
        self.delta0 = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.total_steps = total_steps
        self.lr_decay = lr_decay
        self.delta_decay = delta_decay
        self.consistency_window = consistency_window
        self.clip_norm = clip_norm
        self.chunk_size = chunk_size
        self.device = torch.device(device) if isinstance(device, str) else device
        
        self.m = torch.zeros(dim, device=self.device)
        self.v = torch.zeros(dim, device=self.device)
        self.t = 0
        self._sign_buffer = deque(maxlen=consistency_window) if consistency_window > 0 else None
        
        # Build P_base topological matrix
        total_blocks = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            total_blocks += 2 * l_out + l_in

        self.P_base = torch.zeros((total_blocks, self.dim), device=self.device)
        self.total_k = total_blocks
        
        offset = 0
        b_idx = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            w_offset = offset
            b_offset = offset + l_in * l_out
            
            # 1. Fan-In blocks
            for j in range(l_out):
                w_indices = w_offset + torch.arange(l_in) * l_out + j
                self.P_base[b_idx, w_indices] = 1.0
                self.P_base[b_idx, b_offset + j] = 1.0
                b_idx += 1
                
            # 2. Fan-Out blocks
            for i in range(l_in):
                w_indices = w_offset + i * l_out + torch.arange(l_out)
                self.P_base[b_idx, w_indices] = 1.0
                b_idx += 1
                
            # 3. Bias blocks
            for j in range(l_out):
                self.P_base[b_idx, b_offset + j] = 1.0
                b_idx += 1
                
            offset += l_in * l_out + l_out

    def _cosine(self, v0: float, decay: float) -> float:
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def step(self, f_batched, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        self.t += 1
        lr = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)
        
        is_additive = (self.t % 2 == 0)
        
        P_plus = self.P_base * delta
        
        if is_additive:
            X_batch_plus = x.unsqueeze(0) + P_plus
            X_batch_minus = x.unsqueeze(0) - P_plus
        else:
            X_batch_plus = x.unsqueeze(0) * (1.0 + P_plus)
            X_batch_minus = x.unsqueeze(0) * (1.0 - P_plus)
        
        X_batch = torch.empty((2 * self.total_k, self.dim), device=self.device)
        X_batch[0::2] = X_batch_plus
        X_batch[1::2] = X_batch_minus
        
        if self.chunk_size is not None and X_batch.shape[0] > self.chunk_size:
            losses_list = []
            for i in range(0, X_batch.shape[0], self.chunk_size):
                losses_list.append(f_batched(X_batch[i : i + self.chunk_size]))
            losses = torch.cat(losses_list, dim=0)
        else:
            losses = f_batched(X_batch)
            
        diffs = (losses[0::2] - losses[1::2]) / (2.0 * delta)
        
        # Distribute block scalar gradients back to parameters
        grad_base = diffs @ self.P_base
        
        if is_additive:
            grad = grad_base
        else:
            grad = grad_base * x
        
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        
        mask = 1.0
        if self._sign_buffer is not None:
            self._sign_buffer.append(torch.sign(grad))
            if len(self._sign_buffer) >= 2:
                mask = torch.stack(list(self._sign_buffer)).mean(0).abs()
        
        upd = lr * mask * mh / (torch.sqrt(vh) + self.eps)
        if self.clip_norm is not None:
            un = torch.norm(upd)
            if un > self.clip_norm:
                upd *= self.clip_norm / un
                
        return x - upd, 2 * self.total_k

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARCH          = (784, 128, 64, 10)
BUDGET_DGE    = 2500_000
LR_ZO         = 0.01  # Lower LR for stable combination
DELTA         = 1e-3  
BATCH_SIZE    = 256
SEEDS         = [42]
LOG_INTERVAL  = 25_000

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
    
    # Structural 
    blocks = sum(2 * l_out + l_in for l_in, l_out in zip(ARCH[:-1], ARCH[1:]))
    total_steps = BUDGET_DGE // (2 * blocks)
    opt = opt_cls(
        arch=ARCH, dim=model.dim, lr=LR_ZO, delta=DELTA, 
        total_steps=total_steps, consistency_window=20, 
        device=device, chunk_size=128
    )
    evals_per_step = 2 * blocks

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
    print(f"\n  [{name}]  evals/step = {evals_per_step}")
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
            
            mode_str = "ADD" if (opt.t % 2 == 0) else "MUL"
            print(f"  {evals:>10,}  {te_acc:>8.2%}  {best_test:>8.2%}  {elapsed:>6.0f}s  [Mode: {mode_str}]")
            next_log += LOG_INTERVAL

    wall_time = time.time() - t0
    internal_overhead = wall_time - f_eval_time

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
    print(f"EXPERIMENTO v64: Alternating Structural DGE (Add + Mul)")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print(f"{'='*70}")

    all_results = []
    
    METHODS = [
        ("DGE_Alternating", StructuralAlternatingOptimizer)
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
    out_path = out_dir / "v64_alternating_structural.json"
    
    payload = {
        "experiment": "v64_alternating_structural",
        "budget_dge": BUDGET_DGE,
        "seeds": SEEDS,
        "summary": {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v} for m, v in summary.items()},
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON: {out_path}")
