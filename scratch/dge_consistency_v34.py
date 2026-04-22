"""
dge_consistency_v34.py
=================================
Experiment to test Direction-Consistency V2 (SNR-based).
Features:
  - SNR-based scaling (tanh(SNR)) instead of flat sign consistency.
  - Adaptive T per layer (larger T for first layers).
  - Chronic noise momentum (freezing or 10x penalty for chronically noisy vars).
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
# SNR Consistency Optimizer
# ---------------------------------------------------------------------------

class SNRDGEOptimizer(TorchDGEOptimizer):
    def __init__(self, *args, t_layers=None, chronic_threshold=0.1, chronic_penalty=0.0, **kwargs):
        # We disable the built-in consistency to implement our own
        kwargs.pop('consistency_window', None)
        super().__init__(*args, consistency_window=0, **kwargs)
        
        self.t_layers = t_layers if t_layers else [40, 20, 5]
        assert len(self.t_layers) == len(self.layer_sizes), "t_layers length must match layer_sizes"
        
        self.max_t = max(self.t_layers)
        
        # Buffer circular matricial para gradientes reales
        self.grad_buffer = torch.zeros((self.max_t, self.dim), device=self.device)
        self.buffer_idx = 0
        self.buffer_count = 0
        
        # Máscara a largo plazo (Momentum)
        self.long_term_mask = torch.ones(self.dim, device=self.device)
        self.chronic_threshold = chronic_threshold
        self.chronic_penalty = chronic_penalty

        # Precompute boolean masks for slicing each layer's indices in the buffer
        self.layer_indices = []
        offset = 0
        for sz in self.layer_sizes:
            self.layer_indices.append((offset, offset + sz))
            offset += sz

    def step(self, f_batched, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        self.t += 1
        lr = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)
        
        P_plus = torch.zeros((self.total_k, self.dim), device=self.device)
        offset = 0
        row_offset = 0
        perms = []
        signss = []
        
        for sz, k, pad, grp in zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes):
            perm = torch.randperm(sz, generator=self.rng, device=self.device)
            signs = torch.randint(0, 2, (sz,), generator=self.rng, device=self.device).float() * 2 - 1
            perms.append(perm)
            signss.append(signs)
            if pad > 0:
                perm_pad = torch.cat([perm, torch.zeros(pad, dtype=torch.long, device=self.device)])
                signs_pad = torch.cat([signs, torch.zeros(pad, device=self.device)])
            else:
                perm_pad = perm
                signs_pad = signs
            perm_mat = perm_pad.view(k, grp) + offset
            signs_mat = signs_pad.view(k, grp) * delta
            target_slice = P_plus[row_offset : row_offset + k, :]
            target_slice.scatter_(1, perm_mat, signs_mat)
            if pad > 0:
                target_slice[:, offset] = 0.0
                idx0_mask = (perm == 0)
                idx0_pos = idx0_mask.nonzero(as_tuple=True)[0]
                if len(idx0_pos) > 0:
                    block0 = idx0_pos[0] // grp
                    target_slice[block0, offset] = signs[idx0_pos[0]] * delta
            offset += sz
            row_offset += k
                
        P = torch.empty((2 * self.total_k, self.dim), device=self.device)
        P[0::2] = P_plus
        P[1::2] = -P_plus
        
        X_batch = x.unsqueeze(0) + P
        if self.chunk_size is not None and X_batch.shape[0] > self.chunk_size:
            losses_list = []
            for i in range(0, X_batch.shape[0], self.chunk_size):
                losses_list.append(f_batched(X_batch[i : i + self.chunk_size]))
            losses = torch.cat(losses_list, dim=0)
        else:
            losses = f_batched(X_batch)
        
        diffs = (losses[0::2] - losses[1::2]) / (2.0 * delta)
        grad = torch.zeros(self.dim, device=self.device)
        offset = 0
        row_offset = 0
        for i, (sz, k, pad, grp) in enumerate(zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes)):
            perm = perms[i]
            signs = signss[i]
            layer_diffs = diffs[row_offset : row_offset + k]
            diffs_exp = layer_diffs.unsqueeze(1).expand(k, grp).flatten()
            if pad > 0:
                diffs_exp = diffs_exp[:sz]
            grad[offset + perm] = diffs_exp * signs
            offset += sz
            row_offset += k
        
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        
        # --- SNR CONSISTENCY V2 LOGIC ---
        self.grad_buffer[self.buffer_idx] = grad
        self.buffer_idx = (self.buffer_idx + 1) % self.max_t
        self.buffer_count = min(self.buffer_count + 1, self.max_t)
        
        mask = torch.ones(self.dim, device=self.device)
        
        if self.buffer_count > 2:
            for i, (start, end) in enumerate(self.layer_indices):
                t_layer = self.t_layers[i]
                actual_t = min(t_layer, self.buffer_count)
                
                if actual_t <= 2:
                    continue
                
                # Fetch the last 'actual_t' gradients from the circular buffer
                # It's easier to just take all max_t elements, they are 0 if uninitialized, 
                # but we want exact mean/std over valid items. So we gather properly.
                indices = [(self.buffer_idx - 1 - j) % self.max_t for j in range(actual_t)]
                recent_grads = self.grad_buffer[indices, start:end]
                
                mean_g = recent_grads.mean(dim=0)
                std_g  = recent_grads.std(dim=0) + 1e-8
                
                snr = mean_g.abs() / std_g
                # Scale mask with tanh
                layer_mask = torch.tanh(snr)
                mask[start:end] = layer_mask
                
        # --- CHRONIC NOISE MOMENTUM ---
        # Update long-term mask
        self.long_term_mask = 0.99 * self.long_term_mask + 0.01 * mask
        
        # Apply chronic penalty
        chronic_idx = self.long_term_mask < self.chronic_threshold
        mask[chronic_idx] = self.chronic_penalty
        
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
BUDGET_DGE    = 1_000_000
K_BLOCKS      = (1024, 128, 16)
LR_ZO         = 0.05
DELTA         = 1e-3
BATCH_SIZE    = 256
SEEDS         = [42, 43, 44]
LOG_INTERVAL  = 50_000

EST_ZO_STEP      = 2 * sum(K_BLOCKS)
TOTAL_DGE_STEPS  = BUDGET_DGE  // EST_ZO_STEP

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

def run_dge(name, optimizer_class, use_consistency, model, params0, X_tr, y_tr, X_te, y_te, seed, **kwargs):
    opt_kwargs = dict(
        dim=model.dim,
        layer_sizes=model.sizes,
        k_blocks=list(K_BLOCKS),
        lr=LR_ZO,
        delta=DELTA,
        total_steps=TOTAL_DGE_STEPS,
        consistency_window=20 if use_consistency else 0,
        seed=seed,
        device=device,
        chunk_size=128
    )
    opt_kwargs.update(kwargs)
    opt = optimizer_class(**opt_kwargs)
    
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
    print(f"\n  [{name}]  budget={BUDGET_DGE:,}")
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
    print(f"EXPERIMENTO v34: Direction Consistency V2 (SNR + Adaptive T + Momentum)")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print(f"{'='*70}")

    all_results = []
    
    METHODS = [
        ("DGE_Baseline", TorchDGEOptimizer, False, {}),
        ("DGE_Consist_V1", TorchDGEOptimizer, True, {}),
        ("DGE_Consist_V2_Zero", SNRDGEOptimizer, False, {"chronic_penalty": 0.0}),
        ("DGE_Consist_V2_0.1", SNRDGEOptimizer, False, {"chronic_penalty": 0.1})
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

        for name, opt_cls, use_consist, kwargs in METHODS:
            r = run_dge(name, opt_cls, use_consist, model, params0, X_tr_d, y_tr_d, X_te_d, y_te_d, seed, **kwargs)
            all_results.append(r)
            summary[name].append(r["best_test_acc"])

    print(f"\n{'='*70}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*70}")
    for m in [x[0] for x in METHODS]:
        accs = summary[m]
        mean, std = np.mean(accs), np.std(accs)
        print(f"  {m:<20}: {mean:.2%} ± {std:.2%}")

    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v34_consistency_v2.json"
    
    payload = {
        "experiment": "v34_consistency_v2",
        "budget_dge": BUDGET_DGE,
        "seeds": SEEDS,
        "summary": {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v} for m, v in summary.items()},
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON: {out_path}")
