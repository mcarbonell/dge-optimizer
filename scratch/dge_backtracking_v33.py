"""
dge_backtracking_v33.py
=================================
Experiment to test Batched Backtracking Line Search on DGE.
Instead of applying the computed update step blindly, we evaluate
multiple magnitudes of the step [2.0, 1.0, 0.5, 0.25, 0.125, 0.0]
and pick the one that yields the lowest loss on the current minibatch.

This ensures we never overshoot and can potentially accelerate convergence
by taking larger steps when safe.
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
# Backtracking Optimizer
# ---------------------------------------------------------------------------

class BacktrackingDGEOptimizer(TorchDGEOptimizer):
    def __init__(self, *args, search_factors=(2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.0), **kwargs):
        super().__init__(*args, **kwargs)
        self.search_factors = torch.tensor(search_factors, device=self.device)
        
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
                
        # --- LINE SEARCH / BACKTRACKING ---
        upds = upd.unsqueeze(0) * self.search_factors.unsqueeze(1)
        candidates = x.unsqueeze(0) - upds
        
        with torch.no_grad():
            if self.chunk_size is not None and candidates.shape[0] > self.chunk_size:
                c_losses_list = []
                for i in range(0, candidates.shape[0], self.chunk_size):
                    c_losses_list.append(f_batched(candidates[i : i + self.chunk_size]))
                candidate_losses = torch.cat(c_losses_list, dim=0)
            else:
                candidate_losses = f_batched(candidates)
            
        best_idx = torch.argmin(candidate_losses)
        best_x = candidates[best_idx]
        
        evals_used = 2 * self.total_k + len(self.search_factors)
        return best_x, evals_used

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

def run_dge(name, optimizer_class, use_consistency, model, params0, X_tr, y_tr, X_te, y_te, seed):
    opt = optimizer_class(
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
    
    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)

    evals = 0
    best_test = 0.0
    curve_evals, curve_acc = [], []
    next_log = LOG_INTERVAL
    
    # Internal metrics
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
    print(f"EXPERIMENTO v33: Backtracking Line Search")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print(f"{'='*70}")

    all_results = []
    summary = {m: [] for m in ["DGE_Baseline", "DGE_Backtracking"]}

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

        r1 = run_dge("DGE_Baseline", TorchDGEOptimizer, True, model, params0, X_tr_d, y_tr_d, X_te_d, y_te_d, seed)
        all_results.append(r1)
        summary["DGE_Baseline"].append(r1["best_test_acc"])

        r2 = run_dge("DGE_Backtracking", BacktrackingDGEOptimizer, True, model, params0, X_tr_d, y_tr_d, X_te_d, y_te_d, seed)
        all_results.append(r2)
        summary["DGE_Backtracking"].append(r2["best_test_acc"])

    print(f"\n{'='*70}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*70}")
    for m in ["DGE_Baseline", "DGE_Backtracking"]:
        accs = summary[m]
        mean, std = np.mean(accs), np.std(accs)
        print(f"  {m:<18}: {mean:.2%} ± {std:.2%}")

    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v33_backtracking.json"
    
    payload = {
        "experiment": "v33_backtracking",
        "budget_dge": BUDGET_DGE,
        "seeds": SEEDS,
        "summary": {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v} for m, v in summary.items()},
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON: {out_path}")
