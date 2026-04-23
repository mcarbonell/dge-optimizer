"""
dge_lr_schedules_v57.py
=================================
Experiment: Learning Rate Schedules on Dynamic Budget
We take the best optimizer so far (Dynamic Budget v54) and test different
Learning Rate scheduling strategies.

1. Constant LR: No decay.
2. Cosine Decay: Smooth curve from LR_MAX down to LR_MIN.
3. Step Decay: Drops LR by a factor at 50% and 75% of training.
4. Exponential Decay: Continuous exponential decay.
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
# Dynamic Budget Optimizer with Custom Schedulers
# ---------------------------------------------------------------------------
class ScheduledDynamicOptimizer(TorchDGEOptimizer):
    def __init__(self, schedule_type="cosine", lr_min=1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_k = self.total_k
        self.layer_history = []
        self.m = torch.zeros(self.dim, device=self.device)
        self.v = torch.zeros(self.dim, device=self.device)
        self.schedule_type = schedule_type
        self.lr_min = lr_min

    def get_lr(self):
        """Custom learning rate schedules."""
        if self.schedule_type == "constant":
            return self.lr0
            
        elif self.schedule_type == "cosine":
            # This matches our standard cosine decay
            frac = min(self.t / max(self.total_steps, 1), 1.0)
            return self.lr_min + 0.5 * (self.lr0 - self.lr_min) * (1.0 + math.cos(math.pi * frac))
            
        elif self.schedule_type == "step":
            # Drop LR by 10x at 50% and 75% of training
            frac = self.t / max(self.total_steps, 1)
            if frac >= 0.75:
                return self.lr0 * 0.01
            elif frac >= 0.50:
                return self.lr0 * 0.1
            return self.lr0
            
        elif self.schedule_type == "exponential":
            # Exponential decay down to lr_min
            gamma = (self.lr_min / self.lr0) ** (1.0 / max(self.total_steps, 1))
            return self.lr0 * (gamma ** self.t)
            
        return self.lr0

    def _recalculate_k_blocks(self):
        mh = self.m / (1.0 - self.beta1 ** max(1, self.t))
        vh = self.v / (1.0 - self.beta2 ** max(1, self.t))
        snr = torch.abs(mh) / (torch.sqrt(vh) + self.eps)
        
        noise_scores = []
        off = 0
        for sz in self.layer_sizes:
            layer_v = self.v[off : off + sz].mean().item()
            noise_scores.append(layer_v * sz)
            off += sz
            
        total_score = sum(noise_scores) + 1e-12
        new_k = []
        rem = self.base_k
        
        for i, score in enumerate(noise_scores):
            if i == len(noise_scores) - 1:
                alloc = rem
            else:
                alloc = int(round((score / total_score) * self.base_k))
                rem -= alloc
            alloc = max(2, alloc)
            new_k.append(alloc)
            
        diff = sum(new_k) - self.base_k
        if diff > 0:
            max_idx = np.argmax(new_k)
            new_k[max_idx] -= diff
            
        self.k_blocks = new_k
        self.group_sizes = []
        self.pads = []
        for sz, k in zip(self.layer_sizes, self.k_blocks):
            grp = int(math.ceil(sz / k))
            pad = (grp * k) - sz
            self.group_sizes.append(grp)
            self.pads.append(pad)

    def step_scheduled(self, f_batched, x: torch.Tensor):
        self.t += 1
        lr = self.get_lr()
        delta = self._cosine(self.delta0, self.delta_decay) # Delta can stay cosine
        
        if self.t > 1 and self.t % 10 == 0:
            self._recalculate_k_blocks()
            
        P_plus = torch.zeros((self.base_k, self.dim), device=self.device)
        offset = 0; row_offset = 0; perms, signss = [] , []
        for sz, k, pad, grp in zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes):
            perm = torch.randperm(sz, generator=self.rng, device=self.device)
            signs = torch.randint(0, 2, (sz,), generator=self.rng, device=self.device).float() * 2 - 1
            perms.append(perm); signss.append(signs)
            perm_mat = (perm.view(k, grp) if pad==0 else torch.cat([perm, torch.zeros(pad, dtype=torch.long, device=self.device)]).view(k, grp)) + offset
            signs_mat = (signs.view(k, grp) if pad==0 else torch.cat([signs, torch.zeros(pad, device=self.device)]).view(k, grp)) * delta
            P_plus[row_offset : row_offset + k, :].scatter_(1, perm_mat, signs_mat)
            offset += sz; row_offset += k
            
        X_dge = torch.zeros((2 * self.base_k, self.dim), device=self.device)
        X_dge[0::2] = x + P_plus; X_dge[1::2] = x - P_plus
        L_dge_all = f_batched(X_dge)
        
        diffs = (L_dge_all[0::2] - L_dge_all[1::2]) / (2.0 * delta)
        grad_dge = torch.zeros(self.dim, device=self.device)
        off, row_off = 0, 0
        for i, (sz, k, pad, grp) in enumerate(zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes)):
            df_exp = diffs[row_off : row_off + k].unsqueeze(1).expand(k, grp).flatten()
            if pad > 0: df_exp = df_exp[:sz]
            grad_dge[off + perms[i]] = df_exp * signss[i]
            off += sz; row_off += k
            
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_dge
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad_dge ** 2)
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        upd = lr * mh / (torch.sqrt(vh) + self.eps)
        
        if self.clip_norm is not None:
            un = torch.norm(upd) 
            if un > self.clip_norm: upd *= self.clip_norm / un
            
        return x - upd, 2 * self.base_k

def load_full_mnist():
    assert HAS_TV
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True, download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    return ds_tr.data.float().view(-1,784).to(device), ds_tr.targets.to(device), \
           ds_te.data.float().view(-1,784).to(device), ds_te.targets.to(device)

def zo_acc(model, X, y, params, chunk=1000):
    X_norm = ((X/255.0) - 0.1307) / 0.3081
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), chunk):
            lo = model.forward(X_norm[i:i+chunk], params.unsqueeze(0)).squeeze(0)
            correct += (lo.argmax(1) == y[i:i+chunk]).sum().item()
    return correct / len(y)

def run_schedule_experiment(name, schedule_type, lr_max, lr_min, dim, layer_sizes, k_blocks, X_tr, y_tr, X_te, y_te, total_evals, seed):
    torch.manual_seed(seed)
    model = NeuronOrderedMLP(ARCH)
    params0 = torch.zeros(model.dim, device=device)
    offset = 0
    for l_in, l_out in zip(ARCH[:-1], ARCH[1:]):
        std = math.sqrt(2.0 / l_in)
        sz = l_out * (l_in + 1)
        layer_p = torch.zeros((l_out, l_in + 1), device=device)
        layer_p[:, :l_in] = torch.randn((l_out, l_in), device=device) * std
        params0[offset : offset + sz] = layer_p.flatten()
        offset += sz

    opt = ScheduledDynamicOptimizer(
        schedule_type=schedule_type, lr_min=lr_min,
        dim=model.dim, layer_sizes=layer_sizes, k_blocks=k_blocks,
        lr=lr_max, delta=1e-3, total_steps=total_evals//(sum(k_blocks)*2),
        seed=seed, device=device, chunk_size=128
    )

    params = params0.clone()
    rng_mb = torch.Generator(); rng_mb.manual_seed(seed + 100)
    evals = 0
    best_test = 0.0
    t0 = time.time()
    next_log = total_evals // 10

    print(f"\n--- {name} ({schedule_type.upper()}) ---")
    print(f"  LR Schedule: {lr_max} -> {lr_min}")
    print(f"  {'Evals':>10} | {'Test Acc':>9} | {'Best':>9} | {'LR':>8}")
    print("-" * 50)

    while evals < total_evals:
        idx = torch.randperm(60000, generator=rng_mb)[:256]
        Xb_raw, yb = X_tr[idx], y_tr[idx]
        Xb = ((Xb_raw/255.0) - 0.1307) / 0.3081

        def f_batched(p_batch):
            with torch.no_grad():
                logits = model.forward(Xb, p_batch)
                P, B, C = logits.shape
                t = yb.unsqueeze(0).expand(P, -1)
                return F.cross_entropy(logits.reshape(P*B, C), t.reshape(-1), reduction='none').view(P, B).mean(dim=1)
        
        current_lr = opt.get_lr()
        params, n = opt.step_scheduled(f_batched, params)
        evals += n

        if evals >= next_log or evals >= total_evals:
            te_acc  = zo_acc(model, X_te, y_te, params)
            best_test = max(best_test, te_acc)
            print(f"  {evals:>10,} | {te_acc:>9.2%} | {best_test:>9.2%} | {current_lr:>8.5f}")
            next_log += total_evals // 10
            
    print(f"  Time: {time.time() - t0:.1f}s")
    return best_test

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ARCH          = (784, 128, 64, 10)
BUDGET_DGE    = 500_000
K_BLOCKS      = [512, 64, 10]

if __name__ == "__main__":
    X_tr_raw, y_tr_d, X_te_raw, y_te_d = load_full_mnist()
    model = NeuronOrderedMLP(ARCH)
    layer_sizes = model.layer_param_counts

    print("="*60)
    print(f"EXPERIMENTO v57: Learning Rate Schedules on Dynamic Budget")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print("="*60)
    
    # We will test starting with a high LR (0.01) down to a low LR (0.001)
    LR_MAX = 0.02
    LR_MIN = 0.001
    
    results = {}
    
    schedules = ["constant", "cosine", "step", "exponential"]
    for sched in schedules:
        best_acc = run_schedule_experiment(
            name=f"Schedule: {sched}", 
            schedule_type=sched, 
            lr_max=LR_MAX, 
            lr_min=LR_MIN, 
            dim=model.dim, 
            layer_sizes=layer_sizes, 
            k_blocks=K_BLOCKS, 
            X_tr=X_tr_raw, y_tr=y_tr_d, 
            X_te=X_te_raw, y_te=y_te_d, 
            total_evals=BUDGET_DGE, 
            seed=42
        )
        results[sched] = best_acc
        
    print("\n" + "="*50)
    print("SUMMARY OF LR SCHEDULES (Best Accuracy)")
    print("="*50)
    for sched, acc in results.items():
        print(f"  {sched.capitalize():<12} : {acc:>8.2%}")
    print("="*50)
