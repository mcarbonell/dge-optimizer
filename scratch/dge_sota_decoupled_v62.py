"""
dge_sota_decoupled_v62.py
=================================
Experiment: Mathematical Hyperparameter Scaling + FIXED Budget
Applies the decoupling lesson from v60 + math scaling from v61,
but using a FIXED BUDGET instead of the Dynamic Budget.

Hypothesis: In v61 we used Dynamic Budget which collapsed to [582, 2, 2].
By giving layers 2 and 3 only 2 blocks, their gradient estimates were
extremely noisy. Boosting their LR (via math scaling) while they had
no blocks likely injected massive noise. 
If we fix the budget to [512, 64, 10], deep layers will have enough
evaluations to leverage their mathematically scaled, larger Deltas and LRs!
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
# Optimizer: Fixed Budget + Decoupled Math Scaling
# ---------------------------------------------------------------------------
class FixedDecoupledOptimizer(TorchDGEOptimizer):
    def __init__(self, layer_lrs0, layer_lrs_min, layer_deltas0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_k = self.total_k
        self.layer_lrs0 = layer_lrs0
        self.layer_lrs_min = layer_lrs_min
        self.layer_deltas0 = layer_deltas0
        self.g_prev = None
        self.m = torch.zeros(self.dim, device=self.device)
        self.v = torch.zeros(self.dim, device=self.device)

        self.group_sizes = []
        self.pads = []
        for sz, k in zip(self.layer_sizes, self.k_blocks):
            grp = int(math.ceil(sz / k))
            pad = (grp * k) - sz
            self.group_sizes.append(grp)
            self.pads.append(pad)

    def get_layer_lr(self, lr0, lr_min):
        # Exponential LR Schedule
        gamma = (lr_min / lr0) ** (1.0 / max(self.total_steps, 1))
        return lr0 * (gamma ** self.t)

    def step_sota(self, f_batched, x: torch.Tensor, sniff_ratio=0.1):
        self.t += 1
        
        # 1. Block Perturbations (Per-Layer Delta)
        P_plus = torch.zeros((self.base_k, self.dim), device=self.device)
        offset = 0; row_offset = 0; perms = []; signss = []; current_deltas = []
        for i, (sz, k, pad, grp) in enumerate(zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes)):
            layer_delta = self._cosine(self.layer_deltas0[i], self.delta_decay)
            current_deltas.append(layer_delta)
            
            perm = torch.randperm(sz, generator=self.rng, device="cpu").to(self.device)
            signs = torch.randint(0, 2, (sz,), generator=self.rng, device="cpu").float().to(self.device) * 2 - 1
            perms.append(perm); signss.append(signs)
            
            perm_mat = (perm.view(k, grp) if pad==0 else torch.cat([perm, torch.zeros(pad, dtype=torch.long, device=self.device)]).view(k, grp)) + offset
            signs_mat = (signs.view(k, grp) if pad==0 else torch.cat([signs, torch.zeros(pad, device=self.device)]).view(k, grp)) * layer_delta
            P_plus[row_offset : row_offset + k, :].scatter_(1, perm_mat, signs_mat)
            offset += sz; row_offset += k
            
        # 2. Forward DGE
        X_dge = torch.zeros((2 * self.base_k, self.dim), device=self.device)
        X_dge[0::2] = x + P_plus; X_dge[1::2] = x - P_plus
        L_dge_all = f_batched(X_dge)
        
        # 3. Compute grad_dge (Per-Layer Delta Scaling)
        diffs_unscaled = (L_dge_all[0::2] - L_dge_all[1::2]) / 2.0
        grad_dge = torch.zeros(self.dim, device=self.device)
        off, row_off = 0, 0
        for i, (sz, k, pad, grp) in enumerate(zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes)):
            df_exp = (diffs_unscaled[row_off : row_off + k] / current_deltas[i]).unsqueeze(1).expand(k, grp).flatten()
            if pad > 0: df_exp = df_exp[:sz]
            grad_dge[off + perms[i]] = df_exp * signss[i]
            off += sz; row_off += k
            
        # 4. Hybrid Lateral Sniff (Megazord Curvature)
        sniff_delta = sum(current_deltas) / len(current_deltas)
        v_perp = torch.zeros(self.dim, device=self.device)
        if self.g_prev is not None:
            off = 0
            for i, (sz, k, pad, grp) in enumerate(zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes)):
                for j in range(k):
                    start = off + j * grp
                    end = min(off + (j + 1) * grp, off + sz)
                    if start >= end: break
                    g_curr_b = grad_dge[start:end]
                    g_prev_b = self.g_prev[start:end]
                    v_curv_b = g_curr_b - g_prev_b
                    gn2 = torch.dot(g_curr_b, g_curr_b)
                    if gn2 > 1e-12:
                        v_p_b = v_curv_b - (torch.dot(v_curv_b, g_curr_b) / gn2) * g_curr_b
                        n_p_b = torch.norm(v_p_b)
                        if n_p_b > 1e-12: v_p_b = v_p_b / n_p_b
                        v_perp[start:end] = v_p_b
                off += sz

        n_v_perp = torch.norm(v_perp)
        if n_v_perp < 1e-9:
            off = 0
            for i, (sz, k, pad, grp) in enumerate(zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes)):
                for j in range(k):
                    start = off + j * grp
                    end = min(off + (j + 1) * grp, off + sz)
                    if start >= end: break
                    g_curr_b = grad_dge[start:end]
                    r_b = torch.randn_like(g_curr_b)
                    gn2 = torch.dot(g_curr_b, g_curr_b)
                    if gn2 > 1e-12:
                        v_p_b = r_b - (torch.dot(r_b, g_curr_b) / gn2) * g_curr_b
                        n_p_b = torch.norm(v_p_b)
                        if n_p_b > 1e-12: v_p_b = v_p_b / n_p_b
                    else:
                        v_p_b = r_b / (torch.norm(r_b) + 1e-12)
                    v_perp[start:end] = v_p_b
                off += sz
            v_perp = v_perp / (torch.norm(v_perp) + 1e-12)
        else:
            v_perp = v_perp / n_v_perp

        P_sniff = torch.stack([v_perp * sniff_delta, -v_perp * sniff_delta])
        L_sniff = f_batched(x.unsqueeze(0) + P_sniff)
        grad_perp = ((L_sniff[0] - L_sniff[1]) / (2.0 * sniff_delta)) * v_perp
        
        n_dge = torch.norm(grad_dge)
        n_perp = torch.norm(grad_perp)
        if n_perp > 1e-9 and n_dge > 1e-9:
            grad_perp_s = grad_perp * (n_dge / n_perp) * sniff_ratio
        else:
            grad_perp_s = torch.zeros_like(grad_perp)
            
        self.g_prev = grad_dge.clone()
            
        # 5. Asymmetric Adam EMA
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_dge
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad_dge ** 2)
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        
        # 6. Apply Per-Layer LR & Update
        upd = torch.zeros_like(grad_dge)
        off = 0
        for i, sz in enumerate(self.layer_sizes):
            layer_lr = self.get_layer_lr(self.layer_lrs0[i], self.layer_lrs_min[i])
            layer_mh = mh[off:off+sz]
            layer_vh = vh[off:off+sz]
            upd[off:off+sz] = layer_lr * layer_mh / (torch.sqrt(layer_vh) + self.eps)
            off += sz
            
        if self.clip_norm is not None:
            un = torch.norm(upd) 
            if un > self.clip_norm: upd *= self.clip_norm / un
            
        upd += grad_perp_s * (sum(self.layer_lrs0)/len(self.layer_lrs0)) 
        
        return x - upd, 2 * self.base_k + 2

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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ARCH          = (784, 128, 64, 10)
BUDGET_DGE    = 500_000
BATCH_SIZE    = 256
LOG_INTERVAL  = 25_000

if __name__ == "__main__":
    X_tr_raw, y_tr_d, X_te_raw, y_te_d = load_full_mnist()
    model = NeuronOrderedMLP(ARCH)
    layer_sizes = model.layer_param_counts
    
    # FIXED BUDGET from v60
    k_blocks = [512, 64, 10]
    
    # -----------------------------------------------------------------------
    # MATHEMATICAL SCALING RULE: Proportional to initialization standard deviation
    # -----------------------------------------------------------------------
    scales = [math.sqrt(2.0 / l_in) for l_in in ARCH[:-1]]
    norm_scales = [s / scales[0] for s in scales]
    
    BASE_LR_MAX = 0.01
    BASE_LR_MIN = 0.001
    BASE_DELTA  = 1e-3
    
    LRS_0   = [BASE_LR_MAX * s for s in norm_scales]
    LRS_MIN = [BASE_LR_MIN * s for s in norm_scales]
    DELTAS  = [BASE_DELTA * s for s in norm_scales]

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v62: MATHEMATICAL SCALING + FIXED BUDGET")
    print(f"Fix: Disabling Dynamic Budget so deep layers get proper evaluations.")
    for i in range(len(ARCH)-1):
        print(f"  L{i+1} (k={k_blocks[i]:>3}): Scale={norm_scales[i]:>4.2f}x | LR={LRS_0[i]:>6.4f} | DELTA={DELTAS[i]:>7.5f}")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
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

    opt = FixedDecoupledOptimizer(
        layer_lrs0=LRS_0,
        layer_lrs_min=LRS_MIN,
        layer_deltas0=DELTAS,
        dim=model.dim, layer_sizes=layer_sizes, k_blocks=k_blocks,
        total_steps=BUDGET_DGE//(sum(k_blocks)*2 + 2),
        seed=seed, device=device, chunk_size=128
    )

    params = params0.clone()
    rng_mb = torch.Generator(); rng_mb.manual_seed(seed + 100)
    evals = 0
    best_test = 0.0
    t0 = time.time()
    next_log = LOG_INTERVAL

    print(f"\n  [v62 SOTA Fixed] evals/step = {2 * sum(k_blocks) + 2}")
    print(f"  {'evals':>10}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}  {'k_alloc'}")
    print(f"  {'-'*65}")

    while evals < BUDGET_DGE:
        idx = torch.randperm(60000, generator=rng_mb)[:BATCH_SIZE]
        Xb_raw, yb = X_tr_raw[idx], y_tr_d[idx]
        Xb = ((Xb_raw/255.0) - 0.1307) / 0.3081

        def f_batched(p_batch):
            with torch.no_grad():
                logits = model.forward(Xb, p_batch)
                P, B, C = logits.shape
                t = yb.unsqueeze(0).expand(P, -1)
                return F.cross_entropy(logits.reshape(P*B, C), t.reshape(-1), reduction='none').view(P, B).mean(dim=1)
        
        params, n = opt.step_sota(f_batched, params, sniff_ratio=0.1)
        evals += n

        if evals >= next_log or evals >= BUDGET_DGE:
            te_acc  = zo_acc(model, X_te_raw, y_te_d, params)
            best_test = max(best_test, te_acc)
            alloc_str = f"[{opt.k_blocks[0]}, {opt.k_blocks[1]}, {opt.k_blocks[2]}]"
            print(f"  {evals:>10,}  {te_acc:>8.2%}  {best_test:>8.2%}  {time.time()-t0:>6.0f}s  {alloc_str}")
            next_log += LOG_INTERVAL
            
    # Save results
    out_path = Path(__file__).parent.parent / "results" / "raw" / "v62_mathematical_scaling_fixed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "v62_mathematical_scaling_fixed", 
            "seed": seed, 
            "acc": best_test,
            "final_k_alloc": opt.k_blocks
        }, f, indent=2)
