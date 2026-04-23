"""
dge_canine_curvature_v51.py
=================================
Experiment: Curvature-Guided Canine Sniffing.
Instead of a random orthogonal direction, we use the direction of 
the CHANGE in gradient (Delta G) to identify the curvature of the valley.

Mechanism:
1. Standard DGE step -> grad_dge.
2. v_curvature = grad_dge - grad_prev.
3. v_perp = orthogonal component of v_curvature relative to grad_dge.
4. Sniff nostrils along v_perp.
5. Integrate lateral signal into Adam.
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
# Curvature Canine Optimizer
# ---------------------------------------------------------------------------
class CurvatureCanineOptimizer(TorchDGEOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g_prev = None

    def step_curvature(self, f_batched, x: torch.Tensor, sniff_ratio=0.1):
        self.t += 1
        lr = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)
        
        # 1. Block Perturbations
        P_plus = torch.zeros((self.total_k, self.dim), device=self.device)
        offset = 0; row_offset = 0; perms, signss = [] , []
        for sz, k, pad, grp in zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes):
            perm = torch.randperm(sz, generator=self.rng, device=self.device)
            signs = torch.randint(0, 2, (sz,), generator=self.rng, device=self.device).float() * 2 - 1
            perms.append(perm); signss.append(signs)
            perm_mat = (perm.view(k, grp) if pad==0 else torch.cat([perm, torch.zeros(pad, dtype=torch.long, device=self.device)]).view(k, grp)) + offset
            signs_mat = (signs.view(k, grp) if pad==0 else torch.cat([signs, torch.zeros(pad, device=self.device)]).view(k, grp)) * delta
            P_plus[row_offset : row_offset + k, :].scatter_(1, perm_mat, signs_mat)
            offset += sz; row_offset += k
            
        # 2. Forward DGE
        X_dge = torch.zeros((2 * self.total_k, self.dim), device=self.device)
        X_dge[0::2] = x + P_plus; X_dge[1::2] = x - P_plus
        L_dge_all = f_batched(X_dge)
        
        # Compute grad_dge
        diffs = (L_dge_all[0::2] - L_dge_all[1::2]) / (2.0 * delta)
        grad_dge = torch.zeros(self.dim, device=self.device)
        off, row_off = 0, 0
        for i, (sz, k, pad, grp) in enumerate(zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes)):
            df_exp = diffs[row_off : row_off + k].unsqueeze(1).expand(k, grp).flatten()
            if pad > 0: df_exp = df_exp[:sz]
            grad_dge[off + perms[i]] = df_exp * signss[i]
            off += sz; row_off += k

        # 3. Lateral Sniff (Curvature Guided)
        v_perp = torch.zeros(self.dim, device=self.device)
        if self.g_prev is not None:
            v_curv = grad_dge - self.g_prev
            # Project out the component in the direction of grad_dge
            gn2 = torch.dot(grad_dge, grad_dge)
            if gn2 > 1e-12:
                v_perp = v_curv - (torch.dot(v_curv, grad_dge) / gn2) * grad_dge
                v_perp = v_perp / (torch.norm(v_perp) + 1e-12)
        
        if torch.norm(v_perp) < 1e-9:
            # Fallback to random orthogonal if no curvature signal
            r = torch.randn_like(grad_dge)
            gn2 = torch.dot(grad_dge, grad_dge)
            if gn2 > 1e-12:
                v_perp = r - (torch.dot(r, grad_dge) / gn2) * grad_dge
                v_perp = v_perp / (torch.norm(v_perp) + 1e-12)

        P_sniff = torch.stack([v_perp * delta, -v_perp * delta])
        L_sniff = f_batched(x.unsqueeze(0) + P_sniff)
        grad_perp = ((L_sniff[0] - L_sniff[1]) / (2.0 * delta)) * v_perp
        
        # 4. Normalization
        n_dge = torch.norm(grad_dge)
        n_perp = torch.norm(grad_perp)
        if n_perp > 1e-9 and n_dge > 1e-9:
            grad_perp_s = grad_perp * (n_dge / n_perp) * sniff_ratio
        else:
            grad_perp_s = torch.zeros_like(grad_perp)
            
        grad_total = grad_dge + grad_perp_s
        self.g_prev = grad_dge.clone()
        
        # 5. Adam Update
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_total
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad_total ** 2)
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        upd = lr * mh / (torch.sqrt(vh) + self.eps)
        if self.clip_norm is not None:
            un = torch.norm(upd); 
            if un > self.clip_norm: upd *= self.clip_norm / un
            
        return x - upd, 2 * self.total_k + 2

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
LR_ZO         = 0.05
DELTA         = 1e-3
BATCH_SIZE    = 256
LOG_INTERVAL  = 25_000

if __name__ == "__main__":
    X_tr_raw, y_tr_d, X_te_raw, y_te_d = load_full_mnist()
    model = NeuronOrderedMLP(ARCH)
    layer_sizes = model.layer_param_counts
    k_blocks = [512, 64, 10]

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v51: Curvature-Guided Canine Sniffing")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print(f"Mechanism: Lateral direction derived from Delta-Gradient (Curvature)")
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

    opt = CurvatureCanineOptimizer(
        dim=model.dim, layer_sizes=layer_sizes, k_blocks=k_blocks,
        lr=LR_ZO, delta=DELTA, total_steps=BUDGET_DGE//(sum(k_blocks)*2 + 2),
        seed=seed, device=device, chunk_size=128
    )

    params = params0.clone()
    rng_mb = torch.Generator(); rng_mb.manual_seed(seed + 100)
    evals = 0
    best_test = 0.0
    t0 = time.time()
    next_log = LOG_INTERVAL

    print(f"\n  [Curvature-Sniff] evals/step = {2 * sum(k_blocks) + 2}")
    print(f"  {'evals':>10}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}")
    print(f"  {'-'*45}")

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
        
        # Curvature Sniff
        params, n = opt.step_curvature(f_batched, params, sniff_ratio=0.1)
        evals += n

        if evals >= next_log or evals >= BUDGET_DGE:
            te_acc  = zo_acc(model, X_te_raw, y_te_d, params)
            best_test = max(best_test, te_acc)
            print(f"  {evals:>10,}  {te_acc:>8.2%}  {best_test:>8.2%}  {time.time()-t0:>6.0f}s")
            next_log += LOG_INTERVAL
