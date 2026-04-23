"""
dge_canine_adam_v48.py
=================================
Experiment: Integrated Canine-Adam.
We merge the "Stereo Sniffing" signal directly into the Adam EMA.
The lateral gradient from the "nostrils" is added to the block-gradient 
before the update, so the momentum remembers the lateral scent.

Base: v45 (Subdivided Neurons).
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
# Integrated Canine Optimizer
# ---------------------------------------------------------------------------
class IntegratedCanineOptimizer(TorchDGEOptimizer):
    def step_integrated(self, f_batched, x: torch.Tensor, alpha_sniff=1.0):
        # This implementation requires modifying the internal logic of step()
        # to combine gradients before the Adam update.
        
        # We'll re-implement the core logic here for clarity
        self.t += 1
        lr = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)
        
        # 1. Block Perturbations (Structural DGE)
        P_plus = torch.zeros((self.total_k, self.dim), device=self.device)
        offset = 0
        row_offset = 0
        perms, signss = [], []
        
        for sz, k, pad, grp in zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes):
            perm = torch.randperm(sz, generator=self.rng, device=self.device)
            signs = torch.randint(0, 2, (sz,), generator=self.rng, device=self.device).float() * 2 - 1
            perms.append(perm); signss.append(signs)
            perm_mat = (perm.view(k, grp) if pad==0 else torch.cat([perm, torch.zeros(pad, dtype=torch.long, device=self.device)]).view(k, grp)) + offset
            signs_mat = (signs.view(k, grp) if pad==0 else torch.cat([signs, torch.zeros(pad, device=self.device)]).view(k, grp)) * delta
            P_plus[row_offset : row_offset + k, :].scatter_(1, perm_mat, signs_mat)
            offset += sz; row_offset += k
            
        # 2. Lateral Sniff (Nostrils)
        m_curr = self.m
        v_perp = torch.zeros(self.dim, device=self.device)
        if torch.norm(m_curr) > 1e-9:
            r = torch.randn_like(m_curr)
            v_perp = r - (torch.dot(r, m_curr) / torch.dot(m_curr, m_curr)) * m_curr
            v_perp = v_perp / (torch.norm(v_perp) + 1e-8)
            
        P_sniff = torch.stack([v_perp * delta, -v_perp * delta])
        
        # Combine all perturbations into one batch
        # 2*total_k (DGE) + 2 (Sniff)
        P_all = torch.zeros((2 * self.total_k + 2, self.dim), device=self.device)
        P_all[0 : 2*self.total_k : 2] = P_plus
        P_all[1 : 2*self.total_k : 2] = -P_plus
        P_all[2*self.total_k] = P_sniff[0]
        P_all[2*self.total_k + 1] = P_sniff[1]
        
        # Forward Pass
        X_batch = x.unsqueeze(0) + P_all
        losses = f_batched(X_batch)
        
        # 3. Compute Combined Gradient
        # Block Grads
        diffs = (losses[0 : 2*self.total_k : 2] - losses[1 : 2*self.total_k : 2]) / (2.0 * delta)
        grad_dge = torch.zeros(self.dim, device=self.device)
        off, row_off = 0, 0
        for i, (sz, k, pad, grp) in enumerate(zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes)):
            layer_diffs = diffs[row_off : row_off + k]
            diffs_exp = layer_diffs.unsqueeze(1).expand(k, grp).flatten()
            if pad > 0: diffs_exp = diffs_exp[:sz]
            grad_dge[off + perms[i]] = diffs_exp * signss[i]
            off += sz; row_off += k
            
        # Lateral Grad
        L_sniff = losses[2*self.total_k : 2*self.total_k + 2]
        grad_perp = ((L_sniff[0] - L_sniff[1]) / (2.0 * delta)) * v_perp
        
        # INTEGRATION: Combine signals
        grad_total = grad_dge + alpha_sniff * grad_perp
        
        # 4. Adam Update
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_total
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad_total ** 2)
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        
        upd = lr * mh / (torch.sqrt(vh) + self.eps)
        if self.clip_norm is not None:
            un = torch.norm(upd)
            if un > self.clip_norm: upd *= self.clip_norm / un
            
        return x - upd, 2 * self.total_k + 2

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
    X_tr_d, y_tr_d, X_te_d, y_te_d = load_full_mnist()
    X_tr_d, y_tr_d = X_tr_d.to(device), y_tr_d.to(device)
    X_te_d, y_te_d = X_te_d.to(device), y_te_d.to(device)

    model = NeuronOrderedMLP(ARCH)
    layer_sizes = model.layer_param_counts
    k_blocks = [512, 64, 10]

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v48: Integrated Canine-Adam")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print(f"Mechanism: Combined Grad (Block + Lateral) -> Adam EMA")
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

    opt = IntegratedCanineOptimizer(
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

    print(f"\n  [Canine-Adam] evals/step = {2 * sum(k_blocks) + 2}")
    print(f"  {'evals':>10}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}")
    print(f"  {'-'*45}")

    while evals < BUDGET_DGE:
        idx = torch.randperm(60000, generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr_d[idx], y_tr_d[idx]

        def f_batched(p_batch):
            with torch.no_grad():
                logits = model.forward(Xb, p_batch)
                P, B, C = logits.shape
                t = yb.unsqueeze(0).expand(P, -1)
                return F.cross_entropy(logits.reshape(P*B, C), t.reshape(-1), reduction='none').view(P, B).mean(dim=1)
        
        params, n = opt.step_integrated(f_batched, params, alpha_sniff=1.0)
        evals += n

        if evals >= next_log or evals >= BUDGET_DGE:
            te_acc  = zo_acc(model, X_te_d, y_te_d, params)
            best_test = max(best_test, te_acc)
            print(f"  {evals:>10,}  {te_acc:>8.2%}  {best_test:>8.2%}  {time.time()-t0:>6.0f}s")
            next_log += LOG_INTERVAL
