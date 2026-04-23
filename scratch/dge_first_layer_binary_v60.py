"""
dge_first_layer_binary_v60.py
=================================
Experiment: Hybrid Binary-Continuous Network (Decoupled Delta & LR)
First layer weights are strictly binary {0, 1}.
The rest of the layers are standard continuous floats.

Fixes from v59:
- The global DELTA=0.05 destroyed the continuous layers because finite 
  differences assume small perturbations. 0.05 is massive for a continuous weight.
- Solution: Per-Layer Hyperparameters!
  - Layer 1 (Binary): High Delta (0.05) to cross thresholds, High LR (0.1)
  - Layer 2 & 3 (Continuous): Low Delta (1e-3) for accurate gradients, Low LR (0.005)
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
class HybridBinaryMLP:
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
            
            # --- TWIST: Primera capa con cables binarios {0, 1} ---
            if i == 0:
                W = (W > 0).float()
            
            h = torch.bmm(h, W) + b
            if l_out != self.arch[-1]: 
                h = torch.relu(h)
            
            offset += sz
        return h

# ---------------------------------------------------------------------------
# Optimizer: Decoupled SOTA Megazord
# ---------------------------------------------------------------------------
class DecoupledHybridOptimizer(TorchDGEOptimizer):
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
        gamma = (lr_min / lr0) ** (1.0 / max(self.total_steps, 1))
        return lr0 * (gamma ** self.t)

    def step_decoupled(self, f_batched, x: torch.Tensor, sniff_ratio=0.1):
        self.t += 1
        
        # 1. Block Perturbations (Per-Layer Delta)
        P_plus = torch.zeros((self.base_k, self.dim), device=self.device)
        offset = 0; row_offset = 0; perms = []; signss = []; current_deltas = []
        for i, (sz, k, pad, grp) in enumerate(zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes)):
            layer_delta = self._cosine(self.layer_deltas0[i], self.delta_decay)
            current_deltas.append(layer_delta)
            
            perm = torch.randperm(sz, generator=self.rng, device=self.device)
            signs = torch.randint(0, 2, (sz,), generator=self.rng, device=self.device).float() * 2 - 1
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
            
        # 4. Asymmetric Adam EMA
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_dge
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad_dge ** 2)
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        
        # 5. Apply Per-Layer LR
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ARCH          = (784, 128, 64, 10)
BUDGET_DGE    = 500_000
BATCH_SIZE    = 256
LOG_INTERVAL  = 25_000

if __name__ == "__main__":
    X_tr_raw, y_tr_d, X_te_raw, y_te_d = load_full_mnist()
    model = HybridBinaryMLP(ARCH)
    layer_sizes = model.layer_param_counts
    k_blocks = [512, 64, 10]
    
    # PER-LAYER HYPERPARAMETERS
    LRS_0   = [0.1,   0.005, 0.005]  # High LR for Binary, Low for Continuous
    LRS_MIN = [0.005, 0.001, 0.001]
    DELTAS  = [0.05,  1e-3,  1e-3]   # High Delta for Binary, Low for Continuous

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v60: HYBRID BINARY (Decoupled Layer Hyperparameters)")
    print(f"L1 (Binary):     LR={LRS_0[0]}->{LRS_MIN[0]} | DELTA={DELTAS[0]}")
    print(f"L2 (Continuous): LR={LRS_0[1]}->{LRS_MIN[1]} | DELTA={DELTAS[1]}")
    print(f"L3 (Continuous): LR={LRS_0[2]}->{LRS_MIN[2]} | DELTA={DELTAS[2]}")
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

    opt = DecoupledHybridOptimizer(
        layer_lrs0=LRS_0,
        layer_lrs_min=LRS_MIN,
        layer_deltas0=DELTAS,
        dim=model.dim, layer_sizes=layer_sizes, k_blocks=k_blocks,
        total_steps=BUDGET_DGE//(sum(k_blocks)*2),
        seed=seed, device=device, chunk_size=128
    )

    params = params0.clone()
    rng_mb = torch.Generator(); rng_mb.manual_seed(seed + 100)
    evals = 0
    best_test = 0.0
    t0 = time.time()
    next_log = LOG_INTERVAL

    print(f"\n  [v60 Decoupled] evals/step = {2 * sum(k_blocks)}")
    print(f"  {'evals':>10}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}")
    print(f"  {'-'*55}")

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
        
        params, n = opt.step_decoupled(f_batched, params, sniff_ratio=0.1)
        evals += n

        if evals >= next_log or evals >= BUDGET_DGE:
            te_acc  = zo_acc(model, X_te_raw, y_te_d, params)
            best_test = max(best_test, te_acc)
            print(f"  {evals:>10,}  {te_acc:>8.2%}  {best_test:>8.2%}  {time.time()-t0:>6.0f}s")
            next_log += LOG_INTERVAL
            
    # Save results
    out_path = Path(__file__).parent.parent / "results" / "raw" / "v60_hybrid_binary_decoupled.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "v60_hybrid_binary_decoupled", 
            "seed": seed, 
            "acc": best_test,
            "final_k_alloc": opt.k_blocks
        }, f, indent=2)
