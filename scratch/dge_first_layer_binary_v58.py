"""
dge_first_layer_binary_v58.py
=================================
Experiment: Hybrid Binary-Continuous Network
First layer weights are strictly binary {0, 1}.
The rest of the layers are standard continuous floats.

Uses the state-of-the-art DGE combination (from v55/v57):
- Subdivided Neuron Architecture
- Dynamic Noise Budget
- Asymmetric Adam EMA & Hybrid Curvature (Megazord)
- Exponential LR Schedule
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
                # Binarizamos los pesos latentes a 0 o 1
                # Usamos el umbral en 0. Si el peso latente es > 0, es 1, sino 0.
                W = (W > 0).float()
                # Dejamos el bias continuo o lo fijamos a 0? Lo dejamos continuo para que la neurona pueda ajustar su umbral de activación.
            
            h = torch.bmm(h, W) + b
            if l_out != self.arch[-1]: 
                h = torch.relu(h)
            
            offset += sz
        return h

# ---------------------------------------------------------------------------
# Optimizer: SOTA Megazord + Exponential LR
# ---------------------------------------------------------------------------
class SOTADGEOptimizer(TorchDGEOptimizer):
    def __init__(self, lr_min=1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_k = self.total_k
        self.lr_min = lr_min
        self.g_prev = None
        self.m = torch.zeros(self.dim, device=self.device)
        self.v = torch.zeros(self.dim, device=self.device)

    def get_lr(self):
        # Exponential LR Schedule (from v57)
        gamma = (self.lr_min / self.lr0) ** (1.0 / max(self.total_steps, 1))
        return self.lr0 * (gamma ** self.t)

    def _recalculate_k_blocks(self):
        # Dynamic Noise Budget (from v54/v55)
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

    def step_sota(self, f_batched, x: torch.Tensor, sniff_ratio=0.1):
        self.t += 1
        lr = self.get_lr()
        delta = self._cosine(self.delta0, self.delta_decay)
        
        if self.t > 1 and self.t % 10 == 0:
            self._recalculate_k_blocks()
            
        # 1. Block Perturbations
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
            
        # 2. Forward DGE
        X_dge = torch.zeros((2 * self.base_k, self.dim), device=self.device)
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
            
        # 3. Hybrid Lateral Sniff (Asymmetric EMA from Megazord)
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

        P_sniff = torch.stack([v_perp * delta, -v_perp * delta])
        L_sniff = f_batched(x.unsqueeze(0) + P_sniff)
        grad_perp = ((L_sniff[0] - L_sniff[1]) / (2.0 * delta)) * v_perp
        
        n_dge = torch.norm(grad_dge)
        n_perp = torch.norm(grad_perp)
        if n_perp > 1e-9 and n_dge > 1e-9:
            grad_perp_s = grad_perp * (n_dge / n_perp) * sniff_ratio
        else:
            grad_perp_s = torch.zeros_like(grad_perp)
            
        self.g_prev = grad_dge.clone()
            
        # 4. Asymmetric Adam
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_dge
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad_dge ** 2)
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        upd = lr * mh / (torch.sqrt(vh) + self.eps)
        if self.clip_norm is not None:
            un = torch.norm(upd) 
            if un > self.clip_norm: upd *= self.clip_norm / un
            
        upd += lr * grad_perp_s
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
LR_MAX        = 0.02
LR_MIN        = 0.001
DELTA         = 1e-3
BATCH_SIZE    = 256
LOG_INTERVAL  = 25_000

if __name__ == "__main__":
    X_tr_raw, y_tr_d, X_te_raw, y_te_d = load_full_mnist()
    model = HybridBinaryMLP(ARCH)
    layer_sizes = model.layer_param_counts
    k_blocks = [512, 64, 10]

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v58: HYBRID BINARY-CONTINUOUS NETWORK")
    print(f"Twist: First layer weights are strictly {{0, 1}}.")
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

    opt = SOTADGEOptimizer(
        lr_min=LR_MIN,
        dim=model.dim, layer_sizes=layer_sizes, k_blocks=k_blocks,
        lr=LR_MAX, delta=DELTA, total_steps=BUDGET_DGE//(sum(k_blocks)*2 + 2),
        seed=seed, device=device, chunk_size=128
    )

    params = params0.clone()
    rng_mb = torch.Generator(); rng_mb.manual_seed(seed + 100)
    evals = 0
    best_test = 0.0
    t0 = time.time()
    next_log = LOG_INTERVAL

    print(f"\n  [v58 SOTA] evals/step = {2 * sum(k_blocks) + 2}")
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
    out_path = Path(__file__).parent.parent / "results" / "raw" / "v58_hybrid_binary_first_layer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "v58_hybrid_binary_first_layer", 
            "seed": seed, 
            "acc": best_test,
            "final_k_alloc": opt.k_blocks
        }, f, indent=2)
