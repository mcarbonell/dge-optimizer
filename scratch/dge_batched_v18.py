import torch
import torch.nn as nn
import numpy as np
import time
import math
import os
import json

try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML Device: {device}")
except ImportError:
    device = torch.device("cpu")
    print(f"DirectML not found, using: {device}")

# =============================================================================
# OPTIMIZADOR DGE v18 (Optimized Parallel Engine)
# =============================================================================
class DGEOptimizerV18:
    def __init__(self, dim, lr=1.0, delta=1e-3, beta1=0.95, beta2=0.999,
                 k=16, clip_norm=1.0):
        self.dim = dim
        self.lr = lr
        self.delta = delta
        self.beta1, self.beta2 = beta1, beta2
        self.k = k
        self.clip_norm = clip_norm
        self.m = torch.zeros(dim, device=device)
        self.v = torch.zeros(dim, device=device)
        self.t = 0
        
        # PRE-CALCULATE BLOCK STRUCTURE (Crucial for Speed)
        self.perm = torch.randperm(self.dim, device=device)
        self.block_map = torch.zeros(self.dim, dtype=torch.long, device=device)
        groups = torch.chunk(self.perm, self.k)
        for i, idx in enumerate(groups):
            self.block_map[idx] = i

    def get_batched_perturbations(self):
        perts = torch.zeros((2 * self.k, self.dim), device=device)
        signs = torch.randint(0, 2, (self.dim,), device=device).float() * 2 - 1
        
        # Optimized perturbation generation
        # Row 2*i: signs[block_i] * delta
        # Row 2*i+1: -signs[block_i] * delta
        # This is hard to vectorize fully without complex indexing, so we stick to the loop 
        # but only for 2*k rows, which is small (e.g. 512).
        groups = torch.chunk(self.perm, self.k)
        for i, idx in enumerate(groups):
            perts[2*i, idx] = signs[idx] * self.delta
            perts[2*i+1, idx] = -signs[idx] * self.delta
        return perts, signs

    def update_adam(self, g_step):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_step
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g_step ** 2)
        mh = self.m / (1 - self.beta1 ** self.t + 1e-10)
        vh = self.v / (1 - self.beta2 ** self.t + 1e-10)
        upd = self.lr * mh / (torch.sqrt(vh) + 1e-8)
        norm = torch.norm(upd)
        if norm > self.clip_norm: upd *= self.clip_norm / norm
        return upd

class BatchedMLP:
    def __init__(self, arch=[784, 128, 10]):
        self.arch = arch
        self.dim = sum(arch[i]*arch[i+1] + arch[i+1] for i in range(len(arch)-1))
        
    def forward_batched(self, X, params_batch):
        num_perts = params_batch.shape[0]
        curr_x = X.unsqueeze(0).expand(num_perts, -1, -1)
        i = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            w_size = l_in * l_out
            W = params_batch[:, i:i+w_size].view(num_perts, l_in, l_out)
            i += w_size
            b = params_batch[:, i:i+l_out].view(num_perts, 1, l_out)
            i += l_out
            curr_x = torch.bmm(curr_x, W) + b
            if l_out != self.arch[-1]: curr_x = torch.relu(curr_x)
        return curr_x

def loss_fn_batched(logits, targets):
    targets_exp = targets.unsqueeze(0).expand(logits.shape[0], -1)
    P, B, C = logits.shape
    l = torch.nn.functional.cross_entropy(logits.view(P*B, C), targets_exp.reshape(-1), reduction='none')
    return l.view(P, B).mean(dim=1)

if __name__ == "__main__":
    print(f"DGE v18: BATCHED PARALLEL TEST (Extreme Scale Optimization)")
    
    # 1. SETUP - 1 MILLION PARAMETERS
    D_TARGET = 1_000_000
    K = 256 # 512 evaluations per step
    BATCH_SIZE = 32
    
    X = torch.randn(BATCH_SIZE, 784, device=device)
    y = torch.randint(0, 10, (BATCH_SIZE,), device=device)
    
    # Architecture for ~1M params: 784 -> 1200 -> 10
    model = BatchedMLP(arch=[784, 1260, 10]) 
    print(f"  Model Dimensions: {model.dim:,} parameters")
    params = torch.randn(model.dim, device=device) * 0.1
    
    opt = DGEOptimizerV18(model.dim, k=K)
    
    # -------------------------------------------------------------------------
    # TEST A: SEQUENTIAL (Extrapolated)
    # -------------------------------------------------------------------------
    perts, signs = opt.get_batched_perturbations()
    t0 = time.time()
    num_to_test = 32
    for j in range(num_to_test):
        perturbed_p = params + perts[j]
        logits = model.forward_batched(X, perturbed_p.unsqueeze(0))
        loss = loss_fn_batched(logits, y)
    t_seq = (time.time() - t0) * (2*K / num_to_test)
    print(f"\n>>> SEQUENTIAL (Extrapolated): {t_seq:.4f}s ({2*K/t_seq:.1f} evals/s)")

    # -------------------------------------------------------------------------
    # TEST B: BATCHED PARALLEL
    # -------------------------------------------------------------------------
    t0 = time.time()
    params_batched = params.unsqueeze(0) + perts 
    logits_batched = model.forward_batched(X, params_batched) 
    losses_par = loss_fn_batched(logits_batched, y) 
    t_par = time.time() - t0
    print(f">>> BATCHED PARALLEL: {t_par:.4f}s ({2*K/t_par:.1f} evals/s)")
    print(f"    SPEEDUP (Forward): {t_seq/t_par:.1f}x")

    # -------------------------------------------------------------------------
    # 4. OPTIMIZED ADAM UPDATE (No more loops!)
    # -------------------------------------------------------------------------
    t0 = time.time()
    fp, fm = losses_par[0::2], losses_par[1::2]
    sg = (fp - fm) / (2.0 * opt.delta)
    
    # Pure Tensor Assembly
    g_step = sg[opt.block_map] * signs
    upd = opt.update_adam(g_step)
    params -= upd
    t_step = time.time() - t0
    
    print(f"\n>>> OPTIMIZED UPDATE LOGIC: {t_step:.4f}s")
    print(f"    Total Step Time: {t_par + t_step:.4f}s")
    print(f"    FINAL THROUGHPUT: {2*K / (t_par + t_step):.0f} evals/s")
