import torch
import torch.nn as nn
import numpy as np
import time
import math
import os
import json

# Try to import directml for the user's AMD iGPU, fallback to CPU
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML Device: {device}")
except ImportError:
    device = torch.device("cpu")
    print(f"DirectML not found, using: {device}")

# =============================================================================
# OPTIMIZADOR DGE v18 (Batched Parallel Engine)
# =============================================================================
class DGEOptimizerV18:
    """
    DGE v18: Designed for massive parallel evaluation of blocks.
    Evaluates all 2k perturbations in a single tensor operation.
    """
    def __init__(self, dim, lr=1.0, delta=1e-3, beta1=0.95, beta2=0.999,
                 k=16, clip_norm=1.0):
        self.dim = dim
        self.lr = lr
        self.delta = delta
        self.beta1, self.beta2 = beta1, beta2
        self.k = k
        self.clip_norm = clip_norm
        
        # State tensors on device
        self.m = torch.zeros(dim, device=device)
        self.v = torch.zeros(dim, device=device)
        self.t = 0

    def get_batched_perturbations(self):
        """
        Creates a tensor of shape [2*k, dim] containing all positive and negative
        perturbations for the orthogonal blocks.
        """
        # 1. Create orthogonal blocks
        perm = torch.randperm(self.dim, device=device)
        groups = torch.chunk(perm, self.k)
        
        # 2. Prepare perturbations tensor [2*k, dim]
        # We want: 
        # Row 0: +delta in block 0
        # Row 1: -delta in block 0
        # Row 2: +delta in block 1 ...
        perts = torch.zeros((2 * self.k, self.dim), device=device)
        
        signs = torch.randint(0, 2, (self.dim,), device=device).float() * 2 - 1
        
        for i, idx in enumerate(groups):
            # Pos row
            perts[2*i, idx] = signs[idx] * self.delta
            # Neg row
            perts[2*i+1, idx] = -signs[idx] * self.delta
            
        return perts, signs, groups

    def update_adam(self, g_step):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_step
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g_step ** 2)
        
        mh = self.m / (1 - self.beta1 ** self.t + 1e-10)
        vh = self.v / (1 - self.beta2 ** self.t + 1e-10)
        
        upd = self.lr * mh / (torch.sqrt(vh) + 1e-8)
        
        norm = torch.norm(upd)
        if norm > self.clip_norm:
            upd *= self.clip_norm / norm
        return upd

# =============================================================================
# BATCHED MODEL (Manual Vectorization for speed)
# =============================================================================
class BatchedMLP:
    def __init__(self, arch=[784, 128, 10]):
        self.arch = arch
        self.dim = sum(arch[i]*arch[i+1] + arch[i+1] for i in range(len(arch)-1))
        
    def forward_batched(self, X, params_batch):
        """
        X: [DataBatchSize, InDim]
        params_batch: [Perturbations, TotalParams]
        Returns: [Perturbations, DataBatchSize, OutDim]
        """
        num_perts = params_batch.shape[0]
        curr_x = X.unsqueeze(0).expand(num_perts, -1, -1) # [P, B, In]
        
        i = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            # Extract weights and biases for all perturbations at once
            w_size = l_in * l_out
            W = params_batch[:, i:i+w_size].view(num_perts, l_in, l_out)
            i += w_size
            b = params_batch[:, i:i+l_out].view(num_perts, 1, l_out)
            i += l_out
            
            # Batched matrix multiplication: [P, B, In] @ [P, In, Out] -> [P, B, Out]
            curr_x = torch.bmm(curr_x, W) + b
            
            if l_out != self.arch[-1]:
                curr_x = torch.relu(curr_x)
        return curr_x

def loss_fn_batched(logits, targets):
    """
    logits: [P, B, 10]
    targets: [B]
    Returns: [P] losses
    """
    # Expand targets to match P
    targets_exp = targets.unsqueeze(0).expand(logits.shape[0], -1)
    # Cross entropy over the B dimension for each P
    # PyTorch's cross_entropy expects [N, C] so we flatten P and B
    P, B, C = logits.shape
    l = F.cross_entropy(logits.view(P*B, C), targets_exp.reshape(-1), reduction='none')
    return l.view(P, B).mean(dim=1)

import torch.nn.functional as F

if __name__ == "__main__":
    print(f"DGE v18: BATCHED PARALLEL TEST (Simulating GPU-like throughput)")
    
    # 1. SETUP
    D = 100_000
    K = 128
    BATCH_SIZE = 256
    
    # Mock data and model
    X = torch.randn(BATCH_SIZE, 784, device=device)
    y = torch.randint(0, 10, (BATCH_SIZE,), device=device)
    model = BatchedMLP(arch=[784, 120, 10]) # ~100k params
    params = torch.randn(model.dim, device=device) * 0.1
    
    opt = DGEOptimizerV18(model.dim, k=K)
    
    # -------------------------------------------------------------------------
    # TEST A: SEQUENTIAL (Traditional DGE loop)
    # -------------------------------------------------------------------------
    print(f"\n>>> TEST A: SEQUENTIAL LOOP (2*k={2*K} evaluations)")
    perts, signs, groups = opt.get_batched_perturbations()
    
    t0 = time.time()
    losses_seq = []
    for j in range(2 * K):
        perturbed_p = params + perts[j]
        # We simulate a single forward pass
        logits = model.forward_batched(X, perturbed_p.unsqueeze(0))
        loss = loss_fn_batched(logits, y)
        losses_seq.append(loss.item())
    
    t_seq = time.time() - t0
    print(f"    Sequential Time: {t_seq:.4f}s ({2*K/t_seq:.1f} evals/s)")

    # -------------------------------------------------------------------------
    # TEST B: BATCHED (Parallel DGE)
    # -------------------------------------------------------------------------
    print(f"\n>>> TEST B: BATCHED PARALLEL (All 2*k evaluations at once)")
    
    t0 = time.time()
    # 1. Expand params to [2*k, D] and add all perts at once
    params_batched = params.unsqueeze(0) + perts # [2k, D]
    
    # 2. SINGLE FORWARD CALL for all perturbations
    logits_batched = model.forward_batched(X, params_batched) # [2k, B, 10]
    
    # 3. SINGLE LOSS CALL
    losses_par = loss_fn_batched(logits_batched, y) # [2k]
    
    t_par = time.time() - t0
    print(f"    Parallel Time: {t_par:.4f}s ({2*K/t_par:.1f} evals/s)")
    
    # Verification
    diff = np.abs(np.array(losses_seq) - losses_par.cpu().numpy()).max()
    print(f"\nVerification (Max Diff): {diff:.2e}")
    print(f"SPEEDUP: {t_seq/t_par:.1f}x")

    # 4. ADAM UPDATE PERFORMANCE (Final throughput)
    print(f"\n>>> DGE STEP COMPLETION")
    t0 = time.time()
    # Gradient estimation from batched losses
    # fp is evens [0, 2, 4...], fm is odds [1, 3, 5...]
    fp = losses_par[0::2]
    fm = losses_par[1::2]
    sg = (fp - fm) / (2.0 * opt.delta)
    
    g_step = torch.zeros(model.dim, device=device)
    for i, idx in enumerate(groups):
        g_step[idx] = sg[i] * signs[idx]
        
    upd = opt.update_adam(g_step)
    params -= upd
    
    t_step = time.time() - t0
    print(f"    Update Logic Time: {t_step:.4f}s")
    print(f"    Total Step Time: {t_par + t_step:.4f}s")
    print(f"    ESTIMATED THROUGHPUT: {2*K / (t_par + t_step):.0f} evals/s")
