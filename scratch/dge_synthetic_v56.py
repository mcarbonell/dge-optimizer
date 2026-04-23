"""
dge_synthetic_v56.py
=================================
Experiment: Canine Sniffing in Synthetic Landscapes
We test whether the "Curvature Sniffing" (which failed to beat variance-reduction
on MNIST) actually shines in classic continuous optimization problems with 
narrow, curved valleys (like Rosenbrock or Ellipsoid), where geometry matters 
more than stochastic noise.

Since synthetic functions are flat vectors (no layers), the "Dynamic Budget" (v54) 
doesn't apply (nowhere to shift the budget to). Therefore, we compare:
1. Baseline DGE (Standard Zeroth-Order)
2. Curvature Canine DGE (v51/v52 style: orthogonal sniff to find the valley floor)
"""

import json
import math
import time
from pathlib import Path

import torch

try:
    import torch_directml
    device = torch_directml.device()
    print(f"DirectML: {device}")
except ImportError:
    device = torch.device("cpu")
    print("CPU")

# ---------------------------------------------------------------------------
# Synthetic Functions (PyTorch)
# ---------------------------------------------------------------------------
def rosenbrock(x_batch):
    # x_batch: [B, D]
    x_next = x_batch[:, 1:]
    x_curr = x_batch[:, :-1]
    term1 = 100.0 * (x_next - x_curr**2)**2
    term2 = (1.0 - x_curr)**2
    return torch.sum(term1 + term2, dim=1)

def ellipsoid(x_batch):
    # x_batch: [B, D]
    B, D = x_batch.shape
    indices = torch.arange(1, D + 1, device=x_batch.device, dtype=x_batch.dtype)
    weights = 1000.0 ** ((indices - 1) / (D - 1)) if D > 1 else torch.tensor([1.0], device=x_batch.device)
    return torch.sum(weights.unsqueeze(0) * x_batch**2, dim=1)

# ---------------------------------------------------------------------------
# Optimizers (Simplified for 1D flat vectors, D=1000)
# ---------------------------------------------------------------------------
class FlatDGEOptimizer:
    def __init__(self, dim, k, lr, delta, beta1=0.9, beta2=0.999):
        self.dim = dim
        self.k = k
        self.lr = lr
        self.delta = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = torch.zeros(dim, device=device)
        self.v = torch.zeros(dim, device=device)
        self.t = 0
        self.eps = 1e-8

    def step_baseline(self, f, x):
        self.t += 1
        
        # 1. Generate K orthogonal-ish perturbations using Rademacher
        # To avoid large memory, we just use random signs
        P = torch.randint(0, 2, (self.k, self.dim), device=device).float() * 2 - 1
        P *= self.delta
        
        # 2. Forward DGE
        X_plus = x.unsqueeze(0) + P
        X_minus = x.unsqueeze(0) - P
        
        L_plus = f(X_plus)
        L_minus = f(X_minus)
        
        diffs = (L_plus - L_minus) / (2.0 * self.delta) # [K]
        
        # Approximate gradient
        # grad ≈ (1/K) * sum(diffs[i] * P[i] / delta)
        # P[i] / delta is just the sign matrix
        grad_dge = torch.mean(diffs.unsqueeze(1) * (P / self.delta), dim=0)
        
        # 3. Adam Update
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_dge
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad_dge ** 2)
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        upd = self.lr * mh / (torch.sqrt(vh) + self.eps)
        
        return x - upd, 2 * self.k

class FlatCanineOptimizer(FlatDGEOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g_prev = None

    def step_canine(self, f, x, sniff_ratio=0.1):
        self.t += 1
        
        # 1. Standard DGE
        P = torch.randint(0, 2, (self.k, self.dim), device=device).float() * 2 - 1
        P *= self.delta
        
        X_plus = x.unsqueeze(0) + P
        X_minus = x.unsqueeze(0) - P
        
        L_plus = f(X_plus)
        L_minus = f(X_minus)
        diffs = (L_plus - L_minus) / (2.0 * self.delta)
        grad_dge = torch.mean(diffs.unsqueeze(1) * (P / self.delta), dim=0)
        
        # 2. Canine Sniff (Curvature)
        v_perp = torch.zeros(self.dim, device=device)
        if self.g_prev is not None:
            v_curv = grad_dge - self.g_prev
            gn2 = torch.dot(grad_dge, grad_dge)
            if gn2 > 1e-12:
                v_perp = v_curv - (torch.dot(v_curv, grad_dge) / gn2) * grad_dge
                n_p = torch.norm(v_perp)
                if n_p > 1e-12:
                    v_perp = v_perp / n_p

        # If no curvature signal, random orthogonal
        if torch.norm(v_perp) < 1e-9:
            r = torch.randn_like(grad_dge)
            gn2 = torch.dot(grad_dge, grad_dge)
            if gn2 > 1e-12:
                v_p = r - (torch.dot(r, grad_dge) / gn2) * grad_dge
                n_p = torch.norm(v_p)
                if n_p > 1e-12: v_perp = v_p / n_p
            else:
                v_perp = r / (torch.norm(r) + 1e-12)

        # Evaluate sniff
        P_sniff = torch.stack([v_perp * self.delta, -v_perp * self.delta])
        L_sniff = f(x.unsqueeze(0) + P_sniff)
        grad_perp = ((L_sniff[0] - L_sniff[1]) / (2.0 * self.delta)) * v_perp
        
        # Normalize lateral gradient
        n_dge = torch.norm(grad_dge)
        n_perp = torch.norm(grad_perp)
        if n_perp > 1e-9 and n_dge > 1e-9:
            grad_perp_s = grad_perp * (n_dge / n_perp) * sniff_ratio
        else:
            grad_perp_s = torch.zeros_like(grad_perp)
            
        self.g_prev = grad_dge.clone()
        
        # 3. Asymmetric Adam Update
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_dge
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad_dge ** 2)
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        upd = self.lr * mh / (torch.sqrt(vh) + self.eps)
        
        # Add lateral signal directly
        upd += self.lr * grad_perp_s
        
        return x - upd, 2 * self.k + 2

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_experiment(name, f_loss, dim, k, lr, delta, total_evals, use_canine=False):
    # Initialize near origin, but not exactly at zero to break symmetry
    torch.manual_seed(42)
    x = torch.randn(dim, device=device) * 0.1
    
    opt = FlatCanineOptimizer(dim, k, lr, delta) if use_canine else FlatDGEOptimizer(dim, k, lr, delta)
    
    evals = 0
    t0 = time.time()
    
    # Log intervals
    log_every = total_evals // 10
    next_log = log_every
    
    best_loss = float('inf')
    
    print(f"\n--- {name} ---")
    print(f"  {'Evals':>10} | {'Current Loss':>15} | {'Best Loss':>15}")
    print("-" * 50)
    
    while evals < total_evals:
        def f_wrapper(x_batch):
            return f_loss(x_batch)
            
        if use_canine:
            x, n = opt.step_canine(f_wrapper, x)
        else:
            x, n = opt.step_baseline(f_wrapper, x)
            
        evals += n
        
        with torch.no_grad():
            curr_loss = f_loss(x.unsqueeze(0)).item()
            if curr_loss < best_loss:
                best_loss = curr_loss
                
        if evals >= next_log or evals >= total_evals:
            print(f"  {evals:>10,} | {curr_loss:>15.4f} | {best_loss:>15.4f}")
            next_log += log_every
            
    print(f"Time: {time.time() - t0:.1f}s")
    return best_loss

if __name__ == "__main__":
    DIM = 1000
    K = 50        # Number of perturbation directions
    EVALS = 50_000
    LR = 0.05
    DELTA = 1e-3
    
    print("="*60)
    print(f"SYNTHETIC BENCHMARKS (D={DIM}, Budget={EVALS:,})")
    print(f"K={K} samples per step. Baseline evals/step={2*K}, Canine={2*K+2}")
    print("="*60)

    # 1. Rosenbrock (Narrow curved valley)
    # The minimum is at (1, 1, ..., 1) where loss = 0
    print("\n>>> Function: ROSENBROCK (Target: 0.0)")
    rosen_base = run_experiment("Baseline DGE", rosenbrock, DIM, K, LR, DELTA, EVALS, use_canine=False)
    rosen_canine = run_experiment("Curvature Canine (v52)", rosenbrock, DIM, K, LR, DELTA, EVALS, use_canine=True)

    # 2. Ellipsoid (Ill-conditioned quadratic)
    # The minimum is at (0, 0, ..., 0) where loss = 0
    print("\n>>> Function: ELLIPSOID (Target: 0.0)")
    ellip_base = run_experiment("Baseline DGE", ellipsoid, DIM, K, LR, DELTA, EVALS, use_canine=False)
    ellip_canine = run_experiment("Curvature Canine (v52)", ellipsoid, DIM, K, LR, DELTA, EVALS, use_canine=True)
    
    print("\n" + "="*50)
    print("SUMMARY OF BEST LOSSES (Lower is better)")
    print("="*50)
    print(f"{'Function':<15} | {'Baseline DGE':<15} | {'Canine (v52)':<15}")
    print("-" * 50)
    print(f"{'Rosenbrock':<15} | {rosen_base:>15.4f} | {rosen_canine:>15.4f}")
    print(f"{'Ellipsoid':<15} | {ellip_base:>15.4f} | {ellip_canine:>15.4f}")
    print("="*50)
