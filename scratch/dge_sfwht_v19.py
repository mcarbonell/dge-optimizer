import torch
import numpy as np
import time
import json
import os

try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML Device: {device}")
except ImportError:
    device = torch.device("cpu")
    print(f"DirectML not found, using: {device}")

def fwht_tensor(a):
    """ 1D Fast Walsh-Hadamard Transform """
    n = a.shape[-1]
    h = 1
    # flatten the last dimension to ensure view works properly
    original_shape = a.shape
    a = a.view(-1, n)
    while h < n:
        a = a.view(a.shape[0], -1, h * 2)
        x = a[..., :h]
        y = a[..., h:]
        a = torch.cat((x + y, x - y), dim=-1)
        h *= 2
    return a.view(original_shape)

def create_hadamard(B):
    H = torch.tensor([[1.]], device=device)
    while H.shape[0] < B:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H

class SFWHT_Optimizer:
    def __init__(self, D, B, epsilon=1e-3, threshold=None):
        self.D = D
        self.B = B
        self.epsilon = epsilon
        self.log_D = int(np.log2(D))
        self.log_B = int(np.log2(B))
        assert 2**self.log_D == D, "D must be power of 2"
        assert 2**self.log_B == B, "B must be power of 2"
        
        print(f"Initializing SFWHT: D={D}, B={B}")
        self.H_B = create_hadamard(B)
        self.V_0 = self.H_B.repeat(1, D // B)
        self.threshold = threshold
        
    def estimate_gradient(self, f, x, f_x=None):
        # Base test
        if f_x is None:
            f_x = f(x.unsqueeze(0))
            
        # Batch evaluations for base transform
        perts = x.unsqueeze(0) + self.epsilon * self.V_0
        Y_base = f(perts)
        
        U_base = fwht_tensor(Y_base) / self.B
        
        if self.threshold is None:
            self.threshold = torch.median(torch.abs(U_base)) * 3.0 + 1e-6
            
        # Correct bucket 0 (subtract DC offset)
        U_base[0] -= f_x.squeeze()
        
        # Identify active buckets
        active_buckets = torch.where(torch.abs(U_base) > self.threshold)[0]
        
        if len(active_buckets) == 0:
            return torch.zeros(self.D, device=device), U_base, self.B
        
        # High bit tests
        high_bits = self.log_D - self.log_B
        U_shifts = torch.zeros((high_bits, self.B), device=device)
        
        j_indices = torch.arange(self.D, device=device)
        
        evals = self.B
        for i, m in enumerate(range(self.log_B, self.log_D)):
            flip_mask = ((j_indices & (1 << m)) != 0).float() * -2 + 1
            V_m = self.V_0 * flip_mask.unsqueeze(0)
            Y_m = f(x.unsqueeze(0) + self.epsilon * V_m)
            evals += self.B
            U_shifts[i] = fwht_tensor(Y_m) / self.B
            
        grad = torch.zeros(self.D, device=device)
        for b in active_buckets:
            val_base = U_base[b]
            
            # Recover index
            idx = b.item() # Lower bits
            
            for i, m in enumerate(range(self.log_B, self.log_D)):
                val_m = U_shifts[i, b]
                # Compare signs. If signs differ, bit m is 1.
                if torch.sign(val_base) != torch.sign(val_m):
                    idx |= (1 << m)
                    
            g_est = val_base / self.epsilon
            grad[idx] += g_est
            
        return grad, U_base, evals

def test_sparse_sphere():
    D = 2**20 # 1,048,576
    B = 2**10 # 1024
    
    print(f"Testing SFWHT Gradient Estimation on Sparse Sphere (D={D:,})")
    
    x = torch.randn(D, device=device) * 0.1
    
    # sparse sphere: only 15 active variables
    num_active = 15
    active_indices = torch.randperm(D, device=device)[:num_active]
    print(f"Ground Truth Active Indices: {active_indices.cpu().numpy()}")
    
    def f(tx):
        return torch.sum(tx[:, active_indices]**2, dim=1)
        
    true_grad = torch.zeros(D, device=device)
    true_grad[active_indices] = 2 * x[active_indices]
    
    opt = SFWHT_Optimizer(D, B, epsilon=1e-3, threshold=None)
    
    t0 = time.time()
    est_grad, _, evals = opt.estimate_gradient(f, x)
    t1 = time.time()
    
    print("-" * 50)
    # Compare
    mask = true_grad != 0
    print("True Grad (active):", true_grad[mask].cpu().numpy())
    print("Est Grad  (active):", est_grad[mask].cpu().numpy())
    
    err = torch.norm(true_grad - est_grad)
    print(f"\nError (L2): {err.item():.6f}")
    print(f"Evaluations used: {evals:,}")
    print(f"Theoretical Finite Diffs: {D+1:,}")
    print(f"Compression Ratio: {(D+1)/evals:.1f}x")
    print(f"Time: {t1-t0:.4f}s")
    
    # Validate sparsity pattern
    est_active = torch.where(est_grad != 0)[0]
    print(f"\nEstimated Active Indices: {est_active.cpu().numpy()}")
    
    metrics = {
        "D": D,
        "B": B,
        "epsilon": opt.epsilon,
        "active_vars": num_active,
        "evals": evals,
        "error_l2": err.item(),
        "time_s": t1-t0,
        "compression_ratio": (D+1)/evals
    }
    
    os.makedirs("results/raw", exist_ok=True)
    with open(f"results/raw/sfwht_benchmark_v19.json", "w") as f_out:
        json.dump(metrics, f_out, indent=4)
        
if __name__ == "__main__":
    test_sparse_sphere()