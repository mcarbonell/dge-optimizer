"""
scratch/dge_speed_benchmark.py
==============================
Benchmarking DGE implementations to measure overhead and execution speed.
Compares:
  - V2_Numpy (Baseline: Sequential CPU loops + scalar f())
  - V3_Torch_Loop (PyTorch, Sequential loops + scalar f())
  - V3_Torch_Batched_Hybrid (PyTorch, loop for matrix build + Batched f())
  - V3_Torch_Batched_Pure (PyTorch, fully vectorized scatter + Batched f())
"""

import time
import math
import numpy as np
import torch

import torch.nn.functional as F

try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = 109386 # MLP params size
K = 16
LR = 0.1
DELTA = 1e-3
STEPS = 20

print(f"Device: {device}")
print(f"D = {D:,} | K = {K} | STEPS = {STEPS}")
print("-" * 50)

# --- Synthetic Objective Functions ---
# Simulate a 4-layer MLP: (784, 128, 64, 10) -> Total params ~ 109,000
ARCH = [784, 128, 64, 10]
sizes = [a * b + b for a, b in zip(ARCH[:-1], ARCH[1:])]
D_MLP = sum(sizes) # 109386

# Generate a fixed dataset batch (e.g. batch size 256)
X_batch_np = np.random.randn(256, 784).astype(np.float32)
y_batch_np = np.random.randint(0, 10, size=256)

X_batch_th = torch.from_numpy(X_batch_np).to(device)
y_batch_th = torch.from_numpy(y_batch_np).long().to(device)

def batched_mlp_forward_np(params, X):
    # Sequential (1 parameter vector)
    h = X
    i = 0
    for l_in, l_out in zip(ARCH[:-1], ARCH[1:]):
        W = params[i:i + l_in * l_out].reshape(l_in, l_out)
        i += l_in * l_out
        b = params[i:i + l_out]
        i += l_out
        h = np.dot(h, W) + b
        if l_out != ARCH[-1]:
            h = np.maximum(0, h)
    
    # Cross entropy (simplified)
    exp_h = np.exp(h - np.max(h, axis=1, keepdims=True))
    probs = exp_h / np.sum(exp_h, axis=1, keepdims=True)
    return -np.mean(np.log(probs[np.arange(len(y_batch_np)), y_batch_np] + 1e-8))

def f_np(x):
    return batched_mlp_forward_np(x, X_batch_np)

def f_torch(params_batch):
    # params_batch: (D,) or (P, D)
    is_single = params_batch.ndim == 1
    if is_single:
        params_batch = params_batch.unsqueeze(0)
    
    P = params_batch.shape[0]
    h = X_batch_th.unsqueeze(0).expand(P, -1, -1)
    
    i = 0
    for l_in, l_out in zip(ARCH[:-1], ARCH[1:]):
        W = params_batch[:, i:i + l_in * l_out].view(P, l_in, l_out)
        i += l_in * l_out
        b = params_batch[:, i:i + l_out].view(P, 1, l_out)
        i += l_out
        h = torch.bmm(h, W) + b
        if l_out != ARCH[-1]:
            h = torch.relu(h)
            
    # Cross entropy
    B, C = h.shape[1], h.shape[2]
    h = h.reshape(P * B, C)
    t = y_batch_th.unsqueeze(0).expand(P, -1).reshape(-1)
    loss = torch.nn.functional.cross_entropy(h, t, reduction='none').view(P, B).mean(dim=1)
    
    if is_single:
        return loss[0]
    return loss

# --- Implementations ---

def step_v2_numpy(x, m, v):
    perm = np.random.permutation(D)
    blocks = np.array_split(perm, K)
    grad = np.zeros(D, dtype=np.float32)
    
    for block in blocks:
        signs = np.random.choice([-1.0, 1.0], size=len(block)).astype(np.float32)
        pert = np.zeros(D, dtype=np.float32)
        pert[block] = signs * DELTA
        
        fp = f_np(x + pert)
        fm = f_np(x - pert)
        
        grad[block] = (fp - fm) / (2.0 * DELTA) * signs
        
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * (grad**2)
    upd = LR * m / (np.sqrt(v) + 1e-8)
    return x - upd, m, v

def step_v3_torch_loop(x, m, v):
    perm = torch.randperm(D, device=device)
    # torch.tensor_split allows uneven splits like np.array_split
    blocks = torch.tensor_split(perm, K)
    grad = torch.zeros(D, device=device)
    
    for block in blocks:
        signs = torch.randint(0, 2, (len(block),), device=device).float() * 2 - 1
        pert = torch.zeros(D, device=device)
        pert[block] = signs * DELTA
        
        fp = f_torch(x + pert)
        fm = f_torch(x - pert)
        
        grad[block] = (fp - fm) / (2.0 * DELTA) * signs
        
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * (grad**2)
    upd = LR * m / (torch.sqrt(v) + 1e-8)
    return x - upd, m, v

def step_v3_batched_hybrid(x, m, v):
    perm = torch.randperm(D, device=device)
    signs = torch.randint(0, 2, (D,), device=device).float() * 2 - 1
    blocks = torch.tensor_split(perm, K)
    
    P = torch.zeros((2*K, D), device=device)
    for i, block in enumerate(blocks):
        idx = block
        s = signs[idx] * DELTA
        P[2*i, idx] = s
        P[2*i+1, idx] = -s
        
    # Single Batched Forward
    losses = f_torch(x.unsqueeze(0) + P)
    diffs = (losses[0::2] - losses[1::2]) / (2.0 * DELTA)
    
    grad = torch.zeros(D, device=device)
    for i, block in enumerate(blocks):
        grad[block] = diffs[i] * signs[block]
        
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * (grad**2)
    upd = LR * m / (torch.sqrt(v) + 1e-8)
    return x - upd, m, v

def step_v3_batched_pure(x, m, v):
    perm = torch.randperm(D, device=device)
    signs = torch.randint(0, 2, (D,), device=device).float() * 2 - 1
    
    group_size = (D + K - 1) // K
    pad = group_size * K - D
    
    if pad > 0:
        perm_pad = torch.cat([perm, torch.zeros(pad, dtype=torch.long, device=device)])
        signs_pad = torch.cat([signs, torch.zeros(pad, device=device)])
    else:
        perm_pad = perm
        signs_pad = signs
        
    perm_mat = perm_pad.view(K, group_size)
    signs_mat = signs_pad.view(K, group_size) * DELTA
    
    P_plus = torch.zeros((K, D), device=device)
    P_plus.scatter_(1, perm_mat, signs_mat)
    
    if pad > 0:
        P_plus[:, 0] = 0.0
        idx0_mask = (perm == 0)
        idx0_pos = idx0_mask.nonzero(as_tuple=True)[0]
        if len(idx0_pos) > 0:
            block0 = idx0_pos[0] // group_size
            P_plus[block0, 0] = signs[idx0_pos[0]] * DELTA
            
    P = torch.empty((2*K, D), device=device)
    P[0::2] = P_plus
    P[1::2] = -P_plus
    
    # Batched Forward
    losses = f_torch(x.unsqueeze(0) + P)
    diffs = (losses[0::2] - losses[1::2]) / (2.0 * DELTA)
    
    grad = torch.zeros(D, device=device)
    diffs_exp = diffs.unsqueeze(1).expand(K, group_size).flatten()
    if pad > 0:
        diffs_exp = diffs_exp[:D]
        
    grad[perm] = diffs_exp * signs
    
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * (grad**2)
    upd = LR * m / (torch.sqrt(v) + 1e-8)
    return x - upd, m, v

# Compile the pure step
try:
    step_v3_batched_compiled = torch.compile(step_v3_batched_pure, mode="reduce-overhead")
except Exception:
    step_v3_batched_compiled = step_v3_batched_pure

# --- Runner ---
def run_benchmark(name, step_fn, use_torch=False):
    if use_torch:
        x = torch.ones(D, device=device)
        m = torch.zeros(D, device=device)
        v = torch.zeros(D, device=device)
    else:
        x = np.ones(D, dtype=np.float32)
        m = np.zeros(D, dtype=np.float32)
        v = np.zeros(D, dtype=np.float32)
        
    # Warmup
    x, m, v = step_fn(x, m, v)
    
    # Benchmark
    t0 = time.time()
    for _ in range(STEPS):
        x, m, v = step_fn(x, m, v)
    t1 = time.time()
    
    fps = STEPS / (t1 - t0)
    
    # Calculate checksum to ensure parity
    checksum = float(x.sum().cpu().numpy() if use_torch else x.sum())
    
    return {"time": t1 - t0, "fps": fps, "checksum": checksum}

if __name__ == "__main__":
    results = {}
    
    print("Running V2 Numpy...")
    results["V2_Numpy"] = run_benchmark("V2_Numpy", step_v2_numpy, use_torch=False)
    
    print("Running V3 Torch Loop...")
    results["V3_Torch_Loop"] = run_benchmark("V3_Torch_Loop", step_v3_torch_loop, use_torch=True)
    
    print("Running V3 Batched Hybrid...")
    results["V3_Batched_Hybrid"] = run_benchmark("V3_Batched_Hybrid", step_v3_batched_hybrid, use_torch=True)
    
    print("Running V3 Batched Pure...")
    results["V3_Batched_Pure"] = run_benchmark("V3_Batched_Pure", step_v3_batched_pure, use_torch=True)
    
    print("Running V3 Compiled...")
    try:
        results["V3_Compiled"] = run_benchmark("V3_Compiled", step_v3_batched_compiled, use_torch=True)
    except Exception as e:
        print(f"Compilation failed or not supported: {e}")
        
    print("\n" + "="*70)
    print(f"{'Method':<20} | {'Time (s)':<10} | {'Steps/s':<10} | {'Checksum':<15} | {'Speedup'}")
    print("-" * 70)
    
    base_time = results["V2_Numpy"]["time"]
    for name, r in results.items():
        sp = base_time / r['time']
        print(f"{name:<20} | {r['time']:<10.3f} | {r['fps']:<10.2f} | {r['checksum']:<15.5f} | {sp:.1f}x")
    print("="*70)
