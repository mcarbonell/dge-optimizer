import numpy as np
import math
import time
import os
import json

# =============================================================================
# OPTIMIZADOR DGE v16b (LR Scaling Hypothesis)
# =============================================================================
class DGEOptimizerV16b:
    def __init__(self, dim, lr=1.0, delta=1e-3, beta1=0.95, beta2=0.999,
                 eps=1e-8, total_steps=1000, k=10, 
                 greedy_w_init=0.3, greedy_w_final=0.0,
                 clip_norm=1.0, seed=None):
        self.dim = dim
        self.lr0 = lr
        self.delta = delta
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.total_steps = total_steps
        self.greedy_w_init = greedy_w_init
        self.greedy_w_final = greedy_w_final
        self.clip_norm = clip_norm
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0
        self.history_snr = []

    def _cosine_schedule(self, v_init, v_final):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v_final + (v_init - v_final) * 0.5 * (1 + math.cos(math.pi * frac))

    def step(self, f, x):
        self.t += 1
        lr = self._cosine_schedule(self.lr0, self.lr0 * 0.01)
        greedy_w = self._cosine_schedule(self.greedy_w_init, self.greedy_w_final)

        # 1. ORTHOGONAL PARTITIONING
        perm = self.rng.permutation(self.dim)
        groups = np.array_split(perm, self.k)
        signs = self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
        g_step = np.zeros(self.dim, dtype=np.float32)
        best_s, best_dir = -1.0, np.zeros(self.dim, dtype=np.float32)

        for idx in groups:
            pert = np.zeros(self.dim, dtype=np.float32)
            pert[idx] = signs[idx] * self.delta
            fp, fm = f(x + pert), f(x - pert)
            sg = (fp - fm) / (2.0 * self.delta)
            g_step[idx] = sg * signs[idx]
            if abs(sg) > best_s:
                best_s = abs(sg)
                d = np.zeros(self.dim, dtype=np.float32)
                d[idx] = signs[idx]
                best_dir = -np.sign(sg) * d / (np.linalg.norm(d) + 1e-12)

        # 2. Adam Update
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_step
        self.v = self.beta2 * self.v + (1 - self.beta2) * g_step ** 2
        mh = self.m / (1 - self.beta1 ** self.t + 1e-30)
        vh = self.v / (1 - self.beta2 ** self.t + 1e-30)
        upd_ema = lr * mh / (np.sqrt(vh) + self.eps)
        
        un = np.linalg.norm(upd_ema)
        if un > self.clip_norm:
            upd_ema *= self.clip_norm / un

        if self.t % 5 == 0:
            corr = np.corrcoef(g_step, self.m)[0, 1]
            if not np.isnan(corr): self.history_snr.append(float(corr))

        return x - upd_ema - (greedy_w * lr * best_dir), 2 * self.k

# =============================================================================
# EXPERIMENT: LR SCALING vs k (D = 100,000)
# =============================================================================
def run_scaling_test(k=10, lr=0.5, budget=10000, seed=42):
    dim = 100000
    print(f"\n>>> CONFIG: k={k}, LR={lr} | Budget: {budget}")
    
    def f(x):
        return np.sum(x**2) # Dense Sphere

    rng = np.random.default_rng(seed)
    x = rng.uniform(2.0, 5.0, size=dim).astype(np.float32)
    
    total_steps = budget // (2 * k)
    opt = DGEOptimizerV16b(dim, lr=lr, total_steps=total_steps, k=k, seed=seed)
    
    evals = 0
    initial_loss = f(x)
    
    while evals < budget:
        x, n = opt.step(f, x)
        evals += n
    
    final_loss = f(x)
    snr_final = opt.history_snr[-1] if opt.history_snr else 0
    print(f"    Final Loss: {final_loss:.4e} | SNR Final: {snr_final:.4f} | Steps: {total_steps}")
    
    return {"k": k, "lr": lr, "final_loss": final_loss, "snr": snr_final}

if __name__ == "__main__":
    configs = [
        (8, 0.2),   # Low k, Low LR
        (8, 1.0),   # Low k, High LR (might be unstable?)
        (32, 0.2),  # Med k, Low LR
        (32, 1.0),  # Med k, Med LR
        (128, 1.0), # High k, Med LR
        (128, 4.0), # High k, High LR (clean signal allows this?)
    ]
    
    results = []
    for k, lr in configs:
        res = run_scaling_test(k=k, lr=lr, budget=100000, seed=42)
        results.append(res)
    
    print("\n=== COMPARISON TABLE ===")
    print(f"{'k':>5} | {'LR':>5} | {'Final Loss':>12} | {'SNR':>7}")
    print("-" * 40)
    for r in results:
        print(f"{r['k']:>5} | {r['lr']:>5.1f} | {r['final_loss']:>12.4e} | {r['snr']:>7.4f}")
