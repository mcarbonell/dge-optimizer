import numpy as np
import math
import time
import os
import json

# =============================================================================
# OPTIMIZADOR DGE v17c (Extreme Throughput Test)
# =============================================================================
class DGEOptimizerV17c:
    def __init__(self, dim, lr=1.0, delta=1e-3, beta1=0.95, beta2=0.999,
                 eps=1e-8, k=10, clip_norm=1.0, seed=None):
        self.dim = dim
        self.lr = lr
        self.delta = delta
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.clip_norm = clip_norm
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0

    def step(self, f, x):
        self.t += 1
        perm = self.rng.permutation(self.dim)
        groups = np.array_split(perm, self.k)
        signs = self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
        g_step = np.zeros(self.dim, dtype=np.float32)

        for idx in groups:
            pert = np.zeros(self.dim, dtype=np.float32)
            pert[idx] = signs[idx] * self.delta
            fp, fm = f(x + pert), f(x - pert)
            sg = (fp - fm) / (2.0 * self.delta)
            g_step[idx] = sg * signs[idx]

        self.m = self.beta1 * self.m + (1 - self.beta1) * g_step
        self.v = self.beta2 * self.v + (1 - self.beta2) * g_step ** 2
        mh = self.m / (1 - self.beta1 ** self.t + 1e-30)
        vh = self.v / (1 - self.beta2 ** self.t + 1e-30)
        upd_ema = self.lr * mh / (np.sqrt(vh) + self.eps)
        
        if np.linalg.norm(upd_ema) > self.clip_norm:
            upd_ema *= self.clip_norm / np.linalg.norm(upd_ema)

        return x - upd_ema, 2 * self.k

# =============================================================================
# EXPERIMENT: EXTREME THROUGHPUT LIMITS
# =============================================================================
def run_extreme_test(k=10, time_limit=10.0, dim=10000, seed=42):
    print(f"\n>>> EXTREME TEST: D={dim:,}, k={k} | Limit: {time_limit}s")
    
    scales = 1000.0 ** (np.arange(dim) / (dim - 1))
    def f(x):
        return np.sum(scales * (x**2))

    rng = np.random.default_rng(seed)
    x = rng.uniform(2.0, 5.0, size=dim).astype(np.float32)
    
    opt = DGEOptimizerV17c(dim, lr=1.0, k=k, seed=seed)
    
    evals = 0
    t_start = time.time()
    
    while (time.time() - t_start) < time_limit:
        x, n = opt.step(f, x)
        evals += n
    
    t_actual = time.time() - t_start
    final_loss = f(x)
    print(f"    Final Loss: {final_loss:.4e} | Evals: {evals} | Steps: {opt.t} | Actual Time: {t_actual:.2f}s")
    
    return {"k": k, "evals": evals, "steps": opt.t, "final_loss": final_loss, "evals_per_sec": evals / t_actual}

if __name__ == "__main__":
    TIME_BUDGET = 5.0  # Match v17b's 5.0s budget
    DIMENSION = 10000  # Match v17b's 10,000 dimension
    configs = [64, 256, 1024, 4096] # Testing high k range
    
    results = []
    for k in configs:
        res = run_extreme_test(k=k, time_limit=TIME_BUDGET, dim=DIMENSION, seed=42)
        results.append(res)
    
    print(f"\n=== EXTREME THROUGHPUT SUMMARY (D={DIMENSION:,}, {TIME_BUDGET}s) ===")
    print(f"{'k':>5} | {'Evals':>10} | {'Steps':>6} | {'Evals/s':>10} | {'Final Loss':>12}")
    print("-" * 65)
    for r in results:
        print(f"{r['k']:>5} | {r['evals']:>10} | {r['steps']:>6} | {r['evals_per_sec']:>10.0f} | {r['final_loss']:>12.4e}")
