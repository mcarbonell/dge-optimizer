import numpy as np
import math
import time
import os
import json

# =============================================================================
# OPTIMIZADOR DGE v17b (Time-Budget Prototype)
# =============================================================================
class DGEOptimizerV17b:
    def __init__(self, dim, lr=1.0, delta=1e-3, beta1=0.95, beta2=0.999,
                 eps=1e-8, k=10, greedy_w_init=0.3, greedy_w_final=0.0,
                 clip_norm=1.0, seed=None):
        self.dim = dim
        self.lr = lr
        self.delta = delta
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.greedy_w_init = greedy_w_init
        self.greedy_w_final = greedy_w_final
        self.clip_norm = clip_norm
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0

    def step(self, f, x):
        self.t += 1
        # Simple schedule for timed runs (assuming we don't know total steps)
        greedy_w = max(0.0, self.greedy_w_init * (0.99 ** self.t))

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

        self.m = self.beta1 * self.m + (1 - self.beta1) * g_step
        self.v = self.beta2 * self.v + (1 - self.beta2) * g_step ** 2
        mh = self.m / (1 - self.beta1 ** self.t + 1e-30)
        vh = self.v / (1 - self.beta2 ** self.t + 1e-30)
        upd_ema = self.lr * mh / (np.sqrt(vh) + self.eps)
        
        if np.linalg.norm(upd_ema) > self.clip_norm:
            upd_ema *= self.clip_norm / np.linalg.norm(upd_ema)

        return x - upd_ema - (greedy_w * self.lr * best_dir), 2 * self.k

# =============================================================================
# EXPERIMENT: CONSTANT WALL-CLOCK BUDGET
# =============================================================================
def run_timed_test(k=10, lr=0.5, time_limit=3.0, seed=42):
    dim = 10000
    print(f"\n>>> TIMED TEST: k={k}, LR={lr} | Limit: {time_limit}s")
    
    scales = 1000.0 ** (np.arange(dim) / (dim - 1))
    def f(x):
        return np.sum(scales * (x**2))

    rng = np.random.default_rng(seed)
    x = rng.uniform(2.0, 5.0, size=dim).astype(np.float32)
    
    opt = DGEOptimizerV17b(dim, lr=lr, k=k, seed=seed)
    
    evals = 0
    t_start = time.time()
    
    while (time.time() - t_start) < time_limit:
        x, n = opt.step(f, x)
        evals += n
    
    t_actual = time.time() - t_start
    final_loss = f(x)
    print(f"    Final Loss: {final_loss:.4e} | Evals: {evals} | Steps: {opt.t} | Actual Time: {t_actual:.2f}s")
    
    return {"k": k, "lr": lr, "final_loss": final_loss, "evals": evals, "steps": opt.t, "time": t_actual}

if __name__ == "__main__":
    # Test Ellipsoid with fixed 3 seconds wall-clock budget
    TIME_BUDGET = 5.0
    configs = [
        (4, 0.5),   # Large blocks (High overhead per eval)
        (16, 0.5),  # Medium blocks
        (64, 0.5),  # Small blocks (Low overhead per eval)
        (256, 0.5), # Extremely small blocks
    ]
    
    results = []
    for k, lr in configs:
        res = run_timed_test(k=k, lr=lr, time_limit=TIME_BUDGET, seed=42)
        results.append(res)
    
    print(f"\n=== CONSTANT TIME SUMMARY ({TIME_BUDGET}s) ===")
    print(f"{'k':>5} | {'Evals':>8} | {'Steps':>6} | {'Final Loss':>12}")
    print("-" * 45)
    for r in results:
        print(f"{r['k']:>5} | {r['evals']:>8} | {r['steps']:>6} | {r['final_loss']:>12.4e}")
