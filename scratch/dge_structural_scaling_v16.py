import numpy as np
import math
import time
import os
import json

# =============================================================================
# OPTIMIZADOR DGE v16 (Structural Scaling Prototype)
# =============================================================================
class DGEOptimizerV16:
    """
    DGE v16: Focused on Structural Scaling.
    Tests the hypothesis: Does increasing k (decreasing block size) 
    improve SNR and convergence speed in massive dimensions?
    """
    def __init__(self, dim, lr=1.0, delta=1e-3, beta1=0.95, beta2=0.999,
                 eps=1e-8, lr_decay=0.01, total_steps=1000, 
                 greedy_w_init=0.3, greedy_w_final=0.0,
                 k_multiplier=1.0, clip_norm=1.0, seed=None):
        self.dim = dim
        self.lr0 = lr
        self.delta = delta
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.lr_decay = lr_decay
        self.total_steps = total_steps
        self.greedy_w_init = greedy_w_init
        self.greedy_w_final = greedy_w_final
        self.clip_norm = clip_norm
        self.rng = np.random.default_rng(seed)
        
        # k is now log2(D) * multiplier
        base_k = math.ceil(math.log2(dim)) if dim > 1 else 1
        self.k = max(1, math.ceil(base_k * k_multiplier))
        
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0
        self.history_snr = []

    def _cosine_schedule(self, v_init, v_final):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v_final + (v_init - v_final) * 0.5 * (1 + math.cos(math.pi * frac))

    def step(self, f, x):
        self.t += 1
        lr = self._cosine_schedule(self.lr0, self.lr0 * self.lr_decay)
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

        # 3. SNR Calculation
        if self.t % 5 == 0:
            corr = np.corrcoef(g_step, self.m)[0, 1]
            if not np.isnan(corr): self.history_snr.append(float(corr))

        return x - upd_ema - (greedy_w * lr * best_dir), 2 * self.k

# =============================================================================
# MASSIVE DIMENSION TEST (D = 1,000,000)
# =============================================================================
def run_massive_test(dim=1000000, k_mult=1.0, budget=10000, seed=42):
    print(f"\n>>> TESTING D={dim:,} | k_multiplier={k_mult}")
    
    # Target function: Sparse Sphere (Only 100 dimensions matter)
    # This tests if DGE can find the needle in the haystack in massive D
    target_idx = np.arange(100)
    def f(x):
        return np.sum(x[target_idx]**2)

    rng = np.random.default_rng(seed)
    x = rng.uniform(2.0, 5.0, size=dim).astype(np.float32)
    
    base_k = math.ceil(math.log2(dim))
    k = math.ceil(base_k * k_mult)
    total_steps = budget // (2 * k)
    
    opt = DGEOptimizerV16(dim, total_steps=total_steps, k_multiplier=k_mult, seed=seed)
    
    evals = 0
    t0 = time.time()
    t_eval = 0.0
    
    initial_loss = f(x)
    print(f"    Initial Loss: {initial_loss:.4f} | k: {k} ({2*k} evals/step)")
    
    while evals < budget:
        t_e0 = time.time()
        loss_val = f(x) # just for logging
        t_eval += time.time() - t_e0
        
        x, n = opt.step(f, x)
        evals += n
        
        if (evals // (2*k)) % 10 == 0:
            snr = opt.history_snr[-1] if opt.history_snr else 0
            print(f"    Evals: {evals:>7} | Loss: {f(x):.6f} | SNR: {snr:.4f}")

    t_total = time.time() - t0
    final_loss = f(x)
    print(f"    FINAL LOSS: {final_loss:.6e} | SNR Final: {opt.history_snr[-1]:.4f}")
    print(f"    Overhead: {(t_total - t_eval)/t_total:.1%}")
    
    return {
        "dim": dim,
        "k_mult": k_mult,
        "final_loss": float(final_loss),
        "snr_final": float(opt.history_snr[-1]),
        "overhead_pct": (t_total - t_eval)/t_total
    }

if __name__ == "__main__":
    # Test D=1M with different k-multipliers
    # budget is function evaluations
    results = []
    for m in [0.5, 1.0, 2.0]:
        res = run_massive_test(dim=1000000, k_mult=m, budget=5000, seed=42)
        results.append(res)
    
    print("\n=== SUMMARY D=1,000,000 ===")
    for r in results:
        print(f"k_mult: {r['k_mult']} | Loss: {r['final_loss']:.4e} | SNR: {r['snr_final']:.4f} | Overhead: {r['overhead_pct']:.1%}")
