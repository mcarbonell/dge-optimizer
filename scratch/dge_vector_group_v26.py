import numpy as np
import time
import math

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

class PureDGE:
    def __init__(self, dim: int, k_blocks: int, lr: float = 0.1, delta: float = 1e-3, 
                 total_steps: int = 10000, seed: int = 42):
        self.dim = dim
        self.k = k_blocks
        self.lr0 = lr
        self.delta0 = delta
        self.total_steps = total_steps
        self.rng = np.random.default_rng(seed)
        
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)

        grad = np.zeros(self.dim, dtype=np.float32)
        perm = self.rng.permutation(self.dim)
        blocks = np.array_split(perm, self.k)
        
        evals = 0
        for block in blocks:
            if len(block) == 0: continue
            signs = self.rng.choice([-1.0, 1.0], size=len(block)).astype(np.float32)
            pert = np.zeros(self.dim, dtype=np.float32)
            pert[block] = signs * delta
            
            fp = f(x + pert)
            fm = f(x - pert)
            evals += 2
            
            g_est = (fp - fm) / (2.0 * delta) * signs
            grad[block] = g_est

        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * (grad ** 2)

        mh = self.m / (1 - 0.9 ** self.t)
        vh = self.v / (1 - 0.999 ** self.t)

        upd = lr * mh / (np.sqrt(vh) + 1e-8)
        return x - upd, evals

class ScalarVarianceDGE:
    """Uses +/- 1 perturbations, but group-averaged scalar variance."""
    def __init__(self, dim: int, num_groups: int, lr: float = 0.1, delta: float = 1e-3, 
                 total_steps: int = 10000, seed: int = 42):
        self.dim = dim
        self.num_groups = num_groups
        self.groups = np.array_split(np.arange(dim), num_groups)
        
        self.lr0 = lr
        self.delta0 = delta
        self.total_steps = total_steps
        self.rng = np.random.default_rng(seed)
        
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(self.num_groups, dtype=np.float32)
        self.t = 0

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)

        grad = np.zeros(self.dim, dtype=np.float32)
        evals = 0
        
        for g_idx, block in enumerate(self.groups):
            signs = self.rng.choice([-1.0, 1.0], size=len(block)).astype(np.float32)
            pert = np.zeros(self.dim, dtype=np.float32)
            pert[block] = signs * delta
            
            fp = f(x + pert)
            fm = f(x - pert)
            evals += 2
            
            g_est = (fp - fm) / (2.0 * delta) * signs
            grad[block] = g_est

        self.m = 0.9 * self.m + 0.1 * grad
        
        for g_idx, block in enumerate(self.groups):
            sq_norm = np.sum(grad[block]**2) / len(block)
            self.v[g_idx] = 0.999 * self.v[g_idx] + 0.001 * sq_norm

        mh = self.m / (1 - 0.9 ** self.t)
        vh = self.v / (1 - 0.999 ** self.t)

        upd = np.zeros(self.dim, dtype=np.float32)
        for g_idx, block in enumerate(self.groups):
            upd[block] = lr * mh[block] / (np.sqrt(vh[g_idx]) + 1e-8)

        return x - upd, evals

def run_experiment(method, dim, budget, lr=0.1, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=dim).astype(np.float32)
    
    k_blocks = 8
    total_steps = budget // (2 * k_blocks)
    
    if method == "pure":
        opt = PureDGE(dim, k_blocks, lr=lr, delta=1e-3, total_steps=total_steps, seed=seed)
    else:
        opt = ScalarVarianceDGE(dim, num_groups=k_blocks, lr=lr, delta=1e-3, total_steps=total_steps, seed=seed)
        
    evals = 0
    while evals < budget:
        x, e = opt.step(rosenbrock, x)
        evals += e
            
    return rosenbrock(x)

if __name__ == "__main__":
    DIM = 128
    BUDGET = 200_000
    SEEDS = 5
    
    print(f"--- ROSENBROCK D={DIM} | BUDGET={BUDGET} ---")
    
    for method in ["pure", "scalar_var"]:
        losses = [run_experiment(method, DIM, BUDGET, lr=0.1, seed=s) for s in range(SEEDS)]
        print(f"{method.upper():<10} DGE: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
