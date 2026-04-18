import numpy as np
import math

class DGEOptimizer:
    """
    Dichotomous Gradient Estimation (DGE) Optimizer.
    
    A zeroth-order (derivative-free) optimizer that circumvents the O(D) finite-difference 
    curse of dimensionality. It isolates the most active parameters by testing random 
    subsets (blocks) and maintains a historical Exponential Moving Average (EMA) of 
    the gradients to statistically cancel out the cross-variable noise.

    O(log2(D)) function evaluations per optimization step.
    """
    def __init__(self, dim: int, lr: float = 1.0, delta: float = 1e-3, 
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, 
                 lr_decay: float = 0.01, delta_decay: float = 0.05,
                 total_steps: int = 1000, greedy_w: float = 0.3, 
                 clip_norm: float = 1.0, seed: int | None = None):
        self.dim = dim
        self.lr0 = lr
        self.delta0 = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr_decay = lr_decay
        self.delta_decay = delta_decay
        self.total_steps = total_steps
        self.greedy_w = greedy_w
        self.clip_norm = clip_norm
        self.rng = np.random.default_rng(seed)
        
        # Number of random blocks to test per step: k ≈ log2(D)
        self.k = max(1, math.ceil(math.log2(dim))) if dim > 1 else 1
        self.group_size = max(1, math.ceil(dim / self.k))
        
        # Scaling the learning rate so it's invariant to dimension/groups
        self.lr_scale = 1.0 / math.sqrt(self.k)
        
        # EMA Adam state
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0

    def _cosine(self, v0, decay):
        """Cosine annealing schedule for learning rate and delta."""
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))

    def step(self, f, x):
        """
        Executes a single optimization step.
        f: Callable that takes a flattened parameter array x and returns a scalar loss.
           CRITICAL: In stochastic problems (e.g. Minibatches in ML), f() MUST use 
           the exact same data batch for both forward evaluations inside this step.
        x: Flattened parameter array.
        Returns: (x_new, num_evaluations_used)
        """
        self.t += 1
        lr    = self._cosine(self.lr0, self.lr_decay) * self.lr_scale
        delta = self._cosine(self.delta0, self.delta_decay)

        # 1. Create k random blocks of indices
        groups = [self.rng.choice(self.dim, size=self.group_size, replace=False)
                  for _ in range(self.k)]
        
        # SPSA-style signs
        signs = self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)

        g = np.zeros(self.dim, dtype=np.float32)
        g_cnt = np.zeros(self.dim, dtype=np.int32)
        
        best_s = -1.0
        best_dir = np.zeros(self.dim, dtype=np.float32)

        # 2. Test each block (Explotación)
        for idx in groups:
            pert = np.zeros(self.dim, dtype=np.float32)
            pert[idx] = signs[idx] * delta
            
            # 2 evaluations per block -> Total = 2*k evaluations
            fp = f(x + pert)
            fm = f(x - pert)
            
            sg = (fp - fm) / (2.0 * delta)
            
            # Add gradient signal for variables in this block
            g[idx] += sg * signs[idx]
            g_cnt[idx] += 1
            
            # Track the block with the steepest impact (Greedy Step)
            if abs(sg) > best_s:
                best_s = abs(sg)
                d = np.zeros(self.dim, dtype=np.float32)
                d[idx] = signs[idx]
                dn = np.linalg.norm(d)
                best_dir = -np.sign(sg) * d / (dn + 1e-12)

        # Average overlapping dimensions
        ev = g_cnt > 0
        g[ev] /= g_cnt[ev]

        # 3. Adam EMA Accumulation (Exploración temporal y Denoising)
        self.m[ev] = self.beta1 * self.m[ev] + (1 - self.beta1) * g[ev]
        self.v[ev] = self.beta2 * self.v[ev] + (1 - self.beta2) * g[ev] ** 2

        mh = self.m / (1 - self.beta1 ** self.t + 1e-30)
        vh = self.v / (1 - self.beta2 ** self.t + 1e-30)

        upd = np.zeros(self.dim, dtype=np.float32)
        upd[ev] = lr * mh[ev] / (np.sqrt(vh[ev]) + self.eps)
        
        # Gradient Clipping (Crucial for High-Dimensional stability)
        un = np.linalg.norm(upd)
        if un > self.clip_norm:
            upd *= self.clip_norm / un

        # Final Update = Adam History + Greedy Immediate Step
        x_new = x - upd - self.greedy_w * lr * best_dir
        
        return x_new, 2 * self.k
