import numpy as np
import math

class DGEOptimizerV2:
    """
    Denoised Gradient Estimation (DGE) Optimizer V2.
    Optimized for Vectorized execution and Signal-to-Noise recovery.
    """
    def __init__(self, dim: int, lr: float = 1.0, delta: float = 1e-3, 
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, 
                 lr_decay: float = 0.01, delta_decay: float = 0.05,
                 total_steps: int = 1000, greedy_w: float = 0.1, 
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
        
        # O(log2(D)) efficiency
        self.k = max(1, math.ceil(math.log2(dim))) if dim > 1 else 1
        self.group_size = max(1, math.ceil(dim / self.k))
        self.lr_scale = 1.0 / math.sqrt(self.k)
        
        # Adam state
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0
        
        # V2: Pre-allocated masks to reduce RNG overhead
        self._precompute_masks()

    def _precompute_masks(self):
        """Generates a bank of random indices to cycle through."""
        self.mask_bank_size = 100
        self.mask_bank = [self.rng.choice(self.dim, size=self.group_size, replace=False)
                          for _ in range(self.mask_bank_size)]
        self.mask_idx = 0

    def _cosine(self, v0, decay):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr    = self._cosine(self.lr0, self.lr_decay) * self.lr_scale
        delta = self._cosine(self.delta0, self.delta_decay)

        # 1. Selection of random blocks (using the precomputed bank)
        groups = []
        for _ in range(self.k):
            groups.append(self.mask_bank[self.mask_idx])
            self.mask_idx = (self.mask_idx + 1) % self.mask_bank_size
        
        # SPSA signs
        signs = self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)

        g_accum = np.zeros(self.dim, dtype=np.float32)
        g_count = np.zeros(self.dim, dtype=np.float32)
        
        best_impact = -1.0
        best_dir = np.zeros(self.dim, dtype=np.float32)

        # 2. Block-wise Gradient Estimation
        # We evaluate f(x + pert) and f(x - pert) per block
        for idx in groups:
            pert = np.zeros(self.dim, dtype=np.float32)
            pert[idx] = signs[idx] * delta
            
            # Forward and backward evaluations
            # IMPORTANT: f must be deterministic for the same step (same batch)
            fp = f(x + pert)
            fm = f(x - pert)
            
            # Finite difference estimator
            sg = (fp - fm) / (2.0 * delta)
            
            # Update signal
            g_accum[idx] += sg * signs[idx]
            g_count[idx] += 1
            
            # Greedy heuristic (V2: uses normalized impact)
            impact = abs(sg)
            if impact > best_impact:
                best_impact = impact
                d = np.zeros(self.dim, dtype=np.float32)
                d[idx] = signs[idx] * (-np.sign(sg))
                norm_d = np.linalg.norm(d)
                if norm_d > 1e-12:
                    best_dir = d / norm_d

        # Average estimated gradients
        active = g_count > 0
        g_accum[active] /= g_count[active]

        # 3. Enhanced Adam Denoising
        # Variables NOT in this step's blocks get a beta1 decay toward zero 
        # to cancel out long-term alias noise.
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_accum
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g_accum ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t + 1e-10)
        v_hat = self.v / (1 - self.beta2 ** self.t + 1e-10)

        # Update rule
        upd = lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        # Gradient Clipping
        norm_upd = np.linalg.norm(upd)
        if norm_upd > self.clip_norm:
            upd *= self.clip_norm / norm_upd

        # 4. Final step: Combined EMA + Greedy Momentum
        x_new = x - upd + (self.greedy_w * lr * best_dir)
        
        return x_new, 2 * self.k
