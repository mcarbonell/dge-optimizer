import numpy as np
import math

class SPSAOptimizer:
    def __init__(self, dim, lr=1.0, delta=1e-3, lr_decay=0.01, delta_decay=0.05, total_steps=1000, seed=None):
        self.dim = dim
        self.lr0 = lr
        self.delta0 = delta
        self.lr_decay = lr_decay
        self.delta_decay = delta_decay
        self.total_steps = total_steps
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def _cosine(self, v0, decay):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)

        signs = self.rng.choice([-1.0, 1.0], size=self.dim)
        pert = signs * delta

        fp = f(x + pert)
        fm = f(x - pert)

        g = (fp - fm) / (2.0 * delta) * signs
        x_new = x - lr * g
        
        return x_new, 2

class RandomDirectionOptimizer:
    def __init__(self, dim, lr=1.0, lr_decay=0.01, total_steps=1000, seed=None):
        self.dim = dim
        self.lr0 = lr
        self.lr_decay = lr_decay
        self.total_steps = total_steps
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def _cosine(self, v0, decay):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr = self._cosine(self.lr0, self.lr_decay)
        
        d = self.rng.standard_normal(self.dim)
        dn = np.linalg.norm(d)
        if dn > 0:
            d = d / dn

        fp = f(x + lr * d)
        fm = f(x - lr * d)
        
        if fp < fm:
            x_new = x + lr * d
            evals = 2
        else:
            x_new = x - lr * d
            evals = 2
            
        return x_new, evals


class MeZOptimizer:
    """
    MeZO: Memory-Efficient Zeroth-Order Optimizer (Malladi et al., 2023).
    
    Perturbs parameters globally using standard Gaussian noise z ~ N(0, I),
    estimates the projected gradient along z via symmetric finite differences,
    and updates parameters using SGD (with optional momentum and cosine decay).
    """
    def __init__(
        self,
        dim: int,
        lr: float = 1e-3,
        delta: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        lr_decay: float = 0.01,
        delta_decay: float = 0.1,
        total_steps: int = 1000,
        seed: int | None = None,
    ):
        self.dim = dim
        self.lr0 = lr
        self.delta0 = delta
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.delta_decay = delta_decay
        self.total_steps = total_steps
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self.v = np.zeros(dim, dtype=np.float64) if momentum > 0 else None

    def _cosine(self, v0: float, decay: float) -> float:
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)

        # Standard normal Gaussian perturbation z ~ N(0, I)
        z = self.rng.standard_normal(self.dim)
        pert = z * delta

        fp = f(x + pert)
        fm = f(x - pert)

        # Projected gradient estimate along z: (f(x+pert) - f(x-pert)) / (2*delta) * z
        g = ((fp - fm) / (2.0 * delta)) * z

        if self.weight_decay > 0:
            g = g + self.weight_decay * x

        if self.momentum > 0 and self.v is not None:
            self.v = self.momentum * self.v + g
            update = self.v
        else:
            update = g

        x_new = x - lr * update
        return x_new, 2
