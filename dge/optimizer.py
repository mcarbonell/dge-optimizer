"""
dge/optimizer.py
================
Denoised Gradient Estimation (DGE) Optimizer — Canonical Implementation v2.

Based on experimental findings from v1–v28. Core algorithm:
  1. Split parameters into k random non-overlapping blocks per step.
  2. Evaluate loss with +/- perturbation for each block (2k evaluations/step).
  3. Estimate per-coordinate gradient from block responses.
  4. Accumulate with Adam EMA (temporal denoising).
  5. Scale update by Direction-Consistency mask (v27-v28).

Key improvements over v1 (v25b + v27 findings):
  - Dynamic permutation per step (not fixed overlapping groups).
  - Direction-Consistency LR: per-parameter confidence mask based on
    sign consistency over last T steps. Default T=20.
    Improves MNIST accuracy 80% -> 87.56% at same evaluation budget.
  - Removed: greedy step, clip_norm, dense_update flag (all abandoned
    in v14+ experiments as they added complexity without benefit).

Reference experiments:
  - Architecture: scratch/dge_consistency_mnist_v28.py
  - Synthetic benchmarks: scratch/dge_consistency_lr_v27.py
  - Findings: docs/dge_findings_v28_consistency_mnist.md
"""

import math
from collections import deque

import numpy as np


class DGEOptimizer:
    """
    Denoised Gradient Estimation (DGE) with Direction-Consistency LR.

    A zeroth-order optimizer that trains models using only function evaluations
    (no gradients, no backpropagation). Uses 2*k_blocks evaluations per step.

    Parameters
    ----------
    dim : int
        Total number of parameters to optimize.
    k_blocks : int or None
        Number of random blocks per step. Each block uses 2 evaluations.
        If None, defaults to max(1, ceil(log2(dim))).
        Typical good values: 8 for D=128, 16 for D=1000, 1024+ for NNs.
    lr : float
        Base learning rate. Cosine-annealed to lr*lr_decay over training.
    delta : float
        Perturbation magnitude for finite differences.
        Cosine-annealed to delta*delta_decay over training.
    beta1 : float
        Adam first moment decay (EMA of gradient). Default 0.9.
    beta2 : float
        Adam second moment decay (EMA of gradient^2). Default 0.999.
    eps : float
        Adam epsilon for numerical stability. Default 1e-8.
    total_steps : int
        Total number of .step() calls expected. Used for cosine schedule.
    lr_decay : float
        Minimum fraction of lr at end of cosine schedule. Default 0.01.
    delta_decay : float
        Minimum fraction of delta at end of cosine schedule. Default 0.1.
    consistency_window : int
        Number of past steps to track sign consistency per parameter.
        0 = disabled (pure DGE without consistency mask).
        20 = default (recommended, validated in v27-v28).
        The consistency mask scales the effective LR per parameter by
        |mean(sign(grad_i) over last T steps)|, suppressing noisy dimensions.
    seed : int or None
        Random seed for reproducibility.

    Usage
    -----
    >>> opt = DGEOptimizer(dim=len(params), k_blocks=8, lr=0.1, total_steps=5000)
    >>> evals = 0
    >>> while evals < budget:
    ...     x, n = opt.step(loss_fn, x)
    ...     evals += n

    The loss function ``f`` must accept a 1-D array of shape (dim,) and return
    a scalar. For stochastic problems (minibatches), use the same batch for
    both forward evaluations within a single step call.
    """

    def __init__(
        self,
        dim: int,
        k_blocks: int | None = None,
        lr: float = 0.1,
        delta: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        total_steps: int = 10_000,
        lr_decay: float = 0.01,
        delta_decay: float = 0.1,
        consistency_window: int = 20,
        seed: int | None = None,
    ):
        self.dim = dim
        self.k = k_blocks if k_blocks is not None else max(1, math.ceil(math.log2(dim)))
        self.lr0 = lr
        self.delta0 = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.total_steps = total_steps
        self.lr_decay = lr_decay
        self.delta_decay = delta_decay
        self.consistency_window = consistency_window
        self.rng = np.random.default_rng(seed)

        # Adam state
        self.m = np.zeros(dim, dtype=np.float64)
        self.v = np.zeros(dim, dtype=np.float64)
        self.t = 0

        # Direction-Consistency buffer: circular buffer of sign arrays
        self._sign_buffer: deque[np.ndarray] = deque(maxlen=consistency_window) if consistency_window > 0 else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine(self, v0: float, decay: float) -> float:
        """Cosine annealing: v0 at t=0, v0*decay at t=total_steps."""
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def _consistency_mask(self) -> np.ndarray:
        """
        Returns per-parameter LR scale in [0, 1] based on sign consistency.

        consistency[i] = |mean(sign(grad[i]) over last T steps)|
          ~= 1  : gradient direction reliable  -> full LR
          ~= 0  : gradient sign oscillates     -> update suppressed
        """
        if self._sign_buffer is None or len(self._sign_buffer) < 2:
            return np.ones(self.dim, dtype=np.float64)
        stacked = np.stack(list(self._sign_buffer), axis=0)   # (T, dim)
        return np.abs(stacked.mean(axis=0))                    # (dim,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, f, x: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Execute one optimization step.

        Parameters
        ----------
        f : Callable[[np.ndarray], float]
            Objective function to minimize. Must return a scalar.
            For stochastic settings, the same data batch must be used
            for both internal evaluations within this call.
        x : np.ndarray, shape (dim,)
            Current parameter vector. Not modified in-place.

        Returns
        -------
        x_new : np.ndarray
            Updated parameter vector.
        n_evals : int
            Number of function evaluations used (always 2 * k_blocks).
        """
        self.t += 1
        lr    = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)

        grad = np.zeros(self.dim, dtype=np.float64)

        # Dynamic permutation: non-overlapping random blocks, new each step
        perm   = self.rng.permutation(self.dim)
        blocks = np.array_split(perm, self.k)

        for block in blocks:
            if len(block) == 0:
                continue
            signs = self.rng.choice([-1.0, 1.0], size=len(block))
            pert  = np.zeros(self.dim, dtype=np.float64)
            pert[block] = signs * delta

            fp = f(x + pert)
            fm = f(x - pert)

            grad[block] = (fp - fm) / (2.0 * delta) * signs

        # Adam EMA (temporal denoising)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)

        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)

        # Direction-Consistency mask
        if self._sign_buffer is not None:
            self._sign_buffer.append(np.sign(grad))
        mask = self._consistency_mask()

        upd = lr * mask * mh / (np.sqrt(vh) + self.eps)

        return x - upd, 2 * self.k

    def reset(self, seed: int | None = None) -> None:
        """Reset optimizer state (Adam moments, step counter, sign buffer)."""
        self.m[:] = 0.0
        self.v[:] = 0.0
        self.t = 0
        if self._sign_buffer is not None:
            self._sign_buffer.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    @property
    def evals_per_step(self) -> int:
        """Number of function evaluations per .step() call."""
        return 2 * self.k

    def __repr__(self) -> str:
        cw = self.consistency_window if self._sign_buffer is not None else 0
        return (
            f"DGEOptimizer(dim={self.dim}, k_blocks={self.k}, lr={self.lr0}, "
            f"delta={self.delta0}, consistency_window={cw}, "
            f"total_steps={self.total_steps})"
        )
