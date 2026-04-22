"""
dge/torch_optimizer.py
======================
Denoised Gradient Estimation (DGE) Optimizer — PyTorch Batched Native (v3).

This implementation shifts the optimization loop completely to PyTorch and
generates the perturbations using a fully vectorized (loop-free) strategy via
`scatter_`. 

The objective function `f_batched` MUST accept a 2D batch of parameters
of shape (2k, dim) and return a 1D tensor of losses of shape (2k,).
This allows the underlying ML framework to evaluate all blocks concurrently,
massively reducing the Python overhead and fully utilizing GPU bandwidth.
"""

import math
from collections import deque
import torch

class TorchDGEOptimizer:
    """
    Batched Native PyTorch implementation of DGE (v3).
    
    Parameters
    ----------
    dim : int
        Total number of parameters to optimize.
    k_blocks : int or None
        Number of random blocks per step. Each block uses 2 evaluations.
        If None, defaults to max(1, ceil(log2(dim))).
    lr : float
        Base learning rate. Cosine-annealed to lr*lr_decay over training.
    delta : float
        Perturbation magnitude for finite differences.
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
        0 = disabled (default). 20 = Recommended when lr <= 0.1.
    seed : int or None
        Random seed for reproducibility.
    device : str or torch.device
        The device where state tensors reside ('cpu', 'cuda', 'privateuseone:0').
    clip_norm : float or None
        If provided, clips the final update vector norm to this value.
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
        consistency_window: int = 0,
        seed: int | None = None,
        device: torch.device | str = "cpu",
        clip_norm: float | None = None,
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
        self.clip_norm = clip_norm
        
        self.device = torch.device(device) if isinstance(device, str) else device
        
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)
            
        self.m = torch.zeros(dim, device=self.device)
        self.v = torch.zeros(dim, device=self.device)
        self.t = 0
        
        self._sign_buffer = deque(maxlen=consistency_window) if consistency_window > 0 else None
        
        # Precompute padding logic since the dimension doesn't change
        self.group_size = (self.dim + self.k - 1) // self.k
        self.pad = self.group_size * self.k - self.dim

    def _cosine(self, v0: float, decay: float) -> float:
        """Cosine annealing schedule."""
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def step(self, f_batched, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Execute one heavily vectorized optimization step.

        Parameters
        ----------
        f_batched : Callable[[torch.Tensor], torch.Tensor]
            Objective function that accepts a parameter batch of shape (2k, dim)
            and returns a 1D tensor of losses of shape (2k,).
        x : torch.Tensor, shape (dim,)
            Current parameter vector (must be on self.device).

        Returns
        -------
        x_new : torch.Tensor
            Updated parameter vector.
        n_evals : int
            Number of function evaluations used (always 2 * k).
        """
        self.t += 1
        lr = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)
        
        # 1. Generate permutation and signs
        perm = torch.randperm(self.dim, generator=self.rng, device=self.device)
        signs = torch.randint(0, 2, (self.dim,), generator=self.rng, device=self.device).float() * 2 - 1
        
        # Pad to make exactly divisible by k
        if self.pad > 0:
            perm_pad = torch.cat([perm, torch.zeros(self.pad, dtype=torch.long, device=self.device)])
            signs_pad = torch.cat([signs, torch.zeros(self.pad, device=self.device)])
        else:
            perm_pad = perm
            signs_pad = signs
            
        perm_mat = perm_pad.view(self.k, self.group_size)
        signs_mat = signs_pad.view(self.k, self.group_size) * delta
        
        # 2. Build perturbation matrices using scatter_ for O(1) loop-free performance
        P_plus = torch.zeros((self.k, self.dim), device=self.device)
        P_plus.scatter_(1, perm_mat, signs_mat)
        
        if self.pad > 0:
            # Fix the 0th index which was overwritten by padding zeros
            P_plus[:, 0] = 0.0
            idx0_mask = (perm == 0)
            idx0_pos = idx0_mask.nonzero(as_tuple=True)[0]
            if len(idx0_pos) > 0:
                block0 = idx0_pos[0] // self.group_size
                P_plus[block0, 0] = signs[idx0_pos[0]] * delta
                
        P = torch.empty((2 * self.k, self.dim), device=self.device)
        P[0::2] = P_plus
        P[1::2] = -P_plus
        
        # 3. Batched Forward Pass
        # x is (dim,). x.unsqueeze(0) + P leverages broadcasting to make (2k, dim)
        losses = f_batched(x.unsqueeze(0) + P)
        
        # 4. Extract Gradients
        diffs = (losses[0::2] - losses[1::2]) / (2.0 * delta)
        
        grad = torch.zeros(self.dim, device=self.device)
        diffs_exp = diffs.unsqueeze(1).expand(self.k, self.group_size).flatten()
        if self.pad > 0:
            diffs_exp = diffs_exp[:self.dim]
            
        grad[perm] = diffs_exp * signs
        
        # 5. Temporal Denoising (Adam EMA)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        
        # 6. Direction-Consistency Mask
        mask = 1.0
        if self._sign_buffer is not None:
            self._sign_buffer.append(torch.sign(grad))
            if len(self._sign_buffer) >= 2:
                mask = torch.stack(list(self._sign_buffer)).mean(0).abs()
        
        upd = lr * mask * mh / (torch.sqrt(vh) + self.eps)
        
        if self.clip_norm is not None:
            un = torch.norm(upd)
            if un > self.clip_norm:
                upd *= self.clip_norm / un
                
        return x - upd, 2 * self.k
