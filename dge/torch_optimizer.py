"""
dge/torch_optimizer.py
======================
Denoised Gradient Estimation (DGE) Optimizer — PyTorch Batched Native (v3).

This implementation shifts the optimization loop completely to PyTorch and
generates the perturbations using a fully vectorized (loop-free) strategy via
`scatter_`. It natively supports the "Scaling Head" (multi-layer block sizes) 
approach to ensure deep networks converge correctly.

The objective function `f_batched` MUST accept a 2D batch of parameters
of shape (2k, dim) and return a 1D tensor of losses of shape (2k,).
"""

import math
from collections import deque
import torch

class TorchDGEOptimizer:
    """
    Batched Native PyTorch implementation of DGE (v3) with Scaling Head support.
    
    Parameters
    ----------
    dim : int
        Total number of parameters to optimize.
    k_blocks : int or list[int] or None
        Number of random blocks per step. 
        If a list, it must match the length of `layer_sizes` for multi-layer scaling.
    layer_sizes : list[int] or None
        Sizes of the distinct parameter groups (layers). If provided, `k_blocks`
        must be a list of the same length. If None, treats `dim` as a single layer.
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
    chunk_size : int or None
        If provided, splits the forward pass batch into chunks of this size to prevent OOM.
    """
    def __init__(
        self,
        dim: int,
        k_blocks: int | list[int] | None = None,
        layer_sizes: list[int] | None = None,
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
        chunk_size: int | None = None,
    ):
        self.dim = dim
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
        self.chunk_size = chunk_size
        
        self.device = torch.device(device) if isinstance(device, str) else device
        
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)
            
        self.m = torch.zeros(dim, device=self.device)
        self.v = torch.zeros(dim, device=self.device)
        self.t = 0
        
        self._sign_buffer = deque(maxlen=consistency_window) if consistency_window > 0 else None
        
        # Setup multi-layer architecture
        if layer_sizes is None:
            self.layer_sizes = [dim]
            if k_blocks is None:
                self.k_blocks = [max(1, math.ceil(math.log2(dim)))]
            elif isinstance(k_blocks, int):
                self.k_blocks = [k_blocks]
            else:
                self.k_blocks = k_blocks
        else:
            self.layer_sizes = layer_sizes
            self.k_blocks = k_blocks if isinstance(k_blocks, list) else [k_blocks] * len(layer_sizes)
            
        assert len(self.layer_sizes) == len(self.k_blocks), "k_blocks length must match layer_sizes"
        assert sum(self.layer_sizes) == self.dim, "sum of layer_sizes must equal dim"
        
        self.total_k = sum(self.k_blocks)
        
        # Precompute padding logic per layer
        self.group_sizes = []
        self.pads = []
        for sz, k in zip(self.layer_sizes, self.k_blocks):
            grp = (sz + k - 1) // k
            pad = grp * k - sz
            self.group_sizes.append(grp)
            self.pads.append(pad)

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
            Objective function that accepts a parameter batch of shape (P, dim)
            and returns a 1D tensor of losses of shape (P,).
        x : torch.Tensor, shape (dim,)
            Current parameter vector (must be on self.device).

        Returns
        -------
        x_new : torch.Tensor
            Updated parameter vector.
        n_evals : int
            Number of function evaluations used (always 2 * total_k).
        """
        self.t += 1
        lr = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)
        
        P_plus = torch.zeros((self.total_k, self.dim), device=self.device)
        
        offset = 0
        row_offset = 0
        
        perms = []
        signss = []
        
        # 1. Build Multi-Layer Perturbations Matrix
        for sz, k, pad, grp in zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes):
            perm = torch.randperm(sz, generator=self.rng, device=self.device)
            signs = torch.randint(0, 2, (sz,), generator=self.rng, device=self.device).float() * 2 - 1
            
            perms.append(perm)
            signss.append(signs)
            
            if pad > 0:
                perm_pad = torch.cat([perm, torch.zeros(pad, dtype=torch.long, device=self.device)])
                signs_pad = torch.cat([signs, torch.zeros(pad, device=self.device)])
            else:
                perm_pad = perm
                signs_pad = signs
                
            # Shift permutation indices by the layer's offset in the flat parameter vector
            perm_mat = perm_pad.view(k, grp) + offset
            signs_mat = signs_pad.view(k, grp) * delta
            
            target_slice = P_plus[row_offset : row_offset + k, :]
            target_slice.scatter_(1, perm_mat, signs_mat)
            
            if pad > 0:
                target_slice[:, offset] = 0.0
                idx0_mask = (perm == 0)
                idx0_pos = idx0_mask.nonzero(as_tuple=True)[0]
                if len(idx0_pos) > 0:
                    block0 = idx0_pos[0] // grp
                    target_slice[block0, offset] = signs[idx0_pos[0]] * delta
                    
            offset += sz
            row_offset += k
                
        P = torch.empty((2 * self.total_k, self.dim), device=self.device)
        P[0::2] = P_plus
        P[1::2] = -P_plus
        
        # 2. Batched Forward Pass (with optional Chunking to avoid OOM)
        X_batch = x.unsqueeze(0) + P
        if self.chunk_size is not None and X_batch.shape[0] > self.chunk_size:
            losses_list = []
            for i in range(0, X_batch.shape[0], self.chunk_size):
                losses_list.append(f_batched(X_batch[i : i + self.chunk_size]))
            losses = torch.cat(losses_list, dim=0)
        else:
            losses = f_batched(X_batch)
        
        # 3. Extract Gradients Layer by Layer
        diffs = (losses[0::2] - losses[1::2]) / (2.0 * delta)
        
        grad = torch.zeros(self.dim, device=self.device)
        offset = 0
        row_offset = 0
        
        for i, (sz, k, pad, grp) in enumerate(zip(self.layer_sizes, self.k_blocks, self.pads, self.group_sizes)):
            perm = perms[i]
            signs = signss[i]
            
            layer_diffs = diffs[row_offset : row_offset + k]
            diffs_exp = layer_diffs.unsqueeze(1).expand(k, grp).flatten()
            
            if pad > 0:
                diffs_exp = diffs_exp[:sz]
                
            grad[offset + perm] = diffs_exp * signs
            
            offset += sz
            row_offset += k
        
        # 4. Temporal Denoising (Adam EMA)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        
        mh = self.m / (1.0 - self.beta1 ** self.t)
        vh = self.v / (1.0 - self.beta2 ** self.t)
        
        # 5. Direction-Consistency Mask
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
                
        return x - upd, 2 * self.total_k
