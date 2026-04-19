import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys
import os

# =============================================================================
# OPTIMIZADOR DGE v14 (Orthogonal + SNR Tracking)
# =============================================================================
class DGEOptimizerV14:
    """
    DGE v14: Orthogonal Blocks + Adam EMA + SNR Tracking.
    Fixes the cross-variable noise by ensuring each parameter is evaluated 
    exactly once per step in non-overlapping blocks.
    """
    def __init__(self, dim, lr=1.0, delta=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, lr_decay=0.01, total_steps=1000, clip_norm=1.0, seed=None):
        self.dim = dim
        self.lr0 = lr
        self.delta = delta
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.lr_decay = lr_decay
        self.total_steps = total_steps
        self.clip_norm = clip_norm
        self.rng = np.random.default_rng(seed)
        
        # k determines the number of orthogonal blocks
        self.k = max(1, math.ceil(math.log2(dim))) if dim > 1 else 1
        
        # Adam state
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0
        
        # SNR Tracking
        self.history_corr = []

    def _cosine(self, v0, decay):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr = self._cosine(self.lr0, self.lr_decay) # No more lr_scale penaly

        # 1. ORTHOGONAL PARTITIONING (Fixes the overlapping bug)
        perm = self.rng.permutation(self.dim)
        groups = np.array_split(perm, self.k)
        
        signs = self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
        g_step = np.zeros(self.dim, dtype=np.float32)

        # 2. Block evaluation
        for idx in groups:
            pert = np.zeros(self.dim, dtype=np.float32)
            pert[idx] = signs[idx] * self.delta
            
            fp = f(x + pert)
            fm = f(x - pert)
            
            sg = (fp - fm) / (2.0 * self.delta)
            
            # Map the shared block-scalar back to individual coordinates
            g_step[idx] = sg * signs[idx]

        # 3. Adam Update (Each coordinate updated exactly once per step)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_step
        self.v = self.beta2 * self.v + (1 - self.beta2) * g_step ** 2

        mh = self.m / (1 - self.beta1 ** self.t + 1e-30)
        vh = self.v / (1 - self.beta2 ** self.t + 1e-30)

        upd = lr * mh / (np.sqrt(vh) + self.eps)
        
        # Stability: Clipping
        un = np.linalg.norm(upd)
        if un > self.clip_norm:
            upd *= self.clip_norm / un

        # 4. SNR Tracking: Correlation between noisy step-gradient and EMA
        if self.t % 10 == 0:
            # Correlation on the non-zero updates
            corr = np.corrcoef(g_step, mh)[0, 1]
            self.history_corr.append(corr)

        return x - upd, 2 * self.k

# =============================================================================
# MNIST SETUP (Standardized)
# =============================================================================
SEED         = 42
N_TRAIN      = 3_000     
N_TEST       = 600       
BATCH_SIZE   = 256       
TOTAL_EVALS  = 100_000   
ARCH         = (784, 32, 10) 

def load_mnist_subset(n_train=N_TRAIN, n_test=N_TEST):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    X_tr_all = (full_train.data.float().view(-1, 784) / 255.0 - 0.1307) / 0.3081
    X_te_all = (full_test.data.float().view(-1, 784) / 255.0 - 0.1307) / 0.3081
    y_tr_all, y_te_all = full_train.targets, full_test.targets
    rng = np.random.default_rng(SEED)
    tr_idx = rng.choice(len(y_tr_all), size=n_train, replace=False)
    te_idx = rng.choice(len(y_te_all), size=n_test,  replace=False)
    return X_tr_all[tr_idx].numpy(), y_tr_all[tr_idx].numpy(), X_te_all[te_idx].numpy(), y_te_all[te_idx].numpy()

def forward_np(X, params):
    i = 0
    h = X
    for layer_in, layer_out in zip(ARCH[:-1], ARCH[1:]):
        w_size = layer_in * layer_out
        W = params[i:i+w_size].reshape(layer_in, layer_out)
        i += w_size
        b = params[i:i+layer_out]
        i += layer_out
        h = h @ W + b
        if layer_out != ARCH[-1]: h = np.maximum(h, 0)
    return h

def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def loss_on_batch(Xb, yb, params):
    logits = forward_np(Xb, params)
    probs  = np.clip(softmax_np(logits), 1e-7, 1 - 1e-7)
    return float(-np.mean(np.log(probs[np.arange(len(yb)), yb])))

def full_accuracy(X, y, params):
    return float(np.mean(forward_np(X, params).argmax(axis=1) == y))

if __name__ == "__main__":
    print(f"DGE v14: ORTHOGONAL PARTITIONING TEST")
    X_train, y_train, X_test, y_test = load_mnist_subset()
    D = sum(ARCH[i] * ARCH[i+1] + ARCH[i+1] for i in range(len(ARCH) - 1))
    
    rng_init = np.random.default_rng(SEED)
    params = rng_init.normal(0, 0.1, D).astype(np.float32)
    
    k = math.ceil(math.log2(D))
    optimizer = DGEOptimizerV14(
        dim=D, lr=0.5, delta=1e-3, 
        total_steps=TOTAL_EVALS // (2 * k), 
        clip_norm=0.05, seed=SEED
    )
    
    evals = 0
    rng_mb = np.random.default_rng(SEED + 1)
    t0 = time.time()
    next_log = 10_000

    print(f"  Training D={D} parameters with k={k} ({2*k} evals/step)")
    
    while evals < TOTAL_EVALS:
        idx = rng_mb.integers(0, len(y_train), size=BATCH_SIZE)
        Xb, yb = X_train[idx], y_train[idx]
        f = lambda p: loss_on_batch(Xb, yb, p)

        params, n = optimizer.step(f, params)
        evals += n
        
        if evals >= next_log:
            acc = full_accuracy(X_test, y_test, params)
            # Latest correlation (SNR)
            snr = optimizer.history_corr[-1] if optimizer.history_corr else 0
            print(f"    Evals: {evals:>7} | Test Acc: {acc:.2%} | SNR Corr: {snr:.4f}")
            next_log += 10_000

    print(f"  FINAL DGE v14 ACC: {full_accuracy(X_test, y_test, params):.2%} in {time.time()-t0:.1f}s")
    
    if optimizer.history_corr:
        print(f"  SNR Evolution: {optimizer.history_corr[0]:.4f} -> {optimizer.history_corr[-1]:.4f}")
