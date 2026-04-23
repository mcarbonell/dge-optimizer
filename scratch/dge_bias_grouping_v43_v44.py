"""
dge_bias_grouping_v43_v44.py
=================================
Experiment to test Bias Grouping strategies.
v43: Fan-In blocks for weights + 1 Global Bias block.
v44: Fan-In blocks for weights + 3 Per-Layer Bias blocks.

Budget: 500,000 evals
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dge.torch_optimizer import TorchDGEOptimizer

try:
    from torchvision import datasets, transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False

try:
    import torch_directml
    device = torch_directml.device()
    print(f"DirectML: {device}")
except ImportError:
    device = torch.device("cpu")
    print("CPU")

# ---------------------------------------------------------------------------
# Models & Metrics
# ---------------------------------------------------------------------------
class LayerOrderedMLP:
    """
    MLP where the parameter vector is ordered as:
    [W1_layer, b1_layer, W2_layer, b2_layer, W3_layer, b3_layer]
    This matches the standard layout but we explicitly separate them to define 
    different k_blocks for W and b.
    """
    def __init__(self, arch):
        self.arch = list(arch)
        self.layer_sizes = []
        for a, b in zip(arch[:-1], arch[1:]):
            self.layer_sizes.append(a * b) # Weights
            self.layer_sizes.append(b)     # Bias
        self.dim = sum(self.layer_sizes)

    def forward(self, X, params_batch):
        P = params_batch.shape[0]
        h = X.unsqueeze(0).expand(P, -1, -1)
        
        offset = 0
        for i, (l_in, l_out) in enumerate(zip(self.arch[:-1], self.arch[1:])):
            W_sz = l_in * l_out
            b_sz = l_out
            
            W = params_batch[:, offset : offset + W_sz].view(P, l_in, l_out)
            offset += W_sz
            b = params_batch[:, offset : offset + b_sz].view(P, 1, l_out)
            offset += b_sz
            
            h = torch.bmm(h, W) + b
            if l_out != self.arch[-1]:
                h = torch.relu(h)
        return h

def batched_ce(logits, targets):
    P, B, C = logits.shape
    t = targets.unsqueeze(0).expand(P, -1)
    return F.cross_entropy(logits.reshape(P*B, C), t.reshape(-1),
                           reduction='none').view(P, B).mean(dim=1)

def zo_acc(model, X, y, params, chunk=1000):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), chunk):
            lo = model.forward(X[i:i+chunk], params.unsqueeze(0)).squeeze(0)
            correct += (lo.argmax(1) == y[i:i+chunk]).sum().item()
    return correct / len(y)

def load_full_mnist():
    assert HAS_TV
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = ((ds_tr.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    X_te = ((ds_te.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    return X_tr, ds_tr.targets, X_te, ds_te.targets

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------
def run_experiment(name, model, params0, X_tr, y_tr, X_te, y_te, seed, layer_sizes, k_blocks):
    total_k = sum(k_blocks)
    total_steps = BUDGET_DGE // (2 * total_k)
    
    opt = TorchDGEOptimizer(
        dim=model.dim, 
        layer_sizes=layer_sizes, 
        k_blocks=k_blocks,
        lr=LR_ZO, 
        delta=DELTA, 
        total_steps=total_steps,
        consistency_window=20, 
        seed=seed, 
        device=device, 
        chunk_size=128
    )

    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)

    evals = 0
    best_test = 0.0
    curve_evals, curve_acc = [], []
    next_log = LOG_INTERVAL
    
    t0 = time.time()
    print(f"\n  [{name}] evals/step = {2 * total_k}")
    print(f"  {'evals':>10}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}")
    print(f"  {'-'*45}")

    while evals < BUDGET_DGE:
        idx = torch.randperm(60000, generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr[idx], y_tr[idx]

        def f_batched(p_batch):
            with torch.no_grad():
                logits = model.forward(Xb, p_batch)
                return batched_ce(logits, yb)
        
        params, n = opt.step(f_batched, params)
        evals += n

        if evals >= next_log or evals >= BUDGET_DGE:
            te_acc  = zo_acc(model, X_te, y_te, params)
            best_test = max(best_test, te_acc)
            curve_evals.append(evals)
            curve_acc.append(round(te_acc, 4))
            elapsed = time.time() - t0
            print(f"  {evals:>10,}  {te_acc:>8.2%}  {best_test:>8.2%}  {elapsed:>6.0f}s")
            next_log += LOG_INTERVAL

    return {"best_test_acc": best_test}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARCH          = (784, 128, 64, 10)
BUDGET_DGE    = 500_000
LR_ZO         = 0.05
DELTA         = 1e-3
BATCH_SIZE    = 256
LOG_INTERVAL  = 25_000

if __name__ == "__main__":
    X_tr_all, y_tr_all, X_te_all, y_te_all = load_full_mnist()
    X_tr_d = X_tr_all.to(device)
    y_tr_d = y_tr_all.to(device)
    X_te_d = X_te_all.to(device)
    y_te_d = y_te_all.to(device)

    model = LayerOrderedMLP(ARCH)
    
    # 1. v43: Weights partitioned into 128/64/10 blocks, ALL Biases as 1 block
    # We need to reorder to have [W1, W2, W3, b1+b2+b3] for v43?
    # Actually, we can use 6 layers and set k_blocks=1 for bias layers.
    # v44: [W1(128k), b1(1k), W2(64k), b2(1k), W3(10k), b3(1k)]
    
    # v44 Configuration (Per-Layer Bias Blocks)
    # k_blocks for Weights: 128, 64, 10
    # k_blocks for Biases: 1, 1, 1
    v44_layer_sizes = model.layer_sizes
    v44_k_blocks = [128, 1, 64, 1, 10, 1]
    
    # v43 Configuration (Global Bias Block)
    # To do this exactly, we'd need to reorder params.
    # But v44 is already a very strong "denoised bias" test.
    # Let's add a "v43-like" where we group all biases at the end.
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v43/v44: Bias Denoising")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print(f"{'='*70}")

    seed = 42
    torch.manual_seed(seed)
    params0 = torch.zeros(model.dim, device=device)
    
    # Init Weights (He) and Biases (0)
    offset = 0
    for l_in, l_out in zip(ARCH[:-1], ARCH[1:]):
        std = math.sqrt(2.0 / l_in)
        # Weights
        w_sz = l_in * l_out
        params0[offset : offset + w_sz] = torch.randn(w_sz, device=device) * std
        offset += w_sz + l_out # Biases stay at 0

    # Run v44 (Per-Layer Bias Blocks)
    res_v44 = run_experiment("v44_PerLayer_Bias_Blocks", model, params0, X_tr_d, y_tr_d, X_te_d, y_te_d, seed, v44_layer_sizes, v44_k_blocks)

    # Note: v43 is just v44 but with even fewer blocks.
    # Let's try v43 by grouping ALL biases into one layer at the end.
    
    # For v43 we need a different layout: [W1, W2, W3, B1, B2, B3]
    class V43MLP(LayerOrderedMLP):
        def __init__(self, arch):
            super().__init__(arch)
            self.w_sizes = [a*b for a, b in zip(arch[:-1], arch[1:])]
            self.b_sizes = [b for b in arch[1:]]
            self.dim_w = sum(self.w_sizes)
            self.dim_b = sum(self.b_sizes)
        
        def forward(self, X, params_batch):
            P = params_batch.shape[0]
            h = X.unsqueeze(0).expand(P, -1, -1)
            w_off = 0
            b_off = self.dim_w
            for i, (l_in, l_out) in enumerate(zip(self.arch[:-1], self.arch[1:])):
                W = params_batch[:, w_off : w_off + self.w_sizes[i]].view(P, l_in, l_out)
                b = params_batch[:, b_off : b_off + self.b_sizes[i]].view(P, 1, l_out)
                h = torch.bmm(h, W) + b
                if l_out != self.arch[-1]: h = torch.relu(h)
                w_off += self.w_sizes[i]
                b_off += self.b_sizes[i]
            return h

    model_v43 = V43MLP(ARCH)
    params0_v43 = torch.zeros(model_v43.dim, device=device)
    # Copy init from params0
    # ... logic to copy ...
    # Simplified: just re-init with same seed
    torch.manual_seed(seed)
    w_off = 0
    for l_in, l_out in zip(ARCH[:-1], ARCH[1:]):
        std = math.sqrt(2.0 / l_in)
        params0_v43[w_off : w_off + l_in*l_out] = torch.randn(l_in*l_out, device=device) * std
        w_off += l_in*l_out
    
    v43_layer_sizes = [100352, 8192, 640, 202]
    v43_k_blocks = [128, 64, 10, 1] # 1 block for ALL biases
    
    res_v43 = run_experiment("v43_Global_Bias_Block", model_v43, params0_v43, X_tr_d, y_tr_d, X_te_d, y_te_d, seed, v43_layer_sizes, v43_k_blocks)

    out_path = Path(__file__).parent.parent / "results" / "raw" / "v43_v44_bias_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"v43": res_v43["best_test_acc"], "v44": res_v44["best_test_acc"]}, f, indent=2)
