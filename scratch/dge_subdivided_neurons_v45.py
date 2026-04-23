"""
dge_subdivided_neurons_v45.py
=================================
Experiment: Sub-divided Neuron Blocks.
We take the "Neuron-Aggregated" idea but split each neuron's fan-in into 
smaller sub-blocks to reduce noise while maintaining structural alignment.

Layer 1: 128 neurons. Split each into 4 blocks -> 512 blocks (size ~196).
Layer 2: 64 neurons. 1 block each -> 64 blocks (size 129).
Layer 3: 10 neurons. 1 block each -> 10 blocks (size 65).

Total K = 512 + 64 + 10 = 586 blocks.
Budget: 500,000 evals.
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
class NeuronOrderedMLP:
    def __init__(self, arch):
        self.arch = list(arch)
        self.layer_block_sizes = []
        self.layer_param_counts = []
        for a, b in zip(arch[:-1], arch[1:]):
            block_sz = a + 1
            self.layer_block_sizes.append(block_sz)
            self.layer_param_counts.append(block_sz * b)
        self.dim = sum(self.layer_param_counts)

    def forward(self, X, params_batch):
        P = params_batch.shape[0]
        h = X.unsqueeze(0).expand(P, -1, -1)
        offset = 0
        for i, (l_in, l_out) in enumerate(zip(self.arch[:-1], self.arch[1:])):
            sz = self.layer_param_counts[i]
            block_sz = self.layer_block_sizes[i]
            layer_p = params_batch[:, offset : offset + sz].view(P, l_out, block_sz)
            W = layer_p[:, :, :l_in].transpose(1, 2) 
            b = layer_p[:, :, l_in:].view(P, 1, l_out)
            h = torch.bmm(h, W) + b
            if l_out != self.arch[-1]: h = torch.relu(h)
            offset += sz
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

    model = NeuronOrderedMLP(ARCH)
    
    # Layer 1: 128 neurons * 785 params. Split each into 4 blocks -> 512 blocks.
    # Layer 2: 64 neurons * 129 params. 1 block each -> 64 blocks.
    # Layer 3: 10 neurons * 65 params. 1 block each -> 10 blocks.
    layer_sizes = model.layer_param_counts
    k_blocks = [512, 64, 10]

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v45: Sub-divided Neuron Blocks")
    print(f"BUDGET: {BUDGET_DGE:,} evals")
    print(f"K Total = {sum(k_blocks)}")
    print(f"Block size L1 ~ {layer_sizes[0]//k_blocks[0]}")
    print(f"{'='*70}")

    seed = 42
    torch.manual_seed(seed)
    params0 = torch.zeros(model.dim, device=device)
    
    offset = 0
    for l_in, l_out in zip(ARCH[:-1], ARCH[1:]):
        std = math.sqrt(2.0 / l_in)
        sz = l_out * (l_in + 1)
        layer_p = torch.zeros((l_out, l_in + 1), device=device)
        layer_p[:, :l_in] = torch.randn((l_out, l_in), device=device) * std
        params0[offset : offset + sz] = layer_p.flatten()
        offset += sz

    res = run_experiment("Subdivided_Neurons", model, params0, X_tr_d, y_tr_d, X_te_d, y_te_d, seed, layer_sizes, k_blocks)

    out_path = Path(__file__).parent.parent / "results" / "raw" / "v45_subdivided_neurons.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"experiment": "v45_subdivided_neurons", "seed": seed, "acc": res["best_test_acc"]}, f, indent=2)
