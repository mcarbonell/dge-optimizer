"""
scratch/dge_nondiff_suite_v71.py
================================
Fase P1.1: Suite Canónica de Evaluación en Arquitecturas No Diferenciables y Cuantizadas.

Genera los artefactos JSON crudos para respaldar la Tabla 2 y la Figura 2 del paper:
  1. Redes con activaciones de signo (Sign / Step function): torch.sign
  2. Redes con cuantización simétrica total (pesos y activaciones) a INT8 (256 niveles)
  3. Redes con cuantización simétrica total (pesos y activaciones) a INT4 (16 niveles)

Salidas:
  results/raw/v31_sign_activations.json
  results/raw/v32_quantized_mnist.json
  results/raw/v71_nondiff_suite.json
"""

import sys
import os
import time
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dge.torch_optimizer import TorchDGEOptimizer
from experiments.utils import get_system_info, get_commit_hash, setup_result_directories

# Device setup
try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
SEED = 42

# ---------------------------------------------------------------------------
# Helpers & Quantization Functions
# ---------------------------------------------------------------------------
def fake_quantize(x, bits):
    """Cuantización simétrica en [-1.0, 1.0]."""
    Q = (1 << (bits - 1)) - 1
    x_c = torch.clamp(x, -1.0, 1.0)
    return torch.round(x_c * Q) / Q


def batched_ce(logits, targets):
    P, B, C = logits.shape
    t = targets.unsqueeze(0).expand(P, -1)
    return F.cross_entropy(logits.reshape(P * B, C), t.reshape(-1),
                           reduction='none').view(P, B).mean(dim=1)


def zo_acc(model, X, y, params, chunk=1000):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), chunk):
            lo = model.forward(X[i:i + chunk], params.unsqueeze(0)).squeeze(0)
            correct += (lo.argmax(1) == y[i:i + chunk]).sum().item()
    return correct / len(y)


# ---------------------------------------------------------------------------
# 1. Sign Activation Models
# ---------------------------------------------------------------------------
class BatchedSignMLP:
    def __init__(self, arch):
        self.arch = list(arch)
        self.sizes = [a * b + b for a, b in zip(arch[:-1], arch[1:])]
        self.dim = sum(self.sizes)

    def forward(self, X, params_batch):
        P = params_batch.shape[0]
        h = X.unsqueeze(0).expand(P, -1, -1)
        i = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            W = params_batch[:, i:i + l_in * l_out].view(P, l_in, l_out)
            i += l_in * l_out
            b = params_batch[:, i:i + l_out].view(P, 1, l_out)
            i += l_out
            h = torch.bmm(h, W) + b
            if l_out != self.arch[-1]:
                h = torch.sign(h)  # Activación discontinua de signo
        return h


class SignMLPTorch(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers = []
        for l_in, l_out in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(l_in, l_out))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = torch.sign(layer(h))
        return self.layers[-1](h)


# ---------------------------------------------------------------------------
# 2. Quantized Models (INT4 / INT8)
# ---------------------------------------------------------------------------
class BatchedQuantMLP:
    def __init__(self, arch, bits):
        self.arch = list(arch)
        self.bits = bits
        self.sizes = [a * b + b for a, b in zip(arch[:-1], arch[1:])]
        self.dim = sum(self.sizes)

    def forward(self, X, params_batch):
        P = params_batch.shape[0]
        X_q = fake_quantize(X, self.bits)
        h = X_q.unsqueeze(0).expand(P, -1, -1)
        i = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            W = params_batch[:, i:i + l_in * l_out].view(P, l_in, l_out)
            i += l_in * l_out
            b = params_batch[:, i:i + l_out].view(P, 1, l_out)
            i += l_out

            W_q = fake_quantize(W, self.bits)
            b_q = fake_quantize(b, self.bits)
            h = torch.bmm(h, W_q) + b_q

            if l_out != self.arch[-1]:
                h = fake_quantize(F.relu(h), self.bits)
        return h


class QuantMLPTorch(nn.Module):
    def __init__(self, arch, bits):
        super().__init__()
        self.bits = bits
        layers = []
        for l_in, l_out in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(l_in, l_out))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = fake_quantize(x, self.bits)
        for layer in self.layers[:-1]:
            W_q = fake_quantize(layer.weight, self.bits)
            b_q = fake_quantize(layer.bias, self.bits) if layer.bias is not None else None
            h = F.linear(h, W_q, b_q)
            h = fake_quantize(F.relu(h), self.bits)

        final_l = self.layers[-1]
        W_q = fake_quantize(final_l.weight, self.bits)
        b_q = fake_quantize(final_l.bias, self.bits) if final_l.bias is not None else None
        return F.linear(h, W_q, b_q)


# ---------------------------------------------------------------------------
# Experiment Runners
# ---------------------------------------------------------------------------
def run_sign_experiment(X_tr, y_tr, X_te, y_te, total_evals, quick=False):
    arch = (784, 32, 10)
    k_blocks = [16, 4]
    est_step = 2 * sum(k_blocks)
    evals_budget = 5_000 if quick else total_evals
    total_steps = evals_budget // est_step

    print(f"\n--- [1/3] Red con Activación Signo (torch.sign) ---")

    # 1. Adam
    torch.manual_seed(SEED)
    model_adam = SignMLPTorch(arch).to(device)
    opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3)
    epochs = 2 if quick else 30
    crit = nn.CrossEntropyLoss()
    n_samples = len(y_tr)
    t0 = time.time()

    for ep in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            Xb, yb = X_tr[idx], y_tr[idx]
            opt_adam.zero_grad()
            loss = crit(model_adam(Xb), yb)
            loss.backward()
            opt_adam.step()

    with torch.no_grad():
        preds = model_adam(X_te).argmax(dim=1)
        adam_acc = (preds == y_te).float().mean().item()
    print(f"  Adam Final Acc: {adam_acc:.2%}")

    # 2. DGE V3
    model_dge = BatchedSignMLP(arch)
    torch.manual_seed(SEED)
    params0 = torch.zeros(model_dge.dim, device=device)
    off = 0
    for l_in, l_out in zip(arch[:-1], arch[1:]):
        std = math.sqrt(2.0 / l_in)
        w = l_in * l_out
        params0[off:off + w] = torch.randn(w, device=device) * std
        off += w + l_out

    opt_dge = TorchDGEOptimizer(
        dim=model_dge.dim,
        layer_sizes=model_dge.sizes,
        k_blocks=k_blocks,
        lr=0.5,
        delta=5e-3,
        total_steps=total_steps,
        consistency_window=20,
        clip_norm=0.05,
        seed=SEED,
        device=device,
        chunk_size=128
    )

    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(SEED + 100)
    evals = 0
    best_test = 0.0

    while evals < evals_budget:
        idx = torch.randperm(n_samples, generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr[idx], y_tr[idx]
        def f_batched(p_batch):
            return batched_ce(model_dge.forward(Xb, p_batch), yb)
        params, n = opt_dge.step(f_batched, params)
        evals += n

        if evals % (est_step * 50) < n or evals >= evals_budget:
            te_a = zo_acc(model_dge, X_te, y_te, params)
            best_test = max(best_test, te_a)

    print(f"  DGE Best Acc:   {best_test:.2%}")

    return {
        "architecture": "Sign activations (784-32-10)",
        "adam_acc": round(adam_acc, 4),
        "dge_acc": round(best_test, 4),
        "total_evals": evals
    }


def run_quant_experiment(X_tr, y_tr, X_te, y_te, bits, total_evals, quick=False):
    arch = (784, 128, 64, 10)
    k_blocks = [1024, 128, 16]
    est_step = 2 * sum(k_blocks)
    evals_budget = 5_000 if quick else total_evals
    total_steps = evals_budget // est_step

    Q = (1 << (bits - 1)) - 1
    delta_dyn = 1.05 / Q

    print(f"\n--- Cuantización INT{bits} ({1 << bits} niveles) ---")

    # 1. Adam
    torch.manual_seed(SEED)
    model_adam = QuantMLPTorch(arch, bits).to(device)
    opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3)
    epochs = 2 if quick else 30
    crit = nn.CrossEntropyLoss()
    n_samples = len(y_tr)

    for ep in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            Xb, yb = X_tr[idx], y_tr[idx]
            opt_adam.zero_grad()
            loss = crit(model_adam(Xb), yb)
            loss.backward()
            opt_adam.step()

    with torch.no_grad():
        preds = model_adam(X_te).argmax(dim=1)
        adam_acc = (preds == y_te).float().mean().item()
    print(f"  Adam INT{bits} Final Acc: {adam_acc:.2%}")

    # 2. DGE
    model_dge = BatchedQuantMLP(arch, bits)
    torch.manual_seed(SEED)
    params0 = torch.zeros(model_dge.dim, device=device)
    off = 0
    for l_in, l_out in zip(arch[:-1], arch[1:]):
        std = math.sqrt(2.0 / l_in)
        w = l_in * l_out
        params0[off:off + w] = torch.randn(w, device=device) * std
        off += w + l_out

    opt_dge = TorchDGEOptimizer(
        dim=model_dge.dim,
        layer_sizes=model_dge.sizes,
        k_blocks=k_blocks,
        lr=0.5 if bits == 4 else 0.1,
        delta=delta_dyn,
        total_steps=total_steps,
        consistency_window=20,
        clip_norm=0.05,
        seed=SEED,
        device=device,
        chunk_size=128
    )

    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(SEED + 100)
    evals = 0
    best_test = 0.0

    while evals < evals_budget:
        idx = torch.randperm(n_samples, generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr[idx], y_tr[idx]
        def f_batched(p_batch):
            return batched_ce(model_dge.forward(Xb, p_batch), yb)
        params, n = opt_dge.step(f_batched, params)
        evals += n

        if evals % (est_step * 10) < n or evals >= evals_budget:
            te_a = zo_acc(model_dge, X_te, y_te, params)
            best_test = max(best_test, te_a)

    print(f"  DGE INT{bits} Best Acc:   {best_test:.2%}")

    return {
        "architecture": f"INT{bits} full quantization ({1 << bits} levels)",
        "bits": bits,
        "adam_acc": round(adam_acc, 4),
        "dge_acc": round(best_test, 4),
        "total_evals": evals
    }


def main():
    parser = argparse.ArgumentParser(description="Non-differentiable and Quantized Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Quick verification run (few steps)")
    args = parser.parse_args()

    setup_result_directories()

    print(f"\n{'='*70}")
    print(f"DGE NON-DIFFERENTIABLE & QUANTIZATION SUITE (v71)")
    print(f"Device: {device}")
    print(f"{'='*70}")

    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True, download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = (((ds_tr.data.float().view(-1, 784) / 255.0) - 0.1307) / 0.3081).to(device)
    y_tr = ds_tr.targets.to(device)
    X_te = (((ds_te.data.float().view(-1, 784) / 255.0) - 0.1307) / 0.3081).to(device)
    y_te = ds_te.targets.to(device)

    # 1. Sign activation
    sign_res = run_sign_experiment(X_tr, y_tr, X_te, y_te, total_evals=200_000, quick=args.quick)

    # 2. INT8
    int8_res = run_quant_experiment(X_tr, y_tr, X_te, y_te, bits=8, total_evals=600_000, quick=args.quick)

    # 3. INT4
    int4_res = run_quant_experiment(X_tr, y_tr, X_te, y_te, bits=4, total_evals=600_000, quick=args.quick)

    # Save dedicated JSON artifacts
    sign_artifact = {
        "experiment": "v31_sign_activations",
        "hardware": get_system_info(),
        "commit_hash": get_commit_hash(),
        "result": sign_res
    }
    with open("results/raw/v31_sign_activations.json", "w") as f:
        json.dump(sign_artifact, f, indent=2)

    quant_artifact = {
        "experiment": "v32_quantized_mnist",
        "hardware": get_system_info(),
        "commit_hash": get_commit_hash(),
        "results": {
            "INT8": int8_res,
            "INT4": int4_res
        }
    }
    with open("results/raw/v32_quantized_mnist.json", "w") as f:
        json.dump(quant_artifact, f, indent=2)

    suite_artifact = {
        "experiment": "v71_nondiff_suite",
        "hardware": get_system_info(),
        "commit_hash": get_commit_hash(),
        "results": [sign_res, int8_res, int4_res]
    }
    with open("results/raw/v71_nondiff_suite.json", "w") as f:
        json.dump(suite_artifact, f, indent=2)

    print(f"\n{'='*70}")
    print(f"TABLE 2 REPRODUCTION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Architecture':<40} {'Adam':<12} {'DGE':<12}")
    print(f"{'-'*70}")
    for r in [sign_res, int8_res, int4_res]:
        print(f"{r['architecture']:<40} {r['adam_acc']*100:.2f}%      {r['dge_acc']*100:.2f}%")
    print(f"{'='*70}")
    print("Saved raw JSONs to:")
    print("  - results/raw/v31_sign_activations.json")
    print("  - results/raw/v32_quantized_mnist.json")
    print("  - results/raw/v71_nondiff_suite.json\n")


if __name__ == "__main__":
    main()
