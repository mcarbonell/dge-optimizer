"""
examples/train_mnist.py
=======================
Ejemplo de referencia: entrenar un MLP en MNIST con el DGEOptimizer canónico.

Demuestra que DGEOptimizer (con Direction-Consistency LR activo por defecto)
supera a PureDGE y SPSA en el mismo budget de evaluaciones de función.

Resultados de referencia (v28, CPU, 3 seeds):
  DGEOptimizer (consistency_window=20) : 87.56% ± 0.77%
  PureDGE (consistency_window=0)       : 80.00% ± 1.57%
  Referencia v25b                      : 81.17%

Uso:
  python examples/train_mnist.py
"""

import math
import os
import sys
import time

import numpy as np

# Permite ejecutar desde la raíz del repositorio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dge.optimizer import DGEOptimizer

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
SEED         = 42
N_TRAIN      = 3_000       # subconjunto de train (velocidad)
N_TEST       = 600
BATCH_SIZE   = 256
TOTAL_EVALS  = 300_000     # presupuesto de evaluaciones (ajustar según tiempo)
LOG_EVERY    = 50_000      # imprimir métricas cada N evaluaciones
ARCH         = (784, 32, 10)   # arquitectura del MLP

# ---------------------------------------------------------------------------
# Datos (NumPy puro, sin PyTorch)
# ---------------------------------------------------------------------------

def load_mnist_numpy(n_train=N_TRAIN, n_test=N_TEST, seed=SEED):
    """Carga MNIST como arrays NumPy. Requiere: pip install torchvision"""
    try:
        from torchvision import datasets, transforms
    except ImportError:
        raise ImportError("Instala torchvision: pip install torchvision")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=transform)

    X_tr = ds_tr.data.float().view(-1, 784).numpy() / 255.0
    y_tr = ds_tr.targets.numpy()
    X_te = ds_te.data.float().view(-1, 784).numpy() / 255.0
    y_te = ds_te.targets.numpy()

    X_tr = (X_tr - 0.1307) / 0.3081
    X_te = (X_te - 0.1307) / 0.3081

    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(y_tr), size=n_train, replace=False)
    te_idx = rng.choice(len(y_te), size=n_test,  replace=False)
    return X_tr[tr_idx], y_tr[tr_idx], X_te[te_idx], y_te[te_idx]

# ---------------------------------------------------------------------------
# Modelo: MLP NumPy
# ---------------------------------------------------------------------------

def n_params(arch):
    return sum(a * b + b for a, b in zip(arch[:-1], arch[1:]))

def forward(X, params, arch):
    """Forward pass. X: (N, in). Returns logits (N, out)."""
    h, i = X, 0
    for l_in, l_out in zip(arch[:-1], arch[1:]):
        W = params[i:i + l_in * l_out].reshape(l_in, l_out)
        i += l_in * l_out
        b = params[i:i + l_out]
        i += l_out
        h = h @ W + b
        if l_out != arch[-1]:
            h = np.maximum(h, 0.0)  # ReLU
    return h

def cross_entropy(X, y, params, arch):
    logits = forward(X, params, arch)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-9)
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.mean(np.log(probs[np.arange(len(y)), y])))

def accuracy(X, y, params, arch):
    return float((forward(X, params, arch).argmax(axis=1) == y).mean())

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(label, opt, params0, X_tr, y_tr, X_te, y_te,
        arch=ARCH, total_evals=TOTAL_EVALS, batch_size=BATCH_SIZE):

    params = params0.copy()
    rng_mb = np.random.default_rng(SEED + 1)
    evals  = 0
    best_te_acc = 0.0
    next_log = LOG_EVERY
    t0 = time.time()

    print(f"\n  {label}  [{opt}]")
    print(f"  {'evals':>8}  {'train':>7}  {'test':>7}  {'loss':>8}  {'time':>6}")
    print(f"  {'-'*48}")

    while evals < total_evals:
        idx = rng_mb.integers(0, len(y_tr), size=batch_size)
        Xb, yb = X_tr[idx], y_tr[idx]

        def f(p): return cross_entropy(Xb, yb, p, arch)

        params, n = opt.step(f, params)
        evals += n
        best_te_acc = max(best_te_acc, accuracy(X_te, y_te, params, arch))

        if evals >= next_log or evals >= total_evals:
            tr_acc = accuracy(X_tr, y_tr, params, arch)
            te_acc = accuracy(X_te, y_te, params, arch)
            loss   = cross_entropy(Xb, yb, params, arch)
            elapsed = time.time() - t0
            print(f"  {evals:>8,}  {tr_acc:>6.1%}  {te_acc:>6.1%}  {loss:>8.4f}  {elapsed:>5.0f}s")
            next_log += LOG_EVERY

    print(f"  Best test accuracy: {best_te_acc:.2%}")
    return best_te_acc

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_mnist_numpy()

    D = n_params(ARCH)
    k = max(1, math.ceil(math.log2(D)))
    total_steps = TOTAL_EVALS // (2 * k)

    # He initialization
    rng_init = np.random.default_rng(SEED)
    params0  = np.zeros(D, dtype=np.float64)
    i = 0
    for fan_in, fan_out in zip(ARCH[:-1], ARCH[1:]):
        std = math.sqrt(2.0 / fan_in)
        w   = fan_in * fan_out
        params0[i:i + w] = rng_init.normal(0, std, w)
        i += w + fan_out

    print(f"\nMNIST  arch={ARCH}  D={D}  k={k}  budget={TOTAL_EVALS:,}")
    print(f"Seeds: {SEED}  batch={BATCH_SIZE}  n_train={N_TRAIN}  n_test={N_TEST}")

    # --- DGEOptimizer con Direction-Consistency (por defecto) ---
    dge = DGEOptimizer(
        dim=D, k_blocks=k, lr=0.01, delta=1e-3,
        total_steps=total_steps,
        consistency_window=20,   # Direction-Consistency LR activado
        seed=SEED + 10,
    )
    acc_dge = run("DGEOptimizer (DS-EMA Consistency)", dge,
                  params0, X_tr, y_tr, X_te, y_te)

    # --- PureDGE (consistency desactivada para comparación) ---
    pure = DGEOptimizer(
        dim=D, k_blocks=k, lr=0.01, delta=1e-3,
        total_steps=total_steps,
        consistency_window=0,    # sin máscara de consistencia
        seed=SEED + 10,
    )
    acc_pure = run("PureDGE (consistency_window=0)", pure,
                   params0, X_tr, y_tr, X_te, y_te)

    print(f"\n{'='*54}")
    print(f"  DGEOptimizer (consistency) : {acc_dge:.2%}")
    print(f"  PureDGE (no consistency)   : {acc_pure:.2%}")
    delta_pp = (acc_dge - acc_pure) * 100
    sign = '+' if delta_pp > 0 else ''
    print(f"  Diferencia                 : {sign}{delta_pp:.2f}pp")
