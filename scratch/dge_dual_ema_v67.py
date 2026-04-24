"""
dge_dual_ema_v67.py
===================
Experimento: Dual Sign-EMA (DS-EMA) Consistency Mask.

Idea central:
  Sustituir la ventana deslizante (SMA) de consistencia de signos por dos
  Medias Moviles Exponenciales (EMA) de signos con diferentes constantes
  de tiempo (Fast y Slow).

Mecanica:
  1. Mantener EMA_fast (alpha=0.3) y EMA_slow (alpha=0.05) de sign(grad_est).
  2. Aplicar un "Crossover Gate":
     - Si sign(EMA_fast) != sign(EMA_slow) -> confianza = 0 (oscilacion detectada).
     - Si sign(EMA_fast) == sign(EMA_slow) -> confianza = abs(EMA_slow).
  3. update = lr * confianza * Adam_step.

Hipotesis:
  La mascara Dual EMA es mas eficiente en memoria (O(1) vs O(T)) y mas reactiva
  que el SMA, evitando el "lag" de la ventana al cruzar valles y filtrando
  mejor el ruido de alta frecuencia de los bloques aleatorios.

Referencia:
  Inspirado en indicadores financieros (MACD) aplicados a la estabilidad de
  direccion en optimizacion Zeroth-Order.
"""

import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

def rotated_quadratic(x):
    D = len(x)
    if not hasattr(rotated_quadratic, "_Q") or rotated_quadratic._Q.shape[0] != D:
        rng_rot = np.random.default_rng(999)
        A = rng_rot.standard_normal((D, D))
        Q, _ = np.linalg.qr(A)
        lam = np.logspace(0, 2, D)
        rotated_quadratic._Q = Q
        rotated_quadratic._lam = lam
    z = rotated_quadratic._Q.T @ x
    return float(np.sum(rotated_quadratic._lam * z ** 2))

def ellipsoid(x):
    D = len(x)
    scales = np.logspace(0, 6, D)
    return float(np.sum(scales * x ** 2))

def sphere(x):
    return float(np.sum(x ** 2))

BENCHMARKS = {
    "rosenbrock": rosenbrock,
    "rotated_quadratic": rotated_quadratic,
    "ellipsoid": ellipsoid,
    "sphere": sphere,
}

# ---------------------------------------------------------------------------
# Optimizadores
# ---------------------------------------------------------------------------

class PureDGE:
    def __init__(self, dim, k_blocks, lr=0.1, delta=1e-3, total_steps=10000, seed=42):
        self.dim, self.k, self.lr0, self.delta0, self.total_steps = dim, k_blocks, lr, delta, total_steps
        self.rng = np.random.default_rng(seed)
        self.m, self.v, self.t = np.zeros(dim), np.zeros(dim), 0

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr, delta = self._cosine(self.lr0), self._cosine(self.delta0, 0.1)
        grad = np.zeros(self.dim)
        perm = self.rng.permutation(self.dim)
        blocks = np.array_split(perm, self.k)
        for block in blocks:
            if len(block) == 0: continue
            signs = self.rng.choice([-1.0, 1.0], size=len(block))
            pert = np.zeros(self.dim)
            pert[block] = signs * delta
            grad[block] = (f(x + pert) - f(x - pert)) / (2.0 * delta) * signs
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * (grad ** 2)
        mh, vh = self.m / (1.0 - 0.9**self.t), self.v / (1.0 - 0.999**self.t)
        upd = lr * mh / (np.sqrt(vh) + 1e-8)
        return x - upd, 2 * self.k

class ConsistencyDGE:
    """Baseline v27: SMA of signs (Window=20)."""
    def __init__(self, dim, k_blocks, lr=0.1, delta=1e-3, total_steps=10000, seed=42, window=20):
        self.dim, self.k, self.lr0, self.delta0, self.total_steps, self.window = dim, k_blocks, lr, delta, total_steps, window
        self.rng = np.random.default_rng(seed)
        self.m, self.v, self.t = np.zeros(dim), np.zeros(dim), 0
        self.sign_buffer = deque(maxlen=window)

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr, delta = self._cosine(self.lr0), self._cosine(self.delta0, 0.1)
        grad = np.zeros(self.dim)
        perm = self.rng.permutation(self.dim)
        blocks = np.array_split(perm, self.k)
        for block in blocks:
            if len(block) == 0: continue
            signs = self.rng.choice([-1.0, 1.0], size=len(block))
            pert = np.zeros(self.dim)
            pert[block] = signs * delta
            grad[block] = (f(x + pert) - f(x - pert)) / (2.0 * delta) * signs
        self.m, self.v = 0.9 * self.m + 0.1 * grad, 0.999 * self.v + 0.001 * (grad**2)
        mh, vh = self.m / (1.0 - 0.9**self.t), self.v / (1.0 - 0.999**self.t)
        self.sign_buffer.append(np.sign(grad))
        mask = np.abs(np.mean(np.stack(self.sign_buffer), axis=0)) if len(self.sign_buffer) >= 2 else 1.0
        upd = lr * mask * mh / (np.sqrt(vh) + 1e-8)
        return x - upd, 2 * self.k

class DualEMADGE:
    """Prototipo v67: Fast/Slow EMA crossover gate."""
    def __init__(self, dim, k_blocks, lr=0.1, delta=1e-3, total_steps=10000, seed=42, alpha_fast=0.3, alpha_slow=0.05):
        self.dim, self.k, self.lr0, self.delta0, self.total_steps = dim, k_blocks, lr, delta, total_steps
        self.alpha_f, self.alpha_s = alpha_fast, alpha_slow
        self.rng = np.random.default_rng(seed)
        self.m, self.v, self.t = np.zeros(dim), np.zeros(dim), 0
        # EMAs de signos
        self.ema_fast = np.zeros(dim)
        self.ema_slow = np.zeros(dim)

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr, delta = self._cosine(self.lr0), self._cosine(self.delta0, 0.1)
        grad = np.zeros(self.dim)
        perm = self.rng.permutation(self.dim)
        blocks = np.array_split(perm, self.k)
        for block in blocks:
            if len(block) == 0: continue
            signs = self.rng.choice([-1.0, 1.0], size=len(block))
            pert = np.zeros(self.dim)
            pert[block] = signs * delta
            grad[block] = (f(x + pert) - f(x - pert)) / (2.0 * delta) * signs
        
        # Adam base
        self.m, self.v = 0.9 * self.m + 0.1 * grad, 0.999 * self.v + 0.001 * (grad**2)
        mh, vh = self.m / (1.0 - 0.9**self.t), self.v / (1.0 - 0.999**self.t)
        
        # Actualizacion de EMAs de signos
        s = np.sign(grad)
        self.ema_fast = (1 - self.alpha_f) * self.ema_fast + self.alpha_f * s
        self.ema_slow = (1 - self.alpha_s) * self.ema_slow + self.alpha_s * s
        
        # MACD-style Mask: Crossover Gate
        # 1. Acuerdo de signo entre corto y largo plazo
        agreement = (np.sign(self.ema_fast) == np.sign(self.ema_slow)).astype(np.float64)
        # 2. Magnitud de consistencia de la tendencia lenta
        confidence = np.abs(self.ema_slow)
        # Mascara final: Solo se mueve si hay acuerdo, y escala por la consistencia lenta
        mask = agreement * confidence
        
        upd = lr * mask * mh / (np.sqrt(vh) + 1e-8)
        return x - upd, 2 * self.k

# ---------------------------------------------------------------------------
# Execution Logic
# ---------------------------------------------------------------------------

METHODS = {
    "PureDGE": PureDGE,
    "ConsistencyDGE_T20": ConsistencyDGE,
    "DualEMADGE_v67": DualEMADGE,
}

def run_experiment(dim=128, budget=200_000, k=8, lr=0.1, delta=1e-3, seeds=5):
    all_results = []
    print(f"\n{'='*70}\nEXPERIMENTO v67: DGE DUAL SIGN-EMA (DS-EMA)\n{'='*70}")
    
    for b_name, b_fn in BENCHMARKS.items():
        print(f"\n  [{b_name}]")
        for m_name, m_cls in METHODS.items():
            run_losses = []
            t0 = time.time()
            for s in range(seeds):
                total_steps = budget // (2 * k)
                x_rng = np.random.default_rng(s)
                x = x_rng.uniform(-2.0, 2.0, size=dim)
                opt = m_cls(dim=dim, k_blocks=k, lr=lr, delta=delta, total_steps=total_steps, seed=s+1000)
                
                evals = 0
                while evals < budget:
                    x, n = opt.step(b_fn, x)
                    evals += n
                run_losses.append(b_fn(x))
            
            avg = np.mean(run_losses)
            std = np.std(run_losses)
            print(f"    {m_name:<20}: {avg:.4e} +/- {std:.4e}  ({time.time()-t0:.1f}s)")
            
            for i, loss in enumerate(run_losses):
                all_results.append({
                    "benchmark": b_name, "method": m_name, "seed": i, "loss": loss
                })
    return all_results

if __name__ == "__main__":
    results = run_experiment()
    
    # Save Results
    out_path = Path(__file__).parent.parent / "results" / "raw" / "v67_dual_ema.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en {out_path}")
