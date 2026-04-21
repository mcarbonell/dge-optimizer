"""
dge_consistency_t_probe.py
==========================
Sondeo rapido: ¿sigue mejorando ConsistencyDGE con T > 20?

Parametros reducidos para ejecucion rapida:
  - Budget: 100K (vs 500K del v27)
  - T: {20, 30, 50, 100, 200}
  - Seeds: 3
  - Benchmarks: los 4 del v27 (son rapidos a 100K)

Nota: este script es un sondeo exploratorio, no un experimento
formal. Los resultados orientan si vale la pena hacer v28 con T>20.
"""

import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Benchmarks (identicos al v27)
# ---------------------------------------------------------------------------

def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

def rotated_quadratic(x):
    D = len(x)
    if not hasattr(rotated_quadratic, "_Q") or rotated_quadratic._Q.shape[0] != D:
        rng_rot = np.random.default_rng(999)
        A = rng_rot.standard_normal((D, D))
        Q, _ = np.linalg.qr(A)
        rotated_quadratic._Q = Q
        rotated_quadratic._lam = np.logspace(0, 2, D)
    z = rotated_quadratic._Q.T @ x
    return float(np.sum(rotated_quadratic._lam * z ** 2))

def ellipsoid(x):
    return float(np.sum(np.logspace(0, 6, len(x)) * x ** 2))

def sphere(x):
    return float(np.sum(x ** 2))

BENCHMARKS = {
    "rosenbrock": rosenbrock,
    "rotated_quadratic": rotated_quadratic,
    "ellipsoid": ellipsoid,
    "sphere": sphere,
}

# ---------------------------------------------------------------------------
# ConsistencyDGE con T configurable (igual que v27)
# ---------------------------------------------------------------------------

class ConsistencyDGE:
    def __init__(self, dim, k_blocks, lr=0.1, delta=1e-3,
                 total_steps=10000, seed=42, window=20):
        self.dim = dim
        self.k = k_blocks
        self.lr0 = lr
        self.delta0 = delta
        self.total_steps = total_steps
        self.window = window
        self.rng = np.random.default_rng(seed)
        self.m = np.zeros(dim, dtype=np.float64)
        self.v = np.zeros(dim, dtype=np.float64)
        self.t = 0
        self.sign_buffer = deque(maxlen=window)

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)

        grad = np.zeros(self.dim, dtype=np.float64)
        perm = self.rng.permutation(self.dim)
        blocks = np.array_split(perm, self.k)

        evals = 0
        for block in blocks:
            if len(block) == 0:
                continue
            signs = self.rng.choice([-1.0, 1.0], size=len(block))
            pert = np.zeros(self.dim, dtype=np.float64)
            pert[block] = signs * delta
            fp = f(x + pert)
            fm = f(x - pert)
            evals += 2
            grad[block] = (fp - fm) / (2.0 * delta) * signs

        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * (grad ** 2)

        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self.v / (1.0 - 0.999 ** self.t)

        self.sign_buffer.append(np.sign(grad))
        if len(self.sign_buffer) < 2:
            consistency = np.ones(self.dim, dtype=np.float64)
        else:
            consistency = np.abs(np.mean(np.stack(self.sign_buffer), axis=0))

        upd = lr * consistency * mh / (np.sqrt(vh) + 1e-8)
        return x - upd, evals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DIM      = 128
    BUDGET   = 100_000      # reducido para velocidad
    K_BLOCKS = 8
    LR       = 0.1
    DELTA    = 1e-3
    SEEDS    = 3
    T_VALUES = [20, 30, 50, 100, 200]

    print(f"\n{'='*72}")
    print(f"SONDEO: ConsistencyDGE con T en {T_VALUES}")
    print(f"dim={DIM}, budget={BUDGET:,}, k={K_BLOCKS}, seeds={SEEDS}")
    print(f"{'='*72}")

    all_results = []

    for bench_name, bench_fn in BENCHMARKS.items():
        print(f"\n  [{bench_name}]")
        print(f"  {'T':>6}  {'MEDIA':>14}  {'+-STD':>12}  {'vs T20':>10}")
        print(f"  {'-'*50}")

        ref_mean = None
        for T in T_VALUES:
            total_steps = BUDGET // (2 * K_BLOCKS)
            losses = []
            for seed in range(SEEDS):
                x_rng = np.random.default_rng(seed)
                x = x_rng.uniform(-2.0, 2.0, size=DIM)
                opt = ConsistencyDGE(dim=DIM, k_blocks=K_BLOCKS, lr=LR, delta=DELTA,
                                     total_steps=total_steps, seed=seed + 10_000, window=T)
                evals = 0
                while evals < BUDGET:
                    x, e = opt.step(bench_fn, x)
                    evals += e
                losses.append(bench_fn(x))
                all_results.append({
                    "benchmark": bench_name, "T": T, "seed": seed,
                    "final_loss": bench_fn(x), "budget": evals,
                })

            mean = np.mean(losses)
            std  = np.std(losses)

            if T == 20:
                ref_mean = mean
                rel_str = "  (ref)"
            else:
                rel = (mean - ref_mean) / (abs(ref_mean) + 1e-12) * 100
                sign = "+" if rel > 0 else ""
                rel_str = f"  {sign}{rel:.1f}%"

            print(f"  T={T:>4}  {mean:>14.4e}  {std:>12.4e}{rel_str}")

    # Guardar
    out_path = Path(__file__).parent.parent / "results" / "raw" / "v27_t_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"budget": BUDGET, "seeds": SEEDS,
                   "T_values": T_VALUES, "results": all_results}, f, indent=2)
    print(f"\nJSON guardado en: {out_path}")
