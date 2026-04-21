"""
dge_consistency_lr_v27.py
=========================
Experimento: Direction-Consistency Learning Rates (v27).

Idea central:
  Variables cuyo signo de gradiente estimado es CONSISTENTE en las ultimas T
  evaluaciones reciben un multiplicador de LR mas alto. Variables con signo
  OSCILANTE (ruidoso) reciben un LR reducido.

  consistency_i = mean(sign(g_est_i)  para los ultimos T pasos) en [-1, 1]
  lr_scale_i   = abs(consistency_i)                              en [0, 1]
  update       = lr * lr_scale * (m_hat / (sqrt(v_hat) + eps))

Hipotesis:
  El modulo de la consistencia de signo es una senal de confianza local
  que complementa el segundo momento de Adam. Variables con senal robusta
  convergen mas rapido; variables ruidosas se frenan y no oscilan.

Diseno del experimento:
  - Baseline:       PureDGE (v26b, sin consistency)
  - ConsistencyDGE: PureDGE + consistency LR scaling
  - Ablacion de ventana T in {5, 10, 20}
  - 4 benchmarks: Rosenbrock, RotatedQuadratic, Ellipsoid, Sphere
  - D=128, budget=500K, k=8, lr=0.1, delta=1e-3, seeds=5
  - JSON en results/raw/v27_consistency_lr.json

Nota: NO modificar dge_vector_group_v26b.py. PureDGE se reimplementa aqui
como referencia independiente (regla de oro del repositorio).
"""

import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Funciones de Benchmark (identicas a v26b para comparabilidad directa)
# ---------------------------------------------------------------------------

def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rotated_quadratic(x):
    """Quadratic con rotacion fija (seed 999). Misma que v26b."""
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
# Clase 1: PureDGE — referencia exacta del v26b
# ---------------------------------------------------------------------------

class PureDGE:
    """Baseline: Rademacher +/-1, varianza Adam per-coordinate, sin consistency."""

    def __init__(self, dim: int, k_blocks: int, lr: float = 0.1,
                 delta: float = 1e-3, total_steps: int = 10000, seed: int = 42):
        self.dim = dim
        self.k = k_blocks
        self.lr0 = lr
        self.delta0 = delta
        self.total_steps = total_steps
        self.rng = np.random.default_rng(seed)

        self.m = np.zeros(dim, dtype=np.float64)
        self.v = np.zeros(dim, dtype=np.float64)
        self.t = 0

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

        upd = lr * mh / (np.sqrt(vh) + 1e-8)
        return x - upd, evals


# ---------------------------------------------------------------------------
# Clase 2: ConsistencyDGE — PureDGE + direction-consistency LR scaling
# ---------------------------------------------------------------------------

class ConsistencyDGE:
    """PureDGE con multiplicador de LR basado en consistencia de signo de gradiente.

    Para cada variable i:
      consistency_i = mean(sign(g_est_i) para los ultimos T pasos)  en [-1, 1]
      lr_scale_i   = abs(consistency_i)                              en [0, 1]

    El update final es:
      x <- x - lr * lr_scale * (m_hat / (sqrt(v_hat) + eps))

    La perturbacion, EMA y varianza son identicas a PureDGE. Solo cambia
    la escala local del learning rate en cada coordenada.
    """

    def __init__(self, dim: int, k_blocks: int, lr: float = 0.1,
                 delta: float = 1e-3, total_steps: int = 10000,
                 seed: int = 42, window: int = 10):
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

        # Buffer circular de signos de gradiente: deque de arrays (dim,)
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

        # --- EMA identica a PureDGE ---
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * (grad ** 2)

        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self.v / (1.0 - 0.999 ** self.t)

        # --- Consistency LR scaling ---
        self.sign_buffer.append(np.sign(grad))

        if len(self.sign_buffer) < 2:
            # Sin historia suficiente: usar lr_scale=1 (igual que PureDGE)
            consistency = np.ones(self.dim, dtype=np.float64)
        else:
            # Media de signos en [-1,1] -> abs en [0,1]
            stacked = np.stack(self.sign_buffer, axis=0)   # (T, dim)
            consistency = np.abs(np.mean(stacked, axis=0)) # (dim,)

        upd = lr * consistency * mh / (np.sqrt(vh) + 1e-8)
        return x - upd, evals


# ---------------------------------------------------------------------------
# Corridas y experimento
# ---------------------------------------------------------------------------

OPTIMIZERS = {
    "PureDGE": PureDGE,
    "ConsistencyDGE_T5":  lambda **kw: ConsistencyDGE(**kw, window=5),
    "ConsistencyDGE_T10": lambda **kw: ConsistencyDGE(**kw, window=10),
    "ConsistencyDGE_T20": lambda **kw: ConsistencyDGE(**kw, window=20),
}


def run_single(method: str, benchmark_fn, dim: int, budget: int,
               k_blocks: int, lr: float, delta: float, seed: int) -> dict:
    total_steps = budget // (2 * k_blocks)
    optimizer_cls = OPTIMIZERS[method]

    x_rng = np.random.default_rng(seed)
    x = x_rng.uniform(-2.0, 2.0, size=dim)

    opt = optimizer_cls(dim=dim, k_blocks=k_blocks, lr=lr, delta=delta,
                        total_steps=total_steps, seed=seed + 10_000)

    t0 = time.perf_counter()
    evals_total = 0
    while evals_total < budget:
        x, e = opt.step(benchmark_fn, x)
        evals_total += e

    wall_clock = time.perf_counter() - t0
    final_loss = benchmark_fn(x)

    return {
        "method": method,
        "benchmark": benchmark_fn.__name__,
        "k_blocks": k_blocks,
        "seed": seed,
        "final_loss": final_loss,
        "budget": evals_total,
        "wall_clock_time": round(wall_clock, 3),
    }


def run_full_experiment(dim: int = 128, budget: int = 500_000,
                        k_blocks: int = 8, lr: float = 0.1,
                        delta: float = 1e-3, seeds: int = 5) -> list:
    """4 metodos x 4 benchmarks x 5 seeds = 80 corridas."""
    all_results = []
    methods = list(OPTIMIZERS.keys())
    seed_list = list(range(seeds))

    print("\n" + "=" * 72)
    print(f"EXPERIMENTO v27: Direction-Consistency LR")
    print(f"dim={dim}, budget={budget:,}, k={k_blocks}, seeds={seeds}")
    print("=" * 72)

    for bench_name, bench_fn in BENCHMARKS.items():
        print(f"\n  [{bench_name}]")
        for method in methods:
            runs = []
            for seed in seed_list:
                r = run_single(method, bench_fn, dim, budget,
                               k_blocks, lr, delta, seed)
                runs.append(r)
                all_results.append(r)
            losses = [r["final_loss"] for r in runs]
            print(f"    {method:<26}: {np.mean(losses):.4e} +/- {np.std(losses):.4e}")

    return all_results


def print_summary_tables(results: list, k_blocks: int = 8):
    methods = list(OPTIMIZERS.keys())
    benchmarks = sorted({r["benchmark"] for r in results})

    for bench in benchmarks:
        budget = max(r["budget"] for r in results if r["benchmark"] == bench)
        print(f"\n{'=' * 72}")
        print(f"BENCHMARK: {bench}  (D=128, budget={budget:,}, k={k_blocks})")
        print(f"{'=' * 72}")
        hdr = f"{'METHOD':<28} {'MEAN':>14} {'+-STD':>12} {'BEST':>14} {'WORST':>14}"
        print(hdr)
        print("-" * len(hdr))
        for method in methods:
            losses = [r["final_loss"] for r in results
                      if r["benchmark"] == bench and r["method"] == method]
            if not losses:
                continue
            print(f"{method:<28} {np.mean(losses):>14.4e} {np.std(losses):>12.4e} "
                  f"{min(losses):>14.4e} {max(losses):>14.4e}")

    # Tabla de mejora relativa vs PureDGE
    print(f"\n{'=' * 72}")
    print("MEJORA RELATIVA vs PureDGE (negativo = mejor, positivo = peor)")
    print(f"{'=' * 72}")
    ref_method = "PureDGE"
    print(f"{'METHOD':<28}", end="")
    for bench in benchmarks:
        print(f"  {bench[:18]:>18}", end="")
    print()
    print("-" * (28 + 20 * len(benchmarks)))
    for method in methods:
        if method == ref_method:
            continue
        print(f"{method:<28}", end="")
        for bench in benchmarks:
            ref_losses = [r["final_loss"] for r in results
                          if r["benchmark"] == bench and r["method"] == ref_method]
            met_losses = [r["final_loss"] for r in results
                          if r["benchmark"] == bench and r["method"] == method]
            if not ref_losses or not met_losses:
                print(f"  {'N/A':>18}", end="")
                continue
            rel = (np.mean(met_losses) - np.mean(ref_losses)) / (abs(np.mean(ref_losses)) + 1e-12)
            sign = "+" if rel > 0 else ""
            print(f"  {sign}{rel*100:>16.1f}%", end="")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DIM = 128
    BUDGET = 500_000
    K_BLOCKS = 8
    LR = 0.1
    DELTA = 1e-3
    SEEDS = 5

    results = run_full_experiment(
        dim=DIM,
        budget=BUDGET,
        k_blocks=K_BLOCKS,
        lr=LR,
        delta=DELTA,
        seeds=SEEDS,
    )

    print_summary_tables(results, k_blocks=K_BLOCKS)

    # Guardar JSON
    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v27_consistency_lr.json"

    payload = {
        "experiment": "v27_consistency_lr",
        "dim": DIM,
        "budget": BUDGET,
        "k_blocks": K_BLOCKS,
        "lr": LR,
        "delta": DELTA,
        "seeds": SEEDS,
        "methods": list(OPTIMIZERS.keys()),
        "windows_tested": [5, 10, 20],
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nJSON guardado en: {out_path}")
