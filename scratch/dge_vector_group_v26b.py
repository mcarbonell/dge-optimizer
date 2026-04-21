"""
dge_vector_group_v26b.py
========================
Experimento: Vector Group DGE — Ablación completa y justa.

Corrige los errores del v26:
  1. Implementa las 4 variantes (PureDGE, SphericalDGE, ScalarVarianceDGE, VectorGroupDGE).
  2. Grupos DINÁMICOS en todos los optimizadores (permutación aleatoria por paso).
  3. 4 benchmarks: Rosenbrock, Rotated Quadratic, Ellipsoid, Sphere.
  4. Budget = 500_000, Seeds = 5.
  5. Guarda JSON en results/raw/v26b_vector_group.json.

Diseño del experimento (spec: docs/v26b_implementation_spec.md):
  - 4 métodos × 4 benchmarks × 5 seeds = 80 corridas base
  - Ablación de k_blocks ∈ [2, 4, 8, 16, 32] solo para Rosenbrock y Rotated Quadratic
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Funciones de Benchmark
# ---------------------------------------------------------------------------

def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rotated_quadratic(x):
    """Quadratic con rotación fija (seed 999). Los ejes principales NO están alineados
    con los ejes coordenados. Condition number = 100."""
    D = len(x)
    if not hasattr(rotated_quadratic, "_Q") or rotated_quadratic._Q.shape[0] != D:
        rng_rot = np.random.default_rng(999)
        A = rng_rot.standard_normal((D, D))
        Q, _ = np.linalg.qr(A)
        lam = np.logspace(0, 2, D)  # eigenvalues: 1 → 100
        rotated_quadratic._Q = Q
        rotated_quadratic._lam = lam
    z = rotated_quadratic._Q.T @ x
    return float(np.sum(rotated_quadratic._lam * z ** 2))


def ellipsoid(x):
    """Ill-conditioned ellipsoid. Condition number = 10^6."""
    D = len(x)
    scales = np.logspace(0, 6, D)
    return float(np.sum(scales * x ** 2))


def sphere(x):
    """Control negativo: no debería haber diferencia entre métodos."""
    return float(np.sum(x ** 2))


BENCHMARKS = {
    "rosenbrock": rosenbrock,
    "rotated_quadratic": rotated_quadratic,
    "ellipsoid": ellipsoid,
    "sphere": sphere,
}

# ---------------------------------------------------------------------------
# Clase base con cosine schedule (compartida por todos los optimizadores)
# ---------------------------------------------------------------------------

class _BaseDGE:
    def __init__(self, dim: int, k_blocks: int, lr: float, delta: float,
                 total_steps: int, seed: int):
        self.dim = dim
        self.k = k_blocks
        self.lr0 = lr
        self.delta0 = delta
        self.total_steps = total_steps
        self.rng = np.random.default_rng(seed)

        self.m = np.zeros(dim, dtype=np.float64)
        self.v = np.zeros(dim, dtype=np.float64)   # per-coordinate by default
        self.t = 0

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def _get_perturbation(self, block_size: int):
        """Implementado por subclases. Devuelve un vector de perturbación unitario."""
        raise NotImplementedError

    def step(self, f, x):
        self.t += 1
        lr = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)

        grad = np.zeros(self.dim, dtype=np.float64)
        # Grupos dinámicos: permutación nueva en cada paso
        perm = self.rng.permutation(self.dim)
        blocks = np.array_split(perm, self.k)

        evals = 0
        # Guardamos las direcciones/signs para reutilizarlas en la actualización de v
        directions = {}
        for block in blocks:
            if len(block) == 0:
                continue
            direction = self._get_perturbation(len(block))
            directions[id(block)] = (block, direction)

            pert = np.zeros(self.dim, dtype=np.float64)
            pert[block] = direction * delta

            fp = f(x + pert)
            fm = f(x - pert)
            evals += 2

            g_est = (fp - fm) / (2.0 * delta) * direction
            grad[block] = g_est

        self.m = 0.9 * self.m + 0.1 * grad
        self._update_v(grad, directions)

        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self._get_vh(directions)

        upd = lr * mh / (np.sqrt(vh) + 1e-8)
        return x - upd, evals

    def _update_v(self, grad, directions):
        """Per-coordinate (default). Subclases pueden sobreescribir."""
        self.v = 0.999 * self.v + 0.001 * (grad ** 2)

    def _get_vh(self, directions):
        """Per-coordinate bias correction (default)."""
        return self.v / (1.0 - 0.999 ** self.t)


# ---------------------------------------------------------------------------
# Clase 1: PureDGE — Rademacher ±1, varianza per-coordinate
# (Copiada del v26, es la referencia correcta)
# ---------------------------------------------------------------------------

class PureDGE(_BaseDGE):
    """Baseline. Perturbaciones Rademacher ±1, varianza Adam per-coordinate."""

    def _get_perturbation(self, block_size: int):
        return self.rng.choice([-1.0, 1.0], size=block_size)


# ---------------------------------------------------------------------------
# Clase 2: SphericalDGE — dirección esférica unitaria, varianza per-coordinate
# (Aisla el efecto de la perturbación; v permanece per-coordinate)
# ---------------------------------------------------------------------------

class SphericalDGE(_BaseDGE):
    """Perturbaciones de dirección aleatoria uniforme en S^{n-1}.
    Varianza Adam per-coordinate (idéntica a PureDGE)."""

    def _get_perturbation(self, block_size: int):
        d = self.rng.standard_normal(block_size)
        return d / (np.linalg.norm(d) + 1e-12)


# ---------------------------------------------------------------------------
# Clase 3: ScalarVarianceDGE — Rademacher ±1, varianza escalar por grupo
# (Aisla el efecto de la varianza de Adam)
# ---------------------------------------------------------------------------

class ScalarVarianceDGE(_BaseDGE):
    """Perturbaciones Rademacher ±1. Varianza Adam: escalar por grupo
    (mismo valor para todas las coordenadas del grupo en cada paso).

    NOTA: self.v es per-coordinate (dim), pero se actualiza con el promedio
    del grupo en vez del valor individual. Esto mantiene la dimensionalidad
    correcta sin necesidad de mapeos complicados."""

    def _get_perturbation(self, block_size: int):
        return self.rng.choice([-1.0, 1.0], size=block_size)

    def _update_v(self, grad, directions):
        for block, _ in directions.values():
            sq_mean = np.mean(grad[block] ** 2)  # escalar
            self.v[block] = 0.999 * self.v[block] + 0.001 * sq_mean
            # Mismo valor para todas las coordenadas del grupo → Adam escalar por grupo


# ---------------------------------------------------------------------------
# Clase 4: VectorGroupDGE — esférica + varianza escalar por grupo
# (La hipótesis completa del research_shortlist.md)
# ---------------------------------------------------------------------------

class VectorGroupDGE(_BaseDGE):
    """Perturbaciones esféricas (como SphericalDGE) + varianza escalar
    por grupo (como ScalarVarianceDGE). Es el 'Vector Group DGE' completo."""

    def _get_perturbation(self, block_size: int):
        d = self.rng.standard_normal(block_size)
        return d / (np.linalg.norm(d) + 1e-12)

    def _update_v(self, grad, directions):
        for block, _ in directions.values():
            sq_mean = np.mean(grad[block] ** 2)
            self.v[block] = 0.999 * self.v[block] + 0.001 * sq_mean


# ---------------------------------------------------------------------------
# Función de corrida individual
# ---------------------------------------------------------------------------

OPTIMIZERS = {
    "PureDGE": PureDGE,
    "SphericalDGE": SphericalDGE,
    "ScalarVarianceDGE": ScalarVarianceDGE,
    "VectorGroupDGE": VectorGroupDGE,
}


def run_single(method: str, benchmark_fn, dim: int, budget: int,
               k_blocks: int, lr: float, delta: float, seed: int) -> dict:
    """Ejecuta una corrida y devuelve un dict con las métricas."""
    total_steps = budget // (2 * k_blocks)
    optimizer_cls = OPTIMIZERS[method]

    # Inicialización: mismo x0 para todos los optimizadores en la misma seed
    x_rng = np.random.default_rng(seed)
    x = x_rng.uniform(-2.0, 2.0, size=dim)

    # El optimizador tiene su propia semilla separada para no contaminar x0
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


# ---------------------------------------------------------------------------
# Experimento completo
# ---------------------------------------------------------------------------

def run_full_experiment(dim: int = 128, budget: int = 500_000,
                        k_blocks_main: int = 8, lr: float = 0.1,
                        delta: float = 1e-3, seeds: int = 5,
                        run_ablation: bool = True) -> list:
    """
    Ejecuta:
      - 4 métodos × 4 benchmarks × 5 seeds = 80 corridas base
      - Ablación de k_blocks ∈ [2, 4, 8, 16, 32] para Rosenbrock y Rotated Quadratic
    """
    all_results = []
    methods = list(OPTIMIZERS.keys())
    seed_list = list(range(seeds))

    # — Corridas base —
    print("\n" + "=" * 70)
    print(f"EXPERIMENTO BASE: dim={dim}, budget={budget:,}, k={k_blocks_main}, seeds={seeds}")
    print("=" * 70)

    for bench_name, bench_fn in BENCHMARKS.items():
        for method in methods:
            runs = []
            for seed in seed_list:
                r = run_single(method, bench_fn, dim, budget,
                               k_blocks_main, lr, delta, seed)
                runs.append(r)
                all_results.append(r)
            losses = [r["final_loss"] for r in runs]
            print(f"  [{bench_name:20s}] {method:22s}: "
                  f"{np.mean(losses):.4e} ± {np.std(losses):.4e}")

    # — Ablación de k_blocks —
    if run_ablation:
        ablation_benchmarks = {k: v for k, v in BENCHMARKS.items()
                               if k in ("rosenbrock", "rotated_quadratic")}
        k_sweep = [2, 4, 8, 16, 32]

        print("\n" + "=" * 70)
        print(f"ABLACION k_blocks in {k_sweep} para Rosenbrock y Rotated Quadratic")
        print("=" * 70)

        for bench_name, bench_fn in ablation_benchmarks.items():
            for k in k_sweep:
                if k == k_blocks_main:
                    # Ya está en las corridas base; no duplicar
                    continue
                for method in methods:
                    runs = []
                    for seed in seed_list:
                        r = run_single(method, bench_fn, dim, budget,
                                       k, lr, delta, seed)
                        r["ablation"] = True
                        runs.append(r)
                        all_results.append(r)
                    losses = [r["final_loss"] for r in runs]
                    print(f"  [{bench_name:20s}] k={k:2d} {method:22s}: "
                          f"{np.mean(losses):.4e} ± {np.std(losses):.4e}")

    return all_results


# ---------------------------------------------------------------------------
# Formateo de tablas
# ---------------------------------------------------------------------------

def print_summary_tables(results: list, k_main: int = 8):
    """Imprime tablas resumen solo para las corridas base (k == k_main)."""
    base = [r for r in results if r["k_blocks"] == k_main and not r.get("ablation", False)]

    benchmarks = sorted({r["benchmark"] for r in base})
    methods = list(OPTIMIZERS.keys())

    for bench in benchmarks:
        dim = 128
        budget = max(r["budget"] for r in base if r["benchmark"] == bench)
        print(f"\n{'=' * 70}")
        print(f"BENCHMARK: {bench}  (D={dim}, budget={budget:,}, k={k_main})")
        print(f"{'=' * 70}")
        header = f"{'METHOD':<24} {'MEAN':>14} {'±STD':>12} {'BEST':>14} {'WORST':>14}"
        print(header)
        print("-" * len(header))
        for method in methods:
            losses = [r["final_loss"] for r in base
                      if r["benchmark"] == bench and r["method"] == method]
            if not losses:
                continue
            print(f"{method:<24} {np.mean(losses):>14.4e} {np.std(losses):>12.4e} "
                  f"{min(losses):>14.4e} {max(losses):>14.4e}")

    # Tabla ablación
    ablation = [r for r in results if r.get("ablation", False)]
    if not ablation:
        return

    k_sweep = sorted({r["k_blocks"] for r in ablation})
    print(f"\n{'=' * 70}")
    print("ABLACIÓN k_blocks — Mejores métodos en Rosenbrock y Rotated Quadratic")
    print(f"{'=' * 70}")
    for bench in ("rosenbrock", "rotated_quadratic"):
        print(f"\n  {bench}")
        print(f"  {'k':>4}  {'METHOD':<24}  {'MEAN':>14}  {'±STD':>12}")
        print(f"  {'-'*60}")
        for k in k_sweep:
            for method in methods:
                losses = [r["final_loss"] for r in ablation
                          if r["benchmark"] == bench
                          and r["k_blocks"] == k
                          and r["method"] == method]
                if not losses:
                    continue
                print(f"  {k:>4}  {method:<24}  {np.mean(losses):>14.4e}  "
                      f"{np.std(losses):>12.4e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DIM = 128
    BUDGET = 500_000
    K_MAIN = 8
    LR = 0.1
    DELTA = 1e-3
    SEEDS = 5

    results = run_full_experiment(
        dim=DIM,
        budget=BUDGET,
        k_blocks_main=K_MAIN,
        lr=LR,
        delta=DELTA,
        seeds=SEEDS,
        run_ablation=True,
    )

    print_summary_tables(results, k_main=K_MAIN)

    # — Guardar JSON —
    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v26b_vector_group.json"

    payload = {
        "experiment": "v26b_vector_group",
        "dim": DIM,
        "budget": BUDGET,
        "k_blocks_main": K_MAIN,
        "lr": LR,
        "delta": DELTA,
        "seeds": SEEDS,
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n✓ JSON guardado en: {out_path}")
