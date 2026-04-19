"""
DGE - Denoised Gradient Estimation (Prototype v5)
=====================================================
Mejoras sobre v4:
  - lr automaticamente escalado por 1/sqrt(D): estabilidad en alta dimension.
  - Benchmark JUSTO contra SPSA: misma funcion, mismo x0, mismo presupuesto
    de evaluaciones totales (no de pasos). Esta es la comparacion clave.
  - Metricas unificadas por evaluaciones (no por pasos) para comparacion limpia.

Hipotesis a validar:
  - DGE supera a SPSA en funciones con sparsity de gradiente.
  - DGE es comparable o superior a SPSA en funciones densas.
  - DGE mantiene ventaja en alta dimension por presupuesto logaritmico.

Referencia: docs/denoised_gradient_estimation_idea.md  (Seccion 7)
"""

import numpy as np
import math
from dataclasses import dataclass, field


# =============================================================================
# SPSA — Referencia de comparacion
# =============================================================================

class SPSAOptimizer:
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) clasico.

    Usa exactamente 2 evaluaciones por paso (igual que DGE con D=1).
    El gradiente estimado es un vector D-dimensional muy ruidoso:
      g_hat = (f(x+c*delta) - f(x-c*delta)) / (2*c) * 1/delta_i

    Referencia de comparacion justa: mismo presupuesto de evaluaciones que DGE.
    """

    def __init__(self, dim: int, lr: float = 0.1, delta: float = 1e-2,
                 lr_decay: float = 0.1, delta_decay: float = 0.1,
                 total_steps: int = 1000, seed: int | None = None):
        self.dim = dim
        self.lr0 = lr
        self.delta0 = delta
        self.lr_decay = lr_decay
        self.delta_decay = delta_decay
        self.total_steps = total_steps
        self.rng = np.random.default_rng(seed)
        self.iteration = 0

    def _cosine(self, v0, decay):
        t = min(self.iteration / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * t)))

    def step(self, f, x):
        self.iteration += 1
        lr = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)

        # Perturbacion Bernoulli (+-1) en todas las dimensiones
        signs = self.rng.choice([-1.0, 1.0], size=self.dim)
        pert = delta * signs

        f_plus  = f(x + pert)
        f_minus = f(x - pert)

        # Gradiente SPSA: escalar / perturbacion por dimension
        scalar = (f_plus - f_minus) / (2.0 * delta)
        g_hat = scalar * signs   # misma formula que SPSA clasico

        x_new = x - lr * g_hat
        return x_new, {"f_new": float(f(x_new)), "n_evals": 2, "lr": lr}


# =============================================================================
# DGE v5 — con lr escalado por dimension
# =============================================================================

class DGEOptimizerV5:
    """
    DGE v5: Adam + lr escalado por 1/sqrt(D) para estabilidad en alta dimension.

    Cambio clave respecto a v4:
      lr_efectivo = lr_base / sqrt(k)
      donde k = log2(D) es el numero de grupos.
    Esto hace que el lr sea invariante al numero de dimensiones evaluadas por paso.
    """

    def __init__(
        self,
        dim: int,
        lr: float = 0.5,
        delta: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        lr_decay: float = 0.01,
        delta_decay: float = 0.1,
        total_steps: int = 1000,
        greedy_w: float = 0.3,
        seed: int | None = None,
    ):
        self.dim = dim
        self.lr0 = lr
        self.delta0 = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr_decay = lr_decay
        self.delta_decay = delta_decay
        self.total_steps = total_steps
        self.greedy_w = greedy_w
        self.rng = np.random.default_rng(seed)

        self.k = max(1, math.ceil(math.log2(dim))) if dim > 1 else 1
        self.group_size = max(1, math.ceil(dim / self.k))

        # Escala de lr: invariante al numero de grupos (y por tanto a D)
        self.lr_scale = 1.0 / math.sqrt(self.k)

        self.m = np.zeros(dim)
        self.v_adam = np.zeros(dim)
        self.iteration = 0

    def _cosine(self, v0, decay):
        t = min(self.iteration / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * t)))

    def _make_groups(self):
        return [self.rng.choice(self.dim, size=self.group_size, replace=False)
                for _ in range(self.k)]

    def step(self, f, x):
        self.iteration += 1
        lr    = self._cosine(self.lr0, self.lr_decay) * self.lr_scale
        delta = self._cosine(self.delta0, self.delta_decay)

        groups = self._make_groups()
        signs  = self.rng.choice([-1.0, 1.0], size=self.dim)

        g = np.zeros(self.dim)
        g_count = np.zeros(self.dim, dtype=np.int32)

        best_strength = -1.0
        best_direction = np.zeros(self.dim)

        for idx in groups:
            pert = np.zeros(self.dim)
            pert[idx] = signs[idx] * delta

            f_plus  = f(x + pert)
            f_minus = f(x - pert)

            scalar_grad = (f_plus - f_minus) / (2.0 * delta)

            g[idx] += scalar_grad * signs[idx]
            g_count[idx] += 1

            strength = abs(scalar_grad)
            if strength > best_strength:
                best_strength = strength
                direction = np.zeros(self.dim)
                direction[idx] = signs[idx]
                d_norm = np.linalg.norm(direction)
                best_direction = -np.sign(scalar_grad) * direction / (d_norm + 1e-12)

        evaluated = g_count > 0
        g[evaluated] /= g_count[evaluated]

        # Adam (lazy: solo dims evaluadas)
        self.m[evaluated] = (self.beta1 * self.m[evaluated]
                             + (1 - self.beta1) * g[evaluated])
        self.v_adam[evaluated] = (self.beta2 * self.v_adam[evaluated]
                                  + (1 - self.beta2) * g[evaluated] ** 2)

        t = self.iteration
        m_hat = self.m / (1 - self.beta1 ** t + 1e-30)
        v_hat = self.v_adam / (1 - self.beta2 ** t + 1e-30)

        adam_update = np.zeros(self.dim)
        adam_update[evaluated] = lr * m_hat[evaluated] / (np.sqrt(v_hat[evaluated]) + self.eps)

        greedy_update = self.greedy_w * lr * best_direction
        x_new = x - adam_update - greedy_update

        return x_new, {
            "f_new": float(f(x_new)),
            "n_evals": 2 * self.k,
            "lr": lr,
            "delta": delta,
        }


# =============================================================================
# BENCHMARK JUSTO: DGE vs SPSA — mismas evaluaciones totales
# =============================================================================

@dataclass
class BenchResult:
    name: str
    evals: list = field(default_factory=list)   # evaluaciones acumuladas
    values: list = field(default_factory=list)  # f(x) correspondiente


def benchmark(f, dim, x0, optimum, total_evals,
              dge_lr=0.5, dge_delta=1e-2, dge_greedy=0.3,
              spsa_lr=0.1, spsa_delta=1e-2,
              seed=0):
    """
    Ejecuta DGE y SPSA con el MISMO presupuesto de evaluaciones totales.
    Registra f(x) a lo largo del tiempo en unidades de evaluaciones,
    no de pasos — comparacion justa.
    """
    # Calcular numero de pasos de cada algoritmo dado el mismo presupuesto
    k = max(1, math.ceil(math.log2(dim)))
    dge_steps  = total_evals // (2 * k)
    spsa_steps = total_evals // 2

    # --- DGE -----------------------------------------------------------------
    dge_opt = DGEOptimizerV5(
        dim=dim, lr=dge_lr, delta=dge_delta, greedy_w=dge_greedy,
        lr_decay=0.01, delta_decay=0.05, total_steps=dge_steps, seed=seed,
    )
    x_dge = x0.copy()
    dge_result = BenchResult("DGE v5")
    evals_dge = 0
    for _ in range(dge_steps):
        x_dge, info = dge_opt.step(f, x_dge)
        evals_dge += 2 * k
        dge_result.evals.append(evals_dge)
        dge_result.values.append(info["f_new"])

    # --- SPSA ----------------------------------------------------------------
    spsa_opt = SPSAOptimizer(
        dim=dim, lr=spsa_lr, delta=spsa_delta,
        lr_decay=0.01, delta_decay=0.05, total_steps=spsa_steps, seed=seed,
    )
    x_spsa = x0.copy()
    spsa_result = BenchResult("SPSA")
    evals_spsa = 0
    for _ in range(spsa_steps):
        x_spsa, info = spsa_opt.step(f, x_spsa)
        evals_spsa += 2
        spsa_result.evals.append(evals_spsa)
        spsa_result.values.append(info["f_new"])

    return dge_result, spsa_result


def print_comparison(name, dim, optimum, dge: BenchResult, spsa: BenchResult):
    """Imprime tabla comparativa en checkpoints de evaluaciones."""
    total = max(dge.evals[-1], spsa.evals[-1])
    checkpoints = [int(total * p) for p in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    print()
    print("=" * 70)
    print(f"BENCHMARK: {name}  |  D={dim}  |  presupuesto={total} evals")
    print(f"  DGE:  {len(dge.evals)} pasos x {total // len(dge.evals) * 2} evals/paso")
    print(f"  SPSA: {len(spsa.evals)} pasos x 2 evals/paso")
    print("-" * 70)
    print(f"  {'Evals':>8}  |  {'DGE gap':>14}  |  {'SPSA gap':>14}  |  {'Ganador'}")
    print("-" * 70)

    def gap_at(result, eval_target):
        # Encuentra el f mas cercano al punto de evaluacion deseado
        for i, e in enumerate(result.evals):
            if e >= eval_target:
                return max(result.values[i] - optimum, 0.0)
        return max(result.values[-1] - optimum, 0.0)

    for cp in checkpoints:
        dge_gap  = gap_at(dge, cp)
        spsa_gap = gap_at(spsa, cp)
        winner = "DGE  <--" if dge_gap < spsa_gap * 0.99 else (
                 "SPSA <--" if spsa_gap < dge_gap * 0.99 else "empate")
        print(f"  {cp:>8}  |  {dge_gap:>14.4e}  |  {spsa_gap:>14.4e}  |  {winner}")

    print("-" * 70)
    dge_final  = max(dge.values[-1]  - optimum, 0.0)
    spsa_final = max(spsa.values[-1] - optimum, 0.0)
    ratio = spsa_final / (dge_final + 1e-30)
    print(f"  FINAL DGE gap:  {dge_final:.4e}    SPSA gap: {spsa_final:.4e}")
    print(f"  Ratio SPSA/DGE: {ratio:.2f}x  ({'DGE GANA' if ratio > 1.05 else 'SPSA GANA' if ratio < 0.95 else 'EMPATE'})")
    print("=" * 70)


if __name__ == "__main__":
    TOTAL_EVALS = 20_000   # presupuesto comun para todos los tests

    # Nota sobre calibracion de SPSA:
    # SPSA actualiza TODAS las D dimensiones con un escalar ruidoso en cada paso.
    # Para evitar divergencia, su lr debe escalarse aproximadamente como 1/D.
    # Usamos lr_spsa = lr_base / D para una comparacion justa y estable.

    # ------------------------------------------------------------------
    # B1: Esfera D=512 (gradiente denso — caso desfavorable para DGE)
    # ------------------------------------------------------------------
    D = 512
    x0 = np.random.default_rng(42).uniform(-5.0, 5.0, D)
    dge, spsa = benchmark(
        f=lambda x: float(np.dot(x, x)),
        dim=D, x0=x0, optimum=0.0, total_evals=TOTAL_EVALS,
        dge_lr=0.5, dge_delta=1e-2, dge_greedy=0.3,
        spsa_lr=0.5/D, spsa_delta=1e-2, seed=0,
    )
    print_comparison("Esfera D=512 (denso)", D, 0.0, dge, spsa)

    # ------------------------------------------------------------------
    # B2: Esfera Sparse 5% (caso favorable DGE — hipotesis central)
    # ------------------------------------------------------------------
    D = 512
    rng = np.random.default_rng(7)
    weights = np.zeros(D)
    active = rng.choice(D, size=D // 20, replace=False)
    weights[active] = 1.0
    x0 = rng.uniform(-5.0, 5.0, D)
    dge, spsa = benchmark(
        f=lambda x: float(np.dot(weights * x, x)),
        dim=D, x0=x0, optimum=0.0, total_evals=TOTAL_EVALS,
        dge_lr=0.5, dge_delta=1e-2, dge_greedy=0.3,
        spsa_lr=0.5/D, spsa_delta=1e-2, seed=1,
    )
    print_comparison("Esfera Sparse 5% activas (D=512)", D, 0.0, dge, spsa)

    # ------------------------------------------------------------------
    # B3: Alta dimension D=65536 (4096x ahorro de evals vs FD)
    # ------------------------------------------------------------------
    D = 65536
    x0 = np.random.default_rng(3).uniform(-1.0, 1.0, D)
    dge, spsa = benchmark(
        f=lambda x: float(np.dot(x, x)),
        dim=D, x0=x0, optimum=0.0, total_evals=TOTAL_EVALS,
        dge_lr=0.5, dge_delta=1e-2, dge_greedy=0.3,
        spsa_lr=0.5/D, spsa_delta=1e-2, seed=2,
    )
    print_comparison("Esfera D=65536 (alta dimension)", D, 0.0, dge, spsa)

    # ------------------------------------------------------------------
    # B4: Rosenbrock D=10 (valle curvo, dimension intermedia)
    # ------------------------------------------------------------------
    D = 10
    def rosenbrock(x):
        return float(sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                         for i in range(len(x) - 1)))

    x0 = np.random.default_rng(5).uniform(-2.0, 2.0, D)
    dge, spsa = benchmark(
        f=rosenbrock,
        dim=D, x0=x0, optimum=0.0, total_evals=TOTAL_EVALS,
        dge_lr=0.2, dge_delta=1e-3, dge_greedy=0.2,
        spsa_lr=0.2/D, spsa_delta=1e-3, seed=3,
    )
    print_comparison("Rosenbrock D=10", D, 0.0, dge, spsa)

    print("\nBenchmark completado.")
