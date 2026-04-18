"""
DGE - Dichotomous Gradient Estimation (Prototype v3)
=====================================================
Correccion v3:
  - Paso greedy NORMALIZADO: el tamano del paso es siempre lr, solo la
    DIRECCION viene del mejor bloque. Esto evita la divergencia.
  - EMA del gradiente normalizado (unitario) para estabilidad.
  - Warm-up del lr_ema: solo activo tras k*5 pasos de acumulacion.

Referencia: docs/dichotomous_gradient_estimation_idea.md  (Seccion 7)
"""

import numpy as np
import math


class DGEOptimizer:
    """
    DGE v3: pasos normalizados + EMA de gradiente.

    Parametros
    ----------
    dim       : numero de dimensiones.
    lr        : tamano del paso greedy (en unidades de espacio, no de gradiente).
    lr_ema    : tamano del paso EMA (fraccion del espacio).
    delta     : perturbacion para diferencias centrales.
    ema_alpha : factor EMA (0=memoria infinita, 1=sin memoria).
    seed      : semilla RNG.
    """

    def __init__(
        self,
        dim: int,
        lr: float = 0.05,
        lr_ema: float = 0.01,
        delta: float = 1e-3,
        ema_alpha: float = 0.1,
        seed: int | None = None,
    ):
        self.dim = dim
        self.lr = lr
        self.lr_ema = lr_ema
        self.delta = delta
        self.ema_alpha = ema_alpha
        self.rng = np.random.default_rng(seed)

        self.k = max(1, math.ceil(math.log2(dim))) if dim > 1 else 1
        self.group_size = max(1, math.ceil(dim / self.k))

        # EMA del gradiente normalizado por variable
        self.ema_grad = np.zeros(dim)
        self.iteration = 0

    def _make_groups(self) -> list[np.ndarray]:
        return [self.rng.choice(self.dim, size=self.group_size, replace=False)
                for _ in range(self.k)]

    def step(self, f, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Paso DGE normalizado.
        f : funcion objetivo a MINIMIZAR. f(x) -> float
        x : posicion actual.
        Retorna (x_nueva, info).
        """
        self.iteration += 1
        groups = self._make_groups()
        signs = self.rng.choice([-1.0, 1.0], size=self.dim)

        best_norm = -1.0
        best_direction = None           # vector unitario de la mejor direccion

        for idx in groups:
            # Perturbacion centrada en el bloque
            pert = np.zeros(self.dim)
            pert[idx] = signs[idx] * self.delta

            f_plus  = f(x + pert)
            f_minus = f(x - pert)

            # Gradiente escalar del bloque (diferencias centrales)
            scalar_grad = (f_plus - f_minus) / (2.0 * self.delta)

            # Direccion de descenso del bloque (vector unitario)
            direction = np.zeros(self.dim)
            direction[idx] = signs[idx]
            norm_dir = np.linalg.norm(direction[idx])
            if norm_dir > 0:
                direction[idx] /= norm_dir

            # La "fuerza" del bloque es |pendiente| (positivo = mejora potencial)
            block_strength = abs(scalar_grad)

            if block_strength > best_norm:
                best_norm = block_strength
                # Direccion de descenso: opuesta al gradiente
                best_direction = -np.sign(scalar_grad) * direction

            # --- Actualizar EMA con gradiente direccional normalizado ---------
            grad_contrib = np.zeros(self.dim)
            grad_contrib[idx] = scalar_grad * signs[idx] / (norm_dir + 1e-12)
            self.ema_grad = (
                (1.0 - self.ema_alpha) * self.ema_grad
                + self.ema_alpha * grad_contrib
            )

        # --- Paso 1: Greedy (paso de tamano lr en la mejor direccion hoy) ----
        if best_direction is not None:
            x_new = x + self.lr * best_direction
        else:
            x_new = x.copy()

        # --- Paso 2: Suave EMA (descenso suave sobre gradiente acumulado) ----
        if self.iteration > self.k * 5:
            ema_norm = np.linalg.norm(self.ema_grad) + 1e-12
            x_new = x_new - self.lr_ema * self.ema_grad / ema_norm

        info = {
            "iteration": self.iteration,
            "f_new": float(f(x_new)),
            "n_evals": 2 * self.k,
            "ema_grad_norm": float(np.linalg.norm(self.ema_grad)),
        }
        return x_new, info


# =============================================================================
# TESTS
# =============================================================================

def run_test(name, f, dim, x0, optimum, n_steps, **kwargs):
    opt = DGEOptimizer(dim=dim, **kwargs)
    x = x0.copy()
    total_evals = 0
    f0 = f(x)

    print()
    print("=" * 65)
    print(f"Test: {name}  |  D={dim}  |  pasos={n_steps}")
    print(f"  k={opt.k}  |  evals/paso={2*opt.k}  "
          f"|  ahorro vs FD: {dim // max(opt.k,1)}x")
    print(f"  f0 = {f0:.6e}")
    print("=" * 65)

    log_interval = max(1, n_steps // 10)
    f_prev = f0
    for step in range(n_steps):
        x, info = opt.step(f, x)
        total_evals += 2 * opt.k
        f_cur = info["f_new"]
        if step % log_interval == 0 or step == n_steps - 1:
            gap = max(f_cur - optimum, 0.0)
            pct = (f_prev - f_cur) / (abs(f_prev) + 1e-30) * 100
            print(f"  paso {step+1:5d} | f={f_cur:.6e} | gap={gap:.4e} | "
                  f"mejora={pct:+.2f}%")
        f_prev = f_cur

    f_final = f(x)
    gap_final = max(f_final - optimum, 0.0)
    print(f"\n  [FINAL] f={f_final:.6e}  gap={gap_final:.4e}  "
          f"total_evals={total_evals}")
    print(f"  FD habria usado: {2*dim*n_steps} evals")
    return gap_final


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Test 1: Esfera D=512 — gradiente denso (caso desfavorable en teoria)
    # ------------------------------------------------------------------
    D = 512
    x0 = np.random.default_rng(42).uniform(-5.0, 5.0, D)
    run_test(
        "Esfera D=512 (gradiente denso)",
        f=lambda x: float(np.dot(x, x)),
        dim=D, x0=x0, optimum=0.0, n_steps=1000,
        lr=0.15, lr_ema=0.05, delta=1e-3, ema_alpha=0.1, seed=0,
    )

    # ------------------------------------------------------------------
    # Test 2: Esfera Sparse — 5% de dims activas (caso FAVORABLE)
    # ------------------------------------------------------------------
    D = 512
    rng = np.random.default_rng(7)
    weights = np.zeros(D)
    active = rng.choice(D, size=D // 20, replace=False)
    weights[active] = 1.0
    x0 = rng.uniform(-5.0, 5.0, D)
    run_test(
        "Esfera Sparse 5% activas (D=512)",
        f=lambda x: float(np.dot(weights * x, x)),
        dim=D, x0=x0, optimum=0.0, n_steps=500,
        lr=0.15, lr_ema=0.05, delta=1e-3, ema_alpha=0.15, seed=1,
    )

    # ------------------------------------------------------------------
    # Test 3: Alta dimension D=65536  (solo 32 evals/paso vs 131072 FD)
    # ------------------------------------------------------------------
    D = 65536
    x0 = np.random.default_rng(3).uniform(-1.0, 1.0, D)
    run_test(
        "Esfera D=65536 (alta dimension)",
        f=lambda x: float(np.dot(x, x)),
        dim=D, x0=x0, optimum=0.0, n_steps=300,
        lr=0.2, lr_ema=0.02, delta=1e-3, ema_alpha=0.2, seed=2,
    )

    # ------------------------------------------------------------------
    # Test 4: Rosenbrock D=2
    # ------------------------------------------------------------------
    D = 2
    def rosenbrock(x):
        return float(sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                         for i in range(len(x) - 1)))

    x0 = np.array([-1.5, 0.5])
    run_test(
        "Rosenbrock (D=2)",
        f=rosenbrock,
        dim=D, x0=x0, optimum=0.0, n_steps=5000,
        lr=0.02, lr_ema=0.005, delta=1e-4, ema_alpha=0.3, seed=3,
    )

    print("\nTodos los tests completados.")
