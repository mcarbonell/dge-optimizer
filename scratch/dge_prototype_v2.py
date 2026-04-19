"""
DGE - Denoised Gradient Estimation (Prototype v2)
=====================================================
Correccion v2:
  - Paso greedy simplificado: aplica directamente el gradiente del mejor bloque.
  - Paso EMA normalizado correctamente.
  - Sin emojis (compatibilidad Windows cp1252).

Estrategia: Random Group Testing + EMA  (Seccion 7 del whitepaper)
  - Cada paso genera k = ceil(log2(D)) bloques aleatorios de variables.
  - Para cada bloque: estima el gradiente parcial con diferencias centrales.
  - El bloque con estimacion de gradiente de mayor norma da el paso greedy.
  - Todas las evaluaciones alimentan un mapa EMA de gradiente por variable.
  - El update final combina ambos.

Coste por iteracion: 2 * k = 2 * ceil(log2(D))  evaluaciones de f(x).
"""

import numpy as np
import math


class DGEOptimizer:
    """
    DGE v2: Random Group Testing + EMA gradient accumulation.

    Parametros
    ----------
    dim      : numero de dimensiones del problema.
    lr       : tasa de aprendizaje para el paso greedy.
    lr_ema   : tasa de aprendizaje para el paso EMA acumulado.
    delta    : tamano de perturbacion para estimar el gradiente.
    ema_alpha: factor EMA (mayor = mas peso al pasado reciente, mas olvido).
    seed     : semilla para reproducibilidad.
    """

    def __init__(
        self,
        dim: int,
        lr: float = 0.1,
        lr_ema: float = 0.02,
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

        # k grupos por paso = presupuesto logaritmico
        self.k = max(1, math.ceil(math.log2(dim))) if dim > 1 else 1
        # Tamano de cada grupo: ~D/k variables por bloque
        self.group_size = max(1, math.ceil(dim / self.k))

        # Gradiente EMA acumulado: estimacion de nabla_f por variable
        self.ema_grad = np.zeros(dim)
        self.iteration = 0

    def _make_groups(self) -> list[np.ndarray]:
        """Genera k grupos aleatorios sin reemplazo interno (con solapamiento entre grupos)."""
        return [self.rng.choice(self.dim, size=self.group_size, replace=False)
                for _ in range(self.k)]

    def step(self, f, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Realiza un paso DGE.

        f : funcion objetivo a MINIMIZAR.  f(x: ndarray) -> float
        x : posicion actual (no modificada in-place).
        Retorna (x_new, info_dict).
        """
        self.iteration += 1
        groups = self._make_groups()

        # Signos aleatorios por dimension (estilo SPSA, evita cancelaciones simetricas)
        signs = self.rng.choice([-1.0, 1.0], size=self.dim)

        best_block_idx = None
        best_block_norm = -np.inf
        block_grads = []  # gradiente estimado para cada bloque

        for idx in groups:
            # Vector de perturbacion: solo las dims del bloque, con signos aleatorios
            pert = np.zeros(self.dim)
            pert[idx] = signs[idx] * self.delta

            f_plus = f(x + pert)
            f_minus = f(x - pert)

            # Gradiente centrado proyectado sobre el bloque
            scalar_grad = (f_plus - f_minus) / (2.0 * self.delta)

            # Gradiente estimado por variable del bloque (con el signo de la perturbacion)
            grad_block = np.zeros(self.dim)
            grad_block[idx] = scalar_grad * signs[idx]

            block_grads.append((idx, grad_block))

            # Norma del gradiente estimado: que tan grande es la pendiente en este bloque
            block_norm = abs(scalar_grad) * math.sqrt(len(idx))
            if block_norm > best_block_norm:
                best_block_norm = block_norm
                best_block_idx = len(block_grads) - 1

        # --- Actualizacion EMA con TODOS los bloques -------------------------
        # La co-variables en cada bloque actuan como ruido de media cero;
        # el EMA los promedia fuera con el tiempo.
        for idx, grad_block in block_grads:
            self.ema_grad = (
                (1.0 - self.ema_alpha) * self.ema_grad
                + self.ema_alpha * grad_block
            )

        # --- Paso 1: Greedy inmediato (mejor bloque hoy) ---------------------
        _, best_grad_block = block_grads[best_block_idx]
        x_new = x - self.lr * best_grad_block

        # --- Paso 2: Suave sobre gradiente EMA completo ----------------------
        # Solo activo despues de acumular suficiente historia
        if self.iteration > self.k:
            ema_norm = np.linalg.norm(self.ema_grad) + 1e-12
            x_new = x_new - self.lr_ema * self.ema_grad / ema_norm

        f_new = f(x_new)
        info = {
            "iteration": self.iteration,
            "f_before": f(x),
            "f_after": f_new,
            "n_evals": 2 * self.k + 2,   # +2 para f(x) y f(x_new) del report
            "k": self.k,
            "ema_grad_norm": float(np.linalg.norm(self.ema_grad)),
        }
        return x_new, info


# =============================================================================
# UTILIDAD DE TEST
# =============================================================================

def run_test(name, f, dim, x0, optimum, n_steps, lr, lr_ema, delta, ema_alpha, seed=0):
    opt = DGEOptimizer(dim=dim, lr=lr, lr_ema=lr_ema, delta=delta,
                       ema_alpha=ema_alpha, seed=seed)
    x = x0.copy()
    total_evals = 0

    print()
    print("=" * 65)
    print(f"Test: {name}  |  D={dim}  |  pasos={n_steps}")
    print(f"  k={opt.k} grupos/paso  |  evals/paso~{2*opt.k}  "
          f"|  ahorro vs FD: {dim // opt.k}x")
    print("=" * 65)

    log_interval = max(1, n_steps // 10)
    for step in range(n_steps):
        x, info = opt.step(f, x)
        total_evals += 2 * opt.k
        if step % log_interval == 0 or step == n_steps - 1:
            gap = info["f_after"] - optimum
            print(f"  paso {step+1:5d} | f={info['f_after']:+.6f} | "
                  f"gap={gap:.6e} | evals={total_evals}")

    f_final = f(x)
    gap_final = f_final - optimum
    print(f"\n  [FINAL] f={f_final:.6f}  gap={gap_final:.6e}")
    print(f"  Evals DGE: {total_evals}  |  FD habria usado: {2*dim*n_steps}")
    return gap_final, total_evals


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Test 1: Esfera D=512 — gradiente denso (caso desfavorable teorico)
    # ------------------------------------------------------------------
    D = 512
    x0 = np.random.default_rng(42).uniform(-5, 5, D)
    run_test(
        name="Esfera (D=512, gradiente denso)",
        f=lambda x: float(np.dot(x, x)),
        dim=D, x0=x0, optimum=0.0, n_steps=1000,
        lr=0.5, lr_ema=0.05, delta=1e-2, ema_alpha=0.1,
    )

    # ------------------------------------------------------------------
    # Test 2: Esfera Sparse — solo el 5% de dims tiene coeficiente != 0
    #         Caso FAVORABLE: pocas variables dominan el gradiente
    # ------------------------------------------------------------------
    D = 512
    rng = np.random.default_rng(7)
    weights = np.zeros(D)
    active = rng.choice(D, size=D // 20, replace=False)  # 5% = 25 dims
    weights[active] = 1.0
    x0 = rng.uniform(-5, 5, D)
    run_test(
        name="Esfera Sparse 5% dims activas (D=512)",
        f=lambda x: float(np.dot(weights * x, x)),
        dim=D, x0=x0, optimum=0.0, n_steps=500,
        lr=0.5, lr_ema=0.05, delta=1e-2, ema_alpha=0.15,
    )

    # ------------------------------------------------------------------
    # Test 3: Alta dimension D=65536
    #         log2(65536) = 16 -> solo 32 evals/paso vs 131072 de FD
    # ------------------------------------------------------------------
    D = 65536
    x0 = np.random.default_rng(3).uniform(-1, 1, D)
    run_test(
        name="Esfera Alta Dimension (D=65536)",
        f=lambda x: float(np.dot(x, x)),
        dim=D, x0=x0, optimum=0.0, n_steps=500,
        lr=0.8, lr_ema=0.02, delta=5e-3, ema_alpha=0.2,
    )

    # ------------------------------------------------------------------
    # Test 4: Rosenbrock D=2 (valle curvo clasico)
    # ------------------------------------------------------------------
    D = 2
    def rosenbrock(x):
        return float(sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                         for i in range(len(x) - 1)))

    x0 = np.array([-1.5, 0.5])
    run_test(
        name="Rosenbrock (D=2)",
        f=rosenbrock,
        dim=D, x0=x0, optimum=0.0, n_steps=3000,
        lr=0.05, lr_ema=0.01, delta=1e-4, ema_alpha=0.3,
    )

    print("\n\nTodos los tests completados.")
