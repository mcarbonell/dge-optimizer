"""
DGE - Denoised Gradient Estimation (Prototype v1)
=====================================================
Implementación del algoritmo descrito en:
  docs/denoised_gradient_estimation_idea.md  (Sección 7)

Estrategia: Random Group Testing + EMA
  - Cada paso genera k = ceil(log2(D)) bloques aleatorios de variables.
  - Evalúa f(x + δ·mask) y f(x - δ·mask) para cada bloque.
  - El bloque con mayor mejora da el paso inmediato (explotación).
  - TODAS las evaluaciones alimentan un mapa EMA de sensibilidad por variable
    (exploración acumulada).
  - El update final combina: paso greedy (mejor bloque hoy) +
    paso suave sobre el gradiente EMA completo.

Coste por iteración: 2 * k = 2 * ceil(log2(D))  evaluaciones de f(x).
"""

import numpy as np
import math


class DGEOptimizer:
    """
    Optimizador DGE (Random Group Testing + EMA).

    Parámetros
    ----------
    dim : int
        Número de dimensiones del problema.
    lr : float
        Tasa de aprendizaje para el paso greedy (mejor bloque).
    lr_ema : float
        Tasa de aprendizaje para el paso suave del gradiente EMA.
    delta : float
        Tamaño de perturbación para estimar la sensibilidad.
    ema_alpha : float
        Factor de suavizado EMA. Más alto → más olvido del pasado.
    group_size : int | None
        Tamaño de cada bloque. Si None, usa ceil(D / k) automáticamente.
    seed : int | None
        Semilla para reproducibilidad.
    """

    def __init__(
        self,
        dim: int,
        lr: float = 0.05,
        lr_ema: float = 0.01,
        delta: float = 1e-3,
        ema_alpha: float = 0.1,
        group_size: int | None = None,
        seed: int | None = None,
    ):
        self.dim = dim
        self.lr = lr
        self.lr_ema = lr_ema
        self.delta = delta
        self.ema_alpha = ema_alpha
        self.rng = np.random.default_rng(seed)

        self.k = math.ceil(math.log2(dim)) if dim > 1 else 1
        self.group_size = group_size or max(1, math.ceil(dim / self.k))

        # Mapa EMA de sensibilidad por variable: estimación acumulada del |gradiente|
        self.ema_sensitivity = np.zeros(dim)
        # Mapa EMA del signo del gradiente por variable
        self.ema_grad = np.zeros(dim)
        # Contador de cuántas veces ha sido evaluada cada variable
        self.eval_count = np.zeros(dim, dtype=np.int32)

        self.iteration = 0

    def _random_groups(self) -> list[np.ndarray]:
        """Genera k grupos aleatorios con solapamiento permitido."""
        groups = []
        for _ in range(self.k):
            idx = self.rng.choice(self.dim, size=self.group_size, replace=False)
            groups.append(idx)
        return groups

    def step(self, f, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Realiza un paso de optimización.

        Parámetros
        ----------
        f : callable
            Función objetivo a MINIMIZAR. Firma: f(x: np.ndarray) -> float.
        x : np.ndarray
            Posición actual. No se modifica in-place; se devuelve la nueva posición.

        Retorna
        -------
        x_new : np.ndarray
            Nueva posición tras el paso.
        info : dict
            Métricas de diagnóstico del paso.
        """
        self.iteration += 1
        groups = self._random_groups()

        best_improvement = -np.inf
        best_direction = np.zeros(self.dim)
        f_current = f(x)

        # Signos aleatorios para cada variable (evitar cancelaciones fijas)
        signs = self.rng.choice([-1.0, 1.0], size=self.dim)

        for idx in groups:
            # Construcción del vector de perturbación del bloque
            pert = np.zeros(self.dim)
            pert[idx] = signs[idx] * self.delta

            f_plus = f(x + pert)
            f_minus = f(x - pert)

            raw_diff = f_minus - f_plus          # >0 si mover en +pert mejora
            sensitivity = abs(f_plus - f_minus)  # señal cruda del bloque

            # --- Actualización EMA por variable -------------------------------
            # Sensibilidad individual estimada: señal del bloque como proxy
            # Las co-variables actúan como ruido → se cancelan con el tiempo.
            grad_estimate_block = (f_plus - f_minus) / (2.0 * self.delta)
            grad_per_var = grad_estimate_block * signs[idx]  # con signo correcto

            self.ema_sensitivity[idx] = (
                (1 - self.ema_alpha) * self.ema_sensitivity[idx]
                + self.ema_alpha * sensitivity / max(len(idx), 1)
            )
            self.ema_grad[idx] = (
                (1 - self.ema_alpha) * self.ema_grad[idx]
                + self.ema_alpha * grad_per_var
            )
            self.eval_count[idx] += 1
            # ------------------------------------------------------------------

            # Selección del mejor bloque (explotación greedy)
            if raw_diff > best_improvement:
                best_improvement = raw_diff
                direction = np.zeros(self.dim)
                direction[idx] = signs[idx]
                best_direction = direction

        # --- Paso 1: Greedy (mejor bloque de hoy) ----------------------------
        x_new = x - self.lr * best_direction * self.delta * np.sign(best_improvement + 1e-12)

        # --- Paso 2: Suave sobre gradiente EMA completo ----------------------
        # Solo aplicamos si tenemos suficiente historia acumulada
        if self.iteration > self.k * 2:
            ema_norm = np.linalg.norm(self.ema_grad) + 1e-12
            x_new = x_new - self.lr_ema * self.ema_grad / ema_norm

        info = {
            "iteration": self.iteration,
            "f_current": f_current,
            "best_block_improvement": best_improvement,
            "n_evals": 2 * self.k,
            "ema_sensitivity_max": self.ema_sensitivity.max(),
            "ema_grad_norm": np.linalg.norm(self.ema_grad),
        }
        return x_new, info


# =============================================================================
# TESTS
# =============================================================================

def run_test(name: str, f, dim: int, x0: np.ndarray, optimum: float,
             n_steps: int, optimizer: DGEOptimizer):
    """Ejecuta la optimización y muestra resultados."""
    x = x0.copy()
    total_evals = 0
    print(f"\n{'='*60}")
    print(f"Test: {name}  |  D={dim}  |  pasos={n_steps}")
    print(f"  k (grupos/paso) = {optimizer.k}  |  evals/paso = {2*optimizer.k}")
    print(f"  Evals equivalentes FD = {2*dim}  (ahorro {2*dim/(2*optimizer.k):.1f}x)")
    print(f"{'='*60}")

    for step in range(n_steps):
        x, info = optimizer.step(f, x)
        total_evals += info["n_evals"]
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            gap = info["f_current"] - optimum
            print(f"  paso {step+4:5d} | f={info['f_current']:+.6f} | "
                  f"gap={gap:.6f} | evals_totales={total_evals}")

    f_final = f(x)
    gap_final = f_final - optimum
    print(f"\n  Resultado final: f={f_final:.6f}  gap={gap_final:.6f}")
    print(f"  Evaluaciones totales: {total_evals}  "
          f"(FD equivalente habría usado: {2*dim*n_steps})")


if __name__ == "__main__":
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Test 1: Esfera (paraboloide) — gradiente denso, caso teóricamente
    #         desfavorable para DGE (todas las dims contribuyen igual)
    # ------------------------------------------------------------------
    D = 512
    sphere = lambda x: float(np.sum(x**2))
    x0 = np.random.uniform(-5, 5, D)
    opt = DGEOptimizer(dim=D, lr=0.3, lr_ema=0.05, delta=1e-2,
                       ema_alpha=0.15, seed=0)
    run_test("Esfera (D=512)", sphere, D, x0, optimum=0.0, n_steps=500, optimizer=opt)

    # ------------------------------------------------------------------
    # Test 2: Esfera sparse — solo 5% de dims contribuyen
    #         Caso teóricamente FAVORABLE para DGE
    # ------------------------------------------------------------------
    D = 512
    active = np.random.choice(D, size=max(1, D//20), replace=False)  # 5% activas
    def sparse_sphere(x):
        return float(np.sum(x[active]**2))

    x0 = np.random.uniform(-5, 5, D)
    opt2 = DGEOptimizer(dim=D, lr=0.3, lr_ema=0.05, delta=1e-2,
                        ema_alpha=0.15, seed=1)
    run_test("Esfera Sparse 5% activas (D=512)", sparse_sphere, D, x0,
             optimum=0.0, n_steps=300, optimizer=opt2)

    # ------------------------------------------------------------------
    # Test 3: Alta dimensión — D=65536  (¡log2(65536) = 16 evals/paso!)
    # ------------------------------------------------------------------
    D = 65536
    sphere_hd = lambda x: float(np.sum(x**2))
    x0 = np.random.uniform(-1, 1, D)
    opt3 = DGEOptimizer(dim=D, lr=0.5, lr_ema=0.02, delta=1e-2,
                        ema_alpha=0.2, seed=2)
    run_test("Esfera Alta Dimensión (D=65536)", sphere_hd, D, x0,
             optimum=0.0, n_steps=300, optimizer=opt3)

    # ------------------------------------------------------------------
    # Test 4: Rosenbrock 2D — función clásica en valle curvo,
    #         mínimo en (1,1), f=0. Para ver si DGE navega valles.
    # ------------------------------------------------------------------
    D = 2
    def rosenbrock(x):
        return float(sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2
                         for i in range(len(x)-1)))

    x0 = np.array([-1.5, 0.5])
    opt4 = DGEOptimizer(dim=D, lr=0.1, lr_ema=0.02, delta=1e-3,
                        ema_alpha=0.3, seed=3)
    run_test("Rosenbrock (D=2)", rosenbrock, D, x0,
             optimum=0.0, n_steps=2000, optimizer=opt4)

    print("\n\n✅ Todos los tests completados.")
