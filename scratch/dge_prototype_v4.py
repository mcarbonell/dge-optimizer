"""
DGE - Denoised Gradient Estimation (Prototype v4)
=====================================================
Mejoras sobre v3:
  - Adam sobre el gradiente EMA por variable (reemplaza el paso EMA simple).
    Adam incorpora momentum (beta1) y varianza adaptativa (beta2), lo que
    resuelve el problema de alta dimension y la oscilacion en valles curvos.
  - Decay del lr y del delta a lo largo del tiempo (schedule coseno).
  - El paso greedy se mantiene como arranque rapido (ponderado por greedy_w).
  - Bias correction de Adam correcto desde el primer paso.

Resultado esperado:
  - Alta dimension (D=65536): convergencia mucho mas rapida por lr adaptativo.
  - Rosenbrock: sin oscilacion gracias al decay de lr y la varianza de Adam.

Referencia: docs/denoised_gradient_estimation_idea.md  (Seccion 7)
"""

import numpy as np
import math


class DGEOptimizerV4:
    """
    DGE v4: Random Group Testing + Adam sobre gradiente acumulado.

    Parametros
    ----------
    dim        : numero de dimensiones.
    lr         : tasa de aprendizaje base para Adam.
    delta      : perturbacion inicial para diferencias centrales.
    ema_alpha  : factor EMA para suavizado secundario (no usado en Adam, solo diagnostico).
    beta1      : momentum Adam (primer momento).
    beta2      : varianza Adam (segundo momento).
    eps        : epsilon Adam para estabilidad numerica.
    lr_decay   : fraccion de lr final respecto al inicial (cosine decay).
                 1.0 = sin decay. 0.1 = lr final es 10% del inicial.
    delta_decay: igual que lr_decay pero para delta.
    total_steps: pasos totales esperados (para el schedule de decay).
    greedy_w   : peso del paso greedy inmediato (0 = solo Adam, 1 = como v3).
    seed       : semilla RNG.
    """

    def __init__(
        self,
        dim: int,
        lr: float = 0.01,
        delta: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        lr_decay: float = 0.1,
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

        # Adam: primer y segundo momento por variable
        self.m = np.zeros(dim)   # media (momentum)
        self.v = np.zeros(dim)   # varianza

        self.iteration = 0

    def _cosine_schedule(self, value0: float, decay_fraction: float) -> float:
        """Devuelve el valor actual segun schedule coseno."""
        t = min(self.iteration / max(self.total_steps, 1), 1.0)
        # Coseno: empieza en value0, termina en value0 * decay_fraction
        factor = decay_fraction + (1.0 - decay_fraction) * 0.5 * (1.0 + math.cos(math.pi * t))
        return value0 * factor

    def _make_groups(self) -> list[np.ndarray]:
        return [self.rng.choice(self.dim, size=self.group_size, replace=False)
                for _ in range(self.k)]

    def step(self, f, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Paso DGE v4 con Adam.
        f : funcion objetivo a MINIMIZAR. f(x) -> float
        x : posicion actual (no modificada in-place).
        Retorna (x_nueva, info).
        """
        self.iteration += 1

        # Schedules actuales
        lr = self._cosine_schedule(self.lr0, self.lr_decay)
        delta = self._cosine_schedule(self.delta0, self.delta_decay)

        groups = self._make_groups()
        signs = self.rng.choice([-1.0, 1.0], size=self.dim)

        # --- Acumular gradiente sparse del paso actual -----------------------
        # g[i] = estimacion del gradiente en la variable i para este paso.
        # Variables no evaluadas este paso conservan 0 (update lazy).
        g = np.zeros(self.dim)
        g_count = np.zeros(self.dim, dtype=np.int32)  # veces evaluada cada var

        best_block_strength = -1.0
        best_block_direction = np.zeros(self.dim)

        for idx in groups:
            pert = np.zeros(self.dim)
            pert[idx] = signs[idx] * delta

            f_plus  = f(x + pert)
            f_minus = f(x - pert)

            # Gradiente escalar centrado del bloque
            scalar_grad = (f_plus - f_minus) / (2.0 * delta)

            # Contribucion al gradiente por variable del bloque
            g[idx] += scalar_grad * signs[idx]
            g_count[idx] += 1

            # Track del mejor bloque para el paso greedy
            strength = abs(scalar_grad)
            if strength > best_block_strength:
                best_block_strength = strength
                direction = np.zeros(self.dim)
                direction[idx] = signs[idx]
                d_norm = np.linalg.norm(direction)
                best_block_direction = -np.sign(scalar_grad) * direction / (d_norm + 1e-12)

        # Promedio de gradiente si la variable aparecio en varios bloques
        evaluated = g_count > 0
        g[evaluated] /= g_count[evaluated]

        # --- Actualizacion Adam (solo las variables evaluadas este paso) -----
        # Para las variables NO evaluadas, sus momentos no cambian (lazy Adam).
        self.m[evaluated] = (
            self.beta1 * self.m[evaluated]
            + (1.0 - self.beta1) * g[evaluated]
        )
        self.v[evaluated] = (
            self.beta2 * self.v[evaluated]
            + (1.0 - self.beta2) * g[evaluated] ** 2
        )

        # Bias correction (solo para las variables actualizadas)
        # Se usa el numero de veces que Adam ha visto esa variable.
        # Aproximacion: usamos self.iteration como proxy global.
        t = self.iteration
        m_hat = self.m / (1.0 - self.beta1 ** t + 1e-30)
        v_hat = self.v / (1.0 - self.beta2 ** t + 1e-30)

        # Paso Adam completo (solo en dims evaluadas para no mover dims con m=v=0)
        adam_update = np.zeros(self.dim)
        adam_update[evaluated] = lr * m_hat[evaluated] / (np.sqrt(v_hat[evaluated]) + self.eps)

        # --- Paso greedy (arranque rapido) -----------------------------------
        greedy_update = self.greedy_w * lr * best_block_direction

        # --- Actualizacion final ---------------------------------------------
        x_new = x - adam_update - greedy_update

        info = {
            "iteration": t,
            "f_new": float(f(x_new)),
            "n_evals": 2 * self.k,
            "lr": lr,
            "delta": delta,
            "adam_update_norm": float(np.linalg.norm(adam_update)),
            "greedy_update_norm": float(np.linalg.norm(greedy_update)),
            "m_norm": float(np.linalg.norm(self.m)),
        }
        return x_new, info


# =============================================================================
# TESTS — mismos que v3 para comparacion directa
# =============================================================================

def run_test(name, f, dim, x0, optimum, n_steps, **kwargs):
    opt = DGEOptimizerV4(dim=dim, total_steps=n_steps, **kwargs)
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
    f_best = f0
    for step in range(n_steps):
        x, info = opt.step(f, x)
        total_evals += 2 * opt.k
        f_cur = info["f_new"]
        f_best = min(f_best, f_cur)
        if step % log_interval == 0 or step == n_steps - 1:
            gap = max(f_cur - optimum, 0.0)
            print(f"  paso {step+1:5d} | f={f_cur:.6e} | gap={gap:.4e} | "
                  f"lr={info['lr']:.2e} | delta={info['delta']:.2e}")

    f_final = f(x)
    gap_final = max(f_final - optimum, 0.0)
    print(f"\n  [FINAL] f={f_final:.6e}  best={f_best:.6e}  gap={gap_final:.4e}")
    print(f"  Evals DGE: {total_evals}  |  FD habria usado: {2*dim*n_steps}")
    return gap_final


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Test 1: Esfera D=512 (gradiente denso)
    # ------------------------------------------------------------------
    D = 512
    x0 = np.random.default_rng(42).uniform(-5.0, 5.0, D)
    run_test(
        "Esfera D=512 (gradiente denso)",
        f=lambda x: float(np.dot(x, x)),
        dim=D, x0=x0, optimum=0.0, n_steps=1000,
        lr=0.5, delta=1e-2, beta1=0.9, beta2=0.999,
        lr_decay=0.01, delta_decay=0.1, greedy_w=0.5, seed=0,
    )

    # ------------------------------------------------------------------
    # Test 2: Esfera Sparse 5% activas (caso favorable DGE)
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
        lr=0.5, delta=1e-2, beta1=0.9, beta2=0.999,
        lr_decay=0.01, delta_decay=0.1, greedy_w=0.5, seed=1,
    )

    # ------------------------------------------------------------------
    # Test 3: Alta dimension D=65536  (el caso problematico de v3)
    # ------------------------------------------------------------------
    D = 65536
    x0 = np.random.default_rng(3).uniform(-1.0, 1.0, D)
    run_test(
        "Esfera D=65536 (alta dimension)",
        f=lambda x: float(np.dot(x, x)),
        dim=D, x0=x0, optimum=0.0, n_steps=500,
        lr=1.0, delta=1e-2, beta1=0.9, beta2=0.999,
        lr_decay=0.01, delta_decay=0.05, greedy_w=0.5, seed=2,
    )

    # ------------------------------------------------------------------
    # Test 4: Rosenbrock D=2 (el caso que oscilaba en v3)
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
        lr=0.1, delta=1e-4, beta1=0.9, beta2=0.99,
        lr_decay=0.001, delta_decay=0.01, greedy_w=0.2, seed=3,
    )

    # ------------------------------------------------------------------
    # Test 5: Ackley D=100 (multimodal — primer test no convexo)
    # ------------------------------------------------------------------
    D = 100
    def ackley(x):
        a, b, c = 20, 0.2, 2 * math.pi
        n = len(x)
        s1 = np.sum(x**2)
        s2 = np.sum(np.cos(c * x))
        return float(-a * math.exp(-b * math.sqrt(s1 / n))
                     - math.exp(s2 / n) + a + math.e)

    x0 = np.random.default_rng(9).uniform(-30.0, 30.0, D)
    run_test(
        "Ackley D=100 (multimodal)",
        f=ackley,
        dim=D, x0=x0, optimum=0.0, n_steps=2000,
        lr=0.5, delta=1e-2, beta1=0.9, beta2=0.999,
        lr_decay=0.01, delta_decay=0.05, greedy_w=0.5, seed=4,
    )

    print("\nTodos los tests completados.")
