"""
DGE - Dichotomous Gradient Estimation (Prototype v6)
=====================================================
Primera prueba en Machine Learning real: Red XOR sin backprop.

Red neuronal: 2 entradas -> 4 neuronas ocultas -> 1 salida
Activacion: sigmoide en todas las capas
Loss: Binary Cross-Entropy

Entrenamiento EXCLUSIVAMENTE con DGE como estimador de gradiente.
Sin PyTorch. Sin autograd. Sin backpropagation.

Comparacion justa contra SPSA con el mismo presupuesto de evaluaciones.

Objetivo: validar que DGE puede entrenar una red neuronal real
en un paisaje no convexo, cumpliendo la promesa del whitepaper.
"""

import numpy as np
import math


# =============================================================================
# RED NEURONAL MINIMA (solo numpy, sin frameworks)
# =============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class TinyNet:
    """
    Red neuronal 2->hidden->1 serializable como vector plano de pesos.
    Compatible con cualquier optimizador black-box: solo necesita f(pesos).
    """

    def __init__(self, n_hidden: int = 4):
        self.n_in = 2
        self.n_hidden = n_hidden
        self.n_out = 1
        # Indices de los pesos en el vector plano
        self.w1_end = n_in * n_hidden      if (n_in := 2) else 0
        self.b1_end = self.w1_end + n_hidden
        self.w2_end = self.b1_end + n_hidden * 1
        self.b2_end = self.w2_end + 1
        self.n_params = self.b2_end

    def forward(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Forward pass. X: (N, 2), params: vector plano. Retorna (N,)."""
        W1 = params[:self.w1_end].reshape(self.n_in, self.n_hidden)
        b1 = params[self.w1_end:self.b1_end]
        W2 = params[self.b1_end:self.w2_end].reshape(self.n_hidden, self.n_out)
        b2 = params[self.w2_end:self.b2_end]

        h = sigmoid(X @ W1 + b1)          # (N, n_hidden)
        out = sigmoid(h @ W2 + b2)        # (N, 1)
        return out.ravel()                 # (N,)

    def loss(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        pred = self.forward(X, params)
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))

    def accuracy(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
        pred = self.forward(X, params) > 0.5
        return float(np.mean(pred == y))


# =============================================================================
# DATOS XOR
# =============================================================================

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([0, 1, 1, 0], dtype=float)


# =============================================================================
# DGE v5 (copiado de v5 para ser autocontenido)
# =============================================================================

class DGEOptimizerV5:
    def __init__(self, dim, lr=0.5, delta=1e-2, beta1=0.9, beta2=0.999,
                 eps=1e-8, lr_decay=0.01, delta_decay=0.1,
                 total_steps=1000, greedy_w=0.3, seed=None):
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
        self.lr_scale = 1.0 / math.sqrt(self.k)
        self.m = np.zeros(dim)
        self.v_adam = np.zeros(dim)
        self.iteration = 0

    def _cosine(self, v0, decay):
        t = min(self.iteration / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * t)))

    def step(self, f, x):
        self.iteration += 1
        lr = self._cosine(self.lr0, self.lr_decay) * self.lr_scale
        delta = self._cosine(self.delta0, self.delta_decay)
        groups = [self.rng.choice(self.dim, size=self.group_size, replace=False)
                  for _ in range(self.k)]
        signs = self.rng.choice([-1.0, 1.0], size=self.dim)
        g = np.zeros(self.dim)
        g_count = np.zeros(self.dim, dtype=np.int32)
        best_strength = -1.0
        best_direction = np.zeros(self.dim)

        for idx in groups:
            pert = np.zeros(self.dim)
            pert[idx] = signs[idx] * delta
            f_plus = f(x + pert)
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
        self.m[evaluated] = self.beta1 * self.m[evaluated] + (1 - self.beta1) * g[evaluated]
        self.v_adam[evaluated] = self.beta2 * self.v_adam[evaluated] + (1 - self.beta2) * g[evaluated] ** 2
        t = self.iteration
        m_hat = self.m / (1 - self.beta1 ** t + 1e-30)
        v_hat = self.v_adam / (1 - self.beta2 ** t + 1e-30)
        adam_update = np.zeros(self.dim)
        adam_update[evaluated] = lr * m_hat[evaluated] / (np.sqrt(v_hat[evaluated]) + self.eps)
        greedy_update = self.greedy_w * lr * best_direction
        x_new = x - adam_update - greedy_update
        return x_new, {"f_new": float(f(x_new)), "n_evals": 2 * self.k}


# =============================================================================
# SPSA (referencia)
# =============================================================================

class SPSAOptimizer:
    def __init__(self, dim, lr=0.1, delta=1e-2, lr_decay=0.1, delta_decay=0.1,
                 total_steps=1000, seed=None):
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
        signs = self.rng.choice([-1.0, 1.0], size=self.dim)
        pert = delta * signs
        f_plus = f(x + pert)
        f_minus = f(x - pert)
        scalar = (f_plus - f_minus) / (2.0 * delta)
        g_hat = scalar * signs
        x_new = x - lr * g_hat
        return x_new, {"f_new": float(f(x_new)), "n_evals": 2}


# =============================================================================
# ENTRENAMIENTO Y COMPARACION
# =============================================================================

def train(optimizer_name, optimizer, loss_fn, params0, total_evals,
          evals_per_step, X, y, net, log_every_evals=500):
    """Bucle de entrenamiento universal para cualquier optimizador black-box."""
    params = params0.copy()
    evals = 0
    results = []  # (evals, loss, accuracy)

    while evals < total_evals:
        params, info = optimizer.step(loss_fn, params)
        evals += evals_per_step
        if evals % log_every_evals < evals_per_step or evals >= total_evals:
            loss = net.loss(X, y, params)
            acc = net.accuracy(X, y, params)
            results.append((evals, loss, acc))

    return params, results


def print_training_log(name, results, X, y, net, params_final):
    print(f"\n{'='*60}")
    print(f"Optimizador: {name}")
    print(f"{'='*60}")
    print(f"  {'Evals':>8}  |  {'Loss':>10}  |  {'Accuracy':>10}")
    print(f"  {'-'*38}")
    for evals, loss, acc in results:
        print(f"  {evals:>8}  |  {loss:>10.6f}  |  {acc:>10.1%}")

    # Tabla de predicciones finales
    final_loss = net.loss(X, y, params_final)
    final_acc  = net.accuracy(X, y, params_final)
    preds = net.forward(X, params_final)
    print(f"\n  Resultado final: loss={final_loss:.6f}  accuracy={final_acc:.1%}")
    print(f"\n  Predicciones (umbral 0.5):")
    for i, (xi, yi, pi) in enumerate(zip(X, y, preds)):
        status = "OK" if (pi > 0.5) == (yi > 0.5) else "FAIL"
        print(f"    [{int(xi[0])}, {int(xi[1])}] -> y={int(yi)}  pred={pi:.4f}  [{status}]")
    return final_loss, final_acc


if __name__ == "__main__":
    TOTAL_EVALS = 50_000   # presupuesto: 50k evaluaciones del loss
    N_RUNS = 5             # multiples semillas para resultados estadisticamente robustos
    N_HIDDEN = 4

    net = TinyNet(n_hidden=N_HIDDEN)
    D = net.n_params
    print(f"Red XOR: 2->{N_HIDDEN}->1  |  D={D} parametros  |  presupuesto={TOTAL_EVALS} evals")

    loss_fn = lambda params: net.loss(XOR_X, XOR_Y, params)

    dge_k = max(1, math.ceil(math.log2(D)))
    dge_evals_per_step = 2 * dge_k
    spsa_evals_per_step = 2
    dge_steps  = TOTAL_EVALS // dge_evals_per_step
    spsa_steps = TOTAL_EVALS // spsa_evals_per_step
    print(f"  DGE:  k={dge_k}  |  evals/paso={dge_evals_per_step}  |  pasos={dge_steps}")
    print(f"  SPSA: evals/paso=2  |  pasos={spsa_steps}")

    dge_wins = 0
    spsa_wins = 0
    dge_losses_all = []
    spsa_losses_all = []

    for run in range(N_RUNS):
        seed = run * 100
        # Inicializacion aleatoria de pesos (igual para ambos)
        params0 = np.random.default_rng(seed).normal(0, 0.5, D)

        # --- DGE ---
        dge_opt = DGEOptimizerV5(
            dim=D, lr=0.8, delta=1e-3, beta1=0.9, beta2=0.999,
            lr_decay=0.005, delta_decay=0.01,
            total_steps=dge_steps, greedy_w=0.4, seed=seed + 1,
        )
        _, dge_results = train("DGE", dge_opt, loss_fn, params0,
                               TOTAL_EVALS, dge_evals_per_step,
                               XOR_X, XOR_Y, net)

        # --- SPSA ---
        spsa_opt = SPSAOptimizer(
            dim=D, lr=0.5 / D, delta=1e-3,
            lr_decay=0.005, delta_decay=0.01,
            total_steps=spsa_steps, seed=seed + 2,
        )
        _, spsa_results = train("SPSA", spsa_opt, loss_fn, params0,
                                TOTAL_EVALS, spsa_evals_per_step,
                                XOR_X, XOR_Y, net)

        dge_final_loss  = dge_results[-1][1]
        spsa_final_loss = spsa_results[-1][1]
        dge_losses_all.append(dge_final_loss)
        spsa_losses_all.append(spsa_final_loss)

        winner = "DGE" if dge_final_loss < spsa_final_loss else "SPSA"
        if winner == "DGE":
            dge_wins += 1
        else:
            spsa_wins += 1

        print(f"\n  Run {run+1}/{N_RUNS}  seed={seed}"
              f"  DGE_loss={dge_final_loss:.6f}"
              f"  SPSA_loss={spsa_final_loss:.6f}"
              f"  -> {winner}")

    # Mostrar en detalle la ultima run
    params0 = np.random.default_rng(seed).normal(0, 0.5, D)
    dge_opt = DGEOptimizerV5(
        dim=D, lr=0.8, delta=1e-3, beta1=0.9, beta2=0.999,
        lr_decay=0.005, delta_decay=0.01,
        total_steps=dge_steps, greedy_w=0.4, seed=seed + 1,
    )
    spsa_opt = SPSAOptimizer(
        dim=D, lr=0.5 / D, delta=1e-3,
        lr_decay=0.005, delta_decay=0.01,
        total_steps=spsa_steps, seed=seed + 2,
    )
    params_dge,  dge_res  = train("DGE",  dge_opt,  loss_fn, params0,
                                  TOTAL_EVALS, dge_evals_per_step,
                                  XOR_X, XOR_Y, net, log_every_evals=5000)
    params_spsa, spsa_res = train("SPSA", spsa_opt, loss_fn, params0,
                                  TOTAL_EVALS, spsa_evals_per_step,
                                  XOR_X, XOR_Y, net, log_every_evals=5000)

    print_training_log("DGE v5",  dge_res,  XOR_X, XOR_Y, net, params_dge)
    print_training_log("SPSA", spsa_res, XOR_X, XOR_Y, net, params_spsa)

    print(f"\n{'='*60}")
    print(f"RESUMEN ({N_RUNS} runs):")
    print(f"  DGE  gana: {dge_wins}/{N_RUNS}  "
          f"loss medio: {np.mean(dge_losses_all):.6f} +/- {np.std(dge_losses_all):.6f}")
    print(f"  SPSA gana: {spsa_wins}/{N_RUNS}  "
          f"loss medio: {np.mean(spsa_losses_all):.6f} +/- {np.std(spsa_losses_all):.6f}")
    print(f"{'='*60}")
    print("\nv6 completado.")
