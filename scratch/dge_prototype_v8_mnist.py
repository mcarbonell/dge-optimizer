"""
DGE - Dichotomous Gradient Estimation (Prototype v8)
=====================================================
MNIST sin backpropagation.

Red: 784 -> 128 -> 10  (ReLU + CrossEntropy)
D = 784*128 + 128 + 128*10 + 10 = 101,770 parametros

Estrategia para hacerlo manejable:
  - Subconjunto: 10,000 train / 2,000 test
  - Evaluacion del loss sobre minibatch de 256 muestras (no el dataset completo)
  - k = ceil(log2(D)) = 17  ->  34 evals/paso
  - Presupuesto: 500,000 evaluaciones del minibatch-loss

Comparacion: DGE v7 (con gradient clipping) vs SPSA

Para cargar datos se usa torchvision (ya descargado en ./data).
El forward pass usa PyTorch (eficiencia de numpy no seria suficiente para 784->128->10).
PERO: no se llama a .backward() en ningun momento. Cero autograd.
"""

import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# =============================================================================
# CONFIGURACION
# =============================================================================
SEED         = 42
N_TRAIN      = 3_000     # subconjunto train (rapido en CPU)
N_TEST       = 600       # subconjunto test
BATCH_SIZE   = 128       # minibatch para evaluar el loss
TOTAL_EVALS  = 100_000  # presupuesto total
LOG_INTERVAL = 10_000

torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# DATOS MNIST (usando torchvision, ya descargado)
# =============================================================================

def load_mnist_subset(n_train=N_TRAIN, n_test=N_TEST):
    """Carga MNIST usando los archivos binarios directamente (rapido, sin torchvision lento)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=True,  download=False, transform=transform)
    full_test  = datasets.MNIST('./data', train=False, download=False, transform=transform)

    # DataLoader en batch unico = forma rapida de volcar todo a tensor
    X_tr_all = full_train.data.float().view(-1, 784) / 255.0
    y_tr_all = full_train.targets
    X_te_all = full_test.data.float().view(-1, 784) / 255.0
    y_te_all = full_test.targets

    # Normalizacion manual (equivalente al transform)
    X_tr_all = (X_tr_all - 0.1307) / 0.3081
    X_te_all = (X_te_all - 0.1307) / 0.3081

    rng = np.random.default_rng(SEED)
    tr_idx = rng.choice(len(y_tr_all), size=n_train, replace=False)
    te_idx = rng.choice(len(y_te_all), size=n_test,  replace=False)

    # Convertir a numpy para forward pass rapido
    X_tr = X_tr_all[tr_idx].numpy()
    y_tr = y_tr_all[tr_idx].numpy()
    X_te = X_te_all[te_idx].numpy()
    y_te = y_te_all[te_idx].numpy()
    return X_tr, y_tr, X_te, y_te


# =============================================================================
# RED NEURONAL (forward en PyTorch, sin autograd)
# =============================================================================

# Red reducida para CPU: 784 -> 32 -> 10
ARCH = (784, 32, 10)

def n_params(arch):
    total = 0
    for i in range(len(arch) - 1):
        total += arch[i] * arch[i+1] + arch[i+1]
    return total

D = n_params(ARCH)

def forward_np(X, params):
    """
    Forward pass en numpy puro (mas rapido que PyTorch para batches pequeños).
    X: (N, 784) numpy array
    params: vector plano numpy
    Retorna logits (N, n_out).
    """
    i = 0
    h = X
    for layer_in, layer_out in zip(ARCH[:-1], ARCH[1:]):
        w_size = layer_in * layer_out
        W = params[i:i+w_size].reshape(layer_in, layer_out)
        i += w_size
        b = params[i:i+layer_out]
        i += layer_out
        h = h @ W + b
        if layer_out != ARCH[-1]:
            h = np.maximum(h, 0)   # ReLU
    return h   # logits (N, n_out)


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def minibatch_loss(X_train, y_train, params, rng_mb, batch_size=BATCH_SIZE):
    """Cross-entropy sobre minibatch. Todo numpy, sin PyTorch."""
    idx = rng_mb.integers(0, len(y_train), size=batch_size)
    Xb, yb = X_train[idx], y_train[idx]
    logits = forward_np(Xb, params)
    probs  = softmax_np(logits)
    probs  = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(np.log(probs[np.arange(batch_size), yb])))


def full_accuracy(X, y, params):
    """Accuracy sobre el conjunto completo."""
    logits = forward_np(X, params)
    preds  = logits.argmax(axis=1)
    return float(np.mean(preds == y))


# =============================================================================
# DGE v7 (con gradient clipping) — version compacta
# =============================================================================

class DGEOptimizer:
    def __init__(self, dim, lr=1.0, delta=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, lr_decay=0.01, delta_decay=0.05,
                 total_steps=1000, greedy_w=0.3, clip_norm=1.0, seed=None):
        self.dim = dim
        self.lr0, self.delta0 = lr, delta
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.lr_decay, self.delta_decay = lr_decay, delta_decay
        self.total_steps = total_steps
        self.greedy_w, self.clip_norm = greedy_w, clip_norm
        self.rng = np.random.default_rng(seed)
        self.k = max(1, math.ceil(math.log2(dim)))
        self.group_size = max(1, math.ceil(dim / self.k))
        self.lr_scale = 1.0 / math.sqrt(self.k)
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0

    def _cosine(self, v0, decay):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr    = self._cosine(self.lr0, self.lr_decay) * self.lr_scale
        delta = self._cosine(self.delta0, self.delta_decay)

        groups = [self.rng.choice(self.dim, size=self.group_size, replace=False)
                  for _ in range(self.k)]
        signs = self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)

        g = np.zeros(self.dim, dtype=np.float32)
        g_cnt = np.zeros(self.dim, dtype=np.int32)
        best_s, best_dir = -1.0, np.zeros(self.dim, dtype=np.float32)

        for idx in groups:
            pert = np.zeros(self.dim, dtype=np.float32)
            pert[idx] = signs[idx] * delta
            fp = f(x + pert)
            fm = f(x - pert)
            sg = (fp - fm) / (2.0 * delta)
            g[idx] += sg * signs[idx]
            g_cnt[idx] += 1
            if abs(sg) > best_s:
                best_s = abs(sg)
                d = np.zeros(self.dim, dtype=np.float32)
                d[idx] = signs[idx]
                dn = np.linalg.norm(d)
                best_dir = -np.sign(sg) * d / (dn + 1e-12)

        ev = g_cnt > 0
        g[ev] /= g_cnt[ev]

        self.m[ev] = self.beta1 * self.m[ev] + (1 - self.beta1) * g[ev]
        self.v[ev] = self.beta2 * self.v[ev] + (1 - self.beta2) * g[ev] ** 2

        mh = self.m / (1 - self.beta1 ** self.t + 1e-30)
        vh = self.v / (1 - self.beta2 ** self.t + 1e-30)

        upd = np.zeros(self.dim, dtype=np.float32)
        upd[ev] = lr * mh[ev] / (np.sqrt(vh[ev]) + self.eps)
        un = np.linalg.norm(upd)
        if un > self.clip_norm:
            upd *= self.clip_norm / un

        x_new = x - upd - self.greedy_w * lr * best_dir
        return x_new, 2 * self.k


# =============================================================================
# SPSA — referencia
# =============================================================================

class SPSAOptimizer:
    def __init__(self, dim, lr=0.1, delta=1e-3, lr_decay=0.01, delta_decay=0.05,
                 total_steps=1000, seed=None):
        self.dim = dim
        self.lr0, self.delta0 = lr, delta
        self.lr_decay, self.delta_decay = lr_decay, delta_decay
        self.total_steps = total_steps
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def _cosine(self, v0, decay):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))

    def step(self, f, x):
        self.t += 1
        lr    = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)
        signs = self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
        pert  = (delta * signs).astype(np.float32)
        sc    = (f(x + pert) - f(x - pert)) / (2.0 * delta)
        return x - lr * sc * signs, 2


# =============================================================================
# BENCHMARK
# =============================================================================

def run(name, optimizer, params0, X_train, y_train, X_test, y_test,
        total_evals, evals_per_step):
    rng_mb = np.random.default_rng(SEED + 1)
    params = params0.copy()
    evals  = 0
    steps  = 0
    log_evals = []
    log_train_acc = []
    log_test_acc  = []
    t0 = time.time()

    f = lambda p: minibatch_loss(X_train, y_train, p, rng_mb)

    next_log = LOG_INTERVAL
    print(f"\n  {name}: k={getattr(optimizer,'k',1)}  evals/paso={evals_per_step}")

    while evals < total_evals:
        params, n = optimizer.step(f, params)
        evals += n
        steps += 1

        if evals >= next_log or evals >= total_evals:
            tr_acc = full_accuracy(X_train, y_train, params)
            te_acc = full_accuracy(X_test,  y_test,  params)
            mb_loss = minibatch_loss(X_train, y_train, params, rng_mb)
            elapsed = time.time() - t0
            print(f"    evals={evals:>7}  train={tr_acc:.1%}  test={te_acc:.1%}"
                  f"  loss={mb_loss:.4f}  t={elapsed:.0f}s")
            log_evals.append(evals)
            log_train_acc.append(tr_acc)
            log_test_acc.append(te_acc)
            next_log += LOG_INTERVAL

    return params, log_evals, log_train_acc, log_test_acc


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Cargando MNIST subset ({N_TRAIN} train / {N_TEST} test)...")
    X_train, y_train, X_test, y_test = load_mnist_subset()
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print(f"\nRed: {ARCH}  |  D={D:,} parametros")
    print(f"k = ceil(log2({D})) = {math.ceil(math.log2(D))}  "
          f"->  {2*math.ceil(math.log2(D))} evals/paso DGE")
    print(f"Presupuesto: {TOTAL_EVALS:,} evals")
    print(f"  DGE : ~{TOTAL_EVALS//(2*math.ceil(math.log2(D))):,} pasos")
    print(f"  SPSA: ~{TOTAL_EVALS//2:,} pasos")

    # Inicializacion comun (Xavier para estabilidad)
    rng_init = np.random.default_rng(SEED)
    params0 = np.zeros(D, dtype=np.float32)
    i = 0
    for fan_in, fan_out in zip(ARCH[:-1], ARCH[1:]):
        w_size = fan_in * fan_out
        std = math.sqrt(2.0 / fan_in)   # He init (bueno para ReLU)
        params0[i:i+w_size] = rng_init.normal(0, std, w_size).astype(np.float32)
        i += w_size
        # biases a 0
        i += fan_out

    k = math.ceil(math.log2(D))
    dge_steps_total = TOTAL_EVALS // (2 * k)

    # --- DGE ---
    dge_opt = DGEOptimizer(
        dim=D, lr=2.0, delta=1e-2, beta1=0.9, beta2=0.999,
        lr_decay=0.01, delta_decay=0.05,
        total_steps=dge_steps_total, greedy_w=0.3,
        clip_norm=0.5, seed=SEED + 10,
    )
    p_dge, ev_dge, tr_dge, te_dge = run(
        "DGE v7", dge_opt, params0,
        X_train, y_train, X_test, y_test,
        TOTAL_EVALS, 2 * k,
    )

    # --- SPSA ---
    spsa_opt = SPSAOptimizer(
        dim=D, lr=0.1, delta=1e-2,
        lr_decay=0.01, delta_decay=0.05,
        total_steps=TOTAL_EVALS // 2, seed=SEED + 20,
    )
    p_spsa, ev_spsa, tr_spsa, te_spsa = run(
        "SPSA", spsa_opt, params0,
        X_train, y_train, X_test, y_test,
        TOTAL_EVALS, 2,
    )

    # --- Resumen ---
    print(f"\n{'='*60}")
    print(f"RESUMEN FINAL ({TOTAL_EVALS:,} evals totales):")
    print(f"  {'':20}  {'Train':>8}  {'Test':>8}")
    print(f"  {'DGE v7':20}  {tr_dge[-1]:>8.1%}  {te_dge[-1]:>8.1%}")
    print(f"  {'SPSA':20}  {tr_spsa[-1]:>8.1%}  {te_spsa[-1]:>8.1%}")
    print(f"  {'Random baseline':20}  {'10.0%':>8}  {'10.0%':>8}")
    winner = "DGE" if te_dge[-1] > te_spsa[-1] else (
             "SPSA" if te_spsa[-1] > te_dge[-1] else "EMPATE")
    print(f"\n  Ganador: {winner}")
    print(f"{'='*60}")
