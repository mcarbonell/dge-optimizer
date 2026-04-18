"""
DGE - MNIST v8c: Empujando hacia el 90%
========================================
v8b logró 85.7% test en 100k evals / 59s.
v8c: ¿cuánto más lejos llegamos con más budget y ajuste fino?

Cambios:
  - Total evals: 1_000_000 (10x más, ~10 min en CPU)
  - lr_decay más lento: 0.001 (lr no decae tan rápido)
  - N_TRAIN=5000, N_TEST=1000 (más datos = mejor generalización)
  - Mismo batch fijo por paso (fix crítico de v8b)
"""

import numpy as np
import math
import time
from torchvision import datasets

SEED         = 42
N_TRAIN      = 5_000
N_TEST       = 1_000
BATCH_SIZE   = 256
TOTAL_EVALS  = 1_000_000
LOG_INTERVAL = 100_000

np.random.seed(SEED)

# =============================================================================
# DATOS
# =============================================================================

def load_mnist(n_train=N_TRAIN, n_test=N_TEST):
    full_train = datasets.MNIST('./data', train=True,  download=False)
    full_test  = datasets.MNIST('./data', train=False, download=False)
    X_tr = full_train.data.float().numpy().reshape(-1, 784) / 255.0
    y_tr = full_train.targets.numpy()
    X_te = full_test.data.float().numpy().reshape(-1, 784) / 255.0
    y_te = full_test.targets.numpy()
    X_tr = (X_tr - 0.1307) / 0.3081
    X_te = (X_te - 0.1307) / 0.3081
    rng = np.random.default_rng(SEED)
    tr_idx = rng.choice(len(y_tr), n_train, replace=False)
    te_idx = rng.choice(len(y_te), n_test,  replace=False)
    return (X_tr[tr_idx].astype(np.float32), y_tr[tr_idx],
            X_te[te_idx].astype(np.float32), y_te[te_idx])


# =============================================================================
# RED NEURONAL NUMPY
# =============================================================================

ARCH = (784, 32, 10)

def n_params(arch):
    return sum(arch[i]*arch[i+1]+arch[i+1] for i in range(len(arch)-1))

D = n_params(ARCH)


def forward(X, p):
    h, i = X.astype(np.float32), 0
    for lin, lout in zip(ARCH[:-1], ARCH[1:]):
        sz = lin * lout
        W = p[i:i+sz].reshape(lin, lout); i += sz
        b = p[i:i+lout];                  i += lout
        h = h @ W + b
        if lout != ARCH[-1]:
            h = np.maximum(h, 0)
    return h


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -88, 88))
    return e / (e.sum(1, keepdims=True) + 1e-12)


def loss_batch(Xb, yb, p):
    probs = np.clip(softmax(forward(Xb, p)), 1e-7, 1-1e-7)
    return float(-np.mean(np.log(probs[np.arange(len(yb)), yb])))


def accuracy(X, y, p):
    return float(np.mean(forward(X, p).argmax(1) == y))


# =============================================================================
# DGE (con batch fijo por paso — fix crítico v8b)
# =============================================================================

class DGE:
    def __init__(self, dim, lr=0.5, delta=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, lr_decay=0.001, delta_decay=0.02,
                 steps=10000, greedy_w=0.1, clip_norm=0.05, seed=None):
        self.dim = dim
        self.lr0, self.d0 = lr, delta
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.lr_decay, self.d_decay = lr_decay, delta_decay
        self.steps = steps
        self.gw, self.clip = greedy_w, clip_norm
        self.rng = np.random.default_rng(seed)
        self.k  = max(1, math.ceil(math.log2(dim)))
        self.gs = max(1, math.ceil(dim / self.k))
        self.lrs = 1.0 / math.sqrt(self.k)
        self.m = np.zeros(dim, np.float32)
        self.v = np.zeros(dim, np.float32)
        self.t = 0

    def _cos(self, v0, decay):
        frac = min(self.t / max(self.steps, 1), 1.0)
        return v0 * (decay + (1-decay)*0.5*(1+math.cos(math.pi*frac)))

    def step(self, Xb, yb, x):
        """Un paso DGE con batch fijo (Xb, yb)."""
        self.t += 1
        lr = self._cos(self.lr0, self.lr_decay) * self.lrs
        delta = self._cos(self.d0, self.d_decay)

        groups = [self.rng.choice(self.dim, self.gs, replace=False)
                  for _ in range(self.k)]
        signs = self.rng.choice([-1.,1.], self.dim).astype(np.float32)

        g = np.zeros(self.dim, np.float32)
        gc = np.zeros(self.dim, np.int32)
        bs, bd = -1., np.zeros(self.dim, np.float32)

        for idx in groups:
            pert = np.zeros(self.dim, np.float32)
            pert[idx] = signs[idx] * delta
            # MISMO batch para ambas evaluaciones (fix crítico)
            fp = loss_batch(Xb, yb, x + pert)
            fm = loss_batch(Xb, yb, x - pert)
            sg = (fp - fm) / (2 * delta)
            g[idx] += sg * signs[idx]; gc[idx] += 1
            if abs(sg) > bs:
                bs = abs(sg)
                d = np.zeros(self.dim, np.float32); d[idx] = signs[idx]
                bd = -np.sign(sg) * d / (np.linalg.norm(d) + 1e-12)

        ev = gc > 0; g[ev] /= gc[ev]
        self.m[ev] = self.b1*self.m[ev] + (1-self.b1)*g[ev]
        self.v[ev] = self.b2*self.v[ev] + (1-self.b2)*g[ev]**2
        mh = self.m / (1-self.b1**self.t+1e-30)
        vh = self.v / (1-self.b2**self.t+1e-30)
        upd = np.zeros(self.dim, np.float32)
        upd[ev] = lr * mh[ev] / (np.sqrt(vh[ev]) + self.eps)
        n = np.linalg.norm(upd)
        if n > self.clip: upd *= self.clip / n
        return x - upd - self.gw * lr * bd, 2 * self.k


# =============================================================================
# SPSA (con mismo batch fijo por paso)
# =============================================================================

class SPSA:
    def __init__(self, dim, lr=0.01, delta=1e-3, lr_decay=0.001, delta_decay=0.02,
                 steps=10000, seed=None):
        self.dim = dim
        self.lr0, self.d0 = lr, delta
        self.lr_decay, self.d_decay = lr_decay, delta_decay
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def _cos(self, v0, decay):
        frac = min(self.t / max(self.steps, 1), 1.0)
        return v0 * (decay + (1-decay)*0.5*(1+math.cos(math.pi*frac)))

    def step(self, Xb, yb, x):
        self.t += 1
        lr = self._cos(self.lr0, self.lr_decay)
        delta = self._cos(self.d0, self.d_decay)
        s = self.rng.choice([-1.,1.], self.dim).astype(np.float32)
        pert = delta * s
        sc = (loss_batch(Xb, yb, x+pert) - loss_batch(Xb, yb, x-pert)) / (2*delta)
        return x - lr * sc * s, 2


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def run(name, opt, p0, X_tr, y_tr, X_te, y_te):
    rng_mb = np.random.default_rng(SEED + 1)
    p = p0.copy(); evals = 0
    t0 = time.time(); next_log = LOG_INTERVAL
    is_dge = hasattr(opt, 'k')
    evals_per_step = 2 * opt.k if is_dge else 2
    print(f"\n[{name}]  evals/paso={evals_per_step}  "
          f"pasos_total={TOTAL_EVALS//evals_per_step:,}")
    best_te = 0.0

    while evals < TOTAL_EVALS:
        # Batch fijo para este paso (compartido por fp y fm)
        idx = rng_mb.integers(0, len(y_tr), BATCH_SIZE)
        Xb, yb = X_tr[idx], y_tr[idx]
        p, n = opt.step(Xb, yb, p)
        evals += n

        if evals >= next_log or evals >= TOTAL_EVALS:
            tr = accuracy(X_tr, y_tr, p)
            te = accuracy(X_te, y_te, p)
            l  = loss_batch(Xb, yb, p)
            best_te = max(best_te, te)
            print(f"  {evals:>9}  loss={l:.4f}  train={tr:.1%}  "
                  f"test={te:.1%}  best_test={best_te:.1%}  t={time.time()-t0:.0f}s")
            next_log += LOG_INTERVAL

    return p, best_te


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"MNIST v8c — objetivo: >90% test accuracy sin backprop")
    print(f"Red {ARCH}  D={D:,}  budget={TOTAL_EVALS//1000}k evals")
    print(f"Train={N_TRAIN}  Test={N_TEST}  Batch={BATCH_SIZE}")

    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"  Datos cargados.")

    # Inicializacion He
    rng_init = np.random.default_rng(SEED)
    p0 = np.zeros(D, np.float32)
    i = 0
    for fin, fout in zip(ARCH[:-1], ARCH[1:]):
        sz = fin * fout
        p0[i:i+sz] = rng_init.normal(0, math.sqrt(2.0/fin), sz).astype(np.float32)
        i += sz + fout

    k = math.ceil(math.log2(D))

    # DGE v8c
    dge = DGE(D, lr=0.5, delta=1e-3, lr_decay=0.001, delta_decay=0.02,
              steps=TOTAL_EVALS//(2*k), greedy_w=0.1, clip_norm=0.05,
              seed=SEED+10)
    _, dge_best = run("DGE v8c", dge, p0, X_tr, y_tr, X_te, y_te)

    # SPSA v8c (mismas condiciones justas)
    spsa = SPSA(D, lr=0.01, delta=1e-3, lr_decay=0.001, delta_decay=0.02,
                steps=TOTAL_EVALS//2, seed=SEED+20)
    _, spsa_best = run("SPSA v8c", spsa, p0, X_tr, y_tr, X_te, y_te)

    print(f"\n{'='*55}")
    print(f"RESUMEN FINAL v8c ({TOTAL_EVALS//1000}k evals):")
    print(f"  DGE  best test: {dge_best:.1%}")
    print(f"  SPSA best test: {spsa_best:.1%}")
    print(f"  Referencia Adam+backprop ~98% en misma red")
    print(f"  Referencia v8b DGE: 85.7% (100k evals)")
    print(f"{'='*55}")
