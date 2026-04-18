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
BATCH_SIZE   = 256       # minibatch para evaluar el loss
TOTAL_EVALS  = 100_000  # presupuesto total
LOG_INTERVAL = 10_000

torch.manual_seed(SEED)
np.random.seed(SEED)

def load_mnist_subset(n_train=N_TRAIN, n_test=N_TEST):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    full_test  = datasets.MNIST('./data', train=False, download=True, transform=transform)

    X_tr_all = full_train.data.float().view(-1, 784) / 255.0
    y_tr_all = full_train.targets
    X_te_all = full_test.data.float().view(-1, 784) / 255.0
    y_te_all = full_test.targets

    X_tr_all = (X_tr_all - 0.1307) / 0.3081
    X_te_all = (X_te_all - 0.1307) / 0.3081

    rng = np.random.default_rng(SEED)
    tr_idx = rng.choice(len(y_tr_all), size=n_train, replace=False)
    te_idx = rng.choice(len(y_te_all), size=n_test,  replace=False)

    X_tr = X_tr_all[tr_idx].numpy()
    y_tr = y_tr_all[tr_idx].numpy()
    X_te = X_te_all[te_idx].numpy()
    y_te = y_te_all[te_idx].numpy()
    return X_tr, y_tr, X_te, y_te

ARCH = (784, 32, 10)

def n_params(arch):
    total = 0
    for i in range(len(arch) - 1):
        total += arch[i] * arch[i+1] + arch[i+1]
    return total

D = n_params(ARCH)

def forward_np(X, params):
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

def loss_on_batch(Xb, yb, params):
    logits = forward_np(Xb, params)
    probs  = softmax_np(logits)
    probs  = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(np.log(probs[np.arange(len(yb)), yb])))

def full_accuracy(X, y, params):
    logits = forward_np(X, params)
    preds  = logits.argmax(axis=1)
    return float(np.mean(preds == y))

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dge.optimizer import DGEOptimizer

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

    next_log = LOG_INTERVAL
    print(f"\n  {name}: k={getattr(optimizer,'k',1)}  evals/paso={evals_per_step}")

    while evals < total_evals:
        # BUG FIX: Use a fixed minibatch for the entire step!
        idx = rng_mb.integers(0, len(y_train), size=BATCH_SIZE)
        Xb, yb = X_train[idx], y_train[idx]
        
        f = lambda p: loss_on_batch(Xb, yb, p)

        params, n = optimizer.step(f, params)
        evals += n
        steps += 1

        if evals >= next_log or evals >= total_evals:
            tr_acc = full_accuracy(X_train, y_train, params)
            te_acc = full_accuracy(X_test,  y_test,  params)
            mb_loss = loss_on_batch(Xb, yb, params)
            elapsed = time.time() - t0
            print(f"    evals={evals:>7}  train={tr_acc:.1%}  test={te_acc:.1%}"
                  f"  loss={mb_loss:.4f}  t={elapsed:.0f}s")
            log_evals.append(evals)
            log_train_acc.append(tr_acc)
            log_test_acc.append(te_acc)
            next_log += LOG_INTERVAL

    return params, log_evals, log_train_acc, log_test_acc

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_subset()
    rng_init = np.random.default_rng(SEED)
    params0 = np.zeros(D, dtype=np.float32)
    i = 0
    for fan_in, fan_out in zip(ARCH[:-1], ARCH[1:]):
        w_size = fan_in * fan_out
        std = math.sqrt(2.0 / fan_in)
        params0[i:i+w_size] = rng_init.normal(0, std, w_size).astype(np.float32)
        i += w_size
        i += fan_out

    k = math.ceil(math.log2(D))
    
    dge_steps_total = TOTAL_EVALS // (2 * k)

    dge_opt = DGEOptimizer(
        dim=D, lr=0.5, delta=1e-3, beta1=0.9, beta2=0.999,
        lr_decay=0.01, delta_decay=0.05,
        total_steps=dge_steps_total, greedy_w=0.1,
        clip_norm=0.05, seed=SEED + 10,
    )
    run("DGE v8b", dge_opt, params0, X_train, y_train, X_test, y_test, TOTAL_EVALS, 2 * k)

    spsa_opt = SPSAOptimizer(
        dim=D, lr=0.01, delta=1e-3,
        lr_decay=0.01, delta_decay=0.05,
        total_steps=TOTAL_EVALS // 2, seed=SEED + 20,
    )
    run("SPSA", spsa_opt, params0, X_train, y_train, X_test, y_test, TOTAL_EVALS, 2)
