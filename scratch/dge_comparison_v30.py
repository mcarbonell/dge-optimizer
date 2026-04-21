"""
dge_comparison_v30.py
=====================
Experimento de comparacion para el paper:
  - PureDGE       (zeroth-order, sin consistency)
  - ConsistencyDGE (zeroth-order, T=20) ← nuestro metodo
  - SPSA          (zeroth-order clasico, 2 evals/paso)
  - MeZO          (SPSA + Adam update, zeroth-order, memoria O(1))
  - Adam          (gradiente analitico, backprop)
  - SGD+momentum  (gradiente analitico, backprop)

Configuracion identica a v29:
  Arch: [784, 128, 64, 10]  ~109K params
  Budget: 800K function evaluations (DGE/SPSA/MeZO)
  Adam/SGD: mismo numero de pasos de minibatch forward equivalentes
  Seeds: 42-47 (6 seeds)
  n_train: 3000  n_test: 600  batch: 256

Nota sobre el presupuesto de evaluaciones para Adam/SGD:
  Un "paso" de Adam/SGD involucra 1 forward pass + 1 backward pass.
  Para comparacion justa de computo, lo equiparamos a 2 evals (mismo
  coste que 1 par +/- de SPSA). Por tanto Adam/SGD recibe los mismos
  800K/2 = 400K "pasos" equivalentes.

Nota sobre MeZO:
  MeZO (Zhang et al. 2022) es SPSA aplicado a todo el modelo con Adam
  update. Esencialmente: pertubacion global de signos aleatorios, 2 evals,
  Adam step. Es el metodo de referencia para memory-efficient finetuning.
"""

import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from torchvision import datasets, transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False

try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML: {device}")
except ImportError:
    device = torch.device("cpu")
    print("Using CPU")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARCH       = (784, 128, 64, 10)
BUDGET     = 800_000           # evals para metodos zeroth-order
K_BLOCKS   = (1024, 128, 16)
LR_ZO      = 0.05              # lr para metodos zeroth-order (v29)
DELTA      = 1e-3
WINDOW     = 20
LR_ADAM    = 1e-3              # lr tipico de Adam en MNIST
LR_SGD     = 0.01
N_TRAIN    = 3_000
N_TEST     = 600
BATCH_SIZE = 256
SEEDS      = [42, 43, 44, 45, 46, 47]
CHECKPOINTS = list(range(40_000, BUDGET + 1, 40_000))

# ---------------------------------------------------------------------------
# BatchedMLP para metodos zeroth-order (replica exacta de v28/v29)
# ---------------------------------------------------------------------------

class BatchedMLP:
    def __init__(self, arch):
        self.arch = list(arch)
        self.sizes = [a * b + b for a, b in zip(arch[:-1], arch[1:])]
        self.dim = sum(self.sizes)

    def forward(self, X, params_batch):
        P = params_batch.shape[0]
        h = X.unsqueeze(0).expand(P, -1, -1)
        i = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            W = params_batch[:, i:i + l_in * l_out].view(P, l_in, l_out)
            i += l_in * l_out
            b = params_batch[:, i:i + l_out].view(P, 1, l_out)
            i += l_out
            h = torch.bmm(h, W) + b
            if l_out != self.arch[-1]:
                h = torch.relu(h)
        return h


def batched_ce(logits, targets):
    P, B, C = logits.shape
    t = targets.unsqueeze(0).expand(P, -1)
    return F.cross_entropy(logits.reshape(P * B, C), t.reshape(-1),
                           reduction='none').view(P, B).mean(dim=1)


# PyTorch nn.Module equivalente para Adam/SGD
class TorchMLP(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers = []
        for i, (l_in, l_out) in enumerate(zip(arch[:-1], arch[1:])):
            layers.append(nn.Linear(l_in, l_out))
            if l_out != arch[-1]:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def test_accuracy_zo(model_zo, X_te, y_te, params):
    with torch.no_grad():
        logits = model_zo.forward(X_te, params.unsqueeze(0)).squeeze(0)
        return (logits.argmax(dim=1) == y_te).float().mean().item()


def test_accuracy_nn(model_nn, X_te, y_te):
    with torch.no_grad():
        logits = model_nn(X_te)
        return (logits.argmax(dim=1) == y_te).float().mean().item()

# ---------------------------------------------------------------------------
# Datos MNIST
# ---------------------------------------------------------------------------

def load_mnist():
    assert HAS_TV, "pip install torchvision"
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = (ds_tr.data.float().view(-1, 784) / 255.0 - 0.1307) / 0.3081
    y_tr = ds_tr.targets
    X_te = (ds_te.data.float().view(-1, 784) / 255.0 - 0.1307) / 0.3081
    y_te = ds_te.targets
    return X_tr, y_tr, X_te, y_te


def init_zo_params(model, seed):
    params = torch.zeros(model.dim, device=device)
    offset = 0
    torch.manual_seed(seed)
    for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
        std = math.sqrt(2.0 / l_in)
        w = l_in * l_out
        params[offset:offset + w] = torch.randn(w, device=device) * std
        offset += w + l_out
    return params


def init_nn_params(model_nn, seed):
    torch.manual_seed(seed)
    for layer in model_nn.net:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
    return model_nn

# ---------------------------------------------------------------------------
# Zeroth-Order Optimizers
# ---------------------------------------------------------------------------

class _BaseZO:
    def __init__(self, sizes, k_blocks, total_steps, lr0, delta0, seed):
        self.sizes = sizes
        self.k_blocks = list(k_blocks)
        self.total_steps = total_steps
        self.lr0 = lr0
        self.delta0 = delta0
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        total = sum(sizes)
        self.m = torch.zeros(total, device=device)
        self.v = torch.zeros(total, device=device)
        self.t = 0

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def _layer_grad(self, f_layer, xl, size, k, delta):
        grad = torch.zeros(size, device=device)
        perm = torch.randperm(size, generator=self.rng, device=device)
        blocks = torch.tensor_split(perm, k)
        n = 2 * k
        perts = torch.zeros((n, size), device=device)
        sl = []
        for i, bl in enumerate(blocks):
            if len(bl) == 0:
                continue
            s = torch.randint(0, 2, (len(bl),), generator=self.rng, device=device).float() * 2 - 1
            perts[2*i, bl] = s * delta
            perts[2*i+1, bl] = -s * delta
            sl.append((bl, s))
        Y = f_layer(xl.unsqueeze(0) + perts)
        for i, (bl, s) in enumerate(sl):
            grad[bl] = (Y[2*i] - Y[2*i+1]) / (2.0 * delta) * s
        return grad, 2 * k

    def _full_grad(self, f_eval, params, lr, delta):
        fg = torch.zeros_like(params)
        total_evals = 0
        offset = 0
        for size, k in zip(self.sizes, self.k_blocks):
            xl = params[offset:offset + size]
            def fl(pl, _off=offset, _sz=size):
                P = pl.shape[0]
                fp = params.unsqueeze(0).expand(P, -1).clone()
                fp[:, _off:_off + _sz] = pl
                return f_eval(fp)
            g, evs = self._layer_grad(fl, xl, size, k, delta)
            fg[offset:offset + size] = g
            total_evals += evs
            offset += size
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * fg
        self.v = 0.999 * self.v + 0.001 * (fg ** 2)
        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self.v / (1.0 - 0.999 ** self.t)
        return fg, mh, vh, total_evals


class PureDGE_V30(_BaseZO):
    def step(self, f, params):
        lr    = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)
        _, mh, vh, n = self._full_grad(f, params, lr, delta)
        upd = lr * mh / (torch.sqrt(vh) + 1e-8)
        return params - upd, n


class ConsistencyDGE_V30(_BaseZO):
    def __init__(self, *args, window=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.sign_buf = deque(maxlen=window)

    def step(self, f, params):
        lr    = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)
        grad, mh, vh, n = self._full_grad(f, params, lr, delta)
        self.sign_buf.append(torch.sign(grad).cpu())
        if len(self.sign_buf) < 2:
            mask = torch.ones_like(params)
        else:
            mask = torch.stack(list(self.sign_buf)).mean(dim=0).abs().to(device)
        upd = lr * mask * mh / (torch.sqrt(vh) + 1e-8)
        return params - upd, n


class SPSA_V30:
    """Classical SPSA: 2 evals per step, global Rademacher perturbation."""
    def __init__(self, dim, total_steps, lr0=0.05, delta0=1e-3, seed=42):
        self.dim = dim
        self.total_steps = total_steps
        self.lr0 = lr0
        self.delta0 = delta0
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        self.m = torch.zeros(dim, device=device)
        self.v = torch.zeros(dim, device=device)
        self.t = 0

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def step(self, f, params):
        self.t += 1
        lr    = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)
        signs = torch.randint(0, 2, (self.dim,), generator=self.rng, device=device).float() * 2 - 1
        pert  = signs * delta
        # 2 evals for the full vector
        fp = f(params.unsqueeze(0) + pert.unsqueeze(0))
        fm = f(params.unsqueeze(0) - pert.unsqueeze(0))
        grad = (fp[0] - fm[0]) / (2.0 * delta) * signs
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * (grad ** 2)
        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self.v / (1.0 - 0.999 ** self.t)
        upd = lr * mh / (torch.sqrt(vh) + 1e-8)
        return params - upd, 2


class MeZO_V30:
    """MeZO: SPSA with Adam over the full model (Zhang et al., 2022).
    Uses same-seed perturbation trick for memory efficiency (conceptually).
    Equivalent to SPSA+Adam in terms of evals and compute.
    """
    def __init__(self, dim, total_steps, lr0=0.05, delta0=1e-3, seed=42):
        self.dim = dim
        self.total_steps = total_steps
        self.lr0 = lr0
        self.delta0 = delta0
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        self.m = torch.zeros(dim, device=device)
        self.v = torch.zeros(dim, device=device)
        self.t = 0

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def step(self, f, params):
        self.t += 1
        lr    = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)
        # Random perturbation seed (MeZO regenerates from seed, saves memory)
        signs = torch.randint(0, 2, (self.dim,), generator=self.rng, device=device).float() * 2 - 1
        pert  = signs * delta
        fp = f(params.unsqueeze(0) + pert.unsqueeze(0))
        fm = f(params.unsqueeze(0) - pert.unsqueeze(0))
        scalar_grad = (fp[0] - fm[0]) / (2.0 * delta)
        # MeZO projects scalar gradient back to parameter space via perturbation
        grad = scalar_grad * signs
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * (grad ** 2)
        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self.v / (1.0 - 0.999 ** self.t)
        upd = lr * mh / (torch.sqrt(vh) + 1e-8)
        return params - upd, 2

# ---------------------------------------------------------------------------
# Runner: zeroth-order methods
# ---------------------------------------------------------------------------

def run_zo(method_name, opt, model, params0, X_tr, y_tr, X_te, y_te, seed):
    params = params0.clone()
    batch_rng = torch.Generator()
    batch_rng.manual_seed(seed + 100)

    evals = 0
    best_acc = 0.0
    curve_evals, curve_acc = [], []
    next_ckpt = 0
    t0 = time.time()

    while evals < BUDGET:
        idx = torch.randperm(X_tr.shape[0], generator=batch_rng)[:BATCH_SIZE]
        Xb, yb = X_tr[idx], y_tr[idx]

        def f(pb, _Xb=Xb, _yb=yb):
            return batched_ce(model.forward(_Xb, pb), _yb)

        params, n = opt.step(f, params)
        evals += n

        if next_ckpt < len(CHECKPOINTS) and evals >= CHECKPOINTS[next_ckpt]:
            acc = test_accuracy_zo(model, X_te, y_te, params)
            best_acc = max(best_acc, acc)
            curve_evals.append(evals)
            curve_acc.append(round(acc, 4))
            next_ckpt += 1

    if not curve_evals or curve_evals[-1] < evals:
        acc = test_accuracy_zo(model, X_te, y_te, params)
        best_acc = max(best_acc, acc)
        curve_evals.append(evals)
        curve_acc.append(round(acc, 4))

    return {
        "method": method_name,
        "seed": seed,
        "best_test_acc": round(best_acc, 4),
        "curve_evals": curve_evals,
        "curve_acc": curve_acc,
        "total_evals": evals,
        "wall_time": round(time.time() - t0, 1),
    }

# ---------------------------------------------------------------------------
# Runner: gradient-based methods (Adam, SGD)
# ---------------------------------------------------------------------------

def run_grad(method_name, X_tr, y_tr, X_te, y_te, seed, lr):
    torch.manual_seed(seed)
    model_nn = TorchMLP(ARCH).to(device)
    init_nn_params(model_nn, seed)

    if method_name == "Adam":
        opt_nn = optim.Adam(model_nn.parameters(), lr=lr)
    else:
        opt_nn = optim.SGD(model_nn.parameters(), lr=lr, momentum=0.9)

    # For a fair comparison: Adam/SGD gets BUDGET/2 steps
    # (each step = 1 forward + 1 backward ≈ 2 evals equivalent)
    n_steps = BUDGET // 2
    step_checkpoints = set()
    for ck in CHECKPOINTS:
        step_checkpoints.add(ck // 2)  # convert eval-budget to step-budget

    batch_rng = torch.Generator()
    batch_rng.manual_seed(seed + 100)

    step = 0
    best_acc = 0.0
    curve_evals, curve_acc = [], []
    next_ckpt = 0
    eval_equiv = 0
    t0 = time.time()

    while step < n_steps:
        idx = torch.randperm(X_tr.shape[0], generator=batch_rng)[:BATCH_SIZE]
        Xb, yb = X_tr[idx], y_tr[idx]

        opt_nn.zero_grad()
        logits = model_nn(Xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt_nn.step()

        step += 1
        eval_equiv += 2   # 1 forward + 1 backward ~ 2 forward evals

        if next_ckpt < len(CHECKPOINTS) and eval_equiv >= CHECKPOINTS[next_ckpt]:
            acc = test_accuracy_nn(model_nn, X_te, y_te)
            best_acc = max(best_acc, acc)
            curve_evals.append(eval_equiv)
            curve_acc.append(round(acc, 4))
            next_ckpt += 1

    if not curve_evals or curve_evals[-1] < eval_equiv:
        acc = test_accuracy_nn(model_nn, X_te, y_te)
        best_acc = max(best_acc, acc)
        curve_evals.append(eval_equiv)
        curve_acc.append(round(acc, 4))

    return {
        "method": method_name,
        "seed": seed,
        "best_test_acc": round(best_acc, 4),
        "curve_evals": curve_evals,
        "curve_acc": curve_acc,
        "total_evals": eval_equiv,
        "wall_time": round(time.time() - t0, 1),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*68}")
    print(f"EXPERIMENTO v30: Comparacion completa para el paper")
    print(f"Methods: PureDGE, ConsistencyDGE, SPSA, MeZO, Adam, SGD")
    print(f"Seeds: {SEEDS}  Budget ZO: {BUDGET:,}  Budget grad: {BUDGET//2:,} steps")
    print(f"{'='*68}")

    X_tr_all, y_tr_all, X_te_all, y_te_all = load_mnist()

    all_results = []
    est_zo_step = 2 * sum(K_BLOCKS)
    total_steps_zo = BUDGET // est_zo_step

    for seed in SEEDS:
        print(f"\n--- SEED {seed} ---")

        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(len(y_tr_all), N_TRAIN, replace=False)
        te_idx = rng.choice(len(y_te_all), N_TEST,  replace=False)
        X_tr = X_tr_all[tr_idx].to(device)
        y_tr = y_tr_all[tr_idx].to(device)
        X_te = X_te_all[te_idx].to(device)
        y_te = y_te_all[te_idx].to(device)

        model_zo = BatchedMLP(ARCH)
        params0  = torch.zeros(model_zo.dim, device=device)
        torch.manual_seed(seed)
        offset = 0
        for l_in, l_out in zip(model_zo.arch[:-1], model_zo.arch[1:]):
            std = math.sqrt(2.0 / l_in)
            w = l_in * l_out
            params0[offset:offset + w] = torch.randn(w, device=device) * std
            offset += w + l_out

        # Zeroth-order methods
        zo_configs = [
            ("PureDGE",        PureDGE_V30(model_zo.sizes, K_BLOCKS, total_steps_zo, LR_ZO, DELTA, seed + 10_000)),
            ("ConsistencyDGE", ConsistencyDGE_V30(model_zo.sizes, K_BLOCKS, total_steps_zo, LR_ZO, DELTA, seed + 10_000, window=WINDOW)),
            ("SPSA",           SPSA_V30(model_zo.dim, BUDGET // 2, LR_ZO, DELTA, seed + 10_000)),
            ("MeZO",           MeZO_V30(model_zo.dim, BUDGET // 2, LR_ZO, DELTA, seed + 10_000)),
        ]

        for name, opt in zo_configs:
            r = run_zo(name, opt, model_zo, params0, X_tr, y_tr, X_te, y_te, seed)
            all_results.append(r)
            print(f"  [{name:<16}] best={r['best_test_acc']:.2%}  t={r['wall_time']}s")

        # Gradient-based methods
        for name, lr in [("Adam", LR_ADAM), ("SGD", LR_SGD)]:
            r = run_grad(name, X_tr, y_tr, X_te, y_te, seed, lr)
            all_results.append(r)
            print(f"  [{name:<16}] best={r['best_test_acc']:.2%}  t={r['wall_time']}s")

    # Summary
    print(f"\n{'='*68}")
    print(f"RESULTADOS FINALES ({len(SEEDS)} seeds)")
    print(f"{'='*68}")
    all_methods = ["PureDGE", "ConsistencyDGE", "SPSA", "MeZO", "Adam", "SGD"]
    ref = None
    for method in all_methods:
        accs = [r["best_test_acc"] for r in all_results if r["method"] == method]
        if not accs:
            continue
        mean, std = np.mean(accs), np.std(accs)
        if ref is None:
            ref = mean
        rel = f"  ({(mean-ref)/ref*100:+.1f}%)" if method != "PureDGE" else ""
        tag = "← OUR" if method == "ConsistencyDGE" else ""
        print(f"  {method:<18}: {mean:.2%} ± {std:.2%}{rel}  {tag}")

    # Save JSON
    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v30_full_comparison.json"

    payload = {
        "experiment": "v30_full_comparison",
        "arch": list(ARCH),
        "budget_zo": BUDGET,
        "budget_grad_equiv": BUDGET // 2,
        "seeds": SEEDS,
        "summary": {
            m: {
                "mean": float(np.mean([r["best_test_acc"] for r in all_results if r["method"] == m])),
                "std":  float(np.std( [r["best_test_acc"] for r in all_results if r["method"] == m])),
                "values": [r["best_test_acc"] for r in all_results if r["method"] == m],
            }
            for m in all_methods
            if any(r["method"] == m for r in all_results)
        },
        "results": all_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON guardado en: {out_path}")
    print("Para generar figuras con comparacion completa, ejecuta:")
    print("  python scratch/dge_figures_v29.py  (actualizar para incluir v30)")
