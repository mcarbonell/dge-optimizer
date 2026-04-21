"""
dge_fullmnist_comparison_v30b.py
=================================
Comparacion completa en MNIST COMPLETO (60K train, 10K test):

  Zeroth-Order (sin backpropagation):
    - SPSA           — clasico, 2 evals/paso, perturba todo D a la vez
    - MeZO           — SPSA + Adam update (Zhang et al. 2022)
    - PureDGE        — bloques aleatorios + EMA Adam
    - ConsistencyDGE — PureDGE + mascara de consistencia T=20  <- NUESTRO METODO

  Gradient-Based (con backpropagation):
    - SGD (momentum) — gradiente analitico, referencia clasica
    - Adam           — gradiente analitico, referencia moderna

Config:
  Arch:   [784, 128, 64, 10]  ~109K params
  Budget ZO:  3,000,000 evals
  Adam/SGD:   30 epochs
  Seeds:  [42, 43, 44]
  Batch:  256
  Full MNIST: 60K train, 10K test

Tiempos estimados (CPU):
  PureDGE/ConsistencyDGE: ~14 min / seed
  SPSA/MeZO:              ~2 min / seed  (2 evals/paso vs ~2336)
  Adam/SGD:               ~15 s / seed
  Total ~3 seeds:          ~90-100 min
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
    print(f"DirectML: {device}")
except ImportError:
    device = torch.device("cpu")
    print("CPU")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARCH        = (784, 128, 64, 10)
BUDGET      = 3_000_000       # evals ZO
K_BLOCKS    = (1024, 128, 16)
LR_ZO       = 0.05
DELTA       = 1e-3
WINDOW      = 20
BATCH_SIZE  = 256
ADAM_EPOCHS = 30
SGD_EPOCHS  = 30
SEEDS       = [42, 43, 44]
LOG_POINTS  = 10              # checkpoints por run
CHECKPOINTS = [int(BUDGET * i / LOG_POINTS) for i in range(1, LOG_POINTS + 1)]

EST_ZO_STEP_EVALS = 2 * sum(K_BLOCKS)   # 2336 evals/paso DGE
SPSA_BUDGET_STEPS = BUDGET // 2          # SPSA usa 2 evals/paso -> mismos evals

# ---------------------------------------------------------------------------
# BatchedMLP para ZO
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
    return F.cross_entropy(logits.reshape(P*B, C), t.reshape(-1),
                           reduction='none').view(P, B).mean(dim=1)


def zo_accuracy_chunked(model, X_te, y_te, params, chunk=1000):
    """Evaluacion en chunks para no saturar memoria CPU."""
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y_te), chunk):
            lo = model.forward(X_te[i:i+chunk], params.unsqueeze(0)).squeeze(0)
            correct += (lo.argmax(dim=1) == y_te[i:i+chunk]).sum().item()
    return correct / len(y_te)

# ---------------------------------------------------------------------------
# Zeroth-Order Optimizers
# ---------------------------------------------------------------------------

class _BaseZO:
    def __init__(self, dim_or_sizes, k_blocks, total_steps, lr0, delta0, seed,
                 is_full_model=False):
        """
        is_full_model=True  -> perturba todo el modelo de una vez (SPSA/MeZO)
        is_full_model=False -> perturbacion por capa (DGE)
        """
        if is_full_model:
            self.total_dim = dim_or_sizes
            self.sizes = None
        else:
            self.sizes = dim_or_sizes
            self.total_dim = sum(dim_or_sizes)
        self.k_blocks = list(k_blocks) if k_blocks else None
        self.total_steps = total_steps
        self.lr0, self.delta0 = lr0, delta0
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        self.m = torch.zeros(self.total_dim, device=device)
        self.v = torch.zeros(self.total_dim, device=device)
        self.t = 0

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def _adam_update(self, grad, lr):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * (grad ** 2)
        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self.v / (1.0 - 0.999 ** self.t)
        return lr * mh / (torch.sqrt(vh) + 1e-8)


class SPSA_Full(_BaseZO):
    """SPSA clasico: 2 evals/paso, perturba todo D."""
    def step(self, f, params):
        lr    = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)
        signs = torch.randint(0, 2, (self.total_dim,),
                              generator=self.rng, device=device).float() * 2 - 1
        pert = signs * delta
        fp = f(params.unsqueeze(0) + pert.unsqueeze(0))[0]
        fm = f(params.unsqueeze(0) - pert.unsqueeze(0))[0]
        grad = (fp - fm) / (2.0 * delta) * signs
        upd  = self._adam_update(grad, lr)
        return params - upd, 2


class MeZO_Full(_BaseZO):
    """MeZO: SPSA + Adam, escala O(D) en memoria (sin activaciones).
    La diferencia conceptual con SPSA aqui es que MeZO usa el gradiente
    escalar proyectado de vuelta al espacio de parametros, igual que SPSA
    en terminos de implementacion. Su ventaja real en LLMs es de memoria.
    """
    def step(self, f, params):
        lr    = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)
        signs = torch.randint(0, 2, (self.total_dim,),
                              generator=self.rng, device=device).float() * 2 - 1
        pert = signs * delta
        fp = f(params.unsqueeze(0) + pert.unsqueeze(0))[0]
        fm = f(params.unsqueeze(0) - pert.unsqueeze(0))[0]
        scalar_g = (fp - fm) / (2.0 * delta)
        grad = scalar_g * signs
        upd  = self._adam_update(grad, lr)
        return params - upd, 2


class PureDGE_Full(_BaseZO):
    def _layer_grad(self, f_eval, params, delta):
        fg = torch.zeros_like(params)
        total_evals = 0
        offset = 0
        for size, k in zip(self.sizes, self.k_blocks):
            xl = params[offset:offset + size]
            perm = torch.randperm(size, generator=self.rng, device=device)
            blocks = torch.tensor_split(perm, k)
            perts = torch.zeros((2*k, size), device=device)
            sl = []
            for i, bl in enumerate(blocks):
                if len(bl) == 0: continue
                s = torch.randint(0, 2, (len(bl),), generator=self.rng,
                                  device=device).float() * 2 - 1
                perts[2*i, bl] = s * delta
                perts[2*i+1, bl] = -s * delta
                sl.append((bl, s))
            def fl(pl, _off=offset, _sz=size):
                fp = params.unsqueeze(0).expand(pl.shape[0], -1).clone()
                fp[:, _off:_off+_sz] = pl
                return f_eval(fp)
            Y = fl(xl.unsqueeze(0) + perts)
            for i, (bl, s) in enumerate(sl):
                fg[offset + bl] = (Y[2*i] - Y[2*i+1]) / (2.0*delta) * s
            total_evals += 2 * k
            offset += size
        return fg, total_evals

    def step(self, f, params):
        lr    = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)
        grad, n = self._layer_grad(f, params, delta)
        upd = self._adam_update(grad, lr)
        return params - upd, n


class ConsistencyDGE_Full(PureDGE_Full):
    def __init__(self, *args, window=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.sign_buf = deque(maxlen=window)

    def step(self, f, params):
        lr    = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)
        grad, n = self._layer_grad(f, params, delta)
        self.sign_buf.append(torch.sign(grad).cpu())
        if len(self.sign_buf) >= 2:
            mask = torch.stack(list(self.sign_buf)).mean(0).abs().to(device)
        else:
            mask = torch.ones_like(params)
        upd = mask * self._adam_update(grad, lr)
        return params - upd, n

# ---------------------------------------------------------------------------
# PyTorch nn.Module para Adam/SGD
# ---------------------------------------------------------------------------

class TorchMLP(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers = []
        for l_in, l_out in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(l_in, l_out))
            if l_out != arch[-1]:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def nn_accuracy_chunked(model, X_te, y_te, chunk=2000):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y_te), chunk):
            correct += (model(X_te[i:i+chunk]).argmax(1) == y_te[i:i+chunk]).sum().item()
    return correct / len(y_te)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_full_mnist():
    assert HAS_TV, "pip install torchvision"
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = ((ds_tr.data.float().view(-1, 784) / 255.0) - 0.1307) / 0.3081
    X_te = ((ds_te.data.float().view(-1, 784) / 255.0) - 0.1307) / 0.3081
    return X_tr, ds_tr.targets, X_te, ds_te.targets


def init_zo_params(model, seed):
    params = torch.zeros(model.dim, device=device)
    torch.manual_seed(seed)
    off = 0
    for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
        std = math.sqrt(2.0 / l_in)
        w = l_in * l_out
        params[off:off+w] = torch.randn(w, device=device) * std
        off += w + l_out
    return params


def init_nn_params(model, seed):
    torch.manual_seed(seed)
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_zo(name, opt, model, params0, X_tr, y_tr, X_te, y_te, seed):
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
            acc = zo_accuracy_chunked(model, X_te, y_te, params)
            best_acc = max(best_acc, acc)
            curve_evals.append(evals)
            curve_acc.append(round(acc, 4))
            next_ckpt += 1

    if not curve_evals or curve_evals[-1] < evals:
        acc = zo_accuracy_chunked(model, X_te, y_te, params)
        best_acc = max(best_acc, acc)
        curve_evals.append(evals)
        curve_acc.append(round(acc, 4))

    return {
        "method": name, "seed": seed,
        "best_test_acc": round(best_acc, 4),
        "curve_evals": curve_evals,
        "curve_acc": curve_acc,
        "total_evals": evals,
        "wall_time": round(time.time() - t0, 1),
    }


def run_grad(name, X_tr, y_tr, X_te, y_te, seed, lr, epochs):
    torch.manual_seed(seed)
    model = TorchMLP(ARCH).to(device)
    init_nn_params(model, seed)
    opt = (optim.Adam(model.parameters(), lr=lr) if name == "Adam"
           else optim.SGD(model.parameters(), lr=lr, momentum=0.9))

    batch_rng = torch.Generator()
    batch_rng.manual_seed(seed + 100)

    steps_per_epoch = len(y_tr) // BATCH_SIZE
    best_acc = 0.0
    curve_evals, curve_acc = [], []
    eval_equiv = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            idx = torch.randperm(len(y_tr), generator=batch_rng)[:BATCH_SIZE]
            opt.zero_grad()
            F.cross_entropy(model(X_tr[idx].to(device)), y_tr[idx].to(device)).backward()
            opt.step()
            eval_equiv += 2

        acc = nn_accuracy_chunked(model, X_te.to(device), y_te.to(device))
        best_acc = max(best_acc, acc)
        curve_evals.append(eval_equiv)
        curve_acc.append(round(acc, 4))

    return {
        "method": name, "seed": seed,
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
    total_steps_dge  = BUDGET // EST_ZO_STEP_EVALS
    total_steps_spsa = SPSA_BUDGET_STEPS   # ya en steps (2 evals c/u)

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v30b: Comparacion completa MNIST COMPLETO")
    print(f"ZO budget: {BUDGET:,}  DGE steps: ~{total_steps_dge}  SPSA steps: {SPSA_BUDGET_STEPS:,}")
    print(f"Adam/SGD: {ADAM_EPOCHS} epochs  Seeds: {SEEDS}")
    print(f"{'='*70}")

    X_tr_all, y_tr_all, X_te_all, y_te_all = load_full_mnist()

    all_results = []
    summary = {m: [] for m in ["SPSA", "MeZO", "PureDGE", "ConsistencyDGE", "SGD", "Adam"]}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        model = BatchedMLP(ARCH)
        params0 = init_zo_params(model, seed)
        X_tr_d = X_tr_all.to(device)
        y_tr_d = y_tr_all.to(device)
        X_te_d = X_te_all.to(device)
        y_te_d = y_te_all.to(device)

        # --- Zeroth-order methods ---
        zo_configs = [
            ("SPSA",
             SPSA_Full(model.dim, None, total_steps_spsa, LR_ZO, DELTA, seed+10000,
                       is_full_model=True)),
            ("MeZO",
             MeZO_Full(model.dim, None, total_steps_spsa, LR_ZO, DELTA, seed+10000,
                       is_full_model=True)),
            ("PureDGE",
             PureDGE_Full(model.sizes, K_BLOCKS, total_steps_dge, LR_ZO, DELTA, seed+10000)),
            ("ConsistencyDGE",
             ConsistencyDGE_Full(model.sizes, K_BLOCKS, total_steps_dge, LR_ZO, DELTA, seed+10000,
                                 window=WINDOW)),
        ]
        for name, opt in zo_configs:
            print(f"\n  [{name}]  evals/step={'2' if 'SPSA' in name or 'MeZO' in name else str(EST_ZO_STEP_EVALS)}")
            r = run_zo(name, opt, model, params0, X_tr_d, y_tr_d, X_te_d, y_te_d, seed)
            all_results.append(r)
            summary[name].append(r["best_test_acc"])
            print(f"  -> best={r['best_test_acc']:.2%}  final={r['curve_acc'][-1]:.2%}  t={r['wall_time']}s")

        # --- Gradient-based methods ---
        for name, lr in [("SGD", 0.01), ("Adam", 1e-3)]:
            print(f"\n  [{name} ({ADAM_EPOCHS} epochs, backprop)]")
            r = run_grad(name, X_tr_all, y_tr_all, X_te_all, y_te_all,
                         seed, lr, ADAM_EPOCHS)
            all_results.append(r)
            summary[name].append(r["best_test_acc"])
            print(f"  -> best={r['best_test_acc']:.2%}  t={r['wall_time']}s")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"RESULTADOS FINALES ({len(SEEDS)} seeds) — MNIST COMPLETO 60K/10K")
    print(f"{'='*70}")
    ref = None
    ORDER = ["SPSA", "MeZO", "PureDGE", "ConsistencyDGE", "SGD", "Adam"]
    for method in ORDER:
        accs = summary[method]
        if not accs:
            continue
        mean, std = np.mean(accs), np.std(accs)
        if ref is None:
            ref = mean
        rel = f"  ({(mean-ref)/ref*100:+.1f}%)" if method != "SPSA" else ""
        tag = "  ← OUR METHOD" if method == "ConsistencyDGE" else ""
        backprop = "  [backprop]" if method in {"Adam", "SGD"} else "  [zero-order]"
        print(f"  {method:<18}: {mean:.2%} ± {std:.2%}{rel}{backprop}{tag}")

    # ---------------------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------------------
    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v30b_fullmnist_comparison.json"

    payload = {
        "experiment": "v30b_fullmnist_comparison",
        "arch": list(ARCH),
        "budget_zo": BUDGET,
        "adam_sgd_epochs": ADAM_EPOCHS,
        "seeds": SEEDS,
        "summary": {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
            for m, v in summary.items() if v
        },
        "results": all_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON guardado en: {out_path}")
    print("\nSiguiente paso: python scratch/dge_figures_v30b.py  (para generar figuras)")
