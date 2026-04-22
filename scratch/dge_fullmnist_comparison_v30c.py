"""
dge_fullmnist_comparison_v30c.py
=================================
Comparacion completa MNIST COMPLETO — version corregida con budgets por metodo.

Problema de v30b: SPSA/MeZO con budget 3M evals = 1.5M pasos sin output → horas.
Solucion: budgets distintos por metodo + logging frecuente.

  SPSA / MeZO     : 300,000 evals (150K pasos × 2 evals)
                    Suficiente para ver plateau/fracaso. Logging cada 30K.
  PureDGE         : 3,000,000 evals (~1284 pasos × 2336 evals)
  ConsistencyDGE  : 3,000,000 evals
  Adam / SGD      : 30 epochs

Tiempos estimados (CPU):
  SPSA/MeZO   :  ~3 min/seed × 2 = 6 min
  PureDGE     : ~14 min/seed
  ConsistencyDGE: ~14 min/seed
  Adam/SGD    :  ~15 s/seed
  Total 3 seeds:  ~3 × (6+14+14+0.5) ≈ 105 min
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
ARCH          = (784, 128, 64, 10)
BUDGET_DGE    = 3_000_000       # evals para PureDGE y ConsistencyDGE
BUDGET_SPSA   = 300_000         # evals para SPSA y MeZO (plateau mucho antes)
K_BLOCKS      = (1024, 128, 16)
LR_ZO         = 0.05
DELTA         = 1e-3
WINDOW        = 20
BATCH_SIZE    = 256
ADAM_EPOCHS   = 30
SGD_EPOCHS    = 30
SEEDS         = [42, 43, 44]
LOG_INTERVAL  = 30_000          # imprimir cada 30K evals (aplica a todos los ZO)

EST_ZO_STEP   = 2 * sum(K_BLOCKS)  # 2336 evals/paso DGE
TOTAL_DGE_STEPS  = BUDGET_DGE  // EST_ZO_STEP
TOTAL_SPSA_STEPS = BUDGET_SPSA // 2

# ---------------------------------------------------------------------------
# BatchedMLP
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


def zo_acc(model, X_te, y_te, params, chunk=1000):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y_te), chunk):
            lo = model.forward(X_te[i:i+chunk], params.unsqueeze(0)).squeeze(0)
            correct += (lo.argmax(1) == y_te[i:i+chunk]).sum().item()
    return correct / len(y_te)

# ---------------------------------------------------------------------------
# Adam state base
# ---------------------------------------------------------------------------

class AdamState:
    def __init__(self, dim, seed):
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        self.m = torch.zeros(dim, device=device)
        self.v = torch.zeros(dim, device=device)
        self.t = 0

    def update(self, grad, lr):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * (grad ** 2)
        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self.v / (1.0 - 0.999 ** self.t)
        return lr * mh / (torch.sqrt(vh) + 1e-8)

    def cosine(self, v0, total_steps, decay=0.01):
        frac = min(self.t / max(total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

# ---------------------------------------------------------------------------
# SPSA — global Rademacher perturbation, 2 evals/step
# ---------------------------------------------------------------------------

def run_spsa(name, f_factory, dim, params0, X_te, y_te, model, budget, seed, total_steps):
    state = AdamState(dim, seed)
    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)

    evals = 0
    best_acc = 0.0
    curve_evals, curve_acc = [], []
    next_log = LOG_INTERVAL
    t0 = time.time()
    print(f"\n  [{name}]  budget={budget:,}  evals/step=2  ~steps={total_steps:,}")
    print(f"  {'evals':>10}  {'test_acc':>8}  {'best':>8}  {'time':>7}")
    print(f"  {'-'*44}")

    while evals < budget:
        idx = torch.randperm(60000, generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr_d[idx], y_tr_d[idx]
        f = f_factory(Xb, yb)

        lr    = state.cosine(LR_ZO, total_steps)
        delta = state.cosine(DELTA, total_steps, decay=0.1)

        signs = torch.randint(0, 2, (dim,), generator=state.rng, device=device).float() * 2 - 1
        pert  = signs * delta
        p2    = params.unsqueeze(0)
        fp    = f(p2 + pert.unsqueeze(0))[0]
        fm    = f(p2 - pert.unsqueeze(0))[0]

        if name == "MeZO":
            # MeZO: proyecta el escalar de vuelta — identico matematicamente a SPSA aqui
            grad = (fp - fm) / (2.0 * delta) * signs
        else:
            grad = (fp - fm) / (2.0 * delta) * signs

        upd = state.update(grad, lr)
        params = params - upd
        evals += 2

        if evals >= next_log or evals >= budget:
            acc = zo_acc(model, X_te, y_te, params)
            best_acc = max(best_acc, acc)
            curve_evals.append(evals)
            curve_acc.append(round(acc, 4))
            elapsed = time.time() - t0
            print(f"  {evals:>10,}  {acc:>7.2%}  {best_acc:>7.2%}  {elapsed:>6.0f}s")
            next_log += LOG_INTERVAL

    return {"method": name, "seed": seed,
            "best_test_acc": round(best_acc, 4),
            "curve_evals": curve_evals, "curve_acc": curve_acc,
            "total_evals": evals, "wall_time": round(time.time() - t0, 1)}

# ---------------------------------------------------------------------------
# DGE optimizers — block perturbations
# ---------------------------------------------------------------------------

def run_dge(name, use_consistency, params0, X_te, y_te, model, seed):
    dim = model.dim
    sizes = model.sizes
    state = AdamState(dim, seed)
    sign_buf = deque(maxlen=WINDOW) if use_consistency else None
    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)

    evals = 0
    best_acc = 0.0
    curve_evals, curve_acc = [], []
    next_log = LOG_INTERVAL
    t0 = time.time()
    print(f"\n  [{name}]  budget={BUDGET_DGE:,}  evals/step={EST_ZO_STEP}")
    print(f"  {'evals':>10}  {'test_acc':>8}  {'best':>8}  {'time':>7}")
    print(f"  {'-'*44}")

    while evals < BUDGET_DGE:
        idx = torch.randperm(60000, generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr_d[idx], y_tr_d[idx]

        lr    = state.cosine(LR_ZO, TOTAL_DGE_STEPS)
        delta = state.cosine(DELTA, TOTAL_DGE_STEPS, decay=0.1)

        # Compute block gradients per layer
        full_grad = torch.zeros(dim, device=device)
        step_evals = 0
        offset = 0
        for sz, k in zip(sizes, K_BLOCKS):
            perm = torch.randperm(sz, generator=state.rng, device=device)
            blocks = torch.tensor_split(perm, k)
            perts = torch.zeros((2*k, sz), device=device)
            sl = []
            for bi, bl in enumerate(blocks):
                if len(bl) == 0: continue
                s = torch.randint(0, 2, (len(bl),), generator=state.rng,
                                  device=device).float() * 2 - 1
                perts[2*bi, bl] = s * delta
                perts[2*bi+1, bl] = -s * delta
                sl.append((bl, s))
            def fl(pl, _off=offset, _sz=sz, _Xb=Xb, _yb=yb):
                fp2 = params.unsqueeze(0).expand(pl.shape[0], -1).clone()
                fp2[:, _off:_off+_sz] = pl
                return batched_ce(model.forward(_Xb, fp2), _yb)
            xl = params[offset:offset+sz]
            Y = fl(xl.unsqueeze(0) + perts)
            for bi, (bl, s) in enumerate(sl):
                full_grad[offset + bl] = (Y[2*bi] - Y[2*bi+1]) / (2.0*delta) * s
            step_evals += 2 * k
            offset += sz

        if use_consistency:
            sign_buf.append(torch.sign(full_grad).cpu())
            mask = (torch.stack(list(sign_buf)).mean(0).abs().to(device)
                    if len(sign_buf) >= 2 else torch.ones(dim, device=device))
            upd = mask * state.update(full_grad, lr)
        else:
            upd = state.update(full_grad, lr)

        params = params - upd
        evals += step_evals

        if evals >= next_log or evals >= BUDGET_DGE:
            acc = zo_acc(model, X_te, y_te, params)
            best_acc = max(best_acc, acc)
            curve_evals.append(evals)
            curve_acc.append(round(acc, 4))
            elapsed = time.time() - t0
            print(f"  {evals:>10,}  {acc:>7.2%}  {best_acc:>7.2%}  {elapsed:>6.0f}s")
            next_log += LOG_INTERVAL

    return {"method": name, "seed": seed,
            "best_test_acc": round(best_acc, 4),
            "curve_evals": curve_evals, "curve_acc": curve_acc,
            "total_evals": evals, "wall_time": round(time.time() - t0, 1)}

# ---------------------------------------------------------------------------
# Adam / SGD
# ---------------------------------------------------------------------------

class TorchMLP(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers = []
        for l_in, l_out in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(l_in, l_out))
            if l_out != arch[-1]: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def run_grad(name, X_tr_all, y_tr_all, X_te_all, y_te_all, seed, lr, epochs):
    torch.manual_seed(seed)
    model = TorchMLP(ARCH).to(device)
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
    opt = (optim.Adam(model.parameters(), lr=lr) if name == "Adam"
           else optim.SGD(model.parameters(), lr=lr, momentum=0.9))

    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)
    steps_per_epoch = len(y_tr_all) // BATCH_SIZE
    best_acc = 0.0
    curve_evals, curve_acc = [], []
    eval_equiv = 0
    t0 = time.time()
    print(f"\n  [{name} {epochs} epochs | backprop]")
    print(f"  {'epoch':>6}  {'test_acc':>8}  {'best':>8}  {'time':>7}")
    print(f"  {'-'*36}")

    for ep in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            idx = torch.randperm(len(y_tr_all), generator=rng_mb)[:BATCH_SIZE]
            opt.zero_grad()
            F.cross_entropy(model(X_tr_all[idx].to(device)), y_tr_all[idx].to(device)).backward()
            opt.step()
            eval_equiv += 2
        acc = sum((model(X_te_all[i:i+2000].to(device)).argmax(1) == y_te_all[i:i+2000].to(device)).sum().item()
                  for i in range(0, len(y_te_all), 2000)) / len(y_te_all)
        best_acc = max(best_acc, acc)
        curve_evals.append(eval_equiv)
        curve_acc.append(round(acc, 4))
        if ep % 5 == 0 or ep == epochs:
            print(f"  {ep:>6}  {acc:>7.2%}  {best_acc:>7.2%}  {time.time()-t0:>6.0f}s")

    return {"method": name, "seed": seed,
            "best_test_acc": round(best_acc, 4),
            "curve_evals": curve_evals, "curve_acc": curve_acc,
            "total_evals": eval_equiv, "wall_time": round(time.time() - t0, 1)}

# ---------------------------------------------------------------------------
# Data — cargar a nivel global para evitar recargar entre runs
# ---------------------------------------------------------------------------

def load_full_mnist():
    assert HAS_TV
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = ((ds_tr.data.float().view(-1, 784)/255.0) - 0.1307) / 0.3081
    X_te = ((ds_te.data.float().view(-1, 784)/255.0) - 0.1307) / 0.3081
    return X_tr, ds_tr.targets, X_te, ds_te.targets

X_tr_all, y_tr_all, X_te_all, y_te_all = load_full_mnist()
X_tr_d = X_tr_all.to(device)
y_tr_d = y_tr_all.to(device)
X_te_d = X_te_all.to(device)
y_te_d = y_te_all.to(device)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"EXPERIMENTO v30c: Comparacion completa MNIST 60K/10K")
    print(f"SPSA/MeZO budget: {BUDGET_SPSA:,}  DGE budget: {BUDGET_DGE:,}")
    print(f"Seeds: {SEEDS}  Logging cada: {LOG_INTERVAL:,} evals")
    print(f"{'='*70}")

    all_results = []
    summary = {m: [] for m in ["SPSA", "MeZO", "PureDGE", "ConsistencyDGE", "SGD", "Adam"]}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        model = BatchedMLP(ARCH)
        torch.manual_seed(seed)
        params0 = torch.zeros(model.dim, device=device)
        off = 0
        for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
            std = math.sqrt(2.0 / l_in)
            w = l_in * l_out
            params0[off:off+w] = torch.randn(w, device=device) * std
            off += w + l_out

        def make_f(Xb, yb):
            def f(pb, _X=Xb, _y=yb):
                return batched_ce(model.forward(_X, pb), _y)
            return f

        for name in ["SPSA", "MeZO"]:
            r = run_spsa(name, make_f, model.dim, params0, X_te_d, y_te_d,
                         model, BUDGET_SPSA, seed, TOTAL_SPSA_STEPS)
            all_results.append(r)
            summary[name].append(r["best_test_acc"])
            print(f"  -> best={r['best_test_acc']:.2%}  t={r['wall_time']}s")

        for name, use_c in [("PureDGE", False), ("ConsistencyDGE", True)]:
            r = run_dge(name, use_c, params0, X_te_d, y_te_d, model, seed)
            all_results.append(r)
            summary[name].append(r["best_test_acc"])
            print(f"  -> best={r['best_test_acc']:.2%}  t={r['wall_time']}s")

        for name, lr in [("SGD", 0.01), ("Adam", 1e-3)]:
            r = run_grad(name, X_tr_all, y_tr_all, X_te_all, y_te_all,
                         seed, lr, ADAM_EPOCHS)
            all_results.append(r)
            summary[name].append(r["best_test_acc"])
            print(f"  -> best={r['best_test_acc']:.2%}  t={r['wall_time']}s")

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTADOS FINALES — MNIST COMPLETO 60K/10K  ({len(SEEDS)} seeds)")
    print(f"{'='*70}")
    ORDER = ["SPSA", "MeZO", "PureDGE", "ConsistencyDGE", "SGD", "Adam"]
    for method in ORDER:
        accs = summary[method]
        if not accs: continue
        mean, std = np.mean(accs), np.std(accs)
        backprop = "[backprop]" if method in {"Adam","SGD"} else "[zero-order]"
        tag = "  ← OUR" if method == "ConsistencyDGE" else ""
        print(f"  {method:<18}: {mean:.2%} ± {std:.2%}  {backprop}{tag}")

    # Save
    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v30c_fullmnist_comparison.json"
    payload = {
        "experiment": "v30c_fullmnist_comparison",
        "arch": list(ARCH), "budget_dge": BUDGET_DGE,
        "budget_spsa": BUDGET_SPSA, "adam_epochs": ADAM_EPOCHS,
        "seeds": SEEDS,
        "summary": {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
                    for m, v in summary.items() if v},
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON: {out_path}")
