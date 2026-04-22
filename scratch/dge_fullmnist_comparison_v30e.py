"""
dge_fullmnist_comparison_v30e.py
=================================
v30d acelerado usando la clase nativa TorchDGEOptimizer (DGE V3).

Cambios respecto a v30d:
  - Reemplazada la implementación manual del bucle por la clase TorchDGEOptimizer
  - Todo lo demas identico (hiperparámetros, semillas, budgets)

Config identica a v30c/v30d:
  SPSA/MeZO budget : 300,000 evals  (log cada 30K)
  DGE budget       : 3,000,000 evals
  Adam/SGD         : 30 epochs
  Seeds            : [42, 43, 44]
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

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dge.torch_optimizer import TorchDGEOptimizer

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
BUDGET_DGE    = 3_000_000
BUDGET_SPSA   = 300_000
K_BLOCKS      = (1024, 128, 16)
LR_ZO         = 0.05
DELTA         = 1e-3
WINDOW        = 20
BATCH_SIZE    = 256
ADAM_EPOCHS   = 30
SGD_EPOCHS    = 30
SEEDS         = [42, 43, 44]
LOG_INTERVAL  = 30_000
TRAIN_ACC_N   = 5_000    # subsample de train para calcular train_acc rapidamente

EST_ZO_STEP      = 2 * sum(K_BLOCKS)   # 2336 evals/paso DGE
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


def zo_acc(model, X, y, params, chunk=1000):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), chunk):
            lo = model.forward(X[i:i+chunk], params.unsqueeze(0)).squeeze(0)
            correct += (lo.argmax(1) == y[i:i+chunk]).sum().item()
    return correct / len(y)

# ---------------------------------------------------------------------------
# Adam state
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
# SPSA / MeZO
# ---------------------------------------------------------------------------

def run_spsa(name, model, params0, X_tr_subsample, y_tr_sub, X_te, y_te,
             budget, seed, total_steps):
    dim   = model.dim
    state = AdamState(dim, seed)
    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)

    evals = 0
    best_test = 0.0
    curve_evals, curve_acc = [], []
    next_log = LOG_INTERVAL
    t0 = time.time()

    hdr = f"  {'evals':>10}  {'train_acc':>9}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}"
    print(f"\n  [{name}]  budget={budget:,}  evals/step=2  ~steps={total_steps:,}")
    print(hdr)
    print(f"  {'-'*56}")

    while evals < budget:
        idx = torch.randperm(60000, generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr_d[idx], y_tr_d[idx]

        lr    = state.cosine(LR_ZO, total_steps)
        delta = state.cosine(DELTA, total_steps, decay=0.1)

        signs = torch.randint(0, 2, (dim,), generator=state.rng, device=device).float() * 2 - 1
        pert  = signs * delta
        
        # Batch fp and fm to save 1 GPU kernel launch overhead
        P = torch.empty((2, dim), device=device)
        P[0] = pert
        P[1] = -pert
        
        losses = batched_ce(model.forward(Xb, params.unsqueeze(0) + P), yb)
        fp, fm = losses[0], losses[1]
        
        grad  = (fp - fm) / (2.0 * delta) * signs
        upd   = state.update(grad, lr)
        params = params - upd
        evals += 2

        if evals >= next_log or evals >= budget:
            tr_acc  = zo_acc(model, X_tr_subsample, y_tr_sub, params)
            te_acc  = zo_acc(model, X_te, y_te, params)
            best_test = max(best_test, te_acc)
            curve_evals.append(evals)
            curve_acc.append(round(te_acc, 4))
            elapsed = time.time() - t0
            print(f"  {evals:>10,}  {tr_acc:>8.2%}  {te_acc:>8.2%}  {best_test:>8.2%}  {elapsed:>6.0f}s")
            next_log += LOG_INTERVAL

    return {"method": name, "seed": seed,
            "best_test_acc": round(best_test, 4),
            "curve_evals": curve_evals, "curve_acc": curve_acc,
            "total_evals": evals, "wall_time": round(time.time() - t0, 1)}

# ---------------------------------------------------------------------------
# DGE (Pure + Consistency)
# ---------------------------------------------------------------------------

def run_dge(name, use_consistency, model, params0,
            X_tr_subsample, y_tr_sub, X_te, y_te, seed):
    
    opt = TorchDGEOptimizer(
        dim=model.dim,
        layer_sizes=model.sizes,
        k_blocks=list(K_BLOCKS),
        lr=LR_ZO,
        delta=DELTA,
        total_steps=TOTAL_DGE_STEPS,
        consistency_window=WINDOW if use_consistency else 0,
        seed=seed,
        device=device,
        chunk_size=128
    )
    
    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)

    evals = 0
    best_test = 0.0
    curve_evals, curve_acc = [], []
    next_log = LOG_INTERVAL
    t0 = time.time()

    hdr = f"  {'evals':>10}  {'train_acc':>9}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}"
    print(f"\n  [{name}]  budget={BUDGET_DGE:,}  evals/step={EST_ZO_STEP}")
    print(hdr)
    print(f"  {'-'*56}")

    while evals < BUDGET_DGE:
        idx = torch.randperm(60000, generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr_d[idx], y_tr_d[idx]

        def f_batched(p_batch):
            logits = model.forward(Xb, p_batch)
            return batched_ce(logits, yb)
            
        params, n = opt.step(f_batched, params)
        evals += n

        if evals >= next_log or evals >= BUDGET_DGE:
            tr_acc  = zo_acc(model, X_tr_subsample, y_tr_sub, params)
            te_acc  = zo_acc(model, X_te, y_te, params)
            best_test = max(best_test, te_acc)
            curve_evals.append(evals)
            curve_acc.append(round(te_acc, 4))
            elapsed = time.time() - t0
            print(f"  {evals:>10,}  {tr_acc:>8.2%}  {te_acc:>8.2%}  {best_test:>8.2%}  {elapsed:>6.0f}s")
            next_log += LOG_INTERVAL

    return {"method": name, "seed": seed,
            "best_test_acc": round(best_test, 4),
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
    best_test = 0.0
    curve_evals, curve_acc = [], []
    eval_equiv = 0
    t0 = time.time()

    hdr = f"  {'epoch':>6}  {'train_acc':>9}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}"
    print(f"\n  [{name} | {epochs} epochs | backprop]")
    print(hdr)
    print(f"  {'-'*50}")

    for ep in range(1, epochs + 1):
        correct_tr = 0
        for _ in range(steps_per_epoch):
            idx = torch.randperm(len(y_tr_all), generator=rng_mb)[:BATCH_SIZE]
            Xb = X_tr_all[idx].to(device)
            yb = y_tr_all[idx].to(device)
            opt.zero_grad()
            logits = model(Xb)
            F.cross_entropy(logits, yb).backward()
            opt.step()
            correct_tr += (logits.detach().argmax(1) == yb).sum().item()
            eval_equiv += 2

        tr_acc = correct_tr / (steps_per_epoch * BATCH_SIZE)
        te_acc = sum(
            (model(X_te_all[i:i+2000].to(device)).argmax(1) == y_te_all[i:i+2000].to(device)).sum().item()
            for i in range(0, len(y_te_all), 2000)
        ) / len(y_te_all)
        best_test = max(best_test, te_acc)
        curve_evals.append(eval_equiv)
        curve_acc.append(round(te_acc, 4))
        if ep % 5 == 0 or ep == epochs:
            print(f"  {ep:>6}  {tr_acc:>8.2%}  {te_acc:>8.2%}  {best_test:>8.2%}  {time.time()-t0:>6.0f}s")

    return {"method": name, "seed": seed,
            "best_test_acc": round(best_test, 4),
            "curve_evals": curve_evals, "curve_acc": curve_acc,
            "total_evals": eval_equiv, "wall_time": round(time.time() - t0, 1)}

# ---------------------------------------------------------------------------
# Data — carga global
# ---------------------------------------------------------------------------

def load_full_mnist():
    assert HAS_TV
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = ((ds_tr.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    X_te = ((ds_te.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
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
    print(f"EXPERIMENTO v30d: Comparacion completa MNIST 60K/10K + train_acc")
    print(f"SPSA/MeZO: {BUDGET_SPSA:,} evals  DGE: {BUDGET_DGE:,} evals")
    print(f"Train acc: subsample {TRAIN_ACC_N} muestras  Seeds: {SEEDS}")
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

        # Subsample fijo de train para calcular train_acc rapidamente
        rng_tr = np.random.default_rng(seed + 999)
        tr_sub_idx = rng_tr.choice(60000, TRAIN_ACC_N, replace=False)
        X_tr_sub = X_tr_d[tr_sub_idx]
        y_tr_sub = y_tr_d[tr_sub_idx]

        for mname in ["SPSA", "MeZO"]:
            r = run_spsa(mname, model, params0, X_tr_sub, y_tr_sub,
                         X_te_d, y_te_d, BUDGET_SPSA, seed, TOTAL_SPSA_STEPS)
            all_results.append(r)
            summary[mname].append(r["best_test_acc"])
            print(f"  [{mname}] best_test={r['best_test_acc']:.2%}  t={r['wall_time']}s")

        for mname, use_c in [("PureDGE", False), ("ConsistencyDGE", True)]:
            r = run_dge(mname, use_c, model, params0, X_tr_sub, y_tr_sub,
                        X_te_d, y_te_d, seed)
            all_results.append(r)
            summary[mname].append(r["best_test_acc"])
            print(f"  [{mname}] best_test={r['best_test_acc']:.2%}  t={r['wall_time']}s")

        for mname, lr in [("SGD", 0.01), ("Adam", 1e-3)]:
            r = run_grad(mname, X_tr_all, y_tr_all, X_te_all, y_te_all,
                         seed, lr, ADAM_EPOCHS)
            all_results.append(r)
            summary[mname].append(r["best_test_acc"])
            print(f"  [{mname}] best_test={r['best_test_acc']:.2%}  t={r['wall_time']}s")

    # --- Tabla resumen ---
    print(f"\n{'='*70}")
    print(f"RESULTADOS FINALES — MNIST 60K/10K  ({len(SEEDS)} seeds)")
    print(f"{'='*70}")
    ORDER = ["SPSA", "MeZO", "PureDGE", "ConsistencyDGE", "SGD", "Adam"]
    for m in ORDER:
        accs = summary[m]
        if not accs: continue
        mean, std = np.mean(accs), np.std(accs)
        bp  = "[backprop]  " if m in {"Adam","SGD"} else "[zero-order]"
        tag = "  ← OUR" if m == "ConsistencyDGE" else ""
        print(f"  {m:<18}: {mean:.2%} ± {std:.2%}  {bp}{tag}")

    # --- Guardar JSON ---
    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v30e_fullmnist_comparison.json"
    payload = {
        "experiment": "v30e_fullmnist_comparison",
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
