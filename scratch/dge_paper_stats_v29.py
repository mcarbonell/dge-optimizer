"""
dge_paper_stats_v29.py
======================
Recopilacion de estadisticas para el paper:
  1. 6 seeds (42-47) para consolidar la estimacion estadistica.
  2. Curvas de convergencia: accuracy de test en checkpoints cada 40K evals.

Includes seeds 42-44 from v28 (re-run with curve logging) + seeds 45-47 new.

Architecture: idéntica al v28 [784, 128, 64, 10] ~109K params.
Budget: 800K evals, k_blocks=(1024, 128, 16), lr=0.05, T=20.
Output:
  - results/raw/v29_paper_stats.json   (datos crudos por seed/checkpoint)
  - Tabla resumen en consola
  - Curva de convergencia en ASCII para preview rapido
"""

import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
ARCH        = (784, 128, 64, 10)   # mismo que v28
BUDGET      = 800_000
K_BLOCKS    = (1024, 128, 16)
LR          = 0.05
DELTA       = 1e-3
WINDOW      = 20
N_TRAIN     = 3_000
N_TEST      = 600
BATCH_SIZE  = 256
SEEDS       = [42, 43, 44, 45, 46, 47]   # 3 de v28 + 3 nuevas
CHECKPOINTS = list(range(40_000, BUDGET + 1, 40_000))   # cada 40K

# ---------------------------------------------------------------------------
# Modelo: BatchedMLP (igual que v28)
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
        return h  # (P, N, C)


def batched_ce(logits, targets):
    P, B, C = logits.shape
    t = targets.unsqueeze(0).expand(P, -1)
    return F.cross_entropy(logits.reshape(P * B, C), t.reshape(-1),
                           reduction='none').view(P, B).mean(dim=1)


def test_accuracy(model, X_te, y_te, params):
    with torch.no_grad():
        logits = model.forward(X_te, params.unsqueeze(0)).squeeze(0)
        return (logits.argmax(dim=1) == y_te).float().mean().item()


# ---------------------------------------------------------------------------
# Optimizadores (replica exacta de v28 para comparabilidad)
# ---------------------------------------------------------------------------

class _BaseOpt:
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

    def _layer_grad(self, f_layer, x_layer, size, k, delta):
        grad = torch.zeros(size, device=device)
        perm = torch.randperm(size, generator=self.rng, device=device)
        blocks = torch.tensor_split(perm, k)
        n_perts = 2 * k
        perts = torch.zeros((n_perts, size), device=device)
        signs_list = []
        for i, bl in enumerate(blocks):
            if len(bl) == 0:
                continue
            s = torch.randint(0, 2, (len(bl),), generator=self.rng, device=device).float() * 2 - 1
            perts[2 * i, bl] = s * delta
            perts[2 * i + 1, bl] = -s * delta
            signs_list.append((bl, s))
        Y = f_layer(x_layer.unsqueeze(0) + perts)
        for i, (bl, s) in enumerate(signs_list):
            grad[bl] = (Y[2 * i] - Y[2 * i + 1]) / (2.0 * delta) * s
        return grad, 2 * k

    def step(self, f_eval, params):
        lr    = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)
        full_grad = torch.zeros_like(params)
        total_evals = 0
        offset = 0
        for size, k in zip(self.sizes, self.k_blocks):
            xl = params[offset:offset + size]
            def f_layer(pl, _off=offset, _sz=size):
                P = pl.shape[0]
                fp = params.unsqueeze(0).expand(P, -1).clone()
                fp[:, _off:_off + _sz] = pl
                return f_eval(fp)
            g, evs = self._layer_grad(f_layer, xl, size, k, delta)
            full_grad[offset:offset + size] = g
            total_evals += evs
            offset += size
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * full_grad
        self.v = 0.999 * self.v + 0.001 * (full_grad ** 2)
        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self.v / (1.0 - 0.999 ** self.t)
        return full_grad, mh, vh, total_evals


class PureDGE_V29(_BaseOpt):
    def step(self, f_eval, params):
        grad, mh, vh, n = super().step(f_eval, params)
        lr = self._cosine(self.lr0)
        upd = lr * mh / (torch.sqrt(vh) + 1e-8)
        return params - upd, n


class ConsistencyDGE_V29(_BaseOpt):
    def __init__(self, *args, window=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.sign_buffer = deque(maxlen=window)

    def step(self, f_eval, params):
        grad, mh, vh, n = super().step(f_eval, params)
        lr = self._cosine(self.lr0)
        # Consistency mask
        self.sign_buffer.append(torch.sign(grad).cpu())
        if len(self.sign_buffer) < 2:
            mask = torch.ones_like(params)
        else:
            mask = torch.stack(list(self.sign_buffer)).mean(dim=0).abs().to(device)
        upd = lr * mask * mh / (torch.sqrt(vh) + 1e-8)
        return params - upd, n

# ---------------------------------------------------------------------------
# Datos MNIST
# ---------------------------------------------------------------------------

def load_mnist():
    assert HAS_TV, "pip install torchvision"
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = (ds_tr.data.float().view(-1, 784) / 255.0 - 0.1307) / 0.3081
    y_tr = ds_tr.targets
    X_te = (ds_te.data.float().view(-1, 784) / 255.0 - 0.1307) / 0.3081
    y_te = ds_te.targets
    return X_tr, y_tr, X_te, y_te


def init_params(model, seed):
    params = torch.zeros(model.dim, device=device)
    offset = 0
    torch.manual_seed(seed)
    for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
        std = math.sqrt(2.0 / l_in)
        w = l_in * l_out
        params[offset:offset + w] = torch.randn(w, device=device) * std
        offset += w + l_out
    return params

# ---------------------------------------------------------------------------
# Run experiment with convergence curve
# ---------------------------------------------------------------------------

def run_with_curve(method_name, opt_cls, X_tr_all, y_tr_all, X_te_all, y_te_all, seed):
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(y_tr_all), N_TRAIN, replace=False)
    te_idx = rng.choice(len(y_te_all), N_TEST,  replace=False)
    X_tr = X_tr_all[tr_idx].to(device)
    y_tr = y_tr_all[tr_idx].to(device)
    X_te = X_te_all[te_idx].to(device)
    y_te = y_te_all[te_idx].to(device)

    model = BatchedMLP(ARCH)
    params = init_params(model, seed)

    est_step_evals = 2 * sum(K_BLOCKS)
    total_steps = BUDGET // est_step_evals

    kwargs = dict(sizes=model.sizes, k_blocks=K_BLOCKS, total_steps=total_steps,
                  lr0=LR, delta0=DELTA, seed=seed + 10_000)
    if opt_cls == ConsistencyDGE_V29:
        opt = opt_cls(window=WINDOW, **kwargs)
    else:
        opt = opt_cls(**kwargs)

    batch_rng = torch.Generator()
    batch_rng.manual_seed(seed + 100)

    evals = 0
    best_acc = 0.0
    curve_evals = []
    curve_acc   = []
    next_ckpt_idx = 0
    t0 = time.time()

    while evals < BUDGET:
        idx = torch.randperm(X_tr.shape[0], generator=batch_rng)[:BATCH_SIZE]
        Xb, yb = X_tr[idx], y_tr[idx]

        def f(pb, _Xb=Xb, _yb=yb):
            return batched_ce(model.forward(_Xb, pb), _yb)

        params, n_evals = opt.step(f, params)
        evals += n_evals

        # Checkpoint
        if next_ckpt_idx < len(CHECKPOINTS) and evals >= CHECKPOINTS[next_ckpt_idx]:
            acc = test_accuracy(model, X_te, y_te, params)
            best_acc = max(best_acc, acc)
            curve_evals.append(evals)
            curve_acc.append(round(acc, 4))
            next_ckpt_idx += 1

    # Final accuracy if not already captured
    if not curve_evals or curve_evals[-1] < evals:
        acc = test_accuracy(model, X_te, y_te, params)
        best_acc = max(best_acc, acc)
        curve_evals.append(evals)
        curve_acc.append(round(acc, 4))

    wall = round(time.time() - t0, 1)
    return {
        "method": method_name,
        "seed": seed,
        "best_test_acc": round(best_acc, 4),
        "curve_evals": curve_evals,
        "curve_acc": curve_acc,
        "total_evals": evals,
        "wall_time": wall,
    }

# ---------------------------------------------------------------------------
# ASCII convergence preview
# ---------------------------------------------------------------------------

def ascii_curve(results, width=60, height=12):
    """Print a simple ASCII convergence curve for a quick preview."""
    methods_data = {}
    for r in results:
        m = r["method"]
        if m not in methods_data:
            methods_data[m] = {"evals": [], "acc": []}
        # Aggregate: mean over seeds at each checkpoint index
        for e, a in zip(r["curve_evals"], r["curve_acc"]):
            methods_data[m]["evals"].append(e)
            methods_data[m]["acc"].append(a)

    # Average over seeds per checkpoint
    all_e = sorted(set(e for d in methods_data.values() for e in d["evals"]))
    method_curves = {}
    for m, d in methods_data.items():
        from collections import defaultdict
        by_e = defaultdict(list)
        for e, a in zip(d["evals"], d["acc"]):
            by_e[e].append(a)
        method_curves[m] = {e: np.mean(v) for e, v in by_e.items()}

    all_acc = [a for d in method_curves.values() for a in d.values()]
    if not all_acc:
        return
    min_acc = max(0.1, min(all_acc) - 0.02)
    max_acc = min(1.0, max(all_acc) + 0.02)

    print(f"\n  Convergence Curve (mean over {len(SEEDS)} seeds)")
    print(f"  {'evals':>10}  " + "  ".join(f"{m[:18]:>18}" for m in method_curves))
    print(f"  {'-'*70}")
    for e in all_e[::2]:  # every other checkpoint
        row = f"  {e:>10,}  "
        for m, curve in method_curves.items():
            acc = curve.get(e, float('nan'))
            if not np.isnan(acc):
                bar = int((acc - min_acc) / (max_acc - min_acc) * 20) if max_acc > min_acc else 0
                row += f"  {acc:.2%} {'#'*bar:{20}}"
            else:
                row += f"  {'N/A':>6} {'':{20}}"
        print(row)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*68}")
    print(f"EXPERIMENTO v29: Estadisticas para el paper")
    print(f"Seeds: {SEEDS}  Budget: {BUDGET:,}  T: {WINDOW}")
    print(f"Arch: {ARCH}  k_blocks: {K_BLOCKS}")
    print(f"{'='*68}")

    X_tr_all, y_tr_all, X_te_all, y_te_all = load_mnist()

    all_results = []

    for seed in SEEDS:
        print(f"\n--- SEED {seed} ---")
        for method_name, opt_cls in [("PureDGE", PureDGE_V29),
                                      ("ConsistencyDGE", ConsistencyDGE_V29)]:
            r = run_with_curve(method_name, opt_cls,
                               X_tr_all, y_tr_all, X_te_all, y_te_all, seed)
            all_results.append(r)
            print(f"  [{method_name:<16}] best={r['best_test_acc']:.2%}  "
                  f"final={r['curve_acc'][-1]:.2%}  "
                  f"evals={r['total_evals']:,}  t={r['wall_time']}s")

    # --- Summary table ---
    print(f"\n{'='*68}")
    print(f"RESULTADOS FINALES ({len(SEEDS)} seeds)")
    print(f"{'='*68}")
    ref_mean = None
    for method in ["PureDGE", "ConsistencyDGE"]:
        accs = [r["best_test_acc"] for r in all_results if r["method"] == method]
        mean, std = np.mean(accs), np.std(accs)
        if ref_mean is None:
            ref_mean = mean
            rel = ""
        else:
            delta = (mean - ref_mean) / (ref_mean + 1e-10) * 100
            rel = f"  ({'+' if delta > 0 else ''}{delta:.1f}% vs PureDGE)"
        print(f"  {method:<18}: {mean:.2%} ± {std:.2%}{rel}")
        print(f"    per seed: {' '.join(f'{a:.2%}' for a in accs)}")

    print(f"\n  Referencia v25b PureDGE: 81.17%")
    print(f"  Referencia v28 Consistency (3 seeds): 87.56% ± 0.77%")

    # --- ASCII convergence curve ---
    ascii_curve(all_results)

    # --- Save JSON ---
    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v29_paper_stats.json"

    payload = {
        "experiment": "v29_paper_stats",
        "arch": list(ARCH),
        "budget": BUDGET,
        "k_blocks": list(K_BLOCKS),
        "lr": LR,
        "delta": DELTA,
        "window": WINDOW,
        "seeds": SEEDS,
        "checkpoints": CHECKPOINTS,
        "summary": {
            method: {
                "mean": float(np.mean([r["best_test_acc"] for r in all_results if r["method"] == method])),
                "std":  float(np.std( [r["best_test_acc"] for r in all_results if r["method"] == method])),
                "values": [r["best_test_acc"] for r in all_results if r["method"] == method],
            }
            for method in ["PureDGE", "ConsistencyDGE"]
        },
        "results": all_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON guardado en: {out_path}")
