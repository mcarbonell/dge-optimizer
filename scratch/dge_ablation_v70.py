"""
scratch/dge_ablation_v70.py
===========================
Estudio Riguroso de Ablación y Comparativa de Baselines Reales (Fase P0.1, P0.2, P0.3).

Compara de forma estandarizada y reproducible 5 configuraciones clave en MNIST (MLP 109K):
  1. Global SPSA    — Perturbación Rademacher full + Adam update (2 evals/step)
  2. MeZO Real      — Perturbación Gaussiana z~N(0,I) + Gradiente proyectado + SGD (Malladi et al., 2023)
  3. PureDGE_SGD    — Perturbación por bloques K + SGD directo (sin Adam EMA, sin consistencia)
  4. PureDGE_Adam   — Perturbación por bloques K + Adam EMA (sin consistencia)
  5. DGE_Full       — Perturbación por bloques K + Adam EMA + Máscara de consistencia (DS-EMA/Ventana)

Cumple con el estándar de logging y métricas de GEMINI.md:
  - Registra: final_acc, best_acc, total_evals, wall_time, f_eval_time, internal_overhead_time
  - Salida cruda estructurada en: results/raw/v70_ablation_study.json
"""

import sys
import os
import time
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dge.torch_optimizer import TorchDGEOptimizer
from experiments.utils import get_system_info, get_commit_hash

# ---------------------------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------------------------
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Device: DirectML ({device})")
except ImportError:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

# ---------------------------------------------------------------------------
# Architecture and Default Hyperparameters
# ---------------------------------------------------------------------------
ARCH = (784, 128, 64, 10)
K_BLOCKS = (1024, 128, 16)
LR_ZO = 0.05
DELTA = 1e-3
WINDOW = 20
BATCH_SIZE = 256
LOG_INTERVAL = 30_000
TRAIN_ACC_N = 5_000

EST_ZO_STEP = 2 * sum(K_BLOCKS)  # 2336 evals/paso para DGE

# ---------------------------------------------------------------------------
# Model and Evaluation Helpers
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


def zo_acc(model, X, y, params, chunk=1000):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), chunk):
            lo = model.forward(X[i:i + chunk], params.unsqueeze(0)).squeeze(0)
            correct += (lo.argmax(1) == y[i:i + chunk]).sum().item()
    return correct / len(y)


class AdamState:
    """Adam moment state tracker for vector updates."""
    def __init__(self, dim, seed, dev):
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)
        self.m = torch.zeros(dim, device=dev)
        self.v = torch.zeros(dim, device=dev)
        self.t = 0
        self.dev = dev

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
# Runner for Global Baselines (SPSA and Real MeZO)
# ---------------------------------------------------------------------------
def run_global_baseline(mode, model, params0, X_tr_all, y_tr_all, X_tr_sub, y_tr_sub,
                        X_te, y_te, budget, seed, log_interval):
    """
    Runs either SPSA (Rademacher + Adam) or real MeZO (Gaussian + SGD).
    Both evaluate 2 points per step (symmetric difference).
    """
    dim = model.dim
    params = params0.clone()
    total_steps = max(1, budget // 2)

    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)

    # State
    adam_state = AdamState(dim, seed, device) if mode == "SPSA" else None
    rng_pert = torch.Generator(device="cpu")
    rng_pert.manual_seed(seed + 200)

    evals = 0
    step = 0
    best_test = 0.0
    final_test = 0.0
    curve_evals, curve_acc = [], []
    next_log = log_interval

    f_time = 0.0
    internal_time = 0.0
    t0 = time.time()

    hdr = f"  {'evals':>10}  {'train_acc':>9}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}"
    print(f"\n  [{mode}] budget={budget:,} evals/step=2  steps={total_steps:,}")
    print(hdr)
    print(f"  {'-'*56}")

    P = torch.empty((2, dim), device=device)

    while evals < budget:
        t_step0 = time.time()
        f_time_step_start = f_time

        idx = torch.randperm(len(y_tr_all), generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr_all[idx], y_tr_all[idx]

        frac = min(step / max(total_steps, 1), 1.0)
        lr = LR_ZO * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        delta = DELTA * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))

        if mode == "SPSA":
            # Rademacher signs
            signs = (torch.randint(0, 2, (dim,), generator=rng_pert, device="cpu").float() * 2 - 1).to(device)
            pert = signs * delta
            P[0] = pert
            P[1] = -pert

            t_f0 = time.time()
            losses = batched_ce(model.forward(Xb, params.unsqueeze(0) + P), yb)
            f_time += time.time() - t_f0

            grad = (losses[0] - losses[1]) / (2.0 * delta) * signs
            upd = adam_state.update(grad, lr)
            params = params - upd
        elif mode == "MeZO":
            # Standard Gaussian perturbation z ~ N(0, I)
            z = torch.randn(dim, generator=rng_pert, device="cpu").to(device)
            pert = z * delta
            P[0] = pert
            P[1] = -pert

            t_f0 = time.time()
            losses = batched_ce(model.forward(Xb, params.unsqueeze(0) + P), yb)
            f_time += time.time() - t_f0

            # Projected gradient along z
            scalar_grad = (losses[0] - losses[1]) / (2.0 * delta)
            grad = scalar_grad * z
            # MeZO standard SGD update with cosine schedule
            params = params - lr * grad
        else:
            raise ValueError(f"Unknown global mode: {mode}")

        evals += 2
        step += 1

        step_elapsed = time.time() - t_step0
        step_f = f_time - f_time_step_start
        internal_time += max(0.0, step_elapsed - step_f)

        if evals >= next_log or evals >= budget:
            tr_acc = zo_acc(model, X_tr_sub, y_tr_sub, params)
            te_acc = zo_acc(model, X_te, y_te, params)
            best_test = max(best_test, te_acc)
            final_test = te_acc
            curve_evals.append(evals)
            curve_acc.append(round(te_acc, 4))
            elapsed = time.time() - t0
            print(f"  {evals:>10,}  {tr_acc:>8.2%}  {te_acc:>8.2%}  {best_test:>8.2%}  {elapsed:>6.0f}s")
            next_log += log_interval

    total_wall_time = time.time() - t0

    return {
        "method": mode,
        "seed": seed,
        "best_test_acc": round(best_test, 4),
        "final_test_acc": round(final_test, 4),
        "curve_evals": curve_evals,
        "curve_acc": curve_acc,
        "total_evals": evals,
        "wall_time": round(total_wall_time, 2),
        "f_eval_time": round(f_time, 2),
        "internal_overhead_time": round(internal_time, 2)
    }


# ---------------------------------------------------------------------------
# Runner for Block DGE Variants (PureDGE_SGD, PureDGE_Adam, DGE_Full)
# ---------------------------------------------------------------------------
def run_dge_variant(mode, model, params0, X_tr_all, y_tr_all, X_tr_sub, y_tr_sub,
                    X_te, y_te, budget, seed, log_interval):
    """
    Runs DGE variants with K blocks:
      - PureDGE_SGD: use_adam=False, consistency_window=0
      - PureDGE_Adam: use_adam=True, consistency_window=0
      - DGE_Full: use_adam=True, consistency_window=WINDOW (with DS-EMA)
    """
    total_steps = max(1, budget // EST_ZO_STEP)
    use_adam = (mode != "PureDGE_SGD")
    use_consistency = (mode == "DGE_Full")

    opt = TorchDGEOptimizer(
        dim=model.dim,
        layer_sizes=model.sizes,
        k_blocks=list(K_BLOCKS),
        lr=LR_ZO,
        delta=DELTA,
        total_steps=total_steps,
        consistency_window=WINDOW if use_consistency else 0,
        seed=seed,
        device=device,
        chunk_size=128,
        use_adam=use_adam
    )

    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)

    evals = 0
    best_test = 0.0
    final_test = 0.0
    curve_evals, curve_acc = [], []
    next_log = log_interval

    f_time = 0.0
    internal_time = 0.0
    t0 = time.time()

    hdr = f"  {'evals':>10}  {'train_acc':>9}  {'test_acc':>9}  {'best_test':>9}  {'time':>7}"
    print(f"\n  [{mode}] budget={budget:,} evals/step={EST_ZO_STEP} steps={total_steps:,}")
    print(hdr)
    print(f"  {'-'*56}")

    while evals < budget:
        t_step0 = time.time()
        f_time_step_start = f_time

        idx = torch.randperm(len(y_tr_all), generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr_all[idx], y_tr_all[idx]

        def f_batched(p_batch):
            nonlocal f_time
            t_f0 = time.time()
            logits = model.forward(Xb, p_batch)
            loss = batched_ce(logits, yb)
            f_time += time.time() - t_f0
            return loss

        params, n = opt.step(f_batched, params)
        evals += n

        step_elapsed = time.time() - t_step0
        step_f = f_time - f_time_step_start
        internal_time += max(0.0, step_elapsed - step_f)

        if evals >= next_log or evals >= budget:
            tr_acc = zo_acc(model, X_tr_sub, y_tr_sub, params)
            te_acc = zo_acc(model, X_te, y_te, params)
            best_test = max(best_test, te_acc)
            final_test = te_acc
            curve_evals.append(evals)
            curve_acc.append(round(te_acc, 4))
            elapsed = time.time() - t0
            print(f"  {evals:>10,}  {tr_acc:>8.2%}  {te_acc:>8.2%}  {best_test:>8.2%}  {elapsed:>6.0f}s")
            next_log += log_interval

    total_wall_time = time.time() - t0

    return {
        "method": mode,
        "seed": seed,
        "best_test_acc": round(best_test, 4),
        "final_test_acc": round(final_test, 4),
        "curve_evals": curve_evals,
        "curve_acc": curve_acc,
        "total_evals": evals,
        "wall_time": round(total_wall_time, 2),
        "f_eval_time": round(f_time, 2),
        "internal_overhead_time": round(internal_time, 2)
    }


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_mnist():
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True, download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = ((ds_tr.data.float().view(-1, 784) / 255.0) - 0.1307) / 0.3081
    X_te = ((ds_te.data.float().view(-1, 784) / 255.0) - 0.1307) / 0.3081
    return X_tr, ds_tr.targets, X_te, ds_te.targets


# ---------------------------------------------------------------------------
# Main Execution and Result Aggregation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DGE Ablation Study & Canonical Baselines (v70)")
    parser.add_argument("--methods", nargs="+",
                        default=["SPSA", "MeZO", "PureDGE_SGD", "PureDGE_Adam", "DGE_Full"],
                        help="List of methods to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                        help="Random seeds")
    parser.add_argument("--budget", type=int, default=3_000_000,
                        help="Evaluation budget for DGE methods (and baselines if spsa_budget is not set)")
    parser.add_argument("--spsa_budget", type=int, default=None,
                        help="Evaluation budget for SPSA/MeZO (default: same as budget)")
    parser.add_argument("--log_interval", type=int, default=LOG_INTERVAL,
                        help="Interval between accuracy evaluations")
    parser.add_argument("--output", type=str, default="results/raw/v70_ablation_study.json",
                        help="Path to save raw JSON results")
    args = parser.parse_args()

    spsa_budget = args.spsa_budget if args.spsa_budget is not None else args.budget

    print(f"\n{'='*75}")
    print(f"DGE ABLATION STUDY (v70)")
    print(f"Methods: {args.methods}")
    print(f"Seeds: {args.seeds}")
    print(f"Budget DGE: {args.budget:,} | Budget SPSA/MeZO: {spsa_budget:,}")
    print(f"Output: {args.output}")
    print(f"{'='*75}")

    print("\nLoading MNIST dataset...")
    X_tr_all, y_tr_all, X_te_all, y_te_all = load_mnist()
    X_tr_d = X_tr_all.to(device)
    y_tr_d = y_tr_all.to(device)
    X_te_d = X_te_all.to(device)
    y_te_d = y_te_all.to(device)

    # Load existing results if file exists to allow incremental runs
    existing_data = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r') as f:
                existing_data = json.load(f)
            print(f"Loaded existing results from {args.output}")
        except Exception as e:
            print(f"Warning: could not load existing results: {e}")

    all_results = existing_data.get("results", [])
    model = BatchedMLP(ARCH)

    for seed in args.seeds:
        print(f"\n{'#'*75}")
        print(f"  STARTING SEED {seed}")
        print(f"{'#'*75}")

        # Identical parameter initialization per seed
        torch.manual_seed(seed)
        params0 = torch.zeros(model.dim, device=device)
        off = 0
        for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
            std = math.sqrt(2.0 / l_in)
            w = l_in * l_out
            params0[off:off + w] = torch.randn(w, device=device) * std
            off += w + l_out

        # Subsample for fast training accuracy logging
        rng_tr = np.random.default_rng(seed + 999)
        tr_sub_idx = rng_tr.choice(len(y_tr_all), TRAIN_ACC_N, replace=False)
        X_tr_sub = X_tr_d[tr_sub_idx]
        y_tr_sub = y_tr_d[tr_sub_idx]

        for method in args.methods:
            # Check if this seed and method have already been computed
            already_done = any(r["method"] == method and r["seed"] == seed for r in all_results)
            if already_done:
                print(f"Skipping [{method}] for seed {seed} (already present in results).")
                continue

            if method in ["SPSA", "MeZO"]:
                r = run_global_baseline(
                    method, model, params0, X_tr_d, y_tr_d, X_tr_sub, y_tr_sub,
                    X_te_d, y_te_d, spsa_budget, seed, args.log_interval
                )
            elif method in ["PureDGE_SGD", "PureDGE_Adam", "DGE_Full"]:
                r = run_dge_variant(
                    method, model, params0, X_tr_d, y_tr_d, X_tr_sub, y_tr_sub,
                    X_te_d, y_te_d, args.budget, seed, args.log_interval
                )
            else:
                print(f"Warning: unrecognized method '{method}', skipping.")
                continue

            all_results.append(r)

    # Compute summary statistics
    summary = {}
    methods_present = sorted(list(set(r["method"] for r in all_results)))
    for m in methods_present:
        m_vals = [r["best_test_acc"] for r in all_results if r["method"] == m]
        m_final = [r.get("final_test_acc", r["best_test_acc"]) for r in all_results if r["method"] == m]
        summary[m] = {
            "best_acc_mean": float(np.mean(m_vals)),
            "best_acc_std": float(np.std(m_vals)),
            "final_acc_mean": float(np.mean(m_final)),
            "final_acc_std": float(np.std(m_final)),
            "seeds": [r["seed"] for r in all_results if r["method"] == m],
            "values": m_vals
        }

    out_data = {
        "experiment": "v70_ablation_study",
        "dataset": "mnist_full",
        "arch": list(ARCH),
        "budget_dge": args.budget,
        "budget_spsa": spsa_budget,
        "seeds": args.seeds,
        "hardware": get_system_info(),
        "commit_hash": get_commit_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summary,
        "results": all_results
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSuccessfully wrote results to {args.output}")

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY (Best Test Accuracy):")
    for m, stats in summary.items():
        print(f"  {m:<15}: {stats['best_acc_mean']*100:.2f}% ± {stats['best_acc_std']*100:.2f}%  (n={len(stats['values'])})")
    print("=" * 70)


if __name__ == "__main__":
    main()
