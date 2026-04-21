"""
dge_consistency_mnist_v28.py
============================
Experimento: ConsistencyDGE_T20 en MNIST.

Pregunta: la mejora de ConsistencyDGE en benchmarks sinteticos (v27)
se transfiere a una red neuronal real?

Comparacion directa y justa:
  - PureDGE_V28    : Version v25b limpia, es el baseline (81.17% en v25b)
  - ConsistencyDGE_V28: PureDGE + consistency LR mask (T=20)

Arquitectura identica al v25b para comparabilidad:
  - MLP [784 -> 128 -> 64 -> 10]  (~109K params)
  - n_train=3000, n_test=600 (subsample de MNIST)
  - Batch size=256, budget=800K evaluaciones
  - Seeds: 3 (seeds 42, 43, 44)
  - lr=0.05, delta=1e-3, k_blocks=[1024, 128, 16] (igual que v25b pure)

La unica diferencia entre PureDGE y ConsistencyDGE es la mascara de
consistencia aplicada al update de Adam:
  lr_effective_i = lr * |mean(sign(g_est_i) over last T=20 steps)|

NO se modifica dge_scaling_head_to_head_v25b.py (regla de oro).
"""

import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML Device: {device}")
except ImportError:
    device = torch.device("cpu")
    print(f"DirectML not found, using CPU")

# ---------------------------------------------------------------------------
# Datos MNIST
# ---------------------------------------------------------------------------

def load_mnist(n_train=3000, n_test=600, seed=42):
    assert HAS_TORCHVISION, "torchvision requerido: pip install torchvision"
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

    return X_tr_all, y_tr_all, X_te_all, y_te_all

# ---------------------------------------------------------------------------
# Modelo: BatchedMLP identico al v25b
# ---------------------------------------------------------------------------

class BatchedMLP:
    def __init__(self, arch=(784, 128, 64, 10)):
        self.arch = list(arch)
        self.sizes = []
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            self.sizes.append(l_in * l_out + l_out)
        self.dim = sum(self.sizes)

    def forward_batched(self, X, params_batch):
        """X: (n_samples, 784)  params_batch: (P, total_params)  -> (P, n_samples, 10)"""
        P = params_batch.shape[0]
        curr_x = X.unsqueeze(0).expand(P, -1, -1)
        i = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            w_size = l_in * l_out
            W = params_batch[:, i:i + w_size].view(P, l_in, l_out)
            i += w_size
            b = params_batch[:, i:i + l_out].view(P, 1, l_out)
            i += l_out
            curr_x = torch.bmm(curr_x, W) + b
            if l_out != self.arch[-1]:
                curr_x = torch.relu(curr_x)
        return curr_x   # (P, n_samples, 10)

def loss_fn_batched(logits, targets):
    """logits: (P, B, 10)  targets: (B,)  -> (P,)"""
    P, B, C = logits.shape
    targets_exp = targets.unsqueeze(0).expand(P, -1)
    l = F.cross_entropy(logits.reshape(P * B, C), targets_exp.reshape(-1), reduction='none')
    return l.view(P, B).mean(dim=1)

def accuracy(logits_single, targets):
    """logits_single: (B, 10)  targets: (B,)"""
    preds = logits_single.argmax(dim=1)
    return (preds == targets).float().mean().item()

# ---------------------------------------------------------------------------
# Optimizer 1: PureDGE_V28 — replica exacta del v25b PureDGE
# ---------------------------------------------------------------------------

class PureDGE_V28:
    """Pure DGE con batched evaluation. Replica exacta de PureDGE_V25."""

    def __init__(self, sizes, k_blocks, total_steps, lr0=0.05, delta0=1e-3, seed=42):
        self.sizes = sizes
        self.k_blocks = k_blocks
        self.total_steps = total_steps
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)

        self.lr0 = lr0
        self.delta0 = delta0

        total_params = sum(sizes)
        self.m = torch.zeros(total_params, device=device)
        self.v = torch.zeros(total_params, device=device)
        self.t = 0

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def _estimate_layer_grad(self, f_layer, x_layer, size, k_blocks, delta):
        grad = torch.zeros(size, device=device)
        perm = torch.randperm(size, generator=self.rng, device=device)
        blocks = torch.tensor_split(perm, k_blocks)

        n_perts = 2 * k_blocks
        dge_perts = torch.zeros((n_perts, size), device=device)
        signs_list = []

        for i, block in enumerate(blocks):
            if len(block) == 0:
                continue
            signs = torch.randint(0, 2, (len(block),), generator=self.rng, device=device).float() * 2 - 1
            dge_perts[2 * i, block] = signs * delta
            dge_perts[2 * i + 1, block] = -signs * delta
            signs_list.append((block, signs))

        Y = f_layer(x_layer.unsqueeze(0) + dge_perts)

        for i, (block, signs) in enumerate(signs_list):
            g_est = (Y[2 * i] - Y[2 * i + 1]) / (2.0 * delta) * signs
            grad[block] = g_est

        return grad, 2 * k_blocks

    def step(self, f_eval, params):
        lr    = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)

        full_grad = torch.zeros_like(params)
        total_evals = 0
        offset = 0

        for size, k in zip(self.sizes, self.k_blocks):
            x_layer = params[offset:offset + size]

            def f_layer(p_layer, _off=offset, _sz=size):
                P = p_layer.shape[0]
                full_p = params.unsqueeze(0).expand(P, -1).clone()
                full_p[:, _off:_off + _sz] = p_layer
                return f_eval(full_p)

            g, evs = self._estimate_layer_grad(f_layer, x_layer, size, k, delta)
            full_grad[offset:offset + size] = g
            total_evals += evs
            offset += size

        self.t += 1
        self.m = 0.9 * self.m + 0.1 * full_grad
        self.v = 0.999 * self.v + 0.001 * (full_grad ** 2)
        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self.v / (1.0 - 0.999 ** self.t)
        upd = lr * mh / (torch.sqrt(vh) + 1e-8)

        return params - upd, total_evals

# ---------------------------------------------------------------------------
# Optimizer 2: ConsistencyDGE_V28 — PureDGE + consistency mask (T=20)
# ---------------------------------------------------------------------------

class ConsistencyDGE_V28:
    """PureDGE_V28 + mascara de consistencia de signo de gradiente.

    lr_effective_i = lr * |mean(sign(g_i) over last T steps)|

    Identico a PureDGE en todo lo demas: perturbaciones, EMA, varianza, schedule.
    """

    def __init__(self, sizes, k_blocks, total_steps,
                 lr0=0.05, delta0=1e-3, seed=42, window=20):
        self.sizes = sizes
        self.k_blocks = k_blocks
        self.total_steps = total_steps
        self.window = window
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)

        self.lr0 = lr0
        self.delta0 = delta0

        total_params = sum(sizes)
        self.m = torch.zeros(total_params, device=device)
        self.v = torch.zeros(total_params, device=device)
        self.t = 0

        # Buffer de CPU para acumular signos (evita overhead de GPU para deque)
        self.sign_buffer = deque(maxlen=window)

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def _estimate_layer_grad(self, f_layer, x_layer, size, k_blocks, delta):
        grad = torch.zeros(size, device=device)
        perm = torch.randperm(size, generator=self.rng, device=device)
        blocks = torch.tensor_split(perm, k_blocks)

        n_perts = 2 * k_blocks
        dge_perts = torch.zeros((n_perts, size), device=device)
        signs_list = []

        for i, block in enumerate(blocks):
            if len(block) == 0:
                continue
            signs = torch.randint(0, 2, (len(block),), generator=self.rng, device=device).float() * 2 - 1
            dge_perts[2 * i, block] = signs * delta
            dge_perts[2 * i + 1, block] = -signs * delta
            signs_list.append((block, signs))

        Y = f_layer(x_layer.unsqueeze(0) + dge_perts)

        for i, (block, signs) in enumerate(signs_list):
            g_est = (Y[2 * i] - Y[2 * i + 1]) / (2.0 * delta) * signs
            grad[block] = g_est

        return grad, 2 * k_blocks

    def step(self, f_eval, params):
        lr    = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)

        full_grad = torch.zeros_like(params)
        total_evals = 0
        offset = 0

        for size, k in zip(self.sizes, self.k_blocks):
            x_layer = params[offset:offset + size]

            def f_layer(p_layer, _off=offset, _sz=size):
                P = p_layer.shape[0]
                full_p = params.unsqueeze(0).expand(P, -1).clone()
                full_p[:, _off:_off + _sz] = p_layer
                return f_eval(full_p)

            g, evs = self._estimate_layer_grad(f_layer, x_layer, size, k, delta)
            full_grad[offset:offset + size] = g
            total_evals += evs
            offset += size

        self.t += 1
        self.m = 0.9 * self.m + 0.1 * full_grad
        self.v = 0.999 * self.v + 0.001 * (full_grad ** 2)
        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self.v / (1.0 - 0.999 ** self.t)

        # --- Consistency mask ---
        self.sign_buffer.append(torch.sign(full_grad).cpu())
        if len(self.sign_buffer) < 2:
            consistency = torch.ones_like(params)
        else:
            stacked = torch.stack(list(self.sign_buffer), dim=0)  # (T, dim) on CPU
            consistency = stacked.mean(dim=0).abs().to(device)    # (dim,) on device

        upd = lr * consistency * mh / (torch.sqrt(vh) + 1e-8)

        return params - upd, total_evals

# ---------------------------------------------------------------------------
# Runner de un experimento completo
# ---------------------------------------------------------------------------

def init_params(model, seed):
    """Inicializacion He, identica al v25b."""
    params = torch.zeros(model.dim, device=device)
    offset = 0
    torch.manual_seed(seed)
    for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
        std = math.sqrt(2.0 / l_in)
        w_size = l_in * l_out
        params[offset:offset + w_size] = torch.randn(w_size, device=device) * std
        offset += w_size + l_out
    return params


def run_experiment(method, X_tr_all, y_tr_all, X_te_all, y_te_all,
                   arch=(784, 128, 64, 10), budget=800_000, seed=42,
                   lr=0.05, k_blocks=(1024, 128, 16), window=20):
    rng = np.random.default_rng(seed)
    n_train, n_test = 3000, 600
    tr_idx = rng.choice(len(y_tr_all), size=n_train, replace=False)
    te_idx = rng.choice(len(y_te_all), size=n_test,  replace=False)

    X_tr = X_tr_all[tr_idx].to(device)
    y_tr = y_tr_all[tr_idx].to(device)
    X_te = X_te_all[te_idx].to(device)
    y_te = y_te_all[te_idx].to(device)

    model = BatchedMLP(arch=arch)
    params = init_params(model, seed)

    # Estimar total_steps (igual que v25b)
    est_evals_per_step = 2 * sum(k_blocks)
    total_steps = budget // est_evals_per_step

    if method == "PureDGE":
        opt = PureDGE_V28(model.sizes, list(k_blocks), total_steps,
                          lr0=lr, delta0=1e-3, seed=seed)
    else:  # ConsistencyDGE
        opt = ConsistencyDGE_V28(model.sizes, list(k_blocks), total_steps,
                                 lr0=lr, delta0=1e-3, seed=seed, window=window)

    batch_rng = torch.Generator()
    batch_rng.manual_seed(seed + 100)

    evals = 0
    steps = 0
    best_te_acc = 0.0
    t0 = time.time()

    while evals < budget:
        idx = torch.randperm(X_tr.shape[0], generator=batch_rng)[:256]
        Xb, yb = X_tr[idx], y_tr[idx]

        def f_loss(p_batch, _Xb=Xb, _yb=yb):
            logits = model.forward_batched(_Xb, p_batch)
            return loss_fn_batched(logits, _yb)

        params, n_evals = opt.step(f_loss, params)
        evals += n_evals
        steps += 1

        if steps % 50 == 0 or evals >= budget:
            with torch.no_grad():
                logits_te = model.forward_batched(X_te, params.unsqueeze(0))
                te_acc = accuracy(logits_te.squeeze(0), y_te)
                best_te_acc = max(best_te_acc, te_acc)

    wall_time = time.time() - t0

    # Loss final sobre train (sin gradient)
    with torch.no_grad():
        logits_tr = model.forward_batched(X_tr, params.unsqueeze(0))
        final_loss = F.cross_entropy(logits_tr.squeeze(0), y_tr).item()

    return {
        "method": method,
        "seed": seed,
        "best_test_acc": best_te_acc,
        "final_train_loss": final_loss,
        "evals": evals,
        "steps": steps,
        "wall_time": round(wall_time, 1),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ARCH    = (784, 128, 64, 10)   # ~109K params, igual que v25b
    BUDGET  = 800_000
    LR      = 0.05
    K_BLOCKS = (1024, 128, 16)     # igual que v25b pure DGE
    SEEDS   = [42, 43, 44]
    WINDOW  = 20                    # T=20 confirmado en v27

    print("\n" + "=" * 68)
    print("EXPERIMENTO v28: ConsistencyDGE en MNIST")
    print(f"Arch: {ARCH}  Budget: {BUDGET:,}  LR: {LR}  T: {WINDOW}")
    print(f"k_blocks: {K_BLOCKS}  Seeds: {SEEDS}")
    print("=" * 68)

    X_tr_all, y_tr_all, X_te_all, y_te_all = load_mnist()

    all_results = []
    summary = {"PureDGE": [], "ConsistencyDGE": []}

    for seed in SEEDS:
        print(f"\n--- SEED {seed} ---")
        for method in ["PureDGE", "ConsistencyDGE"]:
            r = run_experiment(
                method, X_tr_all, y_tr_all, X_te_all, y_te_all,
                arch=ARCH, budget=BUDGET, seed=seed,
                lr=LR, k_blocks=K_BLOCKS, window=WINDOW,
            )
            summary[method].append(r["best_test_acc"])
            all_results.append(r)
            print(f"  [{method:<16}] acc={r['best_test_acc']:.2%}  "
                  f"loss={r['final_train_loss']:.4f}  "
                  f"evals={r['evals']:,}  t={r['wall_time']}s")

    print("\n" + "=" * 68)
    print("RESULTADOS FINALES (MEDIA ± STD sobre 3 seeds)")
    print("=" * 68)
    ref = np.mean(summary["PureDGE"])
    for method in ["PureDGE", "ConsistencyDGE"]:
        accs = summary[method]
        mean, std = np.mean(accs), np.std(accs)
        rel = (mean - ref) / (ref + 1e-10) * 100
        sign = f"+{rel:.1f}%" if rel > 0 else f"{rel:.1f}%"
        ref_note = " (baseline)" if method == "PureDGE" else f" ({sign} vs PureDGE)"
        print(f"  {method:<18}: {mean:.2%} ± {std:.2%}{ref_note}")

    # Referencia historica
    print(f"\n  Referencia v25b PureDGE: 81.17%")

    # Guardar JSON
    out_dir = Path(__file__).parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v28_consistency_mnist.json"

    payload = {
        "experiment": "v28_consistency_mnist",
        "arch": list(ARCH),
        "budget": BUDGET,
        "lr": LR,
        "k_blocks": list(K_BLOCKS),
        "window": WINDOW,
        "seeds": SEEDS,
        "summary": {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                        "values": v} for m, v in summary.items()},
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON guardado en: {out_path}")
