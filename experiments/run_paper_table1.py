"""
experiments/run_paper_table1.py
===============================
Canonical Experiment Runner to Reproduce Table 1 of the DGE Paper.

Usage:
    python experiments/run_paper_table1.py
    python experiments/run_paper_table1.py --quick
    python experiments/run_paper_table1.py --config experiments/configs/mnist_full_paper_table1.json

Outputs:
    results/raw/paper_table1_reproduction.json
"""

import os
import sys
import json
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dge.torch_optimizer import TorchDGEOptimizer
from experiments.utils import get_system_info, get_commit_hash, setup_result_directories

# Device setup
try:
    import torch_directml
    device = torch_directml.device()
    device_name = f"DirectML ({device})"
except ImportError:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = "CPU"


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
    def __init__(self, dim, seed, dev):
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)
        self.m = torch.zeros(dim, device=dev)
        self.v = torch.zeros(dim, device=dev)
        self.t = 0

    def update(self, grad, lr):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * (grad ** 2)
        mh = self.m / (1.0 - 0.9 ** self.t)
        vh = self.v / (1.0 - 0.999 ** self.t)
        return lr * mh / (torch.sqrt(vh) + 1e-8)


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


def run_spsa(model, params0, X_tr, y_tr, X_te, y_te, budget, seed, lr_zo, delta, batch_size, log_interval):
    dim = model.dim
    params = params0.clone()
    total_steps = max(1, budget // 2)
    state = AdamState(dim, seed, device)
    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)
    rng_pert = torch.Generator(device="cpu")
    rng_pert.manual_seed(seed + 200)

    evals = 0
    step = 0
    best_test = 0.0
    final_test = 0.0
    t0 = time.time()
    f_time = 0.0
    internal_time = 0.0
    P = torch.empty((2, dim), device=device)

    while evals < budget:
        t_step0 = time.time()
        f_start = f_time

        idx = torch.randperm(len(y_tr), generator=rng_mb)[:batch_size]
        Xb, yb = X_tr[idx], y_tr[idx]

        frac = min(step / max(total_steps, 1), 1.0)
        lr = lr_zo * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        d = delta * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))

        signs = (torch.randint(0, 2, (dim,), generator=rng_pert, device="cpu").float() * 2 - 1).to(device)
        P[0] = signs * d
        P[1] = -signs * d

        t_f0 = time.time()
        losses = batched_ce(model.forward(Xb, params.unsqueeze(0) + P), yb)
        f_time += time.time() - t_f0

        grad = (losses[0] - losses[1]) / (2.0 * d) * signs
        upd = state.update(grad, lr)
        params = params - upd

        evals += 2
        step += 1

        step_elapsed = time.time() - t_step0
        internal_time += max(0.0, step_elapsed - (f_time - f_start))

        if evals % log_interval == 0 or evals >= budget:
            te_acc = zo_acc(model, X_te, y_te, params)
            best_test = max(best_test, te_acc)
            final_test = te_acc

    return {
        "method": "Global SPSA",
        "seed": seed,
        "best_test_acc": round(best_test, 4),
        "final_test_acc": round(final_test, 4),
        "total_evals": evals,
        "wall_time": round(time.time() - t0, 2),
        "f_eval_time": round(f_time, 2),
        "internal_overhead_time": round(internal_time, 2)
    }


def run_dge(name, use_consistency, model, params0, X_tr, y_tr, X_te, y_te,
            budget, seed, k_blocks, lr_zo, delta, window, batch_size, log_interval):
    dim = model.dim
    est_step = 2 * sum(k_blocks)
    total_steps = max(1, budget // est_step)

    opt = TorchDGEOptimizer(
        dim=dim,
        layer_sizes=model.sizes,
        k_blocks=list(k_blocks),
        lr=lr_zo,
        delta=delta,
        total_steps=total_steps,
        consistency_window=window if use_consistency else 0,
        seed=seed,
        device=device,
        chunk_size=128,
        use_adam=True
    )

    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(seed + 100)

    evals = 0
    best_test = 0.0
    final_test = 0.0
    t0 = time.time()
    f_time = 0.0
    internal_time = 0.0

    while evals < budget:
        t_step0 = time.time()
        f_start = f_time

        idx = torch.randperm(len(y_tr), generator=rng_mb)[:batch_size]
        Xb, yb = X_tr[idx], y_tr[idx]

        def f_batched(p_batch):
            nonlocal f_time
            t_f0 = time.time()
            loss = batched_ce(model.forward(Xb, p_batch), yb)
            f_time += time.time() - t_f0
            return loss

        params, n = opt.step(f_batched, params)
        evals += n

        step_elapsed = time.time() - t_step0
        internal_time += max(0.0, step_elapsed - (f_time - f_start))

        if evals % log_interval == 0 or evals >= budget:
            te_acc = zo_acc(model, X_te, y_te, params)
            best_test = max(best_test, te_acc)
            final_test = te_acc

    return {
        "method": name,
        "seed": seed,
        "best_test_acc": round(best_test, 4),
        "final_test_acc": round(final_test, 4),
        "total_evals": evals,
        "wall_time": round(time.time() - t0, 2),
        "f_eval_time": round(f_time, 2),
        "internal_overhead_time": round(internal_time, 2)
    }


def run_backprop(name, arch, X_tr, y_tr, X_te, y_te, seed, lr, epochs, batch_size):
    torch.manual_seed(seed)
    net = TorchMLP(arch).to(device)
    for layer in net.net:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    opt = (torch.optim.Adam(net.parameters(), lr=lr) if name == "Adam"
           else torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9))

    criterion = nn.CrossEntropyLoss()
    n_samples = len(y_tr)
    t0 = time.time()
    best_test = 0.0

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            Xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            loss = criterion(net(Xb), yb)
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            preds = net(X_te).argmax(dim=1)
            te_acc = (preds == y_te).float().mean().item()
            best_test = max(best_test, te_acc)

    return {
        "method": name,
        "seed": seed,
        "best_test_acc": round(best_test, 4),
        "final_test_acc": round(te_acc, 4),
        "total_evals": epochs * (n_samples // batch_size),
        "wall_time": round(time.time() - t0, 2),
        "f_eval_time": 0.0,
        "internal_overhead_time": 0.0
    }


def main():
    parser = argparse.ArgumentParser(description="Canonical Table 1 Reproduction")
    parser.add_argument("--config", type=str, default="experiments/configs/mnist_full_paper_table1.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to run (default: from config)")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to run (default: all)")
    parser.add_argument("--quick", action="store_true", help="Quick verification mode (reduced budget)")
    parser.add_argument("--output", type=str, default="results/raw/paper_table1_reproduction.json")
    args = parser.parse_args()

    setup_result_directories()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    arch = cfg.get("architecture", [784, 128, 64, 10])
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [42, 43, 44])
    all_target_methods = ["Global SPSA", "Block-Adam (PureDGE)", "DGE (DS-EMA)", "SGD + momentum", "Adam"]
    selected_methods = args.methods if args.methods is not None else all_target_methods
    opt_cfg = cfg.get("optimizer", {})
    k_blocks = opt_cfg.get("k_blocks", [1024, 128, 16])
    lr_zo = opt_cfg.get("lr", 0.05)
    delta = opt_cfg.get("delta", 0.001)
    window = opt_cfg.get("consistency_window", 20)
    batch_size = cfg.get("batch_size", 256)

    budget_dge = 10_000 if args.quick else cfg.get("budget_dge", 3_000_000)
    budget_spsa = 5_000 if args.quick else cfg.get("budget_spsa", 300_000)
    log_interval = 2_000 if args.quick else 30_000

    print(f"\n=======================================================")
    print(f"CANONICAL TABLE 1 RUNNER (MNIST MLP 109K)")
    print(f"Device: {device_name}")
    print(f"Budget DGE: {budget_dge:,} | Budget SPSA: {budget_spsa:,}")
    print(f"Seeds: {seeds}")
    print(f"Output: {args.output}")
    print(f"=======================================================\n")

    # Load MNIST
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True, download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = (((ds_tr.data.float().view(-1, 784) / 255.0) - 0.1307) / 0.3081).to(device)
    y_tr = ds_tr.targets.to(device)
    X_te = (((ds_te.data.float().view(-1, 784) / 255.0) - 0.1307) / 0.3081).to(device)
    y_te = ds_te.targets.to(device)

    model = BatchedMLP(arch)
    results = []

    for seed in seeds:
        print(f"--- Running Seed {seed} ---")
        torch.manual_seed(seed)
        params0 = torch.zeros(model.dim, device=device)
        off = 0
        for l_in, l_out in zip(arch[:-1], arch[1:]):
            std = math.sqrt(2.0 / l_in)
            w = l_in * l_out
            params0[off:off + w] = torch.randn(w, device=device) * std
            off += w + l_out

        # Global SPSA
        if "Global SPSA" in selected_methods:
            r = run_spsa(model, params0, X_tr, y_tr, X_te, y_te, budget_spsa, seed, lr_zo, delta, batch_size, log_interval)
            results.append(r)
            print(f"  [Global SPSA]         Best Acc: {r['best_test_acc']:.2%}")

        # Block-Adam (PureDGE)
        if "Block-Adam (PureDGE)" in selected_methods:
            r = run_dge("Block-Adam (PureDGE)", False, model, params0, X_tr, y_tr, X_te, y_te,
                        budget_dge, seed, k_blocks, lr_zo, delta, window, batch_size, log_interval)
            results.append(r)
            print(f"  [Block-Adam (PureDGE)] Best Acc: {r['best_test_acc']:.2%}")

        # DGE Full (ConsistencyDGE)
        if "DGE (DS-EMA)" in selected_methods:
            r = run_dge("DGE (DS-EMA)", True, model, params0, X_tr, y_tr, X_te, y_te,
                        budget_dge, seed, k_blocks, lr_zo, delta, window, batch_size, log_interval)
            results.append(r)
            print(f"  [DGE (DS-EMA)]         Best Acc: {r['best_test_acc']:.2%}")

        # SGD + momentum
        if "SGD + momentum" in selected_methods:
            epochs = 2 if args.quick else 30
            r = run_backprop("SGD + momentum", arch, X_tr, y_tr, X_te, y_te, seed, 0.01, epochs, batch_size)
            results.append(r)
            print(f"  [SGD + momentum]       Best Acc: {r['best_test_acc']:.2%}")

        # Adam
        if "Adam" in selected_methods:
            epochs = 2 if args.quick else 30
            r = run_backprop("Adam", arch, X_tr, y_tr, X_te, y_te, seed, 1e-3, epochs, batch_size)
            results.append(r)
            print(f"  [Adam]                 Best Acc: {r['best_test_acc']:.2%}")

    # Summary
    summary = {}
    methods = [m for m in all_target_methods if any(r["method"] == m for r in results)]
    for m in methods:
        vals = [r["best_test_acc"] for r in results if r["method"] == m]
        summary[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": vals
        }

    out_data = {
        "experiment": "paper_table1_reproduction",
        "config": cfg,
        "hardware": get_system_info(),
        "commit_hash": get_commit_hash(),
        "summary": summary,
        "results": results
    }

    with open(args.output, 'w') as f:
        json.dump(out_data, f, indent=2)

    print(f"\n{'='*65}")
    print(f"TABLE 1 REPRODUCTION SUMMARY (Best Test Accuracy)")
    print(f"{'='*65}")
    print(f"{'Method':<25} {'Type':<15} {'Mean Acc':<12} {'Std':<10}")
    print(f"{'-'*65}")
    for m in methods:
        m_type = "Backprop" if m in ["Adam", "SGD + momentum"] else "Zeroth-order"
        print(f"{m:<25} {m_type:<15} {summary[m]['mean']*100:.2f}%     ±{summary[m]['std']*100:.2f}%")
    print(f"{'='*65}")
    print(f"Raw results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
