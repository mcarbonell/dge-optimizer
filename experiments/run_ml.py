import argparse
import json
import time
import os
import sys
import math
import numpy as np

# Ensure root directory is in path to import dge
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dge.optimizer import DGEOptimizer
from experiments.utils import save_raw_result
from experiments.baselines import SPSAOptimizer, RandomDirectionOptimizer

# Optional Torch dependency for ML datasets
try:
    import torch
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def forward_np(X, params, arch):
    i = 0
    h = X
    for layer_in, layer_out in zip(arch[:-1], arch[1:]):
        w_size = layer_in * layer_out
        W = params[i:i+w_size].reshape(layer_in, layer_out)
        i += w_size
        b = params[i:i+layer_out]
        i += layer_out
        h = h @ W + b
        if layer_out != arch[-1]:
            h = np.maximum(h, 0)   # ReLU
    return h

def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def full_accuracy(X, y, params, arch):
    logits = forward_np(X, params, arch)
    preds  = logits.argmax(axis=1)
    return float(np.mean(preds == y))

def load_mnist_subset(n_train, n_test, seed):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for MNIST benchmark.")
    
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

    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(y_tr_all), size=n_train, replace=False)
    te_idx = rng.choice(len(y_te_all), size=n_test,  replace=False)

    return X_tr_all[tr_idx].numpy(), y_tr_all[tr_idx].numpy(), X_te_all[te_idx].numpy(), y_te_all[te_idx].numpy()

def run_ml_experiment(config, seed):
    print(f"Running ML experiment '{config.get('name', 'unknown')}' with seed {seed}...")
    
    start_time = time.time()
    
    dataset = config.get("dataset", "mnist")
    if dataset != "mnist":
        raise ValueError(f"Unknown dataset {dataset}")
        
    n_train = config.get("n_train", 3000)
    n_test = config.get("n_test", 600)
    batch_size = config.get("batch_size", 256)
    budget = config.get("budget", 100000)
    arch = tuple(config.get("architecture", [784, 32, 10]))
    
    dim = sum(arch[i] * arch[i+1] + arch[i+1] for i in range(len(arch) - 1))
    
    X_train, y_train, X_test, y_test = load_mnist_subset(n_train, n_test, seed)
    
    rng = np.random.default_rng(seed)
    rng_mb = np.random.default_rng(seed + 1)
    
    # Glorot/He initialization
    params = np.zeros(dim, dtype=np.float32)
    i = 0
    for fan_in, fan_out in zip(arch[:-1], arch[1:]):
        w_size = fan_in * fan_out
        std = math.sqrt(2.0 / fan_in)
        params[i:i+w_size] = rng.normal(0, std, w_size).astype(np.float32)
        i += w_size
        i += fan_out
        
    opt_config = config.get("optimizer", {})
    opt_name = opt_config.get("name", "dge")
    
    if opt_name == "dge":
        k = opt_config.get("k", max(1, int(np.ceil(np.log2(dim)))))
        total_steps = budget // (2 * k)
        opt_params = {k: v for k, v in opt_config.items() if k not in ["name", "k"]}
        opt = DGEOptimizer(dim=dim, seed=seed+10, total_steps=total_steps, **opt_params)
    elif opt_name == "spsa":
        total_steps = budget // 2
        opt_params = {k: v for k, v in opt_config.items() if k not in ["name"]}
        opt = SPSAOptimizer(dim=dim, seed=seed+20, total_steps=total_steps, **opt_params)
    elif opt_name == "random":
        total_steps = budget // 2
        opt_params = {k: v for k, v in opt_config.items() if k not in ["name"]}
        opt = RandomDirectionOptimizer(dim=dim, seed=seed+30, total_steps=total_steps, **opt_params)
    else:
        raise ValueError(f"Unknown optimizer {opt_name}")

    evals = 0
    history_evals = []
    history_train_acc = []
    history_test_acc = []
    history_loss = []
    
    log_interval = max(1, budget // 10)
    next_log = 0

    internal_time = 0.0
    f_time = 0.0

    while evals < budget:
        if evals >= next_log:
            tr_acc = full_accuracy(X_train, y_train, params, arch)
            te_acc = full_accuracy(X_test, y_test, params, arch)
            history_evals.append(evals)
            history_train_acc.append(tr_acc)
            history_test_acc.append(te_acc)
            print(f"Evals: {evals}, Test Acc: {te_acc:.2%}")
            next_log += log_interval

        # Sample minibatch once per step (CRITICAL for DGE to cancel spatial noise, not data noise)
        idx = rng_mb.integers(0, len(y_train), size=batch_size)
        Xb, yb = X_train[idx], y_train[idx]

        def tracked_f(p):
            nonlocal f_time
            t_f0 = time.time()
            logits = forward_np(Xb, p, arch)
            probs = softmax_np(logits)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            loss = float(-np.mean(np.log(probs[np.arange(len(yb)), yb])))
            f_time += time.time() - t_f0
            return loss

        t0 = time.time()
        params, evals_used = opt.step(tracked_f, params)
        internal_time += time.time() - t0 - f_time
        
        evals += evals_used
        history_loss.append(tracked_f(params))

    # Final log
    tr_acc = full_accuracy(X_train, y_train, params, arch)
    te_acc = full_accuracy(X_test, y_test, params, arch)
    history_evals.append(evals)
    history_train_acc.append(tr_acc)
    history_test_acc.append(te_acc)
    print(f"Final Evals: {evals}, Final Test Acc: {te_acc:.2%}")

    total_time = time.time() - start_time
    
    history = {
        "evaluations": history_evals,
        "train_accuracy": history_train_acc,
        "test_accuracy": history_test_acc
    }
    metrics = {
        "final_test_accuracy": float(te_acc),
        "final_train_accuracy": float(tr_acc),
        "total_evaluations": evals,
        "wall_clock_time": total_time,
        "internal_overhead_time": internal_time,
        "function_evaluation_time": f_time
    }
    
    return history, metrics

def main():
    parser = argparse.ArgumentParser(description="DGE ML Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return

    with open(args.config, 'r') as f:
        config = json.load(f)

    exp_name = config.get("name", os.path.basename(args.config).split('.')[0])
    history, metrics = run_ml_experiment(config, args.seed)
    
    out_file = save_raw_result(exp_name, args.seed, config, history, metrics)
    print(f"Result saved to {out_file}")

if __name__ == "__main__":
    main()