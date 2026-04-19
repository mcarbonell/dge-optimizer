import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys
import os
import json
from datetime import datetime

# =============================================================================
# OPTIMIZADOR DGE v15b (Orthogonal + Metrics Engine)
# =============================================================================
class DGEOptimizerV15b:
    def __init__(self, dim, lr=1.0, delta=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, lr_decay=0.01, total_steps=1000, 
                 greedy_w_init=0.3, greedy_w_final=0.0,
                 clip_norm=1.0, seed=None):
        self.dim = dim
        self.lr0 = lr
        self.delta = delta
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.lr_decay = lr_decay
        self.total_steps = total_steps
        self.greedy_w_init = greedy_w_init
        self.greedy_w_final = greedy_w_final
        self.clip_norm = clip_norm
        self.rng = np.random.default_rng(seed)
        self.k = max(1, math.ceil(math.log2(dim))) if dim > 1 else 1
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0
        
        # Diagnostics
        self.history_snr = []

    def _cosine_schedule(self, v_init, v_final):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v_final + (v_init - v_final) * 0.5 * (1 + math.cos(math.pi * frac))

    def step(self, f, x):
        self.t += 1
        lr = self._cosine_schedule(self.lr0, self.lr0 * self.lr_decay)
        greedy_w = self._cosine_schedule(self.greedy_w_init, self.greedy_w_final)

        perm = self.rng.permutation(self.dim)
        groups = np.array_split(perm, self.k)
        signs = self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
        g_step = np.zeros(self.dim, dtype=np.float32)
        
        best_s, best_dir = -1.0, np.zeros(self.dim, dtype=np.float32)

        for idx in groups:
            pert = np.zeros(self.dim, dtype=np.float32)
            pert[idx] = signs[idx] * self.delta
            fp, fm = f(x + pert), f(x - pert)
            sg = (fp - fm) / (2.0 * self.delta)
            g_step[idx] = sg * signs[idx]
            if abs(sg) > best_s:
                best_s = abs(sg)
                d = np.zeros(self.dim, dtype=np.float32)
                d[idx] = signs[idx]
                best_dir = -np.sign(sg) * d / (np.linalg.norm(d) + 1e-12)

        # Adam EMA
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_step
        self.v = self.beta2 * self.v + (1 - self.beta2) * g_step ** 2
        mh = self.m / (1 - self.beta1 ** self.t + 1e-30)
        vh = self.v / (1 - self.beta2 ** self.t + 1e-30)
        upd_ema = lr * mh / (np.sqrt(vh) + self.eps)
        
        if np.linalg.norm(upd_ema) > self.clip_norm:
            upd_ema *= self.clip_norm / np.linalg.norm(upd_ema)

        # Diagnostics: Correlation between step and EMA
        if self.t % 5 == 0:
            corr = np.corrcoef(g_step, self.m)[0, 1]
            if not np.isnan(corr): self.history_snr.append(float(corr))

        return x - upd_ema - (greedy_w * lr * best_dir), 2 * self.k

# =============================================================================
# EXPERIMENT ENGINE (v15b)
# =============================================================================
def run_experiment(config, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    X_tr = (train_set.data.float().view(-1, 784) / 255.0 - 0.1307) / 0.3081
    y_tr = train_set.targets
    X_te = (test_set.data.float().view(-1, 784) / 255.0 - 0.1307) / 0.3081
    y_te = test_set.targets
    
    # Subset for speed
    rng = np.random.default_rng(seed)
    idx_tr = rng.choice(len(y_tr), 3000, replace=False)
    X_tr, y_tr = X_tr[idx_tr].numpy(), y_tr[idx_tr].numpy()
    X_te, y_te = X_te[:600].numpy(), y_te[:600].numpy()
    
    # Model
    arch = config['arch']
    dim = sum(arch[i]*arch[i+1] + arch[i+1] for i in range(len(arch)-1))
    params = rng.normal(0, 0.1, dim).astype(np.float32)
    
    def forward(X, p):
        i, h = 0, X
        for l_in, l_out in zip(arch[:-1], arch[1:]):
            w = p[i:i+l_in*l_out].reshape(l_in, l_out)
            i += l_in*l_out
            b = p[i:i+l_out]
            i += l_out
            h = h @ w + b
            if l_out != arch[-1]: h = np.maximum(h, 0)
        return h

    def get_acc(X, y, p):
        return float(np.mean(forward(X, p).argmax(axis=1) == y))

    def loss_fn(Xb, yb, p):
        logits = forward(Xb, p)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))

    # Optimizer
    opt_cfg = config['optimizer']
    total_steps = config['budget'] // (2 * math.ceil(math.log2(dim)))
    opt = DGEOptimizerV15b(dim, total_steps=total_steps, seed=seed, **opt_cfg)
    
    evals, history_acc, history_evals = 0, [], []
    t_start = time.time()
    t_eval = 0.0
    
    rng_mb = np.random.default_rng(seed + 1)
    
    while evals < config['budget']:
        idx = rng_mb.integers(0, len(y_tr), 256)
        Xb, yb = X_tr[idx], y_tr[idx]
        
        def f(p):
            nonlocal t_eval
            t0 = time.time()
            l = loss_fn(Xb, yb, p)
            t_eval += time.time() - t0
            return l
            
        params, n = opt.step(f, params)
        evals += n
        
        if evals % 10000 < n:
            acc = get_acc(X_te, y_te, params)
            history_acc.append(acc)
            history_evals.append(evals)
            print(f"Seed {seed} | Evals {evals} | Acc {acc:.2%}")

    t_total = time.time() - t_start
    
    result = {
        "experiment_name": config['name'],
        "seed": seed,
        "config": config,
        "metrics": {
            "final_accuracy": get_acc(X_te, y_te, params),
            "total_evaluations": evals,
            "wall_clock_time": t_total,
            "function_evaluation_time": t_eval,
            "internal_overhead_time": t_total - t_eval,
            "snr_correlation_final": opt.history_snr[-1] if opt.history_snr else 0
        },
        "history": {"evaluations": history_evals, "objective_value": history_acc}
    }
    
    os.makedirs('results/raw', exist_ok=True)
    with open(f"results/raw/{config['name']}_seed{seed}.json", 'w') as f:
        json.dump(result, f, indent=2)
    return result

if __name__ == "__main__":
    # Hyperparameter Grid Search (v15b)
    # Testing beta1 (EMA depth) and greedy_w_init
    for b1 in [0.9, 0.95, 0.98]:
        for gw in [0.3, 0.5]:
            exp_name = f"mnist_v15b_b1_{b1}_gw{gw}"
            cfg = {
                "name": exp_name,
                "arch": [784, 32, 10],
                "budget": 60000,
                "optimizer": {
                    "beta1": b1,
                    "greedy_w_init": gw,
                    "lr": 0.5,
                    "clip_norm": 0.05
                }
            }
            print(f"\n>>> RUNNING GRID: {exp_name}")
            for s in [1, 2, 3]: # 3 seeds for the grid search speed
                run_experiment(cfg, s)
