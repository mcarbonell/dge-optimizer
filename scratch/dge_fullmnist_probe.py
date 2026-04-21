"""
dge_fullmnist_probe.py
======================
Quick probe: ConsistencyDGE vs PureDGE vs Adam en MNIST COMPLETO (60K train, 10K test).

1 seed, 3 metodos, para estimar tiempos y accuracy antes de correr v30 completo.

Budget ZO: 3,000,000 evals (~1,284 pasos optimizer)
  Con 60K train y batch=256: 1,284 pasos * 256/60,000 ≈ 5.5 epochs de cobertura
Adam: 30 epochs = 30 * (60000//256) = 7,031 steps
"""

import math
import time
from collections import deque

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

ARCH       = (784, 128, 64, 10)
BUDGET     = 3_000_000      # evals ZO
K_BLOCKS   = (1024, 128, 16)
LR_ZO      = 0.05
DELTA      = 1e-3
WINDOW     = 20
BATCH_SIZE = 256
ADAM_EPOCHS = 30
SEED       = 42

# ---------------------------------------------------------------------------
# Modelo BatchedMLP (identico v28/v29)
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


def zo_accuracy(model, X_te, y_te, params):
    with torch.no_grad():
        # Evaluar en chunks de 1000 para no quedarse sin memoria en CPU
        accs = []
        for i in range(0, len(y_te), 1000):
            lo = model.forward(X_te[i:i+1000], params.unsqueeze(0)).squeeze(0)
            accs.append((lo.argmax(dim=1) == y_te[i:i+1000]).float().mean().item())
        return float(np.mean(accs))

# ---------------------------------------------------------------------------
# Optimizadores (replica minimal de v29)
# ---------------------------------------------------------------------------

class _BaseZO:
    def __init__(self, sizes, k_blocks, total_steps, lr0, delta0, seed):
        self.sizes = sizes
        self.k_blocks = list(k_blocks)
        self.total_steps = total_steps
        self.lr0, self.delta0 = lr0, delta0
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        total = sum(sizes)
        self.m = torch.zeros(total, device=device)
        self.v = torch.zeros(total, device=device)
        self.t = 0

    def _cosine(self, v0, decay=0.01):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1.0 - decay) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def _grad(self, f_eval, params, delta):
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
                s = torch.randint(0, 2, (len(bl),), generator=self.rng, device=device).float() * 2 - 1
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
        self.t += 1
        self.m = 0.9*self.m + 0.1*fg
        self.v = 0.999*self.v + 0.001*(fg**2)
        mh = self.m / (1.0 - 0.9**self.t)
        vh = self.v / (1.0 - 0.999**self.t)
        return fg, mh, vh, total_evals


class PureDGE(_BaseZO):
    def step(self, f, params):
        lr = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)
        _, mh, vh, n = self._grad(f, params, delta)
        return params - lr * mh / (torch.sqrt(vh) + 1e-8), n


class ConsistencyDGE(_BaseZO):
    def __init__(self, *args, window=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.buf = deque(maxlen=window)

    def step(self, f, params):
        lr = self._cosine(self.lr0)
        delta = self._cosine(self.delta0, decay=0.1)
        grad, mh, vh, n = self._grad(f, params, delta)
        self.buf.append(torch.sign(grad).cpu())
        mask = torch.stack(list(self.buf)).mean(0).abs().to(device) if len(self.buf) >= 2 else torch.ones_like(params)
        return params - lr * mask * mh / (torch.sqrt(vh) + 1e-8), n

# ---------------------------------------------------------------------------
# Data — full MNIST
# ---------------------------------------------------------------------------

def load_full_mnist():
    assert HAS_TV
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = ((ds_tr.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    y_tr = ds_tr.targets
    X_te = ((ds_te.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    y_te = ds_te.targets
    return X_tr, y_tr, X_te, y_te


def init_zo(model, seed):
    params = torch.zeros(model.dim, device=device)
    torch.manual_seed(seed)
    off = 0
    for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
        std = math.sqrt(2.0/l_in)
        w = l_in*l_out
        params[off:off+w] = torch.randn(w, device=device)*std
        off += w+l_out
    return params

# ---------------------------------------------------------------------------
# ZO runner
# ---------------------------------------------------------------------------

def run_zo(name, opt, model, params0, X_tr, y_tr, X_te, y_te):
    params = params0.clone()
    batch_rng = torch.Generator()
    batch_rng.manual_seed(SEED + 100)
    evals = 0
    best_acc = 0.0
    log_every = BUDGET // 10
    next_log = log_every
    t0 = time.time()
    print(f"\n  [{name}]")
    print(f"  {'evals':>10}  {'test_acc':>8}  {'best':>8}  {'time':>7}")
    print(f"  {'-'*42}")
    while evals < BUDGET:
        idx = torch.randperm(X_tr.shape[0], generator=batch_rng)[:BATCH_SIZE]
        Xb, yb = X_tr[idx].to(device), y_tr[idx].to(device)
        def f(pb, _Xb=Xb, _yb=yb):
            return batched_ce(model.forward(_Xb, pb), _yb)
        params, n = opt.step(f, params)
        evals += n
        if evals >= next_log or evals >= BUDGET:
            acc = zo_accuracy(model, X_te.to(device), y_te.to(device), params)
            best_acc = max(best_acc, acc)
            print(f"  {evals:>10,}  {acc:>7.2%}  {best_acc:>7.2%}  {time.time()-t0:>6.0f}s")
            next_log += log_every
    return best_acc

# ---------------------------------------------------------------------------
# Adam runner
# ---------------------------------------------------------------------------

def run_adam(X_tr, y_tr, X_te, y_te, epochs=ADAM_EPOCHS):
    class MLP(nn.Module):
        def __init__(self, arch):
            super().__init__()
            layers = []
            for l_in, l_out in zip(arch[:-1], arch[1:]):
                layers.append(nn.Linear(l_in, l_out))
                if l_out != arch[-1]: layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    torch.manual_seed(SEED)
    model = MLP(ARCH).to(device)
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    X_tr_d, y_tr_d = X_tr.to(device), y_tr.to(device)
    X_te_d, y_te_d = X_te.to(device), y_te.to(device)
    batch_rng = torch.Generator()
    batch_rng.manual_seed(SEED + 100)

    steps_per_epoch = len(y_tr) // BATCH_SIZE
    best_acc = 0.0
    t0 = time.time()
    print(f"\n  [Adam (backprop, {epochs} epochs)]")
    print(f"  {'epoch':>6}  {'test_acc':>8}  {'best':>8}  {'time':>7}")
    print(f"  {'-'*38}")
    for ep in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            idx = torch.randperm(len(y_tr_d), generator=batch_rng)[:BATCH_SIZE]
            opt.zero_grad()
            logits = model(X_tr_d[idx])
            F.cross_entropy(logits, y_tr_d[idx]).backward()
            opt.step()
        with torch.no_grad():
            acc = 0.0
            for i in range(0, len(y_te_d), 1000):
                lo = model(X_te_d[i:i+1000])
                acc += (lo.argmax(1) == y_te_d[i:i+1000]).float().sum().item()
            acc /= len(y_te_d)
        best_acc = max(best_acc, acc)
        if ep % 5 == 0 or ep == epochs:
            print(f"  {ep:>6}  {acc:>7.2%}  {best_acc:>7.2%}  {time.time()-t0:>6.0f}s")
    return best_acc

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"PROBE: Full MNIST (60K train, 10K test) — 1 seed={SEED}")
    print(f"ZO budget: {BUDGET:,}  ZO_steps: ~{BUDGET//(2*sum(K_BLOCKS))}")
    print(f"Adam: {ADAM_EPOCHS} epochs = ~{ADAM_EPOCHS*(60000//BATCH_SIZE)} steps")
    print(f"{'='*60}")

    X_tr, y_tr, X_te, y_te = load_full_mnist()
    X_tr_d, y_tr_d = X_tr.to(device), y_tr.to(device)
    X_te_d, y_te_d = X_te.to(device), y_te.to(device)

    model = BatchedMLP(ARCH)
    params0 = init_zo(model, SEED)
    total_steps = BUDGET // (2 * sum(K_BLOCKS))

    results = {}

    pure = PureDGE(model.sizes, K_BLOCKS, total_steps, LR_ZO, DELTA, SEED+10000)
    results["PureDGE"] = run_zo("PureDGE", pure, model, params0,
                                X_tr_d, y_tr_d, X_te_d, y_te_d)

    cons = ConsistencyDGE(model.sizes, K_BLOCKS, total_steps, LR_ZO, DELTA, SEED+10000, window=WINDOW)
    results["ConsistencyDGE"] = run_zo("ConsistencyDGE", cons, model, params0,
                                        X_tr_d, y_tr_d, X_te_d, y_te_d)

    results["Adam"] = run_adam(X_tr, y_tr, X_te, y_te)

    print(f"\n{'='*60}")
    print("RESUMEN FINAL — Full MNIST")
    print(f"{'='*60}")
    for m, acc in results.items():
        print(f"  {m:<18}: {acc:.2%}")
