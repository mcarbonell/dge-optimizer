"""
scratch/dge_v3_mnist_test.py
============================
Prueba de integración de la nueva clase TorchDGEOptimizer (DGE V3).
Usa la clase optimizada y comprueba su velocidad y convergencia en MNIST.
"""

import sys
import os
import math
import time
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# Ensure we can import from dge
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dge.torch_optimizer import TorchDGEOptimizer

# Configuración
device = torch.device("cpu")
print(f"Device: {device}", flush=True)

ARCH = (784, 128, 64, 10)
TOTAL_EVALS = 1_000_000
K_BLOCKS = [1024, 128, 16]
LR = 0.05
DELTA = 1e-3
BATCH_SIZE = 256
SEED = 42

print(f"\n--- CONFIGURACION V3 ---", flush=True)
print(f"Arch: {ARCH} | Evals: {TOTAL_EVALS:,} | K: {K_BLOCKS}", flush=True)
print("-" * 50, flush=True)

# 1. Carga de Datos (Subsample para iteración rápida)
t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
ds_tr = datasets.MNIST('./data', train=True, download=True, transform=t)
ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)

X_tr_all = ((ds_tr.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
y_tr_all = ds_tr.targets

X_te_all = ((ds_te.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
y_te_all = ds_te.targets

# Tomamos un subsample para el entrenamiento (como en v9 y similares)
torch.manual_seed(SEED)
idx_tr = torch.randperm(len(y_tr_all))[:5000]
idx_te = torch.randperm(len(y_te_all))[:1000]

X_tr = X_tr_all[idx_tr].to(device)
y_tr = y_tr_all[idx_tr].long().to(device)
X_te = X_te_all[idx_te].to(device)
y_te = y_te_all[idx_te].long().to(device)


# 2. Red Neuronal Batched
class BatchedMLP:
    def __init__(self, arch):
        self.arch = list(arch)
        self.sizes = [a * b + b for a, b in zip(arch[:-1], arch[1:])]
        self.dim = sum(self.sizes)

    def forward(self, X, params_batch):
        # params_batch shape: (P, D)
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
    return F.cross_entropy(logits.reshape(P*B, C), t.reshape(-1), reduction='none').view(P, B).mean(dim=1)


model = BatchedMLP(ARCH)

# Inicialización de pesos base (Kaiming Normal)
torch.manual_seed(SEED)
params0 = torch.zeros(model.dim, device=device)
off = 0
for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
    std = math.sqrt(2.0 / l_in)
    w = l_in * l_out
    params0[off:off+w] = torch.randn(w, device=device) * std
    off += w + l_out

# 3. Inicializar Optimizador V3
opt = TorchDGEOptimizer(
    dim=model.dim,
    layer_sizes=model.sizes,
    k_blocks=K_BLOCKS,
    lr=LR,
    delta=DELTA,
    total_steps=TOTAL_EVALS // (2 * sum(K_BLOCKS)),
    consistency_window=20, # Activamos Direction-Consistency LR
    seed=SEED,
    device=device,
    chunk_size=128 # Añadimos chunking por si acaso el batch de 2336 eval loops consume mucha memoria
)

def evaluate(p):
    with torch.no_grad():
        logits = model.forward(X_te, p.unsqueeze(0)).squeeze(0)
        preds = logits.argmax(dim=1)
        acc = (preds == y_te).float().mean().item()
    return acc

def evaluate_train(p):
    with torch.no_grad():
        logits = model.forward(X_tr, p.unsqueeze(0)).squeeze(0)
        preds = logits.argmax(dim=1)
        acc = (preds == y_tr).float().mean().item()
    return acc


# 4. Bucle de Entrenamiento
rng_mb = torch.Generator(device=device)
rng_mb.manual_seed(SEED + 100)

params = params0.clone()
evals = 0
best_test = 0.0

print(f"\n[{'Evals':>8}] {'Train Acc':>10} | {'Test Acc':>10} | {'Best Test':>10} | {'Time (s)':>8}", flush=True)
print("-" * 60, flush=True)

t0 = time.time()
next_log = 50_000

while evals < TOTAL_EVALS:
    idx = torch.randperm(len(y_tr), generator=rng_mb, device=device)[:BATCH_SIZE]
    Xb, yb = X_tr[idx], y_tr[idx]
    
    # Wrapper para el optimizador V3
    def f_batched(p_batch):
        logits = model.forward(Xb, p_batch)
        return batched_ce(logits, yb)
        
    params, n = opt.step(f_batched, params)
    evals += n
    
    if evals >= next_log or evals >= TOTAL_EVALS:
        tr_acc = evaluate_train(params)
        te_acc = evaluate(params)
        best_test = max(best_test, te_acc)
        elapsed = time.time() - t0
        
        print(f"[{evals:>8,}] {tr_acc:>9.2%} | {te_acc:>9.2%} | {best_test:>9.2%} | {elapsed:>7.1f}s", flush=True)
        next_log += 50_000

print("-" * 60)
print(f"Entrenamiento completado en {time.time() - t0:.1f}s")
