"""
dge_sign_activation_mnist_v31.py
================================
Fase 6: Validacion en redes totalmente no diferenciables (Sign Activations).
El gradiente analítico es nulo casi en todas partes, por lo que Adam fallará.
DGE V3 usará diferencias finitas a un delta moderado para "saltar" el umbral
y aproximar la dirección de descenso.
"""

import math
import time
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
# Configuración (Alineada con v11 para redes discretas)
# ---------------------------------------------------------------------------
ARCH = (784, 32, 10)
TOTAL_EVALS = 200_000
K_BLOCKS = [16, 4]  # Escalonado para 2 capas
LR = 0.5
DELTA = 5e-3
CLIP_NORM = 0.05    # CRITICO para redes discretas
BATCH_SIZE = 256
SEED = 42

EST_ZO_STEP = 2 * sum(K_BLOCKS)
TOTAL_DGE_STEPS = TOTAL_EVALS // EST_ZO_STEP

# ---------------------------------------------------------------------------
# Redes con funciones de activación SIGNO
# ---------------------------------------------------------------------------
class BatchedSignMLP:
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
                # ACTIVACION SIGNO
                h = torch.sign(h)
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
# Red Pytorch estándar para Adam
# ---------------------------------------------------------------------------
class SignActivation(nn.Module):
    def forward(self, x):
        return torch.sign(x)

class TorchSignMLP(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers = []
        for l_in, l_out in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(l_in, l_out))
            if l_out != arch[-1]: 
                layers.append(SignActivation())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def load_mnist_subset(n_train=5000, n_test=1000):
    assert HAS_TV
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    X_tr = ((ds_tr.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    X_te = ((ds_te.data.float().view(-1,784)/255.0) - 0.1307) / 0.3081
    
    rng = torch.Generator().manual_seed(SEED)
    tr_idx = torch.randperm(len(ds_tr), generator=rng)[:n_train]
    te_idx = torch.randperm(len(ds_te), generator=rng)[:n_test]
    
    return X_tr[tr_idx].to(device), ds_tr.targets[tr_idx].to(device), \
           X_te[te_idx].to(device), ds_te.targets[te_idx].to(device)

if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_mnist_subset()
    
    print(f"\n{'='*60}")
    print("FASE 6: SIGN ACTIVATION NETWORKS")
    print(f"{'='*60}")
    print("La activacion Sign(x) hace que el gradiente sea 0.")
    print("PyTorch/Adam deberian fallar miserablemente.")
    print("DGE deberia poder descender cruzando las discontinuidades.")
    print(f"{'-'*60}")
    
    # --- ADAM ---
    print("\n>>> EJECUTANDO ADAM (20 Epochs)...")
    torch.manual_seed(SEED)
    model_adam = TorchSignMLP(ARCH).to(device)
    
    # Copiamos la inicializacion exacta para que sea justa
    rng_mb = torch.Generator()
    rng_mb.manual_seed(SEED + 100)
    
    opt_adam = optim.Adam(model_adam.parameters(), lr=1e-3)
    epochs = 20
    steps_per_epoch = len(y_tr) // BATCH_SIZE
    
    for ep in range(1, epochs + 1):
        correct = 0
        for _ in range(steps_per_epoch):
            idx = torch.randperm(len(y_tr), generator=rng_mb)[:BATCH_SIZE]
            Xb, yb = X_tr[idx], y_tr[idx]
            opt_adam.zero_grad()
            logits = model_adam(Xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt_adam.step()
            correct += (logits.argmax(1) == yb).sum().item()
        
        if ep % 20 == 0 or ep == epochs:
            tr_a = correct / (steps_per_epoch * BATCH_SIZE)
            te_a = (model_adam(X_te).argmax(1) == y_te).float().mean().item()
            print(f"  [Adam] Epoch {ep:>3} | Train Acc: {tr_a:>6.2%} | Test Acc: {te_a:>6.2%}")

    # --- DGE ---
    print(f"\n>>> EJECUTANDO DGE V3 (Budget: {TOTAL_EVALS:,})...")
    model_dge = BatchedSignMLP(ARCH)
    
    # Extraer pesos de model_adam inicial (antes de que se "rompa" para ser justos, 
    # aunque en realidad no aprende nada, los reiniciamos a ka_normal)
    torch.manual_seed(SEED)
    params0 = torch.zeros(model_dge.dim, device=device)
    off = 0
    for l_in, l_out in zip(model_dge.arch[:-1], model_dge.arch[1:]):
        std = math.sqrt(2.0 / l_in)
        w = l_in * l_out
        params0[off:off+w] = torch.randn(w, device=device) * std
        off += w + l_out

    opt_dge = TorchDGEOptimizer(
        dim=model_dge.dim,
        layer_sizes=model_dge.sizes,
        k_blocks=K_BLOCKS,
        lr=LR,
        delta=DELTA,
        total_steps=TOTAL_DGE_STEPS,
        consistency_window=20,
        clip_norm=CLIP_NORM,
        seed=SEED,
        device=device,
        chunk_size=128
    )
    
    params = params0.clone()
    rng_mb.manual_seed(SEED + 100)
    
    evals = 0
    t0 = time.time()
    
    while evals < TOTAL_EVALS:
        idx = torch.randperm(len(y_tr), generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr[idx], y_tr[idx]
        
        def f_batched(p_batch):
            return batched_ce(model_dge.forward(Xb, p_batch), yb)
            
        params, n = opt_dge.step(f_batched, params)
        evals += n
        
        if evals % (EST_ZO_STEP * 20) < n or evals >= TOTAL_EVALS:
            tr_a = zo_acc(model_dge, X_tr, y_tr, params)
            te_a = zo_acc(model_dge, X_te, y_te, params)
            print(f"  [DGE] Evals: {evals:>9,} | Train Acc: {tr_a:>6.2%} | Test Acc: {te_a:>6.2%} | Time: {time.time()-t0:>4.0f}s", flush=True)

    print("\nFIN DEL EXPERIMENTO.")
