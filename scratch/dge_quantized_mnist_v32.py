"""
dge_quantized_mnist_v32.py
================================
Fase 6: Quantization-Aware Training sin Straight-Through Estimator (STE).
Redes completamente cuantizadas (pesos y activaciones) a INT4 e INT8.
El gradiente analítico es nulo en casi todas partes debido a la función de redondeo.
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
# Configuración
# ---------------------------------------------------------------------------
ARCH = (784, 128, 64, 10)
TOTAL_EVALS = 600_000
K_BLOCKS = [1024, 128, 16]
BATCH_SIZE = 256
SEED = 42

EST_ZO_STEP = 2 * sum(K_BLOCKS)
TOTAL_DGE_STEPS = TOTAL_EVALS // EST_ZO_STEP

# ---------------------------------------------------------------------------
# Funciones de Cuantización
# ---------------------------------------------------------------------------
def fake_quantize(x, bits):
    """
    Simula la cuantización simétrica a `bits` bits.
    El rango asumido es [-1.0, 1.0]. 
    Ejemplo INT4 (bits=4): Q = 2^3 - 1 = 7. Niveles: -7/7 a 7/7 (15 niveles).
    """
    Q = (1 << (bits - 1)) - 1
    x_c = torch.clamp(x, -1.0, 1.0)
    return torch.round(x_c * Q) / Q

class QuantizedActivation(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.bits = bits
    def forward(self, x):
        return fake_quantize(torch.relu(x), self.bits)

# ---------------------------------------------------------------------------
# Redes
# ---------------------------------------------------------------------------
class BatchedQuantizedMLP:
    def __init__(self, arch, bits):
        self.arch = list(arch)
        self.sizes = [a * b + b for a, b in zip(arch[:-1], arch[1:])]
        self.dim = sum(self.sizes)
        self.bits = bits

    def forward(self, X, params_batch):
        # 1. Cuantizamos la entrada (Asume X en rango [0, 1])
        X_q = fake_quantize(X, self.bits)
        
        # 2. Cuantizamos TODOS los pesos
        params_q = fake_quantize(params_batch, self.bits)
        
        P = params_q.shape[0]
        h = X_q.unsqueeze(0).expand(P, -1, -1)
        i = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            W = params_q[:, i:i + l_in * l_out].view(P, l_in, l_out)
            i += l_in * l_out
            b = params_q[:, i:i + l_out].view(P, 1, l_out)
            i += l_out
            h = torch.bmm(h, W) + b
            
            # 3. Cuantizamos las activaciones
            if l_out != self.arch[-1]:
                h = torch.relu(h)
                # Opcional: escalar la salida antes de cuantizar si es muy grande, 
                # pero asumiremos que Kaiming init la mantiene ~ O(1)
                h = fake_quantize(h, self.bits)
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
class TorchQuantizedMLP(nn.Module):
    def __init__(self, arch, bits):
        super().__init__()
        self.bits = bits
        self.layers = nn.ModuleList()
        for l_in, l_out in zip(arch[:-1], arch[1:]):
            self.layers.append(nn.Linear(l_in, l_out))
            
    def forward(self, x):
        h = fake_quantize(x, self.bits)
        for i, layer in enumerate(self.layers):
            # Cuantizamos pesos al vuelo para imitar a DGE
            w_q = fake_quantize(layer.weight, self.bits)
            b_q = fake_quantize(layer.bias, self.bits)
            h = F.linear(h, w_q, b_q)
            if i < len(self.layers) - 1:
                h = fake_quantize(torch.relu(h), self.bits)
        return h

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def load_mnist_subset(n_train=5000, n_test=1000):
    assert HAS_TV
    t = transforms.Compose([transforms.ToTensor()])
    ds_tr = datasets.MNIST('./data', train=True,  download=True, transform=t)
    ds_te = datasets.MNIST('./data', train=False, download=True, transform=t)
    
    # Normalizacion ajustada a [0, 1] para evitar que los valores se salgan
    # del rango de cuantización fake_quantize que asume [-1, 1]
    X_tr = ds_tr.data.float().view(-1,784)/255.0
    X_te = ds_te.data.float().view(-1,784)/255.0
    
    rng = torch.Generator().manual_seed(SEED)
    tr_idx = torch.randperm(len(ds_tr), generator=rng)[:n_train]
    te_idx = torch.randperm(len(ds_te), generator=rng)[:n_test]
    
    return X_tr[tr_idx].to(device), ds_tr.targets[tr_idx].to(device), \
           X_te[te_idx].to(device), ds_te.targets[te_idx].to(device)

def run_adam(bits):
    print(f"\n>>> EJECUTANDO ADAM - INT{bits} (100 Epochs)...")
    X_tr, y_tr, X_te, y_te = load_mnist_subset()
    torch.manual_seed(SEED)
    model = TorchQuantizedMLP(ARCH, bits).to(device)
    
    rng_mb = torch.Generator()
    rng_mb.manual_seed(SEED + 100)
    
    opt = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100
    steps_per_epoch = len(y_tr) // BATCH_SIZE
    
    for ep in range(1, epochs + 1):
        correct = 0
        for _ in range(steps_per_epoch):
            idx = torch.randperm(len(y_tr), generator=rng_mb)[:BATCH_SIZE]
            Xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            logits = model(Xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
        
        if ep % 20 == 0 or ep == epochs:
            tr_a = correct / (steps_per_epoch * BATCH_SIZE)
            te_a = (model(X_te).argmax(1) == y_te).float().mean().item()
            print(f"  [Adam INT{bits}] Epoch {ep:>3} | Train Acc: {tr_a:>6.2%} | Test Acc: {te_a:>6.2%}")

def run_dge(bits):
    print(f"\n>>> EJECUTANDO DGE V3 - INT{bits} (Budget: {TOTAL_EVALS:,})...")
    X_tr, y_tr, X_te, y_te = load_mnist_subset()
    model = BatchedQuantizedMLP(ARCH, bits)
    
    # Inicialización de He/Kaiming pero limitando la desviación estandar 
    # para que caiga limpiamente dentro del rango de cuantización [-1, 1]
    torch.manual_seed(SEED)
    params0 = torch.zeros(model.dim, device=device)
    off = 0
    for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
        std = math.sqrt(2.0 / l_in)
        w = l_in * l_out
        # Usar distribucion uniforme recortada
        limit = math.sqrt(3.0) * std
        params0[off:off+w] = (torch.rand(w, device=device) * 2 - 1) * min(limit, 0.99)
        off += w + l_out

    # Hiperparámetros dinámicos según agresividad de cuantización
    q_max = (1 << (bits - 1)) - 1
    delta_necesario = (1.0 / q_max) * 1.1  # 10% más ancho que un escalón
    
    if bits <= 4:
        lr_usado = 0.5
        clip_usado = 0.05
    else:
        lr_usado = 0.05
        clip_usado = 0.1

    print(f"  [Config] Delta: {delta_necesario:.4f} | LR: {lr_usado} | Clip Norm: {clip_usado}")

    opt = TorchDGEOptimizer(
        dim=model.dim,
        layer_sizes=model.sizes,
        k_blocks=K_BLOCKS,
        lr=lr_usado,
        delta=delta_necesario,
        total_steps=TOTAL_DGE_STEPS,
        consistency_window=20,
        clip_norm=clip_usado,
        seed=SEED,
        device=device,
        chunk_size=128
    )
    
    params = params0.clone()
    rng_mb = torch.Generator()
    rng_mb.manual_seed(SEED + 100)
    
    evals = 0
    t0 = time.time()
    
    while evals < TOTAL_EVALS:
        idx = torch.randperm(len(y_tr), generator=rng_mb)[:BATCH_SIZE]
        Xb, yb = X_tr[idx], y_tr[idx]
        
        def f_batched(p_batch):
            return batched_ce(model.forward(Xb, p_batch), yb)
            
        params, n = opt.step(f_batched, params)
        evals += n
        
        if evals % (EST_ZO_STEP * 10) < n or evals >= TOTAL_EVALS:
            tr_a = zo_acc(model, X_tr, y_tr, params)
            te_a = zo_acc(model, X_te, y_te, params)
            print(f"  [DGE INT{bits}] Evals: {evals:>9,} | Train Acc: {tr_a:>6.2%} | Test Acc: {te_a:>6.2%} | Time: {time.time()-t0:>4.0f}s", flush=True)

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("FASE 6: QUANTIZED NETWORKS (INT4 / INT8)")
    print(f"{'='*60}")
    print("Simulación de Full Quantization (Pesos y Activaciones).")
    
    # 1. INT8 (256 niveles)
    print(f"\n{'='*60}\nPRUEBA 1: INT8\n{'='*60}")
    run_adam(8)
    run_dge(8)

    # 2. INT4 (16 niveles)
    print(f"\n{'='*60}\nPRUEBA 2: INT4\n{'='*60}")
    run_adam(4)
    run_dge(4)

    print("\nFIN DEL EXPERIMENTO.")
