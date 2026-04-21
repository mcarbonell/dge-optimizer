import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import time
import json
import os
import math

try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML Device: {device}")
except ImportError:
    device = torch.device("cpu")
    print(f"DirectML not found, using: {device}")

# =============================================================================
# DATA LOADING
# =============================================================================
def load_mnist_tensors(n_train=3000, n_test=600):
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

    rng = np.random.default_rng(42)
    tr_idx = rng.choice(len(y_tr_all), size=n_train, replace=False)
    te_idx = rng.choice(len(y_te_all), size=n_test,  replace=False)

    X_tr = X_tr_all[tr_idx].to(device)
    y_tr = y_tr_all[tr_idx].to(device)
    X_te = X_te_all[te_idx].to(device)
    y_te = y_te_all[te_idx].to(device)
    return X_tr, y_tr, X_te, y_te

# =============================================================================
# SFWHT ENGINE
# =============================================================================
def fwht_tensor(a):
    n = a.shape[-1]
    h = 1
    original_shape = a.shape
    a = a.view(-1, n)
    while h < n:
        a = a.view(a.shape[0], -1, h * 2)
        x = a[..., :h]
        y = a[..., h:]
        a = torch.cat((x + y, x - y), dim=-1)
        h *= 2
    return a.view(original_shape)

def create_hadamard(B):
    H = torch.tensor([[1.]], device=device)
    while H.shape[0] < B:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H

class LayerWiseSFWHT:
    def __init__(self, sizes, B=1024, epsilon=1e-3, threshold=None):
        self.sizes = sizes
        self.B = B
        self.epsilon = epsilon
        self.threshold = threshold
        
        # Prepare padding to power of 2 for each layer
        self.padded_sizes = []
        for s in sizes:
            p = 2**math.ceil(math.log2(max(s, B)))
            self.padded_sizes.append(p)
            
        self.H_B = create_hadamard(B)
        
        # Adam states
        total_params = sum(sizes)
        self.m = torch.zeros(total_params, device=device)
        self.v = torch.zeros(total_params, device=device)
        self.t = 0
        
    def estimate_layer_gradient(self, f_layer, x_layer, D_pad, actual_size):
        log_D = int(math.log2(D_pad))
        log_B = int(math.log2(self.B))
        
        V_0 = self.H_B.repeat(1, D_pad // self.B)
        
        # Base tests
        perts = x_layer.unsqueeze(0) + self.epsilon * V_0[:, :actual_size]
        Y_base = f_layer(perts)
        
        U_base = fwht_tensor(Y_base) / self.B
        f_x = Y_base[0].item() # approximate f(x)
        U_base[0] -= f_x
        
        thresh = self.threshold if self.threshold else (torch.median(torch.abs(U_base)) * 3.0 + 1e-5)
        active_buckets = torch.where(torch.abs(U_base) > thresh)[0]
        
        evals = self.B
        grad = torch.zeros(D_pad, device=device)
        
        if len(active_buckets) > 0:
            high_bits = log_D - log_B
            if high_bits > 0:
                U_shifts = torch.zeros((high_bits, self.B), device=device)
                j_indices = torch.arange(D_pad, device=device)
                
                for i, m in enumerate(range(log_B, log_D)):
                    flip_mask = ((j_indices & (1 << m)) != 0).float() * -2 + 1
                    V_m = V_0 * flip_mask.unsqueeze(0)
                    Y_m = f_layer(x_layer.unsqueeze(0) + self.epsilon * V_m[:, :actual_size])
                    evals += self.B
                    U_shifts[i] = fwht_tensor(Y_m) / self.B
                    
                for b in active_buckets:
                    val_base = U_base[b]
                    idx = b.item()
                    for i, m in enumerate(range(log_B, log_D)):
                        if torch.sign(val_base) != torch.sign(U_shifts[i, b]):
                            idx |= (1 << m)
                    g_est = val_base / self.epsilon
                    grad[idx] += g_est
            else:
                for b in active_buckets:
                    grad[b.item()] += U_base[b] / self.epsilon
                
        return grad[:actual_size], evals

    def step(self, f_eval, params, lr=0.01):
        self.t += 1
        full_grad = torch.zeros_like(params)
        total_evals = 0
        
        offset = 0
        for i, (size, pad_size) in enumerate(zip(self.sizes, self.padded_sizes)):
            x_layer = params[offset:offset+size]
            
            def f_layer(p_layer):
                # Reconstruct full params batch
                num_perts = p_layer.shape[0]
                full_p = params.unsqueeze(0).repeat(num_perts, 1)
                full_p[:, offset:offset+size] = p_layer
                return f_eval(full_p)
                
            g_layer, evs = self.estimate_layer_gradient(f_layer, x_layer, pad_size, size)
            full_grad[offset:offset+size] = g_layer
            total_evals += evs
            offset += size
            
        # Adam Update
        self.m = 0.9 * self.m + 0.1 * full_grad
        self.v = 0.999 * self.v + 0.001 * (full_grad ** 2)
        mh = self.m / (1 - 0.9 ** self.t)
        vh = self.v / (1 - 0.999 ** self.t)
        upd = lr * mh / (torch.sqrt(vh) + 1e-8)
        
        return params - upd, total_evals

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class BatchedMLP:
    def __init__(self, arch=[784, 32, 10]):
        self.arch = arch
        self.sizes = []
        for l_in, l_out in zip(arch[:-1], arch[1:]):
            self.sizes.append(l_in * l_out + l_out)
        self.dim = sum(self.sizes)
        
    def forward_batched(self, X, params_batch):
        num_perts = params_batch.shape[0]
        # X: (Batch, 784). Expand to (Num_Perts, Batch, 784)
        curr_x = X.unsqueeze(0).expand(num_perts, -1, -1)
        i = 0
        for l_in, l_out in zip(self.arch[:-1], self.arch[1:]):
            w_size = l_in * l_out
            W = params_batch[:, i:i+w_size].view(num_perts, l_in, l_out)
            i += w_size
            b = params_batch[:, i:i+l_out].view(num_perts, 1, l_out)
            i += l_out
            curr_x = torch.bmm(curr_x, W) + b
            if l_out != self.arch[-1]: curr_x = torch.relu(curr_x)
        return curr_x

def loss_fn_batched(logits, targets):
    targets_exp = targets.unsqueeze(0).expand(logits.shape[0], -1)
    P, B_size, C = logits.shape
    l = F.cross_entropy(logits.view(P*B_size, C), targets_exp.reshape(-1), reduction='none')
    return l.view(P, B_size).mean(dim=1)

def accuracy(logits, targets):
    # logits shape: (1, B_size, C)
    preds = logits.squeeze(0).argmax(dim=1)
    return (preds == targets).float().mean().item()

# =============================================================================
# MAIN SCRIPT
# =============================================================================
if __name__ == "__main__":
    print("DGE v20: SFWHT Layer-wise on MNIST")
    X_tr, y_tr, X_te, y_te = load_mnist_tensors()
    
    model = BatchedMLP(arch=[784, 32, 10])
    print(f"Model dimensions: {model.sizes} (Total: {model.dim})")
    
    # Initialize weights
    params = torch.zeros(model.dim, device=device)
    offset = 0
    for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
        std = math.sqrt(2.0 / l_in)
        w_size = l_in * l_out
        params[offset:offset+w_size] = torch.randn(w_size, device=device) * std
        offset += w_size + l_out
        
    opt = LayerWiseSFWHT(model.sizes, B=256, epsilon=5e-3)
    
    TOTAL_BUDGET = 200_000
    evals = 0
    steps = 0
    
    log_data = {
        "evals": [],
        "train_acc": [],
        "test_acc": [],
        "loss": []
    }
    
    t0 = time.time()
    
    while evals < TOTAL_BUDGET:
        # Minibatch
        idx = torch.randperm(X_tr.shape[0])[:256]
        Xb, yb = X_tr[idx], y_tr[idx]
        
        def f_loss(p_batch):
            logits = model.forward_batched(Xb, p_batch)
            return loss_fn_batched(logits, yb)
            
        params, n_evals = opt.step(f_loss, params, lr=0.05)
        evals += n_evals
        steps += 1
        
        if steps % 2 == 0 or evals >= TOTAL_BUDGET:
            with torch.no_grad():
                logits_tr = model.forward_batched(X_tr, params.unsqueeze(0))
                logits_te = model.forward_batched(X_te, params.unsqueeze(0))
                tr_acc = accuracy(logits_tr, y_tr)
                te_acc = accuracy(logits_te, y_te)
                loss_val = loss_fn_batched(model.forward_batched(Xb, params.unsqueeze(0)), yb).item()
                
            print(f"Evals: {evals:>7} | Steps: {steps:>4} | Loss: {loss_val:.4f} | Train Acc: {tr_acc:.2%} | Test Acc: {te_acc:.2%}")
            
            log_data["evals"].append(evals)
            log_data["train_acc"].append(tr_acc)
            log_data["test_acc"].append(te_acc)
            log_data["loss"].append(loss_val)
            
    total_time = time.time() - t0
    
    metrics = {
        "final_objective": log_data["loss"][-1],
        "total_evaluations": evals,
        "wall_clock_time": total_time,
        "test_accuracy": log_data["test_acc"][-1],
        "train_accuracy": log_data["train_acc"][-1],
        "d": model.dim,
        "architecture": model.arch
    }
    
    os.makedirs("results/raw", exist_ok=True)
    with open(f"results/raw/sfwht_mnist_v20.json", "w") as f_out:
        json.dump(metrics, f_out, indent=4)
        
    print(f"\nFinished in {total_time:.1f}s. Test Accuracy: {log_data['test_acc'][-1]:.2%}")