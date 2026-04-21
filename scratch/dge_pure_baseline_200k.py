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

class PureDGE_V18:
    def __init__(self, sizes, k_blocks, total_steps, lr0=0.05, delta0=1e-3):
        self.sizes = sizes
        self.k_blocks = k_blocks
        self.total_steps = total_steps
        
        self.lr0 = lr0
        self.delta0 = delta0
        
        self.lr_decay = 0.01
        self.delta_decay = 0.1
        
        total_params = sum(sizes)
        self.m = torch.zeros(total_params, device=device)
        self.v = torch.zeros(total_params, device=device)
        self.t = 0
        
    def _cosine(self, v0, decay):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))
        
    def estimate_layer_gradient(self, f_layer, x_layer, actual_size, k_blocks, delta):
        evals = 0
        grad = torch.zeros(actual_size, device=device)
        
        perm = torch.randperm(actual_size, device=device)
        blocks = torch.tensor_split(perm, k_blocks)
        
        num_perts = 2 * k_blocks
        dge_perts = torch.zeros((num_perts, actual_size), device=device)
        signs_list = []
        
        for i, block in enumerate(blocks):
            if len(block) == 0: continue
            signs = torch.randint(0, 2, (len(block),), device=device).float() * 2 - 1
            dge_perts[2*i, block] = signs * delta
            dge_perts[2*i+1, block] = -signs * delta
            signs_list.append((block, signs))
            
        batch_x = x_layer.unsqueeze(0) + dge_perts
        Y_dge = f_layer(batch_x)
        evals += num_perts
        
        for i, (block, signs) in enumerate(signs_list):
            fp = Y_dge[2*i]
            fm = Y_dge[2*i+1]
            g_est = (fp - fm) / (2.0 * delta) * signs
            grad[block] = g_est
            
        return grad, evals

    def step(self, f_eval, params):
        lr = self._cosine(self.lr0, self.lr_decay)
        delta = self._cosine(self.delta0, self.delta_decay)
        
        full_grad = torch.zeros_like(params)
        total_evals = 0
        
        offset = 0
        for i, (size, k_blocks) in enumerate(zip(self.sizes, self.k_blocks)):
            x_layer = params[offset:offset+size]
            
            def f_layer(p_layer):
                num_perts = p_layer.shape[0]
                full_p = params.unsqueeze(0).repeat(num_perts, 1)
                full_p[:, offset:offset+size] = p_layer
                return f_eval(full_p)
                
            g_layer, evs = self.estimate_layer_gradient(f_layer, x_layer, size, k_blocks, delta)
            full_grad[offset:offset+size] = g_layer
            total_evals += evs
            offset += size
            
        self.m = 0.9 * self.m + 0.1 * full_grad
        self.v = 0.999 * self.v + 0.001 * (full_grad ** 2)
        mh = self.m / (1 - 0.9 ** (self.t + 1))
        vh = self.v / (1 - 0.999 ** (self.t + 1))
        upd = lr * mh / (torch.sqrt(vh) + 1e-8)
        
        self.t += 1
        
        return params - upd, total_evals

class BatchedMLP:
    def __init__(self, arch=[784, 32, 10]):
        self.arch = arch
        self.sizes = []
        for l_in, l_out in zip(arch[:-1], arch[1:]):
            self.sizes.append(l_in * l_out + l_out)
        self.dim = sum(self.sizes)
        
    def forward_batched(self, X, params_batch):
        num_perts = params_batch.shape[0]
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
    preds = logits.squeeze(0).argmax(dim=1)
    return (preds == targets).float().mean().item()

if __name__ == "__main__":
    print("DGE PURE BASELINE 200K BUDGET on MNIST")
    X_tr, y_tr, X_te, y_te = load_mnist_tensors()
    
    model = BatchedMLP(arch=[784, 32, 10])
    
    params = torch.zeros(model.dim, device=device)
    offset = 0
    for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
        std = math.sqrt(2.0 / l_in)
        w_size = l_in * l_out
        params[offset:offset+w_size] = torch.randn(w_size, device=device) * std
        offset += w_size + l_out
        
    TOTAL_BUDGET = 200_000
    
    # Pure DGE setup: similar to v18.
    DGE_BLOCKS = [256, 32]
    ESTIMATED_EVALS_PER_STEP = 2 * DGE_BLOCKS[0] + 2 * DGE_BLOCKS[1]
    total_steps = TOTAL_BUDGET // ESTIMATED_EVALS_PER_STEP
    
    print(f"Config: DGE_Blocks={DGE_BLOCKS}, Steps Expected={total_steps}")
    
    opt = PureDGE_V18(
        sizes=model.sizes, 
        k_blocks=DGE_BLOCKS, 
        total_steps=total_steps, 
        lr0=0.05, delta0=1e-3
    )
    
    evals = 0
    steps = 0
    log_data = {"evals": [], "train_acc": [], "test_acc": [], "loss": []}
    t0 = time.time()
    
    while evals < TOTAL_BUDGET:
        idx = torch.randperm(X_tr.shape[0])[:256]
        Xb, yb = X_tr[idx], y_tr[idx]
        
        def f_loss(p_batch):
            logits = model.forward_batched(Xb, p_batch)
            return loss_fn_batched(logits, yb)
            
        params, n_evals = opt.step(f_loss, params)
        evals += n_evals
        steps += 1
        
        if steps % 20 == 0 or evals >= TOTAL_BUDGET:
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
    with open(f"results/raw/dge_pure_baseline_200k.json", "w") as f_out:
        json.dump(metrics, f_out, indent=4)
        
    print(f"\nFinished in {total_time:.1f}s. Test Accuracy: {log_data['test_acc'][-1]:.2%}")