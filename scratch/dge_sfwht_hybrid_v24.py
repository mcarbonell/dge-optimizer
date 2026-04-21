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

class HybridSFWHT_DGE_V24:
    def __init__(self, sizes, B_list, top_k_buckets, dge_blocks, explore_prob, scan_interval, total_steps, lr0=0.05, eps0=5e-3, delta0=1e-3):
        self.sizes = sizes
        self.B_list = B_list
        self.top_k_buckets = top_k_buckets
        self.dge_blocks = dge_blocks
        self.explore_prob = explore_prob
        self.scan_interval = scan_interval
        self.total_steps = total_steps
        
        self.lr0 = lr0
        self.eps0 = eps0
        self.delta0 = delta0
        
        self.lr_decay = 0.01
        self.eps_decay = 0.1
        self.delta_decay = 0.1
        
        self.padded_sizes = []
        for s, b in zip(sizes, B_list):
            p = 2**math.ceil(math.log2(max(s, b)))
            self.padded_sizes.append(p)
            
        self.H_matrices = {b: create_hadamard(b) for b in set(B_list)}
        
        total_params = sum(sizes)
        self.m = torch.zeros(total_params, device=device)
        self.v = torch.zeros(total_params, device=device)
        self.t = 0
        
        self.cached_bucket_mask = [None] * len(sizes)
        
    def _cosine(self, v0, decay):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))
        
    def estimate_layer_gradient(self, layer_idx, f_layer, x_layer, D_pad, actual_size, B, num_top_buckets, k_blocks, eps, delta, base_explore_p):
        evals = 0
        
        # Progressive scaling
        progress = min(self.t / max(self.total_steps, 1), 1.0)
        # Explore goes from 5% to 15%
        current_explore_p = base_explore_p + (0.15 - base_explore_p) * progress
        # DGE blocks increase by 50% towards the end for finer micro-tuning
        current_k_blocks = int(k_blocks * (1.0 + 0.5 * progress))
        
        # ==========================================
        # PHASE 1: SFWHT Scan (Lazy Evaluation)
        # ==========================================
        if self.t % self.scan_interval == 0 or self.cached_bucket_mask[layer_idx] is None:
            H_B = self.H_matrices[B]
            V_0 = H_B.repeat(1, D_pad // B)
            
            perm = torch.randperm(actual_size, device=device)
            V_0_permuted = V_0[:, perm]
            
            perts_plus = x_layer.unsqueeze(0) + eps * V_0_permuted
            perts_minus = x_layer.unsqueeze(0) - eps * V_0_permuted
            
            Y_plus = f_layer(perts_plus)
            Y_minus = f_layer(perts_minus)
            
            U_base = fwht_tensor(Y_plus - Y_minus) / (2 * eps * B)
            evals += 2 * B
            
            _, top_buckets = torch.topk(torch.abs(U_base), num_top_buckets)
            
            bucket_ids = perm % B
            mask = torch.isin(bucket_ids, top_buckets)
            self.cached_bucket_mask[layer_idx] = mask
        else:
            mask = self.cached_bucket_mask[layer_idx]
        
        # Dynamic exploration mask (fresh every step)
        explore_mask = torch.rand(actual_size, device=device) < current_explore_p
        final_mask = mask | explore_mask
        
        active_indices = torch.arange(actual_size, device=device)[final_mask]
        
        # ==========================================
        # PHASE 2: DGE Refinement
        # ==========================================
        grad = torch.zeros(actual_size, device=device)
        
        if len(active_indices) > 0:
            active_perm = active_indices[torch.randperm(len(active_indices), device=device)]
            actual_k_blocks = min(current_k_blocks, len(active_perm))
            blocks = torch.tensor_split(active_perm, actual_k_blocks)
            
            num_perts = 2 * actual_k_blocks
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
        eps = self._cosine(self.eps0, self.eps_decay)
        delta = self._cosine(self.delta0, self.delta_decay)
        
        full_grad = torch.zeros_like(params)
        total_evals = 0
        
        offset = 0
        for i, (size, pad_size, B, num_top_buckets, k_blocks, explore_p) in enumerate(zip(
            self.sizes, self.padded_sizes, self.B_list, self.top_k_buckets, self.dge_blocks, self.explore_prob)):
            
            x_layer = params[offset:offset+size]
            
            def f_layer(p_layer):
                num_perts = p_layer.shape[0]
                full_p = params.unsqueeze(0).repeat(num_perts, 1)
                full_p[:, offset:offset+size] = p_layer
                return f_eval(full_p)
                
            g_layer, evs = self.estimate_layer_gradient(
                i, f_layer, x_layer, pad_size, size, B, num_top_buckets, k_blocks, eps, delta, explore_p
            )
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
    print("DGE v24: HYBRID SFWHT + DGE (Lazy Scan & Progressive) on MNIST")
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
    
    B_LIST = [512, 128]
    TOP_K_BUCKETS = [50, 12]
    DGE_BLOCKS = [128, 16]
    EXPLORE_PROB = [0.05, 0.05]
    SCAN_INTERVAL = 4
    
    # Evals with scan: 1280 (L1) + 288 (L2) = 1568
    # Evals without scan: 256 (L1) + 32 (L2) = 288
    # Avg evals per step: (1568 + 3*288) / 4 = 608
    ESTIMATED_AVG_EVALS = 608
    total_steps = TOTAL_BUDGET // ESTIMATED_AVG_EVALS
    
    print(f"Config: Interval={SCAN_INTERVAL}, Steps Expected={total_steps}")
    
    opt = HybridSFWHT_DGE_V24(
        sizes=model.sizes, 
        B_list=B_LIST, 
        top_k_buckets=TOP_K_BUCKETS, 
        dge_blocks=DGE_BLOCKS, 
        explore_prob=EXPLORE_PROB,
        scan_interval=SCAN_INTERVAL,
        total_steps=total_steps, 
        lr0=0.05, eps0=5e-3, delta0=1e-3
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
    with open(f"results/raw/sfwht_hybrid_v24.json", "w") as f_out:
        json.dump(metrics, f_out, indent=4)
        
    print(f"\nFinished in {total_time:.1f}s. Test Accuracy: {log_data['test_acc'][-1]:.2%}")