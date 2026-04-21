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

class HybridSFWHT_DGE_V25:
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
        
        progress = min(self.t / max(self.total_steps, 1), 1.0)
        current_explore_p = base_explore_p + (0.15 - base_explore_p) * progress
        current_k_blocks = int(k_blocks * (1.0 + 0.5 * progress))
        
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
        
        explore_mask = torch.rand(actual_size, device=device) < current_explore_p
        final_mask = mask | explore_mask
        
        active_indices = torch.arange(actual_size, device=device)[final_mask]
        
        grad = torch.zeros(actual_size, device=device)
        
        if len(active_indices) > 0:
            active_perm = active_indices[torch.randperm(len(active_indices), device=device)]
            actual_k_blocks = min(current_k_blocks, len(active_perm))
            blocks = torch.tensor_split(active_perm, actual_k_blocks)
            
            num_perts = 2 * actual_k_blocks
            if num_perts > 0:
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
                if num_perts == 0:
                    return torch.empty(0, device=device)
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

class PureDGE_V25:
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
                if num_perts == 0:
                    return torch.empty(0, device=device)
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
    def __init__(self, arch=[784, 128, 64, 10]):
        self.arch = arch
        self.sizes = []
        for l_in, l_out in zip(arch[:-1], arch[1:]):
            self.sizes.append(l_in * l_out + l_out)
        self.dim = sum(self.sizes)
        print(f"Model Architecture: {arch} -> Total Parameters: {self.dim}")
        
    def forward_batched(self, X, params_batch):
        num_perts = params_batch.shape[0]
        if num_perts == 0: return torch.empty(0, device=device)
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
    if logits.numel() == 0: return torch.empty(0, device=device)
    targets_exp = targets.unsqueeze(0).expand(logits.shape[0], -1)
    P, B_size, C = logits.shape
    l = F.cross_entropy(logits.view(P*B_size, C), targets_exp.reshape(-1), reduction='none')
    return l.view(P, B_size).mean(dim=1)

def accuracy(logits, targets):
    if logits.numel() == 0: return 0.0
    preds = logits.squeeze(0).argmax(dim=1)
    return (preds == targets).float().mean().item()

def run_experiment(method, X_tr, y_tr, X_te, y_te, arch=[784, 128, 64, 10], budget=800_000):
    model = BatchedMLP(arch=arch)
    params = torch.zeros(model.dim, device=device)
    offset = 0
    for l_in, l_out in zip(model.arch[:-1], model.arch[1:]):
        std = math.sqrt(2.0 / l_in)
        w_size = l_in * l_out
        params[offset:offset+w_size] = torch.randn(w_size, device=device) * std
        offset += w_size + l_out

    if method == "pure":
        DGE_BLOCKS = [1024, 128, 16]
        est_evals = 2 * sum(DGE_BLOCKS)
        total_steps = budget // est_evals
        print(f"--- PURE DGE ---")
        print(f"Blocks: {DGE_BLOCKS}, Expected steps: {total_steps}, Est evals/step: {est_evals}")
        opt = PureDGE_V25(sizes=model.sizes, k_blocks=DGE_BLOCKS, total_steps=total_steps)
    else:
        B_LIST = [1024, 256, 32]
        TOP_K = [200, 50, 8]
        DGE_BLOCKS = [256, 32, 8]
        EXPLORE_PROB = [0.05, 0.05, 0.05]
        INTERVAL = 4
        # scan evals = 2*1024 + 2*256 + 2*32 = 2624
        # dge evals = 2*256 + 2*32 + 2*8 = 592
        # avg = (2624 + 4*592) / 4 = 1248
        est_evals = int((sum(B_LIST)*2 + INTERVAL*sum(DGE_BLOCKS)*2)/INTERVAL)
        total_steps = budget // est_evals
        print(f"--- HYBRID SFWHT+DGE ---")
        print(f"B_List: {B_LIST}, TopK: {TOP_K}, DGE Blocks: {DGE_BLOCKS}")
        print(f"Expected steps: {total_steps}, Est evals/step: {est_evals}")
        opt = HybridSFWHT_DGE_V25(model.sizes, B_LIST, TOP_K, DGE_BLOCKS, EXPLORE_PROB, INTERVAL, total_steps)

    evals = 0
    steps = 0
    t0 = time.time()
    best_te_acc = 0.0
    
    while evals < budget:
        idx = torch.randperm(X_tr.shape[0])[:256]
        Xb, yb = X_tr[idx], y_tr[idx]
        
        def f_loss(p_batch):
            logits = model.forward_batched(Xb, p_batch)
            return loss_fn_batched(logits, yb)
            
        params, n_evals = opt.step(f_loss, params)
        evals += n_evals
        steps += 1
        
        if steps % 50 == 0 or evals >= budget:
            with torch.no_grad():
                logits_te = model.forward_batched(X_te, params.unsqueeze(0))
                te_acc = accuracy(logits_te, y_te)
                loss_val = f_loss(params.unsqueeze(0)).item()
            best_te_acc = max(best_te_acc, te_acc)
            print(f"[{method.upper()}] Evals: {evals:>7} | Steps: {steps:>4} | Loss: {loss_val:.4f} | Test Acc: {te_acc:.2%}")

    wall_time = time.time() - t0
    return {"method": method, "evals": evals, "steps": steps, "best_test_acc": best_te_acc, "wall_time": wall_time}

if __name__ == "__main__":
    print("V25 SCALING HEAD-TO-HEAD: Medium Model (~109K params)")
    X_tr, y_tr, X_te, y_te = load_mnist_tensors()
    
    BUDGET = 800_000
    ARCH = [784, 128, 64, 10]
    
    res_pure = run_experiment("pure", X_tr, y_tr, X_te, y_te, arch=ARCH, budget=BUDGET)
    print("\n")
    res_hybrid = run_experiment("hybrid", X_tr, y_tr, X_te, y_te, arch=ARCH, budget=BUDGET)
    
    print("\n=== FINAL RESULTS ===")
    print(f"Pure DGE   | Acc: {res_pure['best_test_acc']:.2%} | Time: {res_pure['wall_time']:.1f}s | Evals: {res_pure['evals']}")
    print(f"Hybrid DGE | Acc: {res_hybrid['best_test_acc']:.2%} | Time: {res_hybrid['wall_time']:.1f}s | Evals: {res_hybrid['evals']}")
    
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v25_scaling_medium.json", "w") as f:
        json.dump({"pure": res_pure, "hybrid": res_hybrid}, f, indent=4)
