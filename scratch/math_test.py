import torch

D = 12
K = 3

torch.manual_seed(42)
x = torch.randn(D)

def f(x_batch):
    if x_batch.ndim == 1:
        x_batch = x_batch.unsqueeze(0)
    return torch.sum(x_batch ** 2, dim=1)

# LOOP
torch.manual_seed(123)
perm = torch.randperm(D)
blocks = torch.tensor_split(perm, K)
grad_loop = torch.zeros(D)
for block in blocks:
    signs = torch.randint(0, 2, (len(block),)).float() * 2 - 1
    pert = torch.zeros(D)
    pert[block] = signs * 1e-3
    fp = f(x + pert)
    fm = f(x - pert)
    grad_loop[block] = (fp - fm) / (2.0 * 1e-3) * signs

# BATCHED
torch.manual_seed(123)
perm2 = torch.randperm(D)
signs2 = torch.randint(0, 2, (D,)).float() * 2 - 1

group_size = (D + K - 1) // K
pad = group_size * K - D

if pad > 0:
    perm_pad = torch.cat([perm2, torch.zeros(pad, dtype=torch.long)])
    signs_pad = torch.cat([signs2, torch.zeros(pad)])
else:
    perm_pad = perm2
    signs_pad = signs2

perm_mat = perm_pad.view(K, group_size)
signs_mat = signs_pad.view(K, group_size) * 1e-3

P_plus = torch.zeros((K, D))
P_plus.scatter_(1, perm_mat, signs_mat)

if pad > 0:
    P_plus[:, 0] = 0.0
    idx0_mask = (perm2 == 0)
    idx0_pos = idx0_mask.nonzero(as_tuple=True)[0]
    if len(idx0_pos) > 0:
        block0 = idx0_pos[0] // group_size
        P_plus[block0, 0] = signs2[idx0_pos[0]] * 1e-3

P = torch.empty((2 * K, D))
P[0::2] = P_plus
P[1::2] = -P_plus

losses = f(x.unsqueeze(0) + P)
diffs = (losses[0::2] - losses[1::2]) / (2.0 * 1e-3)

grad_batch = torch.zeros(D)
diffs_exp = diffs.unsqueeze(1).expand(K, group_size).flatten()
if pad > 0:
    diffs_exp = diffs_exp[:D]

grad_batch[perm2] = diffs_exp * signs2

print("Grad Loop:", grad_loop)
print("Grad Batched:", grad_batch)
print("Equal?", torch.allclose(grad_loop, grad_batch))
