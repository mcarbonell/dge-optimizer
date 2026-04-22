# DGE Optimizer (Denoised Gradient Estimation)

**DGE** (**Denoised Gradient Estimation**) is a novel zeroth-order (derivative-free) optimizer that trains Neural Networks **without backpropagation**, using only forward passes (evaluations of the objective function).

It combines randomized block perturbations with temporal denoising via Adam EMA to isolate clean gradient signals from noise — scaling efficiently to 100K+ parameters at a fraction of the cost of finite differences.

---

## 🏆 Latest Result (v28 — April 2026)

**ConsistencyDGE achieves 87.56% ± 0.77% on MNIST** (3000-sample train subset, 800K function evaluations, MLP 784→128→64→10 ~109K params) — a **+7.56 percentage-point improvement** over the Pure DGE baseline (80.00% ± 1.57%) with the **exact same evaluation budget**.

The only change: a 5-line *direction-consistency mask* that scales each parameter's effective learning rate by the confidence of its estimated gradient direction.

| Method | MNIST Acc | Train Loss | Wall-clock |
|---|---|---|---|
| Pure DGE (v25b baseline) | 80.00% ± 1.57% | 0.233 | ~220s |
| **ConsistencyDGE T=20 (v28)** | **87.56% ± 0.77%** | **0.025** | ~221s |

---

## 🚀 Core Innovation

Standard zeroth-order algorithms face a brutal trade-off:

1. **Finite Differences:** $O(D)$ evaluations per step — perfect precision, computationally impossible for large models.
2. **SPSA:** $O(1)$ evaluations — but perturbs all dimensions at once. At high dimension ($D > 10^4$), noise from co-variables drowns the signal.

**DGE's solution: Randomized Block Perturbations + Temporal Denoising**

Per step, DGE evaluates $k$ random non-overlapping blocks of parameters (2 evaluations per block). The raw gradient estimate is noisy. Over time, an **Adam EMA** accumulates these estimates: the noise from co-perturbations cancels to zero, isolating the true latent gradient for each individual parameter.

**Direction-Consistency LR** (introduced in v27–v28) adds a per-parameter confidence weight: if a parameter's estimated gradient sign has been consistent for the last $T=20$ steps, it receives a full learning rate. If its sign oscillates (noise-dominated), its update is suppressed. This adds zero function evaluations and $O(T \cdot D)$ memory.

---

## 📊 Experimental Results

### MNIST Benchmark — Progression

| Version | Architecture | Budget (evals) | Test Acc | Note |
|:---|:---|:---|:---|:---|
| v8c | Shallow (784→32→10) | 1,000,000 | **90.40%** | Early baseline |
| v9 | Shallow (784→32→10) | 100,000 | **87.50%** | Budget-efficient |
| v10 | Deep (784→64×4→10) | 150,000 | **85.67%** | First depth test |
| v11 | Non-diff (Step fn) | 120,000 | **65.00%** | Outperforms Adam (61%) |
| v12 | Binary {-1,1} weights | 150,000 | **73.50%** | 1-bit weight training |
| v13 | Ternary {-1,0,1} | 150,000 | **67.67%** | 50% sparsity |
| v25b | Deep MLP ~109K params | 800,000 | **81.17%** | Pure DGE scaling gate |
| **v28** | Deep MLP ~109K params | 800,000 | **87.56% ± 0.77%** | **ConsistencyDGE — new record** |
| v31 | Sign Activation (784→32→10) | 200,000 | **73.20%** | Adam fails (61%) |
| v32 | INT8 Full Quantization | 600,000 | **82.20%** | Native QAT. Adam fails (8%) |
| v32 | INT4 Full Quantization | 600,000 | **77.80%** | Native QAT. Adam fails (9%) |

### Synthetic Benchmarks — ConsistencyDGE vs Pure DGE (v27, D=128, 500K evals)

| Benchmark | Pure DGE | ConsistencyDGE T=20 | Improvement |
|---|---|---|---|
| Rosenbrock | 89.01 | 79.90 | **-10.2%** |
| Ill-conditioned Ellipsoid (cond=10^6) | 7,742 | 999 | **-87.1%** |
| Rotated Quadratic | 1.29e-02 | 2.68e-04 | **-97.9%** |
| Sphere (isotropic control) | 5.00e-04 | 2.21e-05 | **-95.6%** |

### Key Closed Findings

| Track | Versions | Verdict |
|---|---|---|
| SFWHT Hybrid Scanning | v19–v25 | Fails for dense NNs (D/B>>1). Valid only for K-sparse problems. |
| Vector Group DGE (Spherical perturbations) | v26–v26b | No improvement over Rademacher for groups >= 8 variables. |
| **Direction-Consistency LR** | **v27–v28** | **Universal improvement. New MNIST record.** |

---

## 🎯 Applications

DGE is a universal optimizer for any multi-dimensional landscape where gradients are unavailable or unreliable:

1. **Non-differentiable architectures:** Spiking Neural Networks, logic-gate activations, step-function layers.
2. **Memory-free training:** No activation graphs needed — only forward passes. Enables training on consumer hardware.
3. **Black-box systems:** Hyperparameter tuning in physics simulators, robotics, financial engines.
4. **Quantized / Binary networks:** Native training of 1-bit and ternary weights without straight-through estimators.
5. **Adversarial research:** Efficient black-box attacks with O(log D) queries per step.

---

## 📂 Repository Structure

```
dge/
  optimizer.py         — Canonical algorithm (Pure DGE, stable reference)
scratch/
  dge_consistency_lr_v27.py    — Direction-Consistency LR: synthetic benchmarks
  dge_consistency_mnist_v28.py — ConsistencyDGE on MNIST (87.56% record)
  dge_scaling_head_to_head_v25b.py — Pure DGE scaling to ~109K params
  [v1-v26b]            — Full experimental history
docs/
  research_shortlist.md                  — Active research priorities and empirical context
  dge_findings_v28_consistency_mnist.md  — Latest results (v28)
  dge_findings_v27_consistency_lr.md     — Synthetic benchmark results (v27)
  dge_findings_v25b_scaling_fair.md      — Scaling validation at 110K params
  [v1-v26b findings]   — Complete audit trail
results/raw/           — Raw JSON for every experiment (gitignored)
```

---

## Direction-Consistency LR — 5-line modification

```python
from collections import deque
sign_buffer = deque(maxlen=20)   # T = 20

# Inside the optimizer step, after computing grad:
sign_buffer.append(torch.sign(grad))
if len(sign_buffer) >= 2:
    consistency = torch.stack(list(sign_buffer)).mean(dim=0).abs()  # in [0, 1]
else:
    consistency = torch.ones_like(grad)

# Replace standard Adam update with:
upd = lr * consistency * m_hat / (torch.sqrt(v_hat) + eps)
```

`consistency[i] ~= 1` → stable, reliable gradient direction → full LR applied.
`consistency[i] ~= 0` → oscillating gradient (noise-dominated) → update suppressed.

---

*DGE — Gradient-free optimization that scales.*
