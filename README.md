# DGE: Denoised Gradient Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DirectML%20%7C%20CUDA%20%7C%20CPU-orange.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/dge_paper.pdf)

> **Training Deep Neural Networks Without Backpropagation via Coordinate Block Perturbations and Dual Sign-EMA Temporal Filtering**

📄 **[Read the Full Scientific Paper (PDF)](paper/dge_paper.pdf)** | 📊 **[Master Results Table](docs/results_master.md)**

---

## 📌 Overview

**Denoised Gradient Estimation (DGE)** is a zeroth-order (gradient-free) optimization framework capable of training deep neural networks using **only forward-pass objective evaluations**, without computing analytical backpropagation or saving autodiff activation graphs.

Classical zeroth-order algorithms like SPSA or standard MeZO perturb all parameters simultaneously, suffering from a catastrophic variance explosion ($\mathcal{O}(D)$) that causes optimization to collapse in high dimensions ($D > 10^5$). DGE solves this dimensionality curse through three complementary mechanisms:

1. **Coordinate Block Perturbations:** Partitions the $D$-dimensional parameter manifold into $K \ll D$ coordinate blocks, bounding cross-contamination noise by $1/K$ and proving a **$K$-fold variance reduction law** ($\text{Var}(g_i) \le \frac{1}{K} \|\nabla f\|^2$).
2. **Temporal Momentum Denoising:** Injects block estimates into an Adam-style Exponential Moving Average (EMA) filter, accumulating true gradient signals constructively while destructive random-walk noise cancels out, achieving $\mathcal{O}(\sqrt{T})$ SNR growth.
3. **Dual Sign-EMA (DS-EMA) Consistency Gating:** A MACD-inspired crossover filter tracking fast ($\alpha_f = 0.30$) and slow ($\alpha_s = 0.05$) directional trends, suppressing noisy oscillations with **$\mathcal{O}(2D)$ constant memory** (a 90% memory reduction over sliding-window buffers).

---

## 🚀 Key Results & Highlights

All reported metrics are backed by raw JSON artifacts in `results/raw/` containing hardware and commit provenance.

### 1. Vision Benchmark (Full MNIST, 60K/10K, MLP-109K)
DGE achieves **96.3% of Adam's peak performance** without computing analytical gradients:

| Method | Optimization Paradigm | Best Test Acc. (Mean $\pm$ Std) | Hardware Runtime |
|---|---|---|---|
| **Global SPSA** | Zeroth-Order (Rademacher + Adam) | 20.87% $\pm$ 11.83% *(collapse)* | ~48 min (300K evals) |
| **MeZO** (Malladi et al., 2023) | Zeroth-Order (Gaussian + SGD) | 13.80% $\pm$ 0.00% *(collapse)* | ~18 min (100K evals) |
| **Block-SGD** (PureDGE-SGD) | Zeroth-Order (Coordinate Blocks only) | 88.86% $\pm$ 0.28% | ~12 min (500K evals) |
| **Block-Adam** (PureDGE-Adam) | Zeroth-Order (Blocks + Adam) | 93.00% $\pm$ 0.24% | ~13.8 min (3M evals) |
| **DGE Full (DS-EMA)** | **Zeroth-Order (Blocks + Adam + DS-EMA)** | **94.36% $\pm$ 0.23%** | **~13.5 min (3M evals)** |
| **SGD + Momentum** | Analytical Backpropagation | 97.78% $\pm$ 0.04% | ~18 s (30 epochs) |
| **Adam** | Analytical Backpropagation | 98.00% $\pm$ 0.04% | ~19 s (30 epochs) |

### 2. Native Optimization of Non-Differentiable Architectures
Where backpropagation fails completely (analytical gradients $= 0$), DGE optimizes discrete representations natively **without Straight-Through Estimators (STE)**:

| Architecture / Precision | Adam (Backprop without STE) | DGE (Zeroth-Order) | Advantage |
|---|---|---|---|
| **Sign Activation Networks** (`torch.sign`) | 65.30% *(trains output layer only)* | **70.43%** | **+5.13 pp** |
| **INT8 Full Quantization** (256 discrete levels) | 9.78% *(random guessing collapse)* | **74.24%** | **+64.46 pp** |
| **INT4 Full Quantization** (16 discrete levels) | 9.82% *(random guessing collapse)* | **68.82%** | **+59.00 pp** |

### 3. Memory & Computational Efficiency
* **$\mathcal{O}(\text{params} + \text{batch})$ VRAM Footprint:** Because DGE requires only inference passes, it never stores forward activation tensors for a backward autodiff graph, eliminating >70% of peak training memory.
* **Vectorized Tensor Evaluation:** DGE batches all $2K$ coordinate perturbations into a single parallel tensor pass, evaluating 10$\times$ more candidate directions in less than one-third of SPSA's serial wall-clock time (~13.5 min vs. ~48 min).

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/mcarbonell/dge-optimizer.git
cd dge-optimizer
pip install -r requirements.txt
pip install -e .
```

*Hardware acceleration:* DGE natively supports **NVIDIA CUDA**, **AMD DirectML** (`torch-directml`), and **CPU**.

---

## ⚡ Quickstart

```python
import torch
import torch.nn as nn
from dge import TorchDGEOptimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define network architecture
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

# Total parameter dimension and layer sizes
layer_sizes = [p.numel() for p in model.parameters()]
total_dim = sum(layer_sizes)

# Initialize DGE Optimizer
optimizer = TorchDGEOptimizer(
    dim=total_dim,
    layer_sizes=layer_sizes,
    k_blocks=[128, 16],        # Number of coordinate blocks per layer
    lr=0.05,                   # Learning rate
    delta=1e-3,                # Perturbation magnitude
    total_steps=1000,
    consistency_window=20,     # Temporal consistency window
    clip_norm=0.05,
    device=device
)

# Step function requires evaluating loss on batched parameter perturbations
# (See experiments/run_paper_table1.py for the canonical training loop)
```

---

## 🔬 Reproducibility & Benchmark Suite

Verify that all unit tests pass:

```bash
python tests/test_optimizer.py
```
*(Tests cover unbiasedness on quadratics, $K$-fold variance reduction, DS-EMA bounds, and baseline optimization).*

### Run Paper Benchmarks:

```bash
# 1. Full MNIST Main Comparison (Table 1)
python experiments/run_paper_table1.py --methods "DGE (DS-EMA)" "Block-Adam (PureDGE)"

# 2. Non-Differentiable & Quantization Suite (Table 2, Fig 2)
python scratch/dge_nondiff_suite_v71.py

# 3. Component Ablation Study (Fig 4)
python scratch/dge_ablation_v70.py --methods PureDGE_SGD PureDGE_Adam MeZO

# 4. Regenerate all paper figures (PDF vector graphics)
python paper/figures/generate_figures.py
```

All results are written to `results/raw/` with hardware provenance and commit hash, documented in [docs/results_master.md](docs/results_master.md).

---

## ⚖️ Limitations & Scope

1. **Fully Differentiable Regimes:** Where analytical gradients and memory bandwidth are abundant, backpropagation remains orders of magnitude faster in wall-clock time (~19s for Adam vs. ~13.5 min for DGE). DGE is designed for domains where backprop **cannot operate** (black-box landscapes, non-differentiable physics, analog/neuromorphic hardware, extreme quantization).
2. **Deep Discrete Butterfly Effect:** In very deep discrete networks ($\ge 8$ layers), single-bit perturbation flips in early layers can chaoticly invert all subsequent representations, violating local stationarity and capping performance (~75%).

---

## 📚 Citation

If you use DGE in your research or applications, please cite our paper:

```bibtex
@article{carbonell2026dge,
  title={Denoised Gradient Estimation: Training Deep Neural Networks Without Backpropagation via Coordinate Block Perturbations and Dual Sign-EMA Temporal Filtering},
  author={Carbonell Mart{\'\i}nez, Mario Ra{\'u}l},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/mcarbonell/dge-optimizer}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
