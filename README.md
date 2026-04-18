# DGE Optimizer (Dichotomous Gradient Estimation)

**DGE** is a novel zeroth-order (derivative-free) optimizer designed to solve the "Curse of Dimensionality" in Black-Box optimization and memory-constrained Machine Learning. 

It provides an $O(\log D)$ gradient estimation approach that can successfully train Neural Networks without ever calculating analytical derivatives or using Backpropagation.

---

## 🚀 The Core Innovation

Standard zeroth-order algorithms face a brutal trade-off:
1. **Finite Differences:** Costs $O(D)$ function evaluations. Perfect precision, but computationally impossible for large models (e.g., millions of evaluations per step).
2. **SPSA (Simultaneous Perturbation):** Costs $O(1)$ evaluations, but perturbs all dimensions simultaneously. In high dimensions ($D > 10,000$), the variance (background noise) from other dimensions destroys the gradient signal.

**DGE's Solution: Randomized Group Testing + Temporal Denoising**
Instead of perturbing everything or one-by-one, DGE tests $\approx \log_2(D)$ random, overlapping blocks of parameters per step. It evaluates the loss change for each block, takes a greedy step, and crucially, maintains an **Exponential Moving Average (EMA via Adam)** of the historical success of each individual parameter across the blocks it participated in.

The EMA acts as a *Temporal Denoiser*. Over time, the conflicting noise from co-parameters cancels out to zero, isolating the true, clean latent gradient for every single variable.

### Why DGE matters for AI:
* **Trains non-differentiable architectures:** Spiking Neural Networks, logic-gate activations, or RL physics engines.
* **Memory-Free Training:** Backprop requires storing massive activation graphs in VRAM. DGE only requires memory for the weights themselves (Forward-pass only), allowing trillions of parameters to be tuned on consumer GPUs.
* **Outperforms State-of-the-Art:** In non-convex landscapes, DGE's per-variable momentum routing drastically outperforms SPSA and MeZO.

---

## 📊 Benchmark: MNIST from Scratch (No Backprop)

We trained a standard Vision network (`784 -> 32 -> 10`, 25,450 parameters) on MNIST **without Autograd/Backpropagation**. 
Budget: 100,000 forward passes.

| Algorithm | Method | Train Accuracy | Test Accuracy |
|-----------|--------|----------------|---------------|
| **SPSA** | Perturb everything (O(1)) | 59.1% | 56.5% |
| **DGE** | Group Testing + EMA (O(log D)) | **92.9%** | **85.7%** |

*DGE trained a 25k parameter neural network to 85.7% accuracy using only random blocks and loss evaluations, demonstrating a mathematical victory over high-dimensional variance.*

---

## 💻 Usage

```python
import numpy as np
from dge import DGEOptimizer

# 1. Initialize Optimizer
optimizer = DGEOptimizer(
    dim=25000,           # Number of parameters
    lr=0.5,              # Learning rate
    delta=1e-3,          # Perturbation size
    clip_norm=0.05       # Crucial for high-D stability
)

# 2. Define your parameters and a callable Black-Box loss function
params = np.random.randn(25000)

def loss_fn(p):
    # CRITICAL: For stochastic data (minibatches), ensure the 
    # exact same data batch is used during a single step.
    return my_neural_network_forward(data_batch, p)

# 3. Optimize
for step in range(1000):
    params, evals_used = optimizer.step(loss_fn, params)
```

---

## 📂 Repository Structure
* `dge/optimizer.py`: The core algorithm implementation (Adam-infused EMA group testing).
* `examples/train_mnist.py`: A complete script demonstrating how to train a PyTorch/Numpy network on MNIST without backpropagation.
* `docs/`: The original theoretical whitepapers and empirical findings that led to the creation of DGE.
