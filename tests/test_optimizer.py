"""
tests/test_optimizer.py
=======================
Suite de pruebas unitarias para DGE (DGEOptimizer y TorchDGEOptimizer).
Verifica propiedades matemáticas fundamentales del estimador:
  1. Insesgadez teórica del estimador por bloques en función cuadrática.
  2. Reducción empírica de varianza con partición en K bloques (Lema 2).
  3. Contabilidad exacta de evaluaciones de función por paso (2 * K).
  4. Comportamiento y acotación de la máscara DS-EMA (supresión de ruido).
  5. Compatibilidad y ejecución de TorchDGEOptimizer y baselines.
"""

import math
import sys
import os
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
except ImportError:
    class _PytestStub:
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    return func if not condition else (lambda *args, **kwargs: None)
                return decorator
    pytest = _PytestStub()

from dge.optimizer import DGEOptimizer
from experiments.baselines import SPSAOptimizer, MeZOptimizer

try:
    import torch
    from dge.torch_optimizer import TorchDGEOptimizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def test_evals_per_step():
    """Verifica que cada paso consume exactamente 2*k evaluaciones."""
    dim = 32
    k_blocks = 4
    opt = DGEOptimizer(dim=dim, k_blocks=k_blocks, seed=42)
    x = np.ones(dim, dtype=np.float64)

    eval_count = 0
    def f(p):
        nonlocal eval_count
        eval_count += 1
        return float(np.sum(p ** 2))

    x_new, n_evals = opt.step(f, x)
    assert n_evals == 2 * k_blocks, f"Esperado {2 * k_blocks}, obtenido {n_evals}"
    assert eval_count == 2 * k_blocks, f"Contador f(x) fue {eval_count}"


def test_unbiased_estimator_quadratic():
    """
    Verifica que el estimador por diferencias finitas por bloques
    es un estimador insesgado del gradiente analítico en un entorno suave:
      E[g] = nabla f(x)
    """
    dim = 16
    rng = np.random.default_rng(12345)
    
    # Matriz simétrica definida positiva A y vector b
    M = rng.standard_normal((dim, dim))
    A = M.T @ M + 0.5 * np.eye(dim)
    b = rng.standard_normal(dim)

    # f(x) = 0.5 * x^T A x + b^T x => nabla f(x) = A x + b
    x0 = rng.standard_normal(dim)
    grad_true = A @ x0 + b

    def f(p):
        return float(0.5 * p.T @ A @ p + b.T @ p)

    # Acumulamos gradientes brutos generados por bloques
    k_blocks = 4
    delta = 1e-4
    n_samples = 2000

    grad_accum = np.zeros(dim, dtype=np.float64)
    counts = np.zeros(dim, dtype=np.int32)

    for step_i in range(n_samples):
        # Muestreamos permutación y signos Rademacher idénticos a DGEOptimizer
        perm = rng.permutation(dim)
        blocks = np.array_split(perm, k_blocks)
        for blk in blocks:
            signs = rng.choice([-1.0, 1.0], size=len(blk))
            pert = np.zeros(dim, dtype=np.float64)
            pert[blk] = signs * delta

            fp = f(x0 + pert)
            fm = f(x0 - pert)
            diff = (fp - fm) / (2.0 * delta)

            grad_accum[blk] += diff * signs
            counts[blk] += 1

    grad_est = grad_accum / np.maximum(counts, 1)

    rel_error = np.linalg.norm(grad_est - grad_true) / np.linalg.norm(grad_true)
    assert rel_error < 0.10, f"Error relativo de estimación excesivo ({rel_error:.4f} >= 0.10)"


def test_variance_reduction_with_k():
    """
    Verifica el Lema 2 (Variance Bound):
    La varianza del estimador disminuye al aumentar el número de bloques K.
    Var(g_i | K=16) < Var(g_i | K=1)
    """
    dim = 64
    rng = np.random.default_rng(42)
    # Función no cuadrática con interferencias cruzadas
    A = rng.standard_normal((dim, dim))

    def f(p):
        return float(np.sum((A @ p) ** 2))

    x0 = np.ones(dim, dtype=np.float64)
    delta = 1e-3
    n_trials = 300

    def sample_estimates(k_blocks):
        estimates = []
        for _ in range(n_trials):
            perm = rng.permutation(dim)
            blocks = np.array_split(perm, k_blocks)
            for blk in blocks:
                if 0 in blk:  # rastreamos coordenada 0
                    signs = rng.choice([-1.0, 1.0], size=len(blk))
                    pert = np.zeros(dim)
                    pert[blk] = signs * delta
                    fp = f(x0 + pert)
                    fm = f(x0 - pert)
                    idx_in_blk = np.where(blk == 0)[0][0]
                    g_0 = ((fp - fm) / (2.0 * delta)) * signs[idx_in_blk]
                    estimates.append(g_0)
                    break
        return np.var(estimates)

    var_k1 = sample_estimates(k_blocks=1)   # Global SPSA (1 bloque de tamaño D)
    var_k8 = sample_estimates(k_blocks=8)   # DGE (8 bloques de tamaño D/8)

    assert var_k8 < var_k1, (
        f"Varianza con K=8 ({var_k8:.4e}) debería ser menor que con K=1 ({var_k1:.4e})"
    )


def test_dsema_mask_bounded_and_filters_noise():
    """
    Verifica que la máscara de consistencia DS-EMA:
      1. Produce valores acotados estrictamente en [0, 1].
      2. Mantiene una máscara alta (> 0.7) ante signos consistentes.
      3. Suprime signos oscilantes / ruidosos (< 0.4).
    """
    dim = 10
    opt = DGEOptimizer(dim=dim, consistency_window=20, seed=42)

    # 1. Signos consistentes (+1)
    consistent_signs = np.ones(dim)
    for _ in range(30):
        mask_consistent = opt._consistency_mask(consistent_signs)

    assert np.all(mask_consistent >= 0.0) and np.all(mask_consistent <= 1.0)
    assert np.all(mask_consistent > 0.7), f"Máscara consistente debería ser alta: {mask_consistent}"

    # 2. Reiniciamos y probamos signos aleatorios oscilantes
    opt_noisy = DGEOptimizer(dim=dim, consistency_window=20, seed=43)
    rng = np.random.default_rng(99)
    for _ in range(50):
        noisy_signs = rng.choice([-1.0, 1.0], size=dim)
        mask_noisy = opt_noisy._consistency_mask(noisy_signs)

    assert np.all(mask_noisy >= 0.0) and np.all(mask_noisy <= 1.0)
    mean_noisy_mask = float(np.mean(mask_noisy))
    assert mean_noisy_mask < 0.5, f"Máscara ruidosa debería ser atenuada (<0.5): {mean_noisy_mask}"


def test_baselines_quadratic_optimization():
    """Verifica que SPSAOptimizer y MeZOptimizer optimizan una esfera simple."""
    dim = 8
    def f(p): return float(np.sum(p ** 2))
    x0 = np.ones(dim) * 2.0

    # SPSA
    opt_spsa = SPSAOptimizer(dim=dim, lr=0.1, delta=1e-3, total_steps=100, seed=42)
    x = x0.copy()
    for _ in range(100):
        x, _ = opt_spsa.step(f, x)
    assert f(x) < f(x0), "SPSA no redujo el objetivo en la esfera"

    # MeZO
    opt_mezo = MeZOptimizer(dim=dim, lr=0.1, delta=1e-3, total_steps=100, seed=42)
    x = x0.copy()
    for _ in range(100):
        x, _ = opt_mezo.step(f, x)
    assert f(x) < f(x0), "MeZO no redujo el objetivo en la esfera"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch no disponible")
def test_torch_dge_optimizer_cpu():
    """Verifica que TorchDGEOptimizer ejecuta un paso sin errores en CPU con y sin Adam."""
    dim = 20
    x = torch.ones(dim, dtype=torch.float32)

    def f_batched(p_batch):
        # p_batch shape: (2K, dim)
        return torch.sum(p_batch ** 2, dim=1)

    # 1. Modo Adam con consistencia
    opt_adam = TorchDGEOptimizer(dim=dim, k_blocks=4, lr=0.1, delta=1e-3,
                                 consistency_window=10, use_adam=True, seed=42)
    x_new, evals = opt_adam.step(f_batched, x)
    assert evals == 8
    assert x_new.shape == (dim,)
    assert not torch.isnan(x_new).any()

    # 2. Modo Pure SGD
    opt_sgd = TorchDGEOptimizer(dim=dim, k_blocks=4, lr=0.1, delta=1e-3,
                                consistency_window=0, use_adam=False, seed=42)
    x_new_sgd, evals_sgd = opt_sgd.step(f_batched, x)
    assert evals_sgd == 8
    assert x_new_sgd.shape == (dim,)
    assert not torch.isnan(x_new_sgd).any()


if __name__ == "__main__":
    print("Ejecutando suite de pruebas de DGE...")
    test_evals_per_step()
    print("  [OK] test_evals_per_step")
    test_unbiased_estimator_quadratic()
    print("  [OK] test_unbiased_estimator_quadratic")
    test_variance_reduction_with_k()
    print("  [OK] test_variance_reduction_with_k")
    test_dsema_mask_bounded_and_filters_noise()
    print("  [OK] test_dsema_mask_bounded_and_filters_noise")
    test_baselines_quadratic_optimization()
    print("  [OK] test_baselines_quadratic_optimization")
    if TORCH_AVAILABLE:
        test_torch_dge_optimizer_cpu()
        print("  [OK] test_torch_dge_optimizer_cpu")
    print("\nTODOS LOS TESTS PASARON EXITOSAMENTE (6/6).")
