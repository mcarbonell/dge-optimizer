# Tabla Maestra de Resultados y Trazabilidad — DGE

Este documento constituye la **fuente canónica de verdad** para todas las métricas, benchmarks y afirmaciones reportadas en el `README.md`, el paper (`paper/dge_paper.tex`) y los documentos de hallazgos (`docs/dge_findings_*.md`).

Cada entrada referencia el script que la generó, el archivo JSON de datos crudos en `results/raw/`, el número de semillas y su estado de verificación según la auditoría técnica.

---

## 1. Experimentos Principales en Visión (MNIST)

| Claim / Benchmark | Modelo | Método | Semillas | Métrica (mean ± std) | Script Generador / Runner | Archivo JSON Crudo | Estado de Verificación |
|---|---|---|---|---|---|---|---|
| **Full MNIST (60K/10K)** | MLP (784-128-64-10, ~109K params) | **DGE Full (ConsistencyDGE)** | 3 (42, 43, 44) | 94.36% ± 0.18% (best test acc) | `experiments/run_paper_table1.py`, `scratch/dge_fullmnist_comparison_v30e.py` | `results/raw/v30e_fullmnist_comparison.json` | 🟢 Verificado (JSON presente) |
| Full MNIST (60K/10K) | MLP (~109K params) | **PureDGE** (sin DS-EMA) | 3 (42, 43, 44) | 93.00% ± 0.28% | `experiments/run_paper_table1.py`, `scratch/dge_fullmnist_comparison_v30e.py` | `results/raw/v30e_fullmnist_comparison.json` | 🟢 Verificado (JSON presente) |
| Full MNIST (60K/10K) | MLP (~109K params) | **Global SPSA** (300K evals) | 3 (42, 43, 44) | 20.87% ± 11.83% (colapso) | `experiments/run_paper_table1.py`, `scratch/dge_fullmnist_comparison_v30e.py` | `results/raw/v30e_fullmnist_comparison.json` | 🟢 Verificado (presupuesto 300K declarado) |
| Full MNIST (60K/10K) | MLP (~109K params) | **Adam** (30 epochs) | 3 (42, 43, 44) | 98.00% ± 0.05% | `experiments/run_paper_table1.py`, `scratch/dge_fullmnist_comparison_v30e.py` | `results/raw/v30e_fullmnist_comparison.json` | 🟢 Verificado (JSON presente) |
| Full MNIST (60K/10K) | MLP (~109K params) | **SGD + Momentum** | 3 (42, 43, 44) | 97.78% ± 0.06% | `experiments/run_paper_table1.py`, `scratch/dge_fullmnist_comparison_v30e.py` | `results/raw/v30e_fullmnist_comparison.json` | 🟢 Verificado (JSON presente) |
| **MNIST 3K Subset** | MLP (784-32-10) | ConsistencyDGE vs PureDGE | 6 semillas | Consistency: 87.58% ± 1.23% vs Pure: 82.22% ± 1.70% | `scratch/dge_paper_stats_v29.py` | `results/raw/v29_paper_stats.json` | 🟢 Verificado (JSON presente) |
| MNIST Full (v30d preliminar) | MLP (~109K params) | DGE / SPSA preliminar | 3 semillas | 92.98% / 28.98% | `scratch/dge_fullmnist_comparison_v30d.py` | `results/raw/v30d_fullmnist_comparison.json` | 🟡 Superseded (superado por v30e) |

---

## 2. Regímenes No Diferenciables y Cuantizados (QAT sin STE)

| Claim / Benchmark | Precisión / Operación | Método | Semillas | Métrica Reportada | Script Generador | Archivo JSON Crudo | Estado de Verificación |
|---|---|---|---|---|---|---|---|
| **INT8 QAT Nativo** | 256 niveles (pesos + act) | DGE V3 | 1 (v32, v71) | **74.24%** (v71) / **82.20%** (v32) vs Adam 9.78% | `scratch/dge_nondiff_suite_v71.py` | `results/raw/v71_nondiff_suite.json`, `v32_quantized_mnist.json` | 🟢 Verificado (JSON presente) |
| **INT4 QAT Nativo** | 16 niveles (pesos + act) | DGE V3 | 1 (v32, v71) | **68.82%** (v71) / **77.80%** (v32) vs Adam 9.82% | `scratch/dge_nondiff_suite_v71.py` | `results/raw/v71_nondiff_suite.json`, `v32_quantized_mnist.json` | 🟢 Verificado (JSON presente) |
| **Redes con Activación Signo** | `torch.sign` (step) | DGE | 1 (v31, v71) | **70.43%** (v71) / **73.20%** (v31) vs Adam 65.30% | `scratch/dge_nondiff_suite_v71.py` | `results/raw/v71_nondiff_suite.json`, `v31_sign_activations.json` | 🟢 Verificado (JSON presente) |
| **Pesos Binarios / Ternarios** | $\{-1, 1\}$ / $\{-1, 0, 1\}$ | DGE | 1 | ~73% (binario) | `scratch/dge_binary_weights_v12.py` | `scratch/` findings | 🟡 Documentado en findings preliminares |

---

## 3. Optimizaciones Sintéticas y DS-EMA (Dual Sign-EMA)

| Benchmark | Dimensión / Condición | Comparación | Semillas | Métrica | Script Generador | Archivo JSON Crudo | Estado de Verificación |
|---|---|---|---|---|---|---|---|
| **Ellipsoid** | $D=128$, $\kappa=10^6$ | DS-EMA vs SMA vs Pure DGE | 5 | Pérdida final y SNR | `scratch/dge_dual_ema_v67.py` | `results/raw/v67_dual_ema.json` | 🟢 Verificado (JSON presente) |
| **Rosenbrock** | $D=128$ | DS-EMA vs SMA | 5 | Pérdida final | `scratch/dge_dual_ema_v67.py` | `results/raw/v67_dual_ema.json` | 🟢 Verificado (JSON presente) |
| **MNIST DS-EMA** | $D \approx 109K$ | DS-EMA vs SMA | 5 | Accuracy y convergencia | `scratch/dge_mnist_dual_ema_v68.py` | `results/raw/v68_mnist_dual_ema.json` | 🟢 Verificado (JSON presente) |
| **Escalado $K$ óptimo** | MLP | Estudio de $K = \mathcal{O}(\sqrt{D})$ | 3 | Accuracy | `scratch/dge_k_scaling_v35.py` | `results/raw/v35_k_scaling.json` | 🟢 Verificado (JSON presente) |

---

## 4. Estado de Baselines y Acciones Requeridas

1. **MeZO:**
   - Anteriormente invocaba `run_spsa()` con la misma semilla (B1).
   - Ahora implementado canónicamente en `experiments/baselines.py::MeZOptimizer` (Gaussiana $z \sim \mathcal{N}(0, I)$ + SGD).
   - 🟢 **Verificado en ejecución:** Ejecutado en `scratch/dge_ablation_v70.py` (13.80% test acc, reflejando colapso de varianza $\mathcal{O}(D)$ sin inicialización pre-entrenada).
2. **Ablación:**
   - La Figura 4 de `paper/figures/generate_figures.py` cargaba previamente números manuales.
   - 🟢 **Verificado en ejecución:** La figura 4 ahora lee dinámicamente desde `results/raw/v70_ablation_study.json` con 5 configuraciones reproducibles:
     - **MeZO (Global, SGD):** 13.80%
     - **SPSA (Global, Adam):** 20.87% (300K evals)
     - **Block-SGD (PureDGE_SGD):** 88.86% (500K evals)
     - **Block-Adam (PureDGE_Adam):** 89.62% (500K evals) / 93.00% (3M evals)
     - **DGE Full (+ Consistency):** 89.64% (500K evals) / 94.36% (3M evals)
