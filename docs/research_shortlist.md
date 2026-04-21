# DGE Research Shortlist

Status: active prioritization note — updated 2026-04-21 (post v26b vector group ablation)
Purpose: reduce research sprawl and focus implementation effort on the highest-upside branches

## Why this exists

The project now has enough ideas that the main risk is dilution.

This shortlist defines the top branches worth pursuing next, based on:

- conceptual strength
- implementation feasibility
- experimental clarity
- paper potential
- fit with the current DGE codebase
- **empirical evidence from v1–v25** (ideas must be grounded in past results)

## Empirical Context (What v1–v25 Taught Us)

| Finding | Source | Implication |
|---|---|---|
| Block-SPSA with overlapping groups ≈ same SNR as plain SPSA | `dge_feedback.md`, v14 | Fixing group structure is prerequisite, not optional |
| Orthogonal non-overlapping blocks improved SNR | v14 findings | Structured perturbations > random perturbations |
| Deep MLP stalled at 85% accuracy | v10 | Layer-mixing and axis-aligned perturbations fail at depth |
| Step/Binary/Ternary networks: DGE >> Random Search | v11, v12, v13 | Non-differentiable settings are the strongest niche |
| Batched evaluation on AMD iGPU gave 1.8x speedup | v18 | Parallelism is viable; evaluation budget is the real bottleneck |
| SFWHT recovered 15 active vars from 1M dims (93x compression) | v19 | Works for truly sparse gradients (K << B) |
| SFWHT hybrid matched pure DGE at 25K params | v24 | At small scale, scan overhead ≈ benefit |
| **SFWHT hybrid collapsed at 110K params (65% vs 81%)** | **v25** | **Scan provides corrupted signal when D/B >> 1. Dense NNs are not sparse.** |
| **Pure DGE scaled robustly to 110K params (81.17%)** | **v25** | **The core block-DGE + EMA mechanism is the real workhorse** |
| EMA temporal denoising is critical for convergence | v14–v17 | Temporal aggregation is a core ingredient |
| Greedy step removal improved late-stage convergence | validation roadmap | Greedy accept/reject can trap the optimizer |
| **Spherical perturbations ≤ Rademacher in 3/4 benchmarks** | **v26b** | **Esférico no mejora eje-alineado en grupos ≥ 8 vars; causa regresión en Ellipsoid** |
| **ScalarVarianceDGE ≡ PureDGE con grupos dinámicos** | **v26b** | **Varianza escalar-por-grupo converge a per-coordinate bajo EMA con permutación dinámica** |
| **Sweet spot k_blocks = 8 (grupos de 16) en D=128** | **v26b ablación** | **Grupos muy grandes pierden cobertura; muy pequeños pierden señal por bloque** |

## Closed Research Tracks

### SFWHT Track (v19–v25): CLOSED ❌

The SFWHT integration pathway has been fully explored and its limits are now understood:

| Version | Result | Lesson |
|---|---|---|
| v19 | 93x compression on 1M-dim sparse | ✅ Works for K-sparse gradients where K << B |
| v20 | 53% crash | ❌ Collisions + DC bias in dense regime |
| v21 | 42% stable | ❌ Symmetric eval fixed crash but shifts eat budget |
| v22 | 63% stable | 🟡 Hybrid scan+refine works but SNR too low |
| v23 | 78% stable | ✅ Permutation + massive blocks + exploration |
| v24 | 79.5% ≈ pure DGE 80.5% at 25K | ✅ Tied at small scale |
| **v25** | **65% crash vs pure DGE 81% at 110K** | **❌ GATE FAILED. Does not scale.** |

**Root cause:** For dense neural networks, each SFWHT bucket contains D/B ≈ 100+ variables. The bucket magnitude reflects a random mixture of ~100 gradient components. The scan cannot distinguish important variables from noise — it is essentially random selection with overhead. Only works when the gradient is truly K-sparse with K << B.

**Salvageable niche:** SFWHT remains valid for genuinely sparse optimization problems (pruned networks, L1-regularized models, binary/ternary weight search). But it is not a general-purpose enhancement for DGE on dense models.

## Closed Research Tracks (continuación)

### Vector Group DGE Track (v26–v26b): CLOSED ❌

**Experimento:** 4 variantes × 4 benchmarks × 5 seeds = 80 corridas base + 160 ablación (500K budget, D=128)

| Versión | Problema | Resultado |
|---|---|---|
| v26 | Solo 2 clases, grupos fijos, 1 benchmark, 200K budget | Inválido — implementación incompleta |
| **v26b** | Ablación completa y justa (4 variantes, 4 benchmarks, 5 seeds, 500K) | ❌ Hipótesis no confirmada |

**Resultados v26b (k=8, D=128, budget=500K, 5 seeds):**

| Benchmark | PureDGE | SphericalDGE | VectorGroupDGE | Ganador |
|---|---|---|---|---|
| Rosenbrock | **89.01 ± 5.28** | 101.33 ± 12.55 | 97.29 ± 3.03 | PureDGE ✅ |
| Rotated Quadratic | 1.29e-02 ± 1.12e-02 | 9.19e-03 ± 1.51e-02 | **7.99e-03 ± 9.12e-03** | 🟡 marginal, no robusto |
| Ellipsoid (cond=10^6) | **7,742 ± 1,261** | 12,849 ± 3,301 | 14,473 ± 1,511 | PureDGE ✅ |
| Sphere | **5.00e-04 ± 2.31e-04** | 6.76e-04 ± 7.06e-04 | 3.96e-04 ± 4.47e-04 | Empate |

**Lecciones aprendidas:**
- Una dirección aleatoria en un grupo de n vars proyecta 1/√n en cada eje — idéntico a Rademacher en expectativa. No hay ventaja para grupos ≥ 8 vars.
- Con grupos **dinámicos** (permutación por paso), el EMA vectorial no acumula dirección coherente. Los grupos fijos podrían ser necesarios para que funcione, pero añaden rigidez.
- La varianza escalar-por-grupo (ScalarVarianceDGE) es matemáticamente equivalente a per-coordinate con grupos dinámicos bajo EMA β=0.999 → lección negativa valiosa.
- En Ellipsoid (cond=10^6), la perturbación esférica mezcla coordenadas de escalas muy dispares → Adam pierde su precondición per-coordinate → regresión severa del 87%.

**Salvageable niche:** Si se quisiera retomar, habría que probar grupos **fijos persistentes** (no dinámicos). Pero el overhead conceptual no justifica el experimento según los datos actuales.

## Shortlist: Active Branches (Post-v26b)

## 1. Direction-Consistency Learning Rates ⭐ TOP PRIORITY

Core idea:

Adapt local learning rates from temporal consistency of the estimated direction. Variables with consistent estimated sign across recent steps get a boosted LR; variables with oscillating sign get suppressed.

Why this is now top priority:

- Vector Group DGE (v26b) is closed — this is the next item in the shortlist
- Low implementation cost (~30 min)
- Universally beneficial: mejora PureDGE directamente, sin cambiar la perturbación
- Orthogonal a los resultados de v26b: no depende de perturbaciones esféricas
- Distingue variables con gradiente consistente (útiles) de variables ruidosas (a suprimir)
- Potencial de mejora en Rosenbrock y Ellipsoid, que son sensibles a pasos en dirección errónea

Implementation sketch:

```python
# Track sign consistency per variable
from collections import deque
sign_buffer = deque(maxlen=10)  # últimos 10 pasos
sign_buffer.append(np.sign(g_est))

consistency = np.mean(np.stack(sign_buffer), axis=0)  # en [-1, 1]
lr_scale = np.abs(consistency)  # 0 = signos aleatorios → LR pequeño, 1 = consistente → LR completo

# Aplicar al update de Adam
update = lr * lr_scale * (mh / (np.sqrt(vh) + 1e-8))
```

Hypothesis:

> Variables con signo de gradiente consistente en los últimos T pasos convergen más rápido si reciben un LR efectivo mayor. Variables ruidosas se benefician de un LR reducido que evita oscilaciones.

Success criteria:
- Mejora en Rosenbrock vs PureDGE al mismo budget
- Mejora o empate en Sphere (no regresión)
- Reducción de std entre seeds (convergencia más estable)

Failure criteria:
- Ninguna mejora en ningún benchmark con ventana de 5–20 pasos

Priority:

**Highest. Es el siguiente experimento a ejecutar (v27).**

## 2. Architecture-Aware Grouping (antes item 3)

## 3. Architecture-Aware Grouping (Extension of Vector Group)

Core idea:

Instead of contiguous chunks, use the model's architecture to define groups:
- All weights feeding into neuron j form one group
- All weights in one convolutional filter form one group
- Bias terms are separate groups

Why this matters:

- Neuron-local weights are the most correlated subset in a dense MLP
- A perturbation that moves all incoming weights of one neuron in one direction tests that neuron's response as a unit
- This is the natural intermediate step between "contiguous chunks" and "learned groups"

Priority:

**Medium-high. Test after contiguous grouping is validated, as a refinement.**

## Quick Wins (Tier 0)

Implement alongside Vector Group DGE:

### Half-Step Retry
If a full step worsens the objective, try once with `lr/2`. Cost: 1 eval max.

### Curvature-Preconditioned Perturbations
Scale perturbation magnitude per coordinate by `1/sqrt(v_t)`. Cost: near zero.

### Full-Step Same-Batch Evaluation
Use the same mini-batch for all k perturbations within a step. Cost: zero.

## Medium-Term Ideas

### Layer-Wise Budget Allocation
Assign DGE blocks per layer based on gradient energy. Validated conceptually by the SFWHT layer-wise approach (v20–v24), but now without the scan overhead.

### Gradient Checkpointing Temporal
Reuse past evaluations weighted by geometric distance.

## Preserved Ideas

### SFWHT for Sparse Settings
SFWHT is validated for genuinely K-sparse problems (v19). If the project explores pruned networks, L1-regularized models, or binary weight search, SFWHT could return as a specialized tool. Not for dense model training.

### Temporal Hierarchical SPSA
Deprioritized after SFWHT track closure. The core insight (spread structured probes over time) was partially captured by lazy scanning in v24 but didn't help at scale.

### Fourier / Oscillatory / Tremor Ideas
Still speculative.

## Recommended Execution Order

1. ~~**Vector Group DGE** (v26–v26b)~~ — CLOSED ❌ (perturbaciones esféricas no mejoran Rademacher)
2. **Direction-Consistency LR** (v27): implementar sobre PureDGE, benchmarks sintéticos + MNIST
3. **Quick wins**: half-step retry + curvature perturbations + same-batch
4. **Architecture-aware grouping** (v28): neuron-local groups en MNIST si v27 muestra mejora
5. **Ablations**: ventana de consistencia T ∈ {5, 10, 20}, suavizado vs hard threshold
6. **Paper assembly** once the strongest story is clear

## Decision Gates

- [ ] Does the branch have a clear failure mode and a clear regime where it should help?
- [ ] Can it be benchmarked on synthetic problems within one session?
- [ ] Does it add a genuinely new capability, or only tuning complexity?
- [ ] If it works, can it be explained in one paragraph without hand-waving?
- [ ] If it fails, will we still learn something useful about DGE?
- [ ] Is it grounded in at least one concrete finding from v1–v25?
- [ ] Does answering this question change the strategic direction of the project?

## Final Recommendation

The SFWHT track is closed. The scaling gate (v25) showed definitively that structured scanning cannot overcome the dense-gradient collision problem in real neural networks.

The project's core asset is **Pure DGE**: block perturbations + EMA temporal denoising + Adam. It scales to 110K params with 81% accuracy on MNIST (v25). It is robust, simple, and fast.

The next step is to make the perturbation structure smarter without adding overhead:

> Replace axis-aligned scalar perturbations with random directional perturbations within variable groups. This is the simplest structural upgrade to Pure DGE that addresses its most documented weakness (zig-zag in correlated valleys) at zero extra evaluation cost.

If Vector Group DGE works on Rosenbrock and MNIST, the paper story becomes:

> DGE is a zeroth-order optimizer that combines block perturbations with temporal denoising. We show that replacing scalar perturbations with vector-group directional perturbations eliminates axis-aligned zig-zag and improves convergence on correlated landscapes. The method trains neural networks competitively with pure black-box access, excelling in non-differentiable settings where backpropagation fails.

That is a clean, honest, publishable story.
