# DGE Research Shortlist

Status: active prioritization note — updated 2026-04-21 (post v24)
Purpose: reduce research sprawl and focus implementation effort on the highest-upside branches

## Why this exists

The project now has enough ideas that the main risk is dilution.

This shortlist defines the top branches worth pursuing next, based on:

- conceptual strength
- implementation feasibility
- experimental clarity
- paper potential
- fit with the current DGE codebase
- **empirical evidence from v1–v24** (ideas must be grounded in past results)

The goal is not to say other ideas are bad.
The goal is to decide what deserves serious attention now.

## Empirical Context (What v1–v24 Taught Us)

Every priority decision below is anchored to these validated facts:

| Finding | Source | Implication |
|---|---|---|
| Block-SPSA with overlapping groups ≈ same SNR as plain SPSA | `dge_feedback.md`, v14 | Fixing group structure is prerequisite, not optional |
| Orthogonal non-overlapping blocks improved SNR | v14 findings | Structured perturbations > random perturbations |
| Deep MLP stalled at 85% accuracy | v10 | Layer-mixing and axis-aligned perturbations fail at depth |
| Step/Binary/Ternary networks: DGE >> Random Search | v11, v12, v13 | Non-differentiable settings are the strongest niche |
| Batched evaluation on AMD iGPU gave 1.8x speedup | v18 | Parallelism is viable; evaluation budget is the real bottleneck |
| SFWHT recovered 15 active vars from 1M dims with 11K evals (93x compression) | v19 | Sparse structured estimation works at scale |
| SFWHT puro fails on dense NN gradients (collisions) | v20, v21 | SFWHT cannot be the sole estimator for dense models |
| Hybrid SFWHT-scan + DGE-refine architecture works | v22–v24 | SFWHT as radar + DGE as refiner is the correct paradigm |
| Permutation + massive DGE blocks + exploration fix the 3 bottlenecks | v23 | All three fixes needed together for full effect |
| Lazy scanning (scan every 4 steps) saves 82% of scan budget | v24 | Scan overhead can be amortized effectively |
| **Hybrid v24 matches Pure DGE at 25K params (79.5% vs 80.5%)** | **v24** | **At small scale, hybrid ≈ pure. Crossover point not yet found.** |
| EMA temporal denoising is critical for convergence | v14–v17 ablations | Temporal aggregation is a core ingredient |
| Greedy step removal improved late-stage convergence | validation roadmap | Greedy accept/reject can trap the optimizer |

## SFWHT Track: Status CLOSED (v19–v24)

The SFWHT integration pathway has been fully explored at the 25K-parameter scale:

- v19: proof of concept on synthetic 1M-dim sparse problem ✅
- v20: first MNIST attempt, crashed due to collisions and DC bias ❌
- v21: symmetric evaluation + decay fixed crash but hit 42% ceiling ❌
- v22: hybrid scan+refine architecture broke 63% ✅
- v23: permutation + k=128 + exploration reached 78% ✅
- v24: lazy scanning + progressive refinement reached 79.5%, matched pure DGE ✅

**Key conclusion:** At D=25K, pure DGE is slightly more efficient because the SFWHT scan overhead is not amortized. The hybrid must prove itself at larger scale where scan cost (O(B)) becomes negligible relative to DGE cost (O(D/k)).

**This is the single most important open question in the project.**

## Shortlist: Prioritized Branches

## 1. Scaling Head-to-Head: Hybrid vs Pure DGE at 100K–1M Parameters

Core idea:

Run the v24 head-to-head comparison at progressively larger model sizes to find the crossover point where SFWHT+DGE hybrid surpasses pure DGE in accuracy at equal evaluation budget.

Why this is the absolute top priority:

- The entire SFWHT research line (v19–v24) was built on the hypothesis that structured scanning beats brute-force at scale
- v24 showed a near-tie at 25K params — the crossover must exist at larger D where scan cost is amortized
- If the crossover exists, it is the paper headline result: "SFWHT+DGE outperforms pure block-SPSA above N parameters"
- If the crossover does not exist, the SFWHT line loses its value proposition and resources should go elsewhere
- **This is a gate: everything else should wait until this is answered**

Experimental design:

Three model sizes, same MNIST task, same eval budget scaling:

| Model | Params (D) | Eval Budget | Scan Cost / Step | DGE Cost / Step |
|---|---|---|---|---|
| Small (v24 baseline) | ~25K | 200K | ~1,280 (82%) | ~288 (18%) |
| Medium | ~100K | 800K | ~1,280 (scan unchanged) | ~1,200+ |
| Large | ~500K–1M | 2–4M | ~1,280 (scan unchanged) | ~5,000+ |

Critical prediction:

As D grows, the scan cost stays fixed at ~1,280 evals (B=512+128) while DGE pure needs proportionally more blocks. At some D*, the hybrid should overtake.

Architecture candidates for larger models:

- `784 → 128 → 64 → 10` (~110K params)
- `784 → 256 → 128 → 10` (~230K params)
- `784 → 512 → 256 → 10` (~535K params)

What to measure:

- Test accuracy vs evaluations for both methods at each scale
- Evals/step breakdown (scan vs refine vs pure DGE)
- Crossover point: at which D does hybrid first match, then beat, pure DGE?
- Wall-clock time comparison

Main hypothesis:

> There exists a parameter count D* above which the SFWHT+DGE hybrid achieves higher accuracy than pure DGE at equal evaluation budget. This crossover occurs because scan cost is O(B) (constant) while pure DGE cost is O(D/k) (linear in D).

Success criteria:

- Hybrid clearly beats pure DGE at the largest model size tested
- The crossover point can be identified and reported as a clean figure

Failure criteria:

- Pure DGE remains superior at all tested scales → SFWHT scan does not justify its overhead even at scale → pivot to other branches

Priority:

**Highest. This is a blocking gate for the entire research direction. Do not pursue other branches until this is answered.**

## 2. Vector Group DGE

Core idea:

Treat subsets of variables as vector-valued super-variables and perturb them along random unit directions, while keeping directional EMA per group.

Why it remains high priority:

- directly addresses axis-aligned blindness, independent of SFWHT
- is conceptually clean and easy to explain
- has clear synthetic benchmarks where it should help
- strong paper potential

Evidence that motivates it:

- v10 deep MLP stalled at 85%: axis-aligned updates likely contribute
- Rosenbrock benchmarks (v14–v17) converge slowly due to zig-zag

Synergy with SFWHT hybrid:

If the scaling experiment (branch 1) validates the hybrid, Vector Group can enhance the DGE-refinement phase by replacing scalar perturbations with vector perturbations within the active variable subset. This would combine spatial scan (SFWHT) + structural perturbation (vector groups) + temporal denoising (EMA).

Main hypothesis:

> On problems with correlated variables or diagonal descent structure, vector-group perturbations outperform scalar-coordinate DGE at the same evaluation budget.

Priority:

**High. Pursue after the scaling gate is passed (or failed — vector groups have standalone value regardless).**

## 3. Direction-Consistency Learning Rates

Core idea:

Adapt local learning rates from temporal consistency of the estimated direction.

Why it remains:

- low implementation cost
- universally useful regardless of which structural branch wins
- can be implemented as a quick win alongside any other experiment

Priority:

**Medium-high. Implement as a stabilizer once the primary branch is determined.**

## 4. Temporal Hierarchical SPSA / Hadamard-Cycled Probing

Status update:

The v22–v24 hybrid track partially validated this concept: lazy scanning (every 4 steps) is a temporal distribution of the structured measurement. The core insight — "spread structured probes over time" — is already active in v24.

This branch is now less urgent unless the scaling experiment reveals that even lazy scanning is too expensive, in which case a fully temporal approach (one Hadamard probe per step, never a full scan) may be the right middle ground.

Priority:

**Medium. Revisit only if the scaling experiment shows scan cost remains problematic at large D.**

## Quick Wins (Tier 0)

These remain valid and can be tested alongside the scaling experiment:

### Half-Step Retry

If a full step worsens the objective, try once with `lr/2`. Cost: 1 eval max.

### Curvature-Preconditioned Perturbations

Scale perturbation magnitude per coordinate by `1/sqrt(v_t)`. Cost: near zero.

### Full-Step Same-Batch Evaluation

Use the same mini-batch for all k perturbations within a step. Cost: zero.

## Medium-Term Ideas

These become relevant after the scaling gate:

### Layer-Wise Budget Allocation

Assign scan and DGE budgets per layer based on gradient energy. Especially relevant for the larger models in the scaling experiment.

### Warm-Start with Partial Finite Differences

Bootstrap EMA with targeted FD on the coordinates SFWHT identifies as active. Synergizes with the hybrid architecture.

### Gradient Checkpointing Temporal

Reuse past evaluations weighted by geometric distance. Most impactful at large D where each eval is expensive.

## Preserved Ideas (Not Prioritized)

### Walsh-Hadamard Binary Coding as a General Family

Fully explored via the v19–v24 SFWHT track. The hybrid scan-only approach (no Peeling, no shift decoding) is the validated survivor. Full WHT coding with Peeling works only for truly sparse gradients (v19 synthetic).

### Backtracking Line Search

Absorbed into Quick Win "Half-Step Retry".

### Fourier / Oscillatory / Tremor Ideas

Still speculative. May become relevant for RL or continuous-time settings.

## Recommended Execution Order

1. **Scaling head-to-head** (v25): Hybrid vs Pure DGE at 100K+ params — THE blocking gate
2. **Quick wins**: half-step retry + curvature perturbations + same-batch (can run in parallel)
3. If hybrid wins at scale → optimize the hybrid (layer-wise budget, warm-start, vector groups in refine phase)
4. If hybrid loses at scale → pivot to Vector Group DGE + Direction-Consistency LR as the primary research line
5. **Ablations** across whatever branch wins
6. **Paper assembly** once the strongest experimental story is clear

## Suggested Decision Gates

- [ ] Does the branch have a clear failure mode and a clear regime where it should help?
- [ ] Can it be benchmarked on synthetic problems within one session?
- [ ] Does it add a genuinely new capability, or only tuning complexity?
- [ ] If it works, can it be explained in one paragraph without hand-waving?
- [ ] If it fails, will we still learn something useful about DGE?
- [ ] **Is it grounded in at least one concrete finding from v1–v24?**
- [ ] **Does answering this question change the strategic direction of the project?** (new gate — prioritize experiments that are decision-relevant)

## Final Recommendation

The v19–v24 SFWHT track produced the project's strongest result: a hybrid optimizer that combines structured scanning with targeted refinement and matches pure DGE at small scale.

But the strategic thesis — that structured scanning amortizes at large D — remains unproven.

**The single highest-value experiment right now is scaling the head-to-head to 100K–1M parameters.** Everything else is secondary until this gate is cleared.

If the crossover exists, the paper story is:

> SFWHT+DGE is a hybrid zeroth-order optimizer that uses Walsh-Hadamard scanning to identify active gradient regions at O(B) cost, then applies targeted block perturbations for precise estimation. At scale (D > D*), this hybrid outperforms both pure block-SPSA and SFWHT-only approaches, achieving structured gradient estimation in high-dimensional black-box settings.

If the crossover does not exist, the honest story is still publishable:

> We explored structured scanning as a complement to zeroth-order optimization. While SFWHT scanning works for truly sparse gradients (93x compression at 1M dims), the overhead is not justified for dense neural network training at any tested scale. The DGE framework with orthogonal blocks and temporal denoising remains the most efficient approach.

Either outcome is valuable. That is what makes this the right next experiment.
