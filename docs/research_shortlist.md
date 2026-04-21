# DGE Research Shortlist

Status: active prioritization note — updated 2026-04-21
Purpose: reduce research sprawl and focus implementation effort on the highest-upside branches

## Why this exists

The project now has enough ideas that the main risk is dilution.

This shortlist defines the top branches worth pursuing next, based on:

- conceptual strength
- implementation feasibility
- experimental clarity
- paper potential
- fit with the current DGE codebase
- **empirical evidence from v1–v19** (new criterion: ideas must be grounded in past results)

The goal is not to say other ideas are bad.
The goal is to decide what deserves serious attention now.

## Empirical Context (What v1–v19 Taught Us)

Every priority decision below is anchored to these validated facts:

| Finding | Source | Implication |
|---|---|---|
| Block-SPSA with overlapping groups ≈ same SNR as plain SPSA | `dge_feedback.md`, v14 | Fixing group structure is prerequisite, not optional |
| Orthogonal non-overlapping blocks improved SNR | v14 findings | Structured perturbations > random perturbations |
| Deep MLP stalled at 85% accuracy | v10 | Layer-mixing and axis-aligned perturbations fail at depth |
| Step/Binary/Ternary networks: DGE >> Random Search | v11, v12, v13 | Non-differentiable settings are the strongest niche |
| Batched evaluation on AMD iGPU gave 1.8x speedup | v18 | Parallelism is viable; evaluation budget is the real bottleneck |
| SFWHT recovered 15 active vars from 1M dims with 11K evals (93x compression) | **v19** | Sparse structured estimation works. This is no longer speculative. |
| EMA temporal denoising is critical for convergence | v14–v17 ablations | Temporal aggregation is a core ingredient, not a nice-to-have |
| Greedy step removal improved late-stage convergence | validation roadmap | Greedy accept/reject can trap the optimizer; softer mechanisms needed |

## Shortlist: Top 4 Branches

## 1. SFWHT-Gradient Integration into DGE Training Loop

Core idea:

Replace random block perturbations with structured SFWHT perturbation patterns inside the DGE training loop. Use the Peeling decoder to recover per-variable gradient estimates at O(B · log(D)) cost instead of O(D).

Why it is now the top priority:

- v19 validated the core mechanism: phase recovery works, compression is 93x on sparse 1M-dim problems
- the next step (`v20`) is already defined: integrate SFWHT into `BatchedMLP` for MNIST
- this is the single idea with the strongest empirical backing *and* the strongest theoretical story
- if it works on ML training, it is a paper headline result
- it directly leverages the batched GPU pipeline from v18

Evidence that supports it:

- `dge_findings_v19_sfwht.md`: 11,264 evals for 1M dims, L2 error 0.0149
- `sfwht-gradient.md`: full technical specification already written

Main hypothesis:

> SFWHT-based gradient estimation enables DGE to train neural networks with fewer evaluations than block-SPSA, especially when effective gradients are sparse or layer-structured.

Main risk:

- Neural network gradients are not truly K-sparse in the Walsh-Hadamard basis. Performance may degrade with dense gradients.
- The sparsity assumption must be validated per-layer; some layers may need fallback to standard DGE.

Best first tests:

- MNIST MLP with SFWHT replacing random perturbations (same eval budget as v18)
- Layer-by-layer decomposition: apply SFWHT per layer with B scaled to layer size
- Ablation: SFWHT on output layer only vs all layers

Required comparison:

- v18 batched DGE (random blocks) at same eval budget
- SPSA at same eval budget

Priority:

**Highest. This is the branch with the most momentum and the clearest path to a paper-worthy result.**

## 2. Vector Group DGE

Core idea:

Treat subsets of variables as vector-valued super-variables and perturb them along random unit directions, while keeping directional EMA per group.

Why it made the shortlist:

- directly addresses axis-aligned blindness, the structural weakness identified in `dge_feedback.md`
- is conceptually clean and easy to explain
- is close enough to the current algorithm to implement incrementally
- has clear synthetic benchmarks where it should help
- has strong paper potential if it improves correlated landscapes

Evidence that motivates it:

- v10 deep MLP stalled at 85%: axis-aligned updates in correlated weight space likely contribute
- Rosenbrock benchmarks (v14–v17) converge slowly due to zig-zag on the diagonal valley

Main hypothesis:

> On problems with correlated variables or diagonal descent structure, vector-group perturbations outperform scalar-coordinate DGE at the same evaluation budget.

Synergy with SFWHT:

Vector groups define *how* to structure the parameter space. SFWHT defines *how* to measure it.
These are orthogonal improvements. A combined version (SFWHT within vector groups) is a natural v21 candidate.

Best first tests:

- Rosenbrock
- rotated quadratics
- ill-conditioned correlated synthetic functions
- MLP with neuron-local grouping

Priority:

**Very high. Implement after SFWHT integration is stable, or in parallel if capacity allows.**

## 3. Direction-Consistency Learning Rates

Core idea:

Adapt local learning rates from temporal consistency of the estimated direction rather than relying only on global schedules or gradient magnitude.

Why it made the shortlist:

- low implementation cost
- high practical upside
- fits the existing DGE philosophy extremely well
- likely useful regardless of which structural branch wins later
- good candidate to stabilize other future variants

Evidence that motivates it:

- All versions v10–v18 use a single global LR with cosine/linear decay
- The feedback doc notes that variables receive uneven exposure across blocks
- v17 ellipsoid tests showed convergence depends heavily on LR schedule

Main hypothesis:

> Using directional consistency to scale local step sizes improves stability and sample efficiency relative to a single global learning rate.

Best first tests:

- current DGE on synthetic functions
- current DGE on MNIST subset
- ablation against fixed-LR DGE

Priority:

**High. This is a stabilizing improvement that benefits all other branches. Can be implemented as a quick win (~1h).**

## 4. Temporal Hierarchical SPSA / Hadamard-Cycled Probing

Core idea:

Use structured hierarchical perturbation patterns over time, with one two-evaluation probe per iteration and immediate or smoothed updates across a `log(D)` cycle.

Why it made the shortlist:

- tackles a central optimization tradeoff: richer estimation versus more frequent action
- offers a meaningful alternative to current DGE rather than a minor tweak
- has a stronger measurement-theory angle than random grouping alone
- could produce better early progress if low-frequency structure dominates

Relationship to SFWHT:

This is conceptually a *temporal* version of what SFWHT does *spatially* in one shot.
If SFWHT integration succeeds (branch 1), this branch may become less urgent — SFWHT already solves the measurement efficiency problem more directly.
However, the temporal variant may still be valuable when evaluation cost per step is very high and streaming updates are preferred.

Main hypothesis:

> Temporally distributed hierarchical probing can outperform original DGE in early convergence or progress-per-evaluation on problems with persistent low-frequency structure.

Best first tests:

- correlated synthetic landscapes
- structured quadratics
- compare reactive vs memory-augmented variants

Priority:

**Medium-high. Pursue after SFWHT integration reveals whether the spatial approach already solves the measurement problem.**

## Quick Wins (Tier 0)

These are low-cost ideas that can be implemented and validated in under an hour each.
They should be tried before or alongside the main branches.

### Half-Step Retry

Instead of binary greedy accept/reject, when a full step worsens the objective, try once with `lr/2`. Accept if it improves; reject otherwise.

- **Cost:** 1 extra eval in worst case, 0 in normal case
- **Motivation:** v14+ ablations showed greedy step hurts late convergence. This is a softer mechanism.
- **Estimated effort:** 30 minutes

### Curvature-Preconditioned Perturbations

Scale perturbation magnitude per coordinate by `1/sqrt(v_t)` where `v_t` is Adam's second moment (already computed).
This makes perturbations larger in flat directions and smaller in steep directions.

- **Cost:** Near zero — reuses existing second-moment buffer
- **Motivation:** Current perturbations assume isotropic space; ill-conditioned problems suffer
- **Estimated effort:** 30 minutes

### Full-Step Same-Batch Evaluation

Use the same mini-batch for all k perturbations within a single DGE step, not just for each ±δ pair.
This makes inter-block comparisons meaningful by eliminating data noise between blocks.

- **Cost:** Zero extra evaluations
- **Motivation:** v18 uses paired evaluation per perturbation; extending to the full step isolates weight signal from data noise more completely
- **Estimated effort:** 30 minutes

## Medium-Term Ideas

These are strong but require more thought or infrastructure before implementation.

### Layer-Wise Budget Allocation

Assign evaluation budget per layer proportionally to estimated gradient energy:

```
budget_layer_i ∝ ||EMA_gradient_layer_i||₂ / Σ_j ||EMA_gradient_layer_j||₂
```

Layers with more signal get more perturbations; dormant layers get fewer. Re-evaluate allocation every N steps.

- **Motivation:** `sfwht-gradient.md` already notes layers have different sensitivity. v10 deep MLP stalling suggests some layers starve.
- **Synergy with SFWHT:** Bucket count B can be set per-layer based on expected sparsity.
- **Estimated effort:** 2 hours

### Warm-Start with Partial Finite Differences

In the first epoch (or periodically), evaluate the top-K coordinates by estimated energy with direct finite differences to bootstrap the EMA with clean signal.

- **Motivation:** Early DGE iterations are near-pure noise; warm-starting could accelerate initial convergence significantly
- **Synergy with SFWHT:** SFWHT Peeling can identify active dimensions; FD can then refine their values precisely
- **Estimated effort:** 1 hour

### Gradient Checkpointing Temporal

Maintain a circular buffer of recent evaluations and reuse them with geometric distance-weighted decay:

```
weight_k = exp(-α · ||x_t - x_{t-k}||² / ||δ||²)
```

If the optimizer has moved little, old measurements are still relevant and boost effective SNR at zero extra cost.

- **Motivation:** DGE discards all evaluation information after each step. On slowly-changing landscapes, this is wasteful.
- **Estimated effort:** 3 hours

## Preserved Ideas (Not Prioritized)

These ideas remain in `brainstorming.md` and `brainstorming_refined.md` for future reference.

### Walsh-Hadamard Binary Coding as a General Family

Status update: The core mechanism has been validated by SFWHT v19. This is no longer speculative — it is being absorbed into Branch 1.

### Backtracking Line Search

Absorbed into the Quick Win "Half-Step Retry" above as a minimal version.

### Fourier / Oscillatory / Tremor Ideas

Still speculative. May become relevant if continuous-time optimization or RL settings are explored.

## Recommended Execution Order

1. **Quick wins** (half-step retry + curvature perturbations + same-batch): implement all three in one session
2. **SFWHT integration** into BatchedMLP for MNIST training (`v20`)
3. **Vector Group DGE** on synthetic correlated problems (`v21`)
4. **Direction-Consistency LR** as a stabilizer for both SFWHT and Vector Group variants
5. **Layer-wise budget allocation** once training loop is stable with SFWHT
6. **Ablations** across branches to determine which components drive the gains
7. **Temporal Hierarchical SPSA** only if SFWHT integration reveals gaps in streaming/online settings

## Suggested Decision Gates

Use these gates before committing heavily to a branch.

- [ ] Does the branch have a clear failure mode and a clear regime where it should help?
- [ ] Can it be benchmarked on synthetic problems within one session?
- [ ] Does it add a genuinely new capability, or only tuning complexity?
- [ ] If it works, can it be explained in one paragraph without hand-waving?
- [ ] If it fails, will we still learn something useful about DGE?
- [ ] **Is it grounded in at least one concrete finding from v1–v19?** (new gate)

If a branch does not pass these gates, keep it in brainstorming rather than promoting it.

## Final Recommendation

The v19 SFWHT result changes the strategic picture fundamentally.

Before v19, the project was evolving DGE incrementally: better groups, better LR schedules.
After v19, there is a credible path to a **qualitatively different estimator** that extracts gradients from structured measurements rather than random probing.

The recommended strategy is:

- **Core bet:** SFWHT-Gradient integration (Branch 1)
- **Structural complement:** Vector Group DGE (Branch 2)
- **Stabilizer:** Direction-Consistency LR (Branch 3)
- **Quick wins:** Half-step retry + curvature perturbations + same-batch (implement immediately)

If SFWHT integration succeeds on MNIST, the paper story becomes:

> DGE evolved from a Block-SPSA with temporal denoising into a structured gradient estimator that exploits Walsh-Hadamard sparsity to achieve O(B·log D) gradient estimation in high-dimensional non-differentiable settings — validated on 1M+ parameter models with 93x compression over brute-force finite differences.

That is a paper worth writing.
