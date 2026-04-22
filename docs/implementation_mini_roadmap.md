# DGE Implementation Mini Roadmap

Status: handoff document
Audience: another coding agent or future session
Purpose: translate the research shortlist into concrete implementation work

## Context

This roadmap is intentionally pragmatic.
It is not a theory note and not a broad validation plan.
It is a build order for the next highest-value branches, assuming implementation time and iteration budget are limited.

The three target branches are:

1. Direction-Consistency Learning Rates
2. Vector Group DGE
3. Temporal Hierarchical SPSA / Hadamard-Cycled Probing

The recommended order is deliberate:

- start with the lowest-risk stabilizing extension
- then implement the strongest structural improvement
- only then implement the most ambitious alternative estimator branch

## Global Instructions for the Implementing Agent

- Do not replace the current DGE implementation immediately.
- Add each branch as an isolated variant behind a clean interface.
- Keep the baseline reproducible and runnable at all times.
- Prefer small, comparable experiment scripts over broad refactors.
- Stop after each branch and summarize whether it earned the right to continue.

Before coding, read:

- `docs/dge_validation_roadmap.md`
- `docs/brainstorming_refined.md`
- `docs/research_shortlist.md`
- `dge/optimizer.py`

## Deliverable Philosophy

For each branch, the implementing agent should produce:

- one implementation variant
- one or two focused benchmark scripts or configs
- one short result memo
- one go / no-go recommendation for deeper investment

Avoid building large frameworks unless clearly necessary.

## Branch 1: Direction-Consistency Learning Rates

Goal:

Add local or semi-local step adaptation using temporal consistency of update direction.

Why first:

- lowest implementation risk
- likely useful even if later branches win
- may stabilize current DGE and improve benchmark quality

### Implementation Tasks

- [x] Create a variant of DGE with direction-consistency-based LR adaptation.
- [x] Keep the existing optimizer untouched or subclassed cleanly.
- [x] Decide whether adaptation is per-variable or per-group; start with the simplest viable option.
- [x] Track a lightweight consistency statistic over time:
  - sign agreement
  - directional persistence
  - or flip frequency
- [x] Increase local LR when consistency stays high.
- [x] Decrease local LR when direction flips repeatedly.
- [x] Add caps and floors to avoid exploding or vanishing local LRs.
- [x] Log enough internal stats to inspect whether adaptation is actually doing anything.

### Suggested Minimal API

- `DGEOptimizer(...)`
- `DGEConsistencyLR(...)`

Do not overdesign.

### Required Benchmarks

- [ ] Rosenbrock
- [ ] ill-conditioned quadratic
- [ ] current MNIST subset benchmark if cheap enough

### Required Comparisons

- [ ] baseline DGE
- [ ] consistency-LR DGE

### Acceptance Criteria

- [ ] Variant runs stably across multiple seeds.
- [ ] Local LR adaptation visibly changes optimizer behavior.
- [ ] At least one benchmark shows improved stability or final performance.

### Stop / Continue Decision

Continue only if:

- there is a measurable benefit on at least one benchmark
- and the added complexity is still easy to explain

If not, keep as optional experimental code and move on.

## Branch 2: Vector Group DGE

Goal:

Replace scalar per-coordinate perturbation logic inside selected groups with vector-valued directional perturbations and directional EMA.

Why second:

- strongest structural idea in the current shortlist
- likely strongest paper candidate
- clearer upside on correlated landscapes

### Implementation Tasks

- [x] Design a grouping abstraction without overengineering.
- [x] Start with fixed groups, not learned groups.
- [x] Support at least one grouping strategy:
  - contiguous chunks
  - neuron-local groups
  - fixed-size blocks
- [x] Sample a random unit direction within each selected group.
- [x] Replace scalar group signal accumulation with directional accumulation.
- [x] Maintain EMA or moment estimates at the group-direction level.
- [x] Convert group-direction updates back into parameter updates cleanly.
- [x] Keep a scalar-DGE path available for direct comparison.

### Important Scope Control

Do not solve architecture-aware grouping perfectly in version one.
Use one or two simple grouping strategies first.

### Required Benchmarks

- [ ] Rosenbrock
- [ ] rotated quadratic
- [ ] correlated synthetic benchmark of your choice
- [ ] optional MNIST MLP with neuron-local grouping if feasible

### Required Comparisons

- [ ] baseline DGE
- [ ] consistency-LR DGE if Branch 1 was useful
- [ ] vector-group DGE

### Diagnostics to Log

- [ ] group-level update norms
- [ ] directional persistence
- [ ] whether groups align better with correlated descent directions

### Acceptance Criteria

- [ ] Variant is stable and not dramatically slower per evaluation.
- [ ] It improves at least one correlated benchmark versus baseline DGE.
- [ ] The benefit is attributable to grouping, not only to incidental tuning.

### Stop / Continue Decision

This branch deserves deeper follow-up only if:

- it clearly helps on correlated landscapes
- and its explanation remains clean enough for a paper section

If results are mixed, keep the synthetic evidence and postpone ML scaling.

## Branch 3: Temporal Hierarchical SPSA / Hadamard-Cycled Probing

Goal:

Implement a structured perturbation schedule over time where each iteration uses one hierarchical pattern and immediately or gradually updates the parameters.

Why third:

- most ambitious branch
- strongest departure from original DGE
- depends on having a strong baseline before evaluation

### Implementation Tasks

- [ ] Define a simple hierarchical perturbation schedule.
- [ ] Start with deterministic or semi-deterministic low-frequency patterns.
- [ ] Implement one cycle over approximately `log(D)` steps.
- [ ] Build two variants:
  - reactive immediate-update variant
  - memory-augmented variant with EMA over probes or levels
- [ ] Keep evaluation cost per iteration at 2 function calls.
- [ ] Ensure the cycle structure is inspectable and logged.

### Scope Control

Do not build a full general Walsh-Hadamard framework in the first pass.
Implement only enough structure to test the central hypothesis.

### Required Benchmarks

- [ ] correlated synthetic landscapes
- [ ] structured quadratics
- [ ] Rosenbrock-like valley benchmark

### Required Comparisons

- [ ] SPSA
- [ ] baseline DGE
- [ ] hierarchical full-cycle accumulation before update
- [ ] reactive temporal hierarchy
- [ ] temporal hierarchy with memory

### Required Metrics

- [ ] progress in the first part of training
- [ ] final performance at fixed budget
- [ ] stability across seeds
- [ ] progress per evaluation
- [ ] optional progress per committed optimizer step

### Acceptance Criteria

- [ ] At least one temporal hierarchical variant shows better early progress than baseline DGE or SPSA on a structured benchmark.
- [ ] The reactive and memory-based variants can be meaningfully compared.
- [ ] The result can be explained as a real tradeoff, not noise.

### Stop / Continue Decision

Continue only if this branch shows a distinctive regime where frequent structured action beats slower richer estimation.

If it does not, preserve the code and lessons learned, but do not let it dominate the project.

## Suggested File / Structure Approach

This is only a suggestion.
The implementing agent may adjust it if there is a cleaner fit.

- `dge/optimizer.py`
  - keep current baseline stable
- `dge/variants/consistency_lr.py`
- `dge/variants/vector_group.py`
- `dge/variants/temporal_hierarchy.py`
- `experiments/` or `scratch/`
  - one script per focused benchmark family
- `docs/`
  - add one short memo per branch after running it

If introducing `dge/variants/` is too much overhead, use a simpler structure, but keep variants isolated.

## Benchmark Execution Order

Recommended order for another agent:

1. Implement Branch 1
2. Benchmark Branch 1 on 2-3 synthetic tasks
3. Write a short memo
4. Implement Branch 2
5. Benchmark Branch 2 on correlated synthetic tasks
6. Write a short memo
7. Only then implement Branch 3
8. Benchmark Branch 3 against SPSA and baseline DGE
9. Write a short memo

Do not implement all three branches before benchmarking.

## Memo Template

After each branch, create a brief note in `docs/` with:

- what was implemented
- what benchmarks were run
- whether the branch helped
- where it failed
- whether it should continue

Suggested filenames:

- `docs/dge_consistency_lr_results.md`
- `docs/dge_vector_group_results.md`
- `docs/dge_temporal_hierarchy_results.md`

## Hard Rules

- Do not silently change the baseline behavior while adding a variant.
- Do not declare a branch successful from one seed.
- Do not compare only against a weak SPSA baseline when evaluating Branch 3.
- Do not add many speculative sub-variants before one core version works.
- Do not optimize presentation before getting benchmark signal.

## Final Recommendation to the Implementing Agent

If limited time forces a hard choice:

- implement Branch 1 for the fastest practical return
- implement Branch 2 for the strongest research upside

Branch 3 should be treated as a serious but higher-risk bet, not as the default next step unless the first two are already under control.
