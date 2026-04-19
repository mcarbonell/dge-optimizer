# DGE Validation Roadmap

Status: active working document
Audience: future sessions, collaborators, and coding agents
Goal: turn DGE from a promising idea into a credible research project and paper candidate

## Purpose

This roadmap defines a serious validation program for DGE. The objective is not to produce a few good demos, but to establish:

- what DGE actually does better than existing zeroth-order baselines
- under which assumptions DGE works well
- which parts of the method matter most
- which claims are validated, preliminary, or speculative

This document should be updated as work progresses. Keep it operational, not aspirational.

## Core Research Thesis

Working thesis:

> DGE is a zeroth-order optimizer based on randomized block perturbations plus temporal aggregation. It may improve sample efficiency and signal quality relative to SPSA-style methods in some high-dimensional, structured, or non-differentiable settings.

Do not claim stronger statements until the evidence exists.

Examples of claims that should remain speculative until proven:

- "solves the curse of dimensionality"
- "replaces backpropagation"
- "scales to trillions of parameters on consumer hardware"
- "universally outperforms black-box baselines"

## Success Gates

These gates decide whether the project is strong enough to continue toward a paper.

- [ ] Gate A: On synthetic functions, DGE shows a repeatable advantage over SPSA in at least one clear regime.
- [ ] Gate B: Ablations show that temporal aggregation contributes materially to performance.
- [ ] Gate C: Results hold across multiple seeds, not just best-case runs.
- [ ] Gate D: DGE is competitive against several zeroth-order baselines, not only SPSA.
- [ ] Gate E: There is at least one compelling regime where DGE is clearly useful, ideally non-differentiable or structured black-box optimization.

If Gate A or Gate B fails, rethink the method before scaling experiments.

## Working Principles

- [ ] Use fixed experiment configs and save them with results.
- [ ] Use multiple seeds by default.
- [ ] Compare on equal evaluation budgets first; report wall-clock time separately.
- [ ] Treat Adam/SGD as analytic references, not the main target baseline.
- [ ] Separate validated evidence from intuition and future vision.
- [ ] Log failures and negative results, not just wins.

## Phase 0: Research Infrastructure

Objective: make the repo capable of running reproducible experiments and aggregating results.

Deliverables:

- a standard experiment runner
- config files for repeatable experiments
- structured logs
- plotting utilities
- results directory layout

Checklist:

- [ ] Create a standard experiment entrypoint such as `experiments/run.py`.
- [ ] Define a config format for experiments, preferably YAML or JSON.
- [ ] Save seed, commit hash, hardware info, config, and timestamp with each run.
- [ ] Standardize output paths, for example:
  - `results/raw/`
  - `results/summary/`
  - `results/figures/`
- [ ] Add an aggregation script to combine runs across seeds.
- [ ] Add a plotting script for curves with mean and variance bands.
- [ ] Document how to rerun a full benchmark from scratch.

Acceptance criteria:

- [ ] A full experiment can be rerun from a config file without manual editing.
- [ ] Results from multiple seeds can be aggregated automatically.
- [ ] Another agent can run one benchmark family without reading old chat context.

## Phase 1: Hypotheses and Metrics

Objective: define what is being tested before running larger experiments.

Checklist:

- [ ] Write down the primary hypotheses explicitly in the repo.
- [ ] Map each benchmark to at least one hypothesis.
- [ ] Define primary metrics and secondary metrics.
- [ ] Define what counts as success, neutral, or failure for each benchmark family.

Recommended hypotheses:

- [ ] H1: DGE beats SPSA on final objective value at equal evaluation budget in at least one high-dimensional regime.
- [ ] H2: Temporal aggregation is necessary for the observed advantage.
- [ ] H3: DGE benefits from structured or sparse effective gradients more than SPSA does.
- [ ] H4: DGE loses advantage in dense isotropic regimes.
- [ ] H5: DGE remains viable in some non-differentiable settings where analytic gradients are unavailable or degraded.

Recommended metrics:

- [ ] Objective value vs function evaluations
- [ ] Objective value vs wall-clock time
- [ ] Final performance at fixed budget
- [ ] Variance across seeds
- [ ] For synthetic benchmarks with known gradients:
  - [ ] cosine similarity between estimated and true gradient
  - [ ] sign agreement
  - [ ] estimator bias and variance

Acceptance criteria:

- [ ] Every experiment in the repo can be justified as testing a defined hypothesis.
- [ ] All reported figures use metrics defined here.

## Phase 2: Synthetic Benchmark Suite

Objective: understand the mechanism under controlled conditions before relying on ML tasks.

Priority benchmarks:

- [ ] Sphere
- [ ] Ellipsoid / ill-conditioned quadratic
- [ ] Rosenbrock
- [ ] Rastrigin
- [ ] Ackley
- [ ] Sparse-gradient synthetic function
- [ ] Piecewise or step-based non-differentiable synthetic function

Dimension sweep:

- [ ] D = 32
- [ ] D = 128
- [ ] D = 512
- [ ] D = 2048
- [ ] D = 8192
- [ ] D = 32768 if computationally feasible

Condition sweep:

- [ ] noiseless objective
- [ ] additive noise
- [ ] stochastic minibatch-like noise
- [ ] dense gradients
- [ ] sparse effective gradients
- [ ] well-conditioned
- [ ] ill-conditioned

Required outputs:

- [ ] loss vs evaluations for every benchmark family
- [ ] seed-aggregated summary tables
- [ ] gradient-estimation diagnostics where ground truth exists
- [ ] a short written conclusion for each regime

Acceptance criteria:

- [ ] We can state clearly where DGE helps and where it does not.
- [ ] At least one synthetic regime shows a repeatable advantage over SPSA.
- [ ] At least one synthetic regime exposes a limitation or failure mode.

## Phase 3: Baseline Expansion

Objective: ensure DGE is being compared to a serious zeroth-order baseline set.

Baselines to include:

- [ ] SPSA
- [ ] SPSA + momentum or Adam-style accumulator
- [ ] random direction search
- [ ] simple evolution strategies / NES-style estimator
- [ ] coordinate or block coordinate finite differences where feasible
- [ ] partial finite differences under matched evaluation budgets

Rules:

- [ ] Tune baselines fairly.
- [ ] Do not compare a heavily tuned DGE against naive untuned baselines only.
- [ ] Record tuning ranges and the selection policy.

Acceptance criteria:

- [ ] DGE is no longer evaluated only against one weak comparator.
- [ ] The repo can produce a table of results across multiple black-box methods.

## Phase 4: Method Ablations

Objective: identify which DGE components are actually responsible for performance.

Mandatory ablations:

- [ ] Remove temporal aggregation entirely.
- [ ] Remove the greedy step.
- [ ] Replace Adam-style moments with EMA-only momentum.
- [ ] Remove clipping.
- [ ] Vary `k` from fixed small values to `log2(D)` and beyond.
- [ ] Vary group size independently of `k`.
- [ ] Compare overlapping vs non-overlapping groups.
- [ ] Compare random signs vs simpler perturbation rules.
- [ ] Compare fixed minibatch-per-step vs changing minibatch inside the same step.
- [ ] Compare with and without learning-rate / delta schedules.

Key question to answer:

> Is DGE's advantage due to the new estimator logic, or mostly due to optimizer heuristics layered on top?

Acceptance criteria:

- [ ] At least one ablation study shows temporal aggregation is materially important.
- [ ] The contribution of the greedy step is quantified instead of assumed.
- [ ] The method can be described in a simpler final form if some components do not matter.

## Phase 5: Small and Medium ML Benchmarks

Objective: test DGE on practical learning problems without jumping straight to oversized claims.

Candidate tasks:

- [ ] Iris classification
- [ ] MNIST subset
- [ ] Fashion-MNIST subset

Architectures:

- [ ] shallow MLP
- [ ] deeper MLP with several hidden layers
- [ ] optional narrow and wide variants

Comparisons:

- [ ] DGE vs black-box baselines at equal evaluation budgets
- [ ] DGE vs Adam/SGD as analytic references
- [ ] Analyze impact of batch size and paired evaluation (evaluating all perturbations on the exact same batch to isolate weight signal from data noise)

Required outputs:

- [ ] accuracy vs evaluations
- [ ] accuracy vs time
- [ ] final accuracy table with variance across seeds
- [ ] notes on tuning sensitivity

Acceptance criteria:

- [ ] DGE demonstrates stable training on at least one standard ML task.
- [ ] Any claim of competitiveness against analytic training is framed conservatively and supported by multi-seed data.

## Phase 6: Non-Differentiable and Discrete Settings

Objective: test the regime where DGE is most likely to have genuine research value.

Priority tasks:

- [ ] step activation networks
- [ ] sign activation networks
- [ ] binary weight networks
- [ ] ternary weight networks
- [ ] optional discrete or simulator-like toy black-box problem

Questions to answer:

- [ ] Does DGE remain trainable when gradients are unavailable or useless?
- [ ] Does DGE outperform zeroth-order baselines in these settings?
- [ ] Is this the strongest niche for the method?

Acceptance criteria:

- [ ] At least one non-differentiable setting produces a clearly positive result worth featuring.
- [ ] The result is reproducible across seeds and not dependent on one fragile hyperparameter setting.

## Phase 7: Statistical Reporting and Reproducibility

Objective: make the results paper-grade rather than anecdotal.

Checklist:

- [ ] Use at least 10 seeds for cheap experiments.
- [ ] Use as many seeds as feasible for expensive experiments; justify lower counts.
- [ ] Report mean and standard deviation or confidence intervals.
- [ ] Plot variance bands, not only best runs.
- [ ] Keep train, validation, and test protocol fixed.
- [ ] Freeze dataset subsets for exact reruns where appropriate.
- [ ] Record failed runs and divergence cases.

Acceptance criteria:

- [ ] Reported results are based on seed-aggregated summaries.
- [ ] It is possible to reproduce a figure from a saved config and raw logs.

## Phase 8: Theory and Analytical Support

Objective: provide enough formal grounding to support the empirical story.

Minimum theoretical targets:

- [ ] Define the DGE estimator formally.
- [ ] Derive expected estimator behavior under a simplified model.
- [ ] Analyze how bias and variance depend on dimension, group size, and sparsity assumptions.
- [ ] Explain when temporal aggregation should improve signal-to-noise ratio.
- [ ] Document at least one regime where DGE should not be expected to help.

Nice-to-have targets:

- [ ] simplified convergence argument under strong assumptions
- [ ] comparison against SPSA under a toy probabilistic model

Acceptance criteria:

- [ ] The paper can explain why DGE might work without relying only on intuition.
- [ ] Theoretical discussion includes limitations, not just optimistic cases.

## Phase 9: Claim Discipline

Objective: align repository language with the actual evidence.

Checklist:

- [ ] Rewrite README claims into validated, preliminary, and speculative sections.
- [ ] Separate "what we measured" from "why this might matter".
- [ ] Remove or soften universal claims unless supported.
- [ ] Add a limitations section.
- [ ] Add a reproducibility section with exact commands or scripts.

Suggested evidence labels:

- [ ] Validated: replicated with multiple seeds and documented configs
- [ ] Preliminary: observed, but not yet fully replicated or ablated
- [ ] Speculative: plausible future direction, not yet validated

Acceptance criteria:

- [ ] A skeptical reader can tell which claims are solid and which are exploratory.

## Phase 10: Paper Assembly

Objective: convert validated work into a paper-quality narrative.

Draft structure:

- [ ] Introduction: problem definition and motivation
- [ ] Related work: zeroth-order optimization, SPSA, ES, coordinate methods
- [ ] Method: DGE algorithm and complexity discussion
- [ ] Theory: bias/variance or signal-to-noise analysis
- [ ] Experiments: synthetic, black-box baselines, ML tasks, non-differentiable tasks
- [ ] Ablations: what matters and why
- [ ] Limitations: honest failure modes and open questions
- [ ] Conclusion: scoped claims only

Figures to target:

- [ ] one core synthetic scaling figure
- [ ] one ablation figure
- [ ] one main ML benchmark figure
- [ ] one non-differentiable highlight figure
- [ ] one summary table across baselines

Acceptance criteria:

- [ ] The paper narrative matches the strongest validated evidence, not the most ambitious speculation.

## Immediate Next Actions

This is the recommended execution order for the next implementation sessions.

- [ ] Create experiment infrastructure in a new `experiments/` area.
- [ ] Define benchmark configs for a first synthetic suite.
- [ ] Implement SPSA, random directions, and at least one ES-style baseline in a comparable framework.
- [ ] Add result logging and aggregation scripts.
- [ ] Run the first synthetic dimension sweep with multiple seeds.
- [ ] Write a short results memo after the first sweep before expanding into more benchmarks.

## Session Handoff Notes

If another agent picks this up, start here:

1. Read this roadmap.
2. Inspect current `dge/optimizer.py` and existing `scratch/` experiments.
3. Do not add new grand claims before Phase 2 and Phase 4 are underway.
4. Prioritize infrastructure and synthetic benchmarks before adding more MNIST-like demos.
5. Update this document after each meaningful milestone.

## Progress Log

Use this section to leave short updates between sessions.

- [ ] Roadmap created
- [ ] Infrastructure bootstrapped
- [ ] Synthetic suite implemented
- [ ] Baseline suite expanded
- [ ] Ablations completed
- [ ] ML benchmarks stabilized
- [ ] Non-differentiable benchmarks stabilized
- [ ] Theory notes drafted
- [ ] README claims revised
- [ ] Paper outline drafted
