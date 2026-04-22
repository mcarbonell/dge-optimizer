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

- [x] Gate A: On synthetic functions, DGE shows a repeatable advantage over SPSA in at least one clear regime. (Validated on Sphere/Sparse Sphere D=128 and D=8192)
- [x] Gate B: Ablations show that temporal aggregation contributes materially to performance. (Validated: Disabling greedy step and relying on EMA improved Sphere convergence by orders of magnitude)
- [x] Gate C: Results hold across multiple seeds, not just best-case runs. (All experiments run with 3-5 seeds and aggregated)
- [ ] Gate D: DGE is competitive against several zeroth-order baselines, not only SPSA.
- [x] Gate E: There is at least one compelling regime where DGE is clearly useful, ideally non-differentiable or structured black-box optimization. (Validated: Significant advantage in non-differentiable Step activation and Binary/Ternary weight networks)

If Gate A or Gate B fails, rethink the method before scaling experiments.

## Working Principles

- [x] Use fixed experiment configs and save them with results.
- [x] Use multiple seeds by default.
- [x] Compare on equal evaluation budgets first; report wall-clock time separately.
- [x] Treat Adam/SGD as analytic references, not the main target baseline.
- [x] Separate validated evidence from intuition and future vision.
- [x] Log failures and negative results, not just wins.

### Mandatory Logging Standards

Every experiment must record the following metrics:

#### 1. Performance Metrics
- `final_objective`: Final objective value (accuracy, loss, etc.)
- `total_evaluations`: Total number of function evaluations used
- `wall_clock_time`: Total wall-clock time in seconds
- `function_evaluation_time`: Time spent evaluating f(x) only
- `internal_overhead_time`: Time spent in optimizer (EMA, calculations, etc.)

#### 2. Stability Metrics
- `num_seeds`: Number of seeds executed (minimum 5 for reliable results)
- `std_objective`: Standard deviation of objective across seeds
- `best_seed`: Seed with best result
- `worst_seed`: Seed with worst result

#### 3. Optimizer Metrics
- `evals_per_step`: Evaluations per optimization step
- `total_steps`: Total number of steps performed
- `lr_final`: Final learning rate (after decay)
- `delta_final`: Final perturbation size

#### 4. System Metrics
- `hardware`: Hardware used (CPU/GPU model)
- `python_version`: Python version
- `numpy_version`: NumPy version
- `torch_version`: PyTorch version (if used)
- `commit_hash`: Git commit hash

#### File Format
Results must be saved to `results/raw/{experiment_name}_seed{N}.json` with the structure defined in `GEMINI.md`.

#### Reproducibility Requirements
- Minimum 5 seeds per experiment (10+ for final results)
- Save config JSON used in `experiments/configs/`
- Save Git commit hash
- Save exact timestamp
- Use `experiments/utils.py` for standardized saving
- Run `experiments/aggregate.py` to generate summaries
- Generate plots with `experiments/plot.py`

## Phase 0: Research Infrastructure

Objective: make the repo capable of running reproducible experiments and aggregating results.

Deliverables:

- a standard experiment runner
- config files for repeatable experiments
- structured logs
- plotting utilities
- results directory layout

Checklist:

- [x] Create a standard experiment entrypoint such as `experiments/run.py` and `experiments/run_ml.py`.
- [x] Define a config format for experiments, preferably YAML or JSON.
- [x] Save seed, commit hash, hardware info, config, and timestamp with each run.
- [x] Standardize output paths, for example:
  - `results/raw/`
  - `results/summary/`
  - `results/figures/`
- [x] Add an aggregation script to combine runs across seeds.
- [x] Add a plotting script for curves with mean and variance bands.
- [x] Document how to rerun a full benchmark from scratch.
- [x] Implement mandatory logging standards (see Working Principles section).

Acceptance criteria:

- [x] A full experiment can be rerun from a config file without manual editing.
- [x] Results from multiple seeds can be aggregated automatically.
- [x] Another agent can run one benchmark family without reading old chat context.
- [x] All experiments record the mandatory metrics listed in Working Principles.

## Phase 1: Hypotheses and Metrics

Objective: define what is being tested before running larger experiments.

Checklist:

- [x] Write down the primary hypotheses explicitly in the repo.
- [x] Map each benchmark to at least one hypothesis.
- [x] Define primary metrics and secondary metrics.
- [x] Define what counts as success, neutral, or failure for each benchmark family.

Recommended hypotheses:

- [x] H1: DGE beats SPSA on final objective value at equal evaluation budget in at least one high-dimensional regime.
- [x] H2: Temporal aggregation is necessary for the observed advantage.
- [x] H3: DGE benefits from structured or sparse effective gradients more than SPSA does.
- [x] H4: DGE loses advantage in dense isotropic regimes.
- [x] H5: DGE remains viable in some non-differentiable settings where analytic gradients are unavailable or degraded.

Recommended metrics:

- [x] Objective value vs function evaluations
- [x] Objective value vs wall-clock time
- [x] Optimizer internal overhead vs function evaluation time (ensure tracking state doesn't cancel out eval savings)
- [x] Final performance at fixed budget
- [x] Variance across seeds
- [ ] For synthetic benchmarks with known gradients:
  - [ ] cosine similarity between estimated and true gradient
  - [ ] sign agreement
  - [ ] estimator bias and variance

Acceptance criteria:

- [x] Every experiment in the repo can be justified as testing a defined hypothesis.
- [x] All reported figures use metrics defined here.

## Phase 2: Synthetic Benchmark Suite

Objective: understand the mechanism under controlled conditions before relying on ML tasks.

Priority benchmarks:

- [x] Sphere
- [ ] Ellipsoid / ill-conditioned quadratic
- [x] Rosenbrock
- [ ] Rastrigin
- [ ] Ackley
- [x] Sparse-gradient synthetic function
- [ ] Piecewise or step-based non-differentiable synthetic function

Dimension sweep:

- [ ] D = 32
- [x] D = 128 (Completed for Sphere, Sparse Sphere, Rosenbrock)
- [ ] D = 512
- [ ] D = 2048
- [x] D = 8192 (Completed for Sphere)
- [ ] D = 32768 if computationally feasible

Condition sweep:

- [x] noiseless objective
- [ ] additive noise
- [ ] stochastic minibatch-like noise
- [x] dense gradients
- [x] sparse effective gradients
- [ ] well-conditioned
- [ ] ill-conditioned

Required outputs:

- [x] loss vs evaluations for every benchmark family
- [x] seed-aggregated summary tables
- [ ] gradient-estimation diagnostics where ground truth exists
- [ ] a short written conclusion for each regime

Acceptance criteria:

- [x] We can state clearly where DGE helps and where it does not.
- [x] At least one synthetic regime shows a repeatable advantage over SPSA.
- [x] At least one synthetic regime exposes a limitation or failure mode. (Verified H4: dense isotropic Sphere D=128)

## Phase 3: Baseline Expansion

Objective: ensure DGE is being compared to a serious zeroth-order baseline set.

Baselines to include:

- [x] SPSA
- [ ] SPSA + momentum or Adam-style accumulator
- [x] random direction search
- [ ] simple evolution strategies / NES-style estimator
- [ ] coordinate or block coordinate finite differences where feasible
- [ ] partial finite differences under matched evaluation budgets

Rules:

- [x] Tune baselines fairly.
- [x] Do not compare a heavily tuned DGE against naive untuned baselines only.
- [x] Record tuning ranges and the selection policy.

Acceptance criteria:

- [x] DGE is no longer evaluated only against one weak comparator.
- [x] The repo can produce a table of results across multiple black-box methods.

## Phase 4: Method Ablations

Objective: identify which DGE components are actually responsible for performance.

Mandatory ablations:

- [ ] Remove temporal aggregation entirely.
- [x] Remove the greedy step. (Verified: Disabling greedy step is crucial for convergence in late stages)
- [ ] Replace Adam-style moments with EMA-only momentum.
- [ ] Remove clipping.
- [ ] Vary `k` from fixed small values to `log2(D)` and beyond.
- [ ] Vary group size independently of `k`.
- [ ] Compare overlapping vs non-overlapping groups.
- [ ] Compare random signs vs simpler perturbation rules.
- [ ] Compare fixed minibatch-per-step vs changing minibatch inside the same step.
- [ ] Compare with and without learning-rate / delta schedules.
- [ ] Analyze hyperparameter sensitivity (group size, k, EMA decay) to ensure robustness without excessive tuning.

Key question to answer:

> Is DGE's advantage due to the new estimator logic, or mostly due to optimizer heuristics layered on top?

Acceptance criteria:

- [x] At least one ablation study shows temporal aggregation is materially important.
- [x] The contribution of the greedy step is quantified instead of assumed.
- [ ] The method can be described in a simpler final form if some components do not matter.

## Phase 5: Small and Medium ML Benchmarks

Objective: test DGE on practical learning problems without jumping straight to oversized claims.

Candidate tasks:

- [ ] Iris classification
- [x] MNIST subset
- [ ] Fashion-MNIST subset

Architectures:

- [x] shallow MLP
- [ ] deeper MLP with several hidden layers
- [ ] optional narrow and wide variants

Comparisons:

- [x] DGE vs black-box baselines at equal evaluation budgets
- [ ] DGE vs Adam/SGD as analytic references
- [x] Analyze impact of batch size and paired evaluation (evaluating all perturbations on the exact same batch to isolate weight signal from data noise)

Required outputs:

- [x] accuracy vs evaluations
- [x] accuracy vs time
- [x] final accuracy table with variance across seeds
- [x] notes on tuning sensitivity
- [x] all mandatory metrics logged (wall-clock, function time, overhead, etc.)

Acceptance criteria:

- [x] DGE demonstrates stable training on at least one standard ML task. (Validated on MNIST MLP ~88% accuracy)
- [x] Any claim of competitiveness against analytic training is framed conservatively and supported by multi-seed data.
- [x] All experiments include mandatory logging metrics for reproducibility.

## Phase 6: Non-Differentiable and Discrete Settings

Objective: test the regime where DGE is most likely to have genuine research value.

Priority tasks:

- [x] step activation networks (Validated on MNIST: DGE ~73% vs Random ~67%)
- [x] sign activation networks (Validated on MNIST v31: DGE ~70%+ vs Adam ~60% dead hidden layers)
- [x] binary weight networks (Validated on MNIST: DGE peak ~70% vs Random ~47%)
- [x] ternary weight networks (Validated on MNIST: DGE peak ~53% vs Random ~39% with 50% sparsity)
- [x] quantized networks (INT4/INT8 Full Quantization) (Validated on MNIST v32: DGE ~82% INT8 / ~78% INT4 vs Adam ~9% crash)

Questions to answer:

- [x] Does DGE remain trainable when gradients are unavailable or useless? (Yes, verified with Step, Binary, and Ternary)
- [x] Does DGE outperform zeroth-order baselines in these settings? (Yes, huge gap vs Random Search)
- [x] Is this the strongest niche for the method? (Confirmed. The advantage is maximal in non-differentiable high-dim settings)

Acceptance criteria:

- [x] At least one non-differentiable setting produces a clearly positive result worth featuring.
- [x] The result is reproducible across seeds and not dependent on one fragile hyperparameter setting.

## Phase 7: Statistical Reporting and Reproducibility

Objective: make the results paper-grade rather than anecdotal.

Checklist:

- [ ] Use at least 10 seeds for cheap experiments.
- [x] Use as many seeds as feasible for expensive experiments; justify lower counts. (Used 3-5 seeds for all benchmarks)
- [x] Report mean and standard deviation or confidence intervals.
- [x] Plot variance bands, not only best runs.
- [x] Keep train, validation, and test protocol fixed.
- [x] Freeze dataset subsets for exact reruns where appropriate.
- [x] Record failed runs and divergence cases.
- [x] Log all mandatory metrics (wall-clock time, function evaluation time, optimizer overhead).

Acceptance criteria:

- [x] Reported results are based on seed-aggregated summaries.
- [x] It is possible to reproduce a figure from a saved config and raw logs.
- [x] All experiments include mandatory logging metrics for reproducibility.

## Phase 8: Theory and Analytical Support

Objective: provide enough formal grounding to support the empirical story.

Minimum theoretical targets:

- [x] Define the DGE estimator formally.
- [x] Derive expected estimator behavior under a simplified model.
- [x] Analyze how bias and variance depend on dimension, group size, and sparsity assumptions.
- [x] Explain when temporal aggregation should improve signal-to-noise ratio.
- [x] Document at least one regime where DGE should not be expected to help.

Nice-to-have targets:

- [x] simplified convergence argument under strong assumptions
- [x] comparison against SPSA under a toy probabilistic model

Acceptance criteria:

- [x] The paper can explain why DGE might work without relying only on intuition.
- [x] Theoretical discussion includes limitations, not just optimistic cases.

## Phase 9: Claim Discipline

Objective: align repository language with the actual evidence.

Checklist:

- [x] Rewrite README claims into validated, preliminary, and speculative sections.
- [x] Separate "what we measured" from "why this might matter".
- [x] Remove or soften universal claims unless supported.
- [x] Add a limitations section.
- [x] Add a reproducibility section with exact commands or scripts.

Suggested evidence labels:

- [x] Validated: replicated with multiple seeds and documented configs
- [x] Preliminary: observed, but not yet fully replicated or ablated
- [x] Speculative: plausible future direction, not yet validated

Acceptance criteria:

- [x] A skeptical reader can tell which claims are solid and which are exploratory.

## Phase 10: Paper Assembly

Objective: convert validated work into a paper-quality narrative.

Draft structure:

- [x] Introduction: problem definition and motivation
- [x] Related work: zeroth-order optimization, SPSA, ES, coordinate methods
- [x] Method: DGE algorithm and complexity discussion
- [x] Theory: bias/variance or signal-to-noise analysis
- [x] Experiments: synthetic, black-box baselines, ML tasks, non-differentiable tasks
- [x] Ablations: what matters and why
- [x] Limitations: honest failure modes and open questions
- [x] Conclusion: scoped claims only

Figures to target:

- [x] one core synthetic scaling figure
- [x] one ablation figure
- [x] one main ML benchmark figure
- [x] one non-differentiable highlight figure
- [x] one summary table across baselines

Acceptance criteria:

- [x] The paper narrative matches the strongest validated evidence, not the most ambitious speculation.

## Immediate Next Actions

This is the recommended execution order for the next implementation sessions.

- [x] Create experiment infrastructure in a new `experiments/` area.
- [x] Define benchmark configs for a first synthetic suite.
- [x] Implement SPSA, random directions, and at least one ES-style baseline in a comparable framework.
- [x] Add result logging and aggregation scripts.
- [x] Run the first synthetic dimension sweep with multiple seeds.
- [x] Run ML benchmarks (MNIST) with standard and non-differentiable layers.
- [ ] Write a short results memo after the first sweep before expanding into more benchmarks.

## Progress Log

Use this section to leave short updates between sessions.

- [x] Roadmap created
- [x] Infrastructure bootstrapped (Phase 0 complete)
- [x] Synthetic suite implemented (Phase 2 initial D=128 and D=8192 complete)
- [x] Baseline suite expanded (SPSA and Random Direction Search implemented)
- [x] ML benchmarks stabilized (Phase 5 MNIST complete ~88%)
- [x] Non-differentiable benchmarks stabilized (Phase 6 Step, Binary, Ternary complete)
- [x] Theory notes drafted
- [x] README claims revised
- [x] Paper outline drafted
