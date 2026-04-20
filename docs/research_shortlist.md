# DGE Research Shortlist

Status: active prioritization note
Purpose: reduce research sprawl and focus implementation effort on the highest-upside branches

## Why this exists

The project now has enough ideas that the main risk is dilution.

This shortlist defines the top branches worth pursuing next, based on:

- conceptual strength
- implementation feasibility
- experimental clarity
- paper potential
- fit with the current DGE codebase

The goal is not to say other ideas are bad.
The goal is to decide what deserves serious attention now.

## Shortlist: Top 3 Branches

## 1. Vector Group DGE

Core idea:

Treat subsets of variables as vector-valued super-variables and perturb them along random unit directions, while keeping directional EMA per group.

Why it made the shortlist:

- directly addresses a likely weakness of current DGE: axis-aligned blindness
- is conceptually clean and easy to explain
- is close enough to the current algorithm to implement incrementally
- has clear synthetic benchmarks where it should help
- has strong paper potential if it improves correlated landscapes

Main hypothesis:

> On problems with correlated variables or diagonal descent structure, vector-group perturbations outperform scalar-coordinate DGE at the same evaluation budget.

Best first tests:

- Rosenbrock
- rotated quadratics
- ill-conditioned correlated synthetic functions

Priority:

Highest.
If only one branch gets implemented deeply next, this should probably be it.

## 2. Direction-Consistency Learning Rates

Core idea:

Adapt local learning rates from temporal consistency of the estimated direction rather than relying only on global schedules or gradient magnitude.

Why it made the shortlist:

- low implementation cost
- high practical upside
- fits the existing DGE philosophy extremely well
- likely useful regardless of which structural branch wins later
- good candidate to stabilize other future variants

Main hypothesis:

> Using directional consistency to scale local step sizes improves stability and sample efficiency relative to a single global learning rate.

Best first tests:

- current DGE on synthetic functions
- current DGE on MNIST subset
- ablation against fixed-LR DGE

Priority:

Very high.
This is the most practical branch and the easiest one to validate quickly.

## 3. Temporal Hierarchical SPSA / Hadamard-Cycled Probing

Core idea:

Use structured hierarchical perturbation patterns over time, with one two-evaluation probe per iteration and immediate or smoothed updates across a `log(D)` cycle.

Why it made the shortlist:

- tackles a central optimization tradeoff: richer estimation versus more frequent action
- offers a meaningful alternative to current DGE rather than a minor tweak
- has a stronger measurement-theory angle than random grouping alone
- could produce better early progress if low-frequency structure dominates

Main hypothesis:

> Temporally distributed hierarchical probing can outperform original DGE in early convergence or progress-per-evaluation on problems with persistent low-frequency structure.

Best first tests:

- correlated synthetic landscapes
- structured quadratics
- compare reactive vs memory-augmented variants

Priority:

High, but after at least one lower-risk branch is stabilized.
This is the most ambitious branch in the shortlist.

## Near Misses

These ideas are still interesting, but did not make the top 3.

## Walsh-Hadamard Binary Coding as a General Family

Why it missed:

- strong long-term potential
- but risks becoming a separate project too early
- better revisited after one structured probing branch is stabilized

## Backtracking Line Search

Why it missed:

- useful as a robustness tool
- but not likely to define the research contribution
- risks hurting the evaluation budget story

## Fourier / Oscillatory / Tremor Ideas

Why they missed:

- creative and potentially deep
- but currently too speculative relative to the project's validation stage

## Recommended Execution Order

If the next few sessions are focused on disciplined progress, the recommended order is:

1. Implement and benchmark direction-consistency learning rates
2. Implement and benchmark vector-group DGE
3. Run ablations to see whether one clearly dominates
4. Only then branch into temporal hierarchical SPSA / Hadamard-cycled probing

Reason:

- Branch 2 is cheap and stabilizing
- Branch 1 is the most plausible high-value structural extension
- Branch 3 is the most ambitious and deserves a cleaner baseline before evaluation

## Suggested Decision Gates

Use these gates before committing heavily to a branch.

- [ ] Does the branch have a clear failure mode and a clear regime where it should help?
- [ ] Can it be benchmarked on synthetic problems within one session?
- [ ] Does it add a genuinely new capability, or only tuning complexity?
- [ ] If it works, can it be explained in one paragraph without hand-waving?
- [ ] If it fails, will we still learn something useful about DGE?

If a branch does not pass these gates, keep it in brainstorming rather than promoting it.

## Final Recommendation

If the project wants one practical win and one high-upside research bet, choose:

- Practical win: Direction-Consistency Learning Rates
- High-upside structural bet: Vector Group DGE

If the project wants one more ambitious branch after that, choose:

- Temporal Hierarchical SPSA / Hadamard-Cycled Probing

This combination gives a good balance of:

- immediate optimizer improvement
- strong experimental story
- credible paper evolution
