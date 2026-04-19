# DGE Brainstorming: Refined Research Directions

Status: working draft
Source: refined from `docs/brainstorming.md` without modifying the original
Purpose: turn raw brainstorming into a more actionable research document

## Why this document exists

The original brainstorming file contains several strong ideas, but they are mixed together at different levels of maturity.

This version reorganizes them into:

- practical optimizer improvements
- promising research directions
- speculative ideas worth preserving but not prioritizing yet

For each idea, the goal is to make it easier to answer:

- what problem it solves
- why it might work
- what it costs
- how to test it
- whether it is core to DGE or a side branch

## High-Level Takeaway

The most promising evolution of DGE is not "more randomness", but "more structured signal extraction".

Three strategic directions stand out:

1. Better step adaptation from temporal consistency
2. Better spatial structure through grouped or vector-valued perturbations
3. Better measurement structure through orthogonal or hierarchical perturbation codes

If these are pursued well, DGE may evolve from a zeroth-order heuristic into a broader family of structured gradient estimators.

## Priority Map

### Tier 1: Highest priority

These ideas seem both plausible and implementable, with a realistic chance of producing meaningful gains.

- Vector grouping with directional EMA
- Per-variable or per-group learning-rate adaptation based on directional consistency

### Tier 2: Strong research directions

These ideas may become paper-worthy branches, but need more careful experimental framing.

- Hierarchical SPSA / truncated Walsh-Hadamard perturbations
- Binary orthogonal coding via Walsh-Hadamard-style perturbation schedules

### Tier 3: Preserve, but do not prioritize yet

These ideas are creative and may become useful later, but are currently more speculative or harder to validate cleanly.

- Backtracking line search in black-box mode
- Oscillating variables with snapshots
- Fourier-based frequency decomposition
- Desynchronized tremor / phase-based exploration

## Tier 1 Directions

## 1. Direction-Consistency Learning Rates

Original source:

- `docs/brainstorming.md` section 1.1

### Core idea

Instead of adapting the step size mainly from gradient magnitude, adapt it from the stability of the estimated direction over time.

If a variable or group keeps receiving updates with the same sign or same direction, increase trust and allow larger local steps.
If the direction flips often, interpret that as noise, curvature, or overshooting and reduce the local step size.

### Why this fits DGE

DGE already depends heavily on temporal aggregation.
This idea uses the same signal source, but turns it into explicit step control rather than only denoising.

That makes it conceptually aligned with the current method rather than bolted on.

### What it may help with

- unstable updates caused by noisy block estimates
- overshooting in steep or non-smooth regions
- uneven convergence across variables
- making DGE less dependent on one global learning rate

### Main risk

If the direction estimate is itself biased or noisy, the optimizer may become overconfident on the wrong coordinates.

### Minimal testable hypothesis

> Using sign or direction consistency to scale local learning rates improves stability and final performance relative to a single global learning rate.

### Suggested first implementation

- keep a short moving statistic per variable or per group
- measure sign agreement over time
- increase local LR when agreement stays high
- reduce local LR sharply when sign flips accumulate

### Benchmarks to try first

- Rosenbrock
- ill-conditioned quadratic
- MNIST subset with current DGE

### Recommendation

High priority.
This is a practical extension and a good candidate for a first low-risk improvement.

## 2. Vector Grouping and Directional EMA

Original source:

- `docs/brainstorming.md` section 3.1

### Core idea

Treat some subsets of variables as vector-valued super-variables instead of isolated scalars.
When perturbing a group, sample a random unit direction in that group's subspace rather than independent scalar signs.
Accumulate directional history as a vector EMA, not just scalar per-coordinate history.

### Why this matters

Current DGE still inherits an axis-aligned worldview.
That can be inefficient when the useful descent direction lives in a diagonal or correlated subspace.

Vector grouping may help DGE capture:

- correlated weights
- neuron-local structure
- channel-level structure
- low-dimensional manifolds inside high-dimensional parameter spaces

### Strong intuition

This directly attacks the "zig-zag in correlated valleys" problem.
If the loss landscape prefers diagonal movement, a grouped vector update can express that naturally.

### Main risk

Bad grouping may hide useful fine-grained signals.
If groups are too large or poorly chosen, the direction estimate may become too coarse.

### Minimal testable hypothesis

> On problems with correlated variables, grouped vector perturbations outperform scalar-coordinate DGE at the same evaluation budget.

### Suggested groupings

- contiguous chunks in synthetic problems
- all incoming weights of one neuron
- small blocks within a layer
- learned or data-driven groups later, but not first

### Suggested first experiments

- Rosenbrock in multiple dimensions
- rotated quadratic functions
- MLP layers where neuron-local weights are grouped

### Recommendation

Highest priority.
This is the most compelling idea in the document in terms of novelty, plausibility, and paper potential.

## Tier 2 Directions

## 3. Hierarchical SPSA / Truncated Walsh-Hadamard Probing

Original source:

- `docs/brainstorming.md` section 3.2

### Core idea

Instead of sampling arbitrary random blocks, use structured perturbation patterns that act like low-frequency spatial probes.
The perturbations follow a hierarchical plus/minus structure similar to Walsh-Hadamard basis functions.

Rather than estimating each coordinate directly, the optimizer estimates coarse spatial components of the gradient and uses those to move in structured directions.

### Why this is interesting

This shifts the framing from "estimate many coordinates poorly" to "estimate the dominant coarse structure well".

That is a much stronger story if:

- nearby variables are correlated
- early optimization is dominated by low-frequency structure
- exact coordinate-level detail is unnecessary at first

### Strong research angle

This may lead to a clearer theoretical story than pure random grouping.
You can talk about projecting the gradient onto low-frequency spatial modes under a strict budget.

### Main risk

The assumed spatial locality may not exist in every parameterization.
For many models, parameter adjacency is arbitrary unless the grouping is architecture-aware.

### Minimal testable hypothesis

> Low-frequency structured perturbations provide a better signal-to-noise tradeoff than unstructured random perturbations when the true useful gradient has spatial coherence.

### Suggested first experiments

- synthetic piecewise-smooth functions with local correlation
- small MLPs with grouped layer ordering
- compare against random-group DGE under same number of evaluations

### Recommendation

Strong paper direction, but not the first thing to implement if the infrastructure is still evolving.

## 4. Walsh-Hadamard Binary Coding as a General Estimation Family

Original source:

- `docs/brainstorming.md` section 2.3

### Core idea

Assign binary perturbation codes over time instead of purely random signs.
Recover structured information from the response sequence using Walsh-Hadamard-like transforms or truncated code families.

### Why it fits better than Fourier

Walsh-Hadamard codes are naturally binary and align much better with DGE's current perturbation style than sinusoidal oscillations.
They are also easier to reason about in discrete settings.

### Strategic value

This may become a broader family of methods:

- random coding
- orthogonal coding
- hierarchical coding
- sparse coding

DGE could then be one member of a larger structured-perturbation framework.

### Main risk

If pushed too far, this becomes a new project rather than an incremental evolution of DGE.

### Minimal testable hypothesis

> Binary orthogonal perturbation schedules improve separability of variable contributions relative to purely random perturbation schedules.

### Recommendation

Keep this alive as a medium-term branch.
Promising, but likely best pursued after the current DGE line is experimentally stabilized.

## Tier 3 Directions

## 5. Backtracking Line Search in Black-Box Mode

Original source:

- `docs/brainstorming.md` section 1.2

### Core idea

If a proposed update worsens the objective, keep the direction but reduce the step magnitude until an improvement is found.

### Why it is appealing

It offers a safety mechanism against overshooting and may improve robustness in non-smooth landscapes.

### Why to be careful

Every backtracking stage consumes more function evaluations.
That can destroy the main budget advantage unless used selectively.

### Better framing

Treat this as an optional rescue mechanism, not a default ingredient of DGE.

### Recommendation

Low priority.
Useful if instability becomes a major practical issue, but not central to the identity of the method.

## 6. Oscillating Variables and Snapshot-Based Updates

Original source:

- `docs/brainstorming.md` section 2.1

### Core idea

Let variables oscillate around a center and capture a snapshot when the objective improves.

### What is interesting about it

It reframes optimization as continuous local exploration around a moving center.
There is a nice intuition of probing a local neighborhood without committing immediately.

### Why it feels weak right now

It is not yet clear what signal this extracts more efficiently than simpler randomized exploration.
Without a cleaner estimation story, it risks becoming a complicated form of local noise injection.

### Recommendation

Preserve as a speculative idea, but do not prioritize.

## 7. Fourier-Based Frequency Estimation

Original source:

- `docs/brainstorming.md` section 2.2

### Core idea

Assign unique oscillation frequencies to variables and recover contributions by spectral decomposition of the objective signal over time.

### Why the idea is attractive

It is elegant and ambitious.
In theory, it offers a way to multiplex many variable probes into one temporal signal.

### Why it is risky

- the objective is nonlinear
- discrete optimization steps break the clean continuous-time picture
- frequency separation may require longer observation windows than the evaluation budget allows
- the implementation complexity is high

### Recommendation

Interesting academically, but too speculative for the current stage.
If revisited later, compare it directly against Walsh-Hadamard-style binary coding.

## 8. Desynchronized Tremor / Phase-Based Exploration

Original source:

- `docs/brainstorming.md` section 2.4

### Core idea

Each variable oscillates with its own phase or random temporal perturbation, allowing asynchronous local exploration around the current center.

### What may be useful here

The general intuition of asynchronous exploratory perturbation could inspire a stochastic local probing mechanism.

### What remains unclear

It is still too close to a metaphor.
The link between the temporal behavior and a usable gradient estimate is not yet explicit.

### Recommendation

Keep only as a source of inspiration unless a sharper estimator formulation appears.

## Cross-Cutting Research Themes

These themes cut across multiple ideas and may help organize future work.

## A. Temporal Structure

Main question:

> How much useful information can be extracted by accumulating weak directional signals over time?

Relevant branches:

- directional learning-rate adaptation
- EMA denoising
- coded perturbation schedules

## B. Spatial Structure

Main question:

> Can DGE exploit correlation between nearby or logically related parameters instead of treating every dimension independently?

Relevant branches:

- vector grouping
- architecture-aware grouping
- hierarchical perturbation bases

## C. Measurement Structure

Main question:

> What perturbation patterns extract the most informative signal per function evaluation?

Relevant branches:

- random groups
- orthogonal binary coding
- hierarchical Walsh-Hadamard patterns

## Recommended Next Steps

If the goal is disciplined progress rather than open-ended ideation, the next implementation order should be:

1. Implement direction-consistency learning rates
2. Implement vector-group DGE
3. Benchmark both against current DGE on synthetic correlated problems
4. Only after that, prototype one structured perturbation family such as truncated Walsh-Hadamard probing

## Concrete Research Questions

These are the questions most worth answering next.

- [ ] Does directional consistency improve stability enough to justify local LR adaptation?
- [ ] Does vector grouping reduce zig-zag behavior on correlated landscapes?
- [ ] Does architecture-aware grouping help on neural-network benchmarks?
- [ ] Can low-frequency structured probes outperform random groups under the same evaluation budget?
- [ ] Which improvement is strongest: better temporal adaptation, better spatial grouping, or better perturbation coding?

## Shortlist for Paper-Worthy Branches

If only one or two branches can be developed deeply, these are the best candidates:

- Vector Group DGE
- Hierarchical / Walsh-Hadamard DGE

Those two ideas feel the most likely to produce:

- a clean conceptual contribution
- meaningful experiments
- a stronger theoretical story than the current purely random grouping view

## Final Assessment

The original brainstorming is strong because it does not just propose tweaks.
It points toward a deeper reframing of DGE:

- from scalar to structured updates
- from random probing to coded probing
- from noisy local estimates to signal extraction over space and time

That is the right direction.
The main thing to avoid now is spreading effort across too many speculative branches at once.

The best move is to pick one practical extension and one high-upside research branch, then validate them hard.
