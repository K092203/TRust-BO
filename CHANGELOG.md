# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed

- Removed the deprecated sklearn-based `TandemEngine` / `TandemEngineV2`,
  the `legacy-tandem` extra, and their obsolete benchmark scripts. The native
  Rust residual Micro-GP (`config={"enable_phase2": True}`) is the supported
  replacement. This removal will be released in v0.3.0.

## [0.2.0] - 2026-07-11

### Changed

- **Default acquisition is now `"ts"`** (Thompson-sampling-style randomized
  acquisition: per-candidate draws from the ensemble's predictive marginal
  `N(μ, σ²)`). On a 16-case synthetic suite (Ackley/Rastrigin/Rosenbrock/Levy
  × 50/100D × {noise-free, 5% noise} × 8 seeds, budget 250) it improves final
  regret by ~10–14% geometric mean over EI, winning 27/32 case-comparisons.
  `"ei"` and `"ucb"` remain available via `config={"acquisition": "ei"}`.
- **~2× faster `ask()`**: CEM candidate generation now runs its multi-start /
  multi-TR jobs in parallel with rayon, and constant training tensors are
  hoisted out of the surrogate epoch loop. Same-seed results are bit-identical
  to the previous sequential implementation.
- Constrained `"ts"` scores are shifted non-negative before the
  P(feasible) weighting (signed scores would otherwise favor
  predicted-infeasible points); non-finite predictions rank last
  deterministically.
- **`"ts"` re-validated on a real-CFD-like problem (NeuralFoil, 16D CST
  airfoil, Cl/Cd)**: `"ei"` wins clearly there (ts/ei geometric mean
  0.86–0.92, `"ts"` losing 2/8 seeds at both noise levels), reversing the
  synthetic-suite result above. `"ts"`'s exploration bonus pays off on
  multimodal problems but not on smoother, near-unimodal response surfaces —
  prefer `config={"acquisition": "ei"}` for CFD-shaped objectives.

### Added

- `config={"phase2_ls_prior": True}` (default `False`): switches the Phase 2
  micro-GP's length-scale search from MLE to a dimension-scaled LogNormal MAP
  prior (Hvarfner et al. 2024 / BoTorch reference values, shifted into
  trust-region-local coordinates). Evaluated on the 16-case synthetic suite
  (256 runs) and found to have no benefit (regret ratio 0.996) — kept behind
  the flag, off by default, for reference rather than removed.

### Docs

- Resolved staleness/duplication introduced by the above findings across
  CLAUDE.md, docs/BENCHMARK.md, docs/ALGORITHM.md, docs/DEVELOPMENT.md,
  docs/PERFORMANCE_ASSESSMENT.md, docs/ROADMAP.md, and README.md; flagged
  historical planning docs as superseded snapshots.
- 110 tests total (69 passed + 2 skipped Python, 39 passed Rust), CPU-only.

## [0.1.0] - 2026-06-14

### Added

- **Trust Region Bayesian Optimization engine written in Rust** (TuRBO-style),
  exposed to Python via PyO3. No GPU, no BLAS/LAPACK, no cloud required.
- **MLP Bootstrap Ensemble surrogate** (5 members, warm-started between rounds)
  with CEM acquisition optimization inside the trust region.
- **Native Rust Phase 2** (Tandem Residual-GP): a pure-Rust Matern 5/2 micro-GP
  fits the residuals of the MLP ensemble near the incumbent for endgame
  refinement. Enabled with `config={"enable_phase2": True}`. No sklearn needed.
  +32% over plain TRust-BO at 50D, +55% at 10D (Ackley, 3-seed median).
- **Constraint handling** via a feasibility surrogate (acquisition × P(feasible)).
- **Async parallel / rolling evaluation** (`RollingTRustBOEngine`, SLURM-ready)
  for expensive solvers where evaluations take minutes to hours.
- **Multi-objective optimization** (`MultiObjectiveEngine`): Chebyshev scalarization
  (any number of objectives) and a closed-form 2-objective Expected Hypervolume
  Improvement (EHVI) implemented in Rust. Exposes `pareto_front()` / `hypervolume()`.
- **Real CFD airfoil pipelines**: NeuralFoil (H-1, learned surrogate) and
  SU2 RANS (H-2, real steady Navier-Stokes, Ma=0.3 Re=3e6 SA). Multi-objective
  Cl/Cd optimization validated on SU2 (K-2-8).
- **Save/resume**: `engine.save("study.zip")` / `TRustBOEngine.load(...)`.
- **Optuna sampler** integration (`trust_bo.integrations.optuna`).
- 91 tests passing (63 Python + 28 Rust).

### Deprecated

- `TandemEngine` / `TandemEngineV2` (sklearn-based Phase 2). Use
  `TRustBOEngine(config={"enable_phase2": True})` instead. The old classes remained available behind the `legacy-tandem` extra; their
  originally planned v0.2 removal is completed in the Unreleased v0.3 changes.
