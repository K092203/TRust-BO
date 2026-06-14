# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Trust Region Bayesian Optimization engine written in Rust** (TuRBO-style),
  exposed to Python via PyO3. No GPU, no BLAS/LAPACK, no cloud required.
- **MLP Bootstrap Ensemble surrogate** (5 members, warm-started between rounds)
  with CEM acquisition optimization inside the trust region.
- **Native Rust Phase 2** (Tandem Residual-GP): a pure-Rust Matern 5/2 micro-GP
  fits the residuals of the MLP ensemble near the incumbent for endgame
  refinement. Enabled with `config={"enable_phase2": True}`. No sklearn needed.
  +32% over plain TRust-BO at 50D, +55% at 10D (Ackley, 3-seed median).
- **Constraint handling** via a feasibility surrogate (EI × P(feasible)).
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
- 91 tests passing (63 Python + 28 Rust), CPU-only.

### Deprecated

- `TandemEngine` / `TandemEngineV2` (sklearn-based Phase 2). Use
  `TRustBOEngine(config={"enable_phase2": True})` instead. The old classes
  remain available behind the `legacy-tandem` extra and will be removed in v0.2.
