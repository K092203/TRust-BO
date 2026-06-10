# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-06-11

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
- **CFD-oriented examples**: NACA airfoil and F1 wing optimization pipelines
  with a mock CFD solver, drop-in replaceable with real OpenFOAM runs.
- **Save/resume**: `engine.save("study.zip")` / `TRustBOEngine.load(...)`.
- 71 tests passing (Python + Rust), CPU-only.

### Deprecated

- `TandemEngine` / `TandemEngineV2` (sklearn-based Phase 2). Use
  `TRustBOEngine(config={"enable_phase2": True})` instead. The old classes
  remain available behind the `legacy-tandem` extra and will be removed in v0.2.
