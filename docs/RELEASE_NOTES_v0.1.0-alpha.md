# v0.1.0-alpha

> This is an alpha release.
> Real CFD validation (OpenFOAM/SU2) is the next milestone.

## What's included
- Rust/PyO3 optimization core
- Trust Region CEM + MLP bootstrap ensemble
- Native Residual Micro-GP Phase 2 (pure Rust, no sklearn)
- Python ask/tell API
- Constraint handling (feasibility surrogate)
- Optuna sampler integration (TrustBoOptunaSampler)
- Save/load
- 83 tests passing (71 Python + 12 Rust)
- Benchmarks vs Random/BoTorch/HEBO on synthetic functions
- Mock CFD pipeline (NACA airfoil / F1 wing style)

## Benchmark highlights (synthetic)

### Setting A — large budget (budget=500)
| Method | Ackley 50D median | Time/run |
|---|---|---|
| **TRust-BO + native Phase 2** | **5.07** | **29s** |
| BoTorch TuRBO-1 | 6.38 | 254s |
| HEBO | too slow | — |
| Random Search | 9.02 | ~0s |

### Setting B — CFD-scale budget (budget=50)
| Method | Ackley 50D median | Time/run |
|---|---|---|
| **TRust-BO** | **8.85** | **2.5s** |
| BoTorch TuRBO-1 | 8.83 | 11s |
| HEBO | 9.35 | 22s |
| Random Search | 9.57 | ~0s |

## Known limitations
- Real CFD validation not yet complete
- PyPI not yet published (build from source)
- Single-objective only
- 10D small-budget: GP-based methods (HEBO) have the edge

## Next milestone
OpenFOAM/SU2 integration and real 2D airfoil optimization.
