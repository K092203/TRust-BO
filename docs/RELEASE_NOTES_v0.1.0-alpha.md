# v0.1.0-alpha

> Alpha release. Honest scope and limitations are documented below and in
> [PERFORMANCE_ASSESSMENT.md](PERFORMANCE_ASSESSMENT.md).

## What's included
- Rust/PyO3 optimization core (no GPU, no BLAS/LAPACK, no cloud)
- Trust Region CEM + MLP bootstrap ensemble surrogate
- Native Residual Micro-GP Phase 2 (pure Rust, no sklearn)
- Python ask/tell API + constraint handling (feasibility surrogate)
- Async parallel / rolling evaluation (SLURM-ready)
- Multi-objective: Chebyshev scalarization + closed-form 2-objective EHVI (Rust)
- Real CFD airfoil pipelines: NeuralFoil (H-1) and SU2 RANS (H-2)
- Optuna sampler integration, save/load
- 91 tests passing (63 Python + 28 Rust)

## Benchmark highlights

Compared against **BoTorch TuRBO, CMA-ES, Random, NSGA-II**. SAASBO is future work.

### Synthetic high-dimensional (minimize, median best)
| Condition | TRust-BO+P2 | BoTorch TuRBO | Speed |
|---|---|---|---|
| Ackley 50D, b=300 | **5.96** | 7.25 | **8.9×** |
| Ackley 100D, b=300 | **7.13** | 8.51 | **5.7×** |
| Ackley 50D, b=500 | **4.64** | 6.38 | — |

Wins 16/18 mid-budget conditions and 5/5 at budget=500, 5–10× faster.

### Real CFD (maximize Cl/Cd)
- **NeuralFoil (H-1, 16D, 10 seeds):** BoTorch median 241 vs TRust 228 (BoTorch leads at low-dim smooth; TRust reaches best single design 267).
- **SU2 RANS (H-2, 16D, 3 seeds):** TRust median **171.6** vs BoTorch 126.1 (+36%), only method with all seeds physical.

### Multi-objective (SU2, 2 seeds)
EHVI hypervolume 0.0239 vs Chebyshev 0.0165 (+45%); Chebyshev gives a more diverse Pareto front.

## Known limitations
- Low-dim / smooth / small-budget: GP-based methods (BoTorch, HEBO) typically do better.
- SAASBO not yet benchmarked.
- CFD seeds are few (SU2 3, multi-objective 2) — statistically thin.
- SU2 feasibility check is simplified (no minimum-thickness constraint); ultra-thin shapes
  can produce non-physical Cl/Cd. NeuralFoil pipeline is unaffected.
- EHVI is 2-objective; use Chebyshev for 3+.

## Next milestones
SAASBO comparison, more benchmark seeds, geometric shape constraints for CFD, PyPI release.
