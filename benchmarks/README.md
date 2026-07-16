# Benchmarks

Scripts used to produce the numbers in the main [README](../README.md),
[docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md), and
[docs/BENCHMARK.md](../docs/BENCHMARK.md). All scripts write their CSV/JSONL
output to the current directory (gitignored) and most support `SMOKE=1` for a
short sanity run before a full A/B. Comparison baselines require extra
dependencies not used by TRust-BO itself (`botorch`, `hebo`, `scikit-learn`);
install them as needed per script.

## Synthetic benchmarks (no CFD dependency)

| Script | Compares |
|---|---|
| `benchmark.py` / `benchmark_v2.py` | TRust-BO vs BoTorch (TuRBO-1) vs HEBO vs Random |
| `benchmark_50d.py` | 50D Ackley: TRust-BO vs Random vs GP |
| `benchmark_multi_tr.py` / `_v2.py` | Multi Trust Region (TuRBO-M, experimental) |
| `midbudget_benchmark.py` | Standard synthetic A/B harness (Ackley/Rastrigin/Rosenbrock/Levy, 50/100D). Default entry point for a quick engine A/B; `SMOKE=1` for a ~1 min smoke run |
| `large_budget_benchmark.py` | TRust-BO's strong range: large budget × high dimension |
| `cfd_scale_benchmark.py` | Tiny-budget (CFD-scale) synthetic comparison |
| `zdt_test.py` / `zdt_ehvi_benchmark.py` | Multi-objective (ZDT test problems, EHVI vs Chebyshev) |
| `crossover_analysis.py` | Aggregates 3 CSVs into `crossover_summary.csv` (source data for `docs/BENCHMARK.md` §1-9) |
| `bench_resume.py` | Shared resume/checkpoint helper used by most long-running harnesses above |

## Real CFD — 2-D airfoil (NeuralFoil / SU2 RANS, `su2/` subpackage)

| Script | Purpose |
|---|---|
| `su2/airfoil_mesh.py` | CST (Kulfan) 2-D O-mesh generation + `validate_cst_geometry` fail-fast checks |
| `su2/su2_runner.py` | SU2 RANS 2-D execution wrapper (H-2 pipeline) |
| `su2/su2_evaluator.py` | `JobEvaluator` for `RollingTRustBOEngine` (async parallel SU2 evaluation) |
| `cfd_neuralfoil_benchmark.py` | NeuralFoil (H-1) surrogate airfoil optimization benchmark |
| `su2_cfd_benchmark.py` | SU2 RANS (H-2) real-CFD airfoil optimization benchmark |
| `su2_mo_benchmark.py` | Multi-objective SU2 (Cl↑/Cd↓): EHVI vs Chebyshev |
| `rolling_integration_test.py` | Async/rolling engine integration test against NeuralFoil |

## A/B harnesses behind config flags (see CLAUDE.md "棄却済み" / docs/BENCHMARK.md)

Each of these validates one config flag against the v0.3.0 baseline; most document a
**rejected** (default-off) result — see `docs/BENCHMARK.md` for the numeric verdict.

| Script | Flag under test | BENCHMARK.md |
|---|---|---|
| `cfd_ts_ab_benchmark.py` | `acquisition="ts"` vs `"ei"` on NeuralFoil | §15 |
| `su2_ts_ab_benchmark.py` | `acquisition="ts"` vs `"ei"` on real SU2 RANS | §15-16 |
| `mix_ab_benchmark.py` | `acquisition="ts_ei"` + `phase2_early_frac` | §16 |
| `lsprior_ab_benchmark.py` | `phase2_ls_prior` (Hvarfner-style GP length-scale prior) | §15.1 |
| `dualtr_ab_benchmark.py` | `n_trs=2` (dual Trust Region) | §17 |
| `mf_ab_benchmark.py` | `CascadeMFEngine` (LF→HF multi-fidelity cascade) | §18 |
| `mf_su2_correlation.py` / `mf_su2_coarse_correlation.py` | LF/HF correlation gate for NeuralFoil→SU2 / coarse-mesh→fine-mesh cascades | §19, §21 |
| `bilog_ab_benchmark.py` | `bilog_transform` (SCBO/HEBO-style output transform) | §20.2 |
| `cal_ab_benchmark.py` | `sigma_calibration` (post-hoc ensemble σ calibration) | §22 |
| `goal70_ab_benchmark.py` | Whether budget 75 can match base budget-250 quality (multi-fidelity) | §18 |
| `goal2_su2_ab_benchmark.py` | Candidates 6/1/9/11 (diverse starts / MADS poll / joint batch select / early-termination) on real SU2 RANS | §23 |

## Real CFD — 3-D front-wing (FSAE validation, 65-D)

| Script | Purpose |
|---|---|
| `su2/wing3d_mesh.py` | Gmsh-free 3-D structured mesh (spanwise-extruded 2-D O-mesh, half-model + endplate) |
| `su2/wing3d_runner.py` | SU2 `INC_RANS` 3-D execution wrapper (moving-ground, symmetry plane, windowed-average convergence) |
| `wing3d_benchmark.py` | Resumable 65-D `TRustBOEngine` run harness for the front-wing pipeline (§24) |
