# TRust-BO

**[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md)**

**Bayesian optimization that runs on the hardware you already have.**

> **TRust-BO is not a CFD solver.** It sits in the optimization loop — it does not
> replace the CFD solver; it reduces the number of CFD runs needed during design search.

[![CI](https://github.com/K092203/TRust-BO/actions/workflows/ci.yml/badge.svg)](https://github.com/K092203/TRust-BO/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)]()
[![Rust](https://img.shields.io/badge/built%20with-Rust-orange)]()

---

## Why this exists

The best design should not require the best hardware.

CFD optimization has long been gated by HPC clusters, workstations, and institutional budgets — tools that are excellent, but out of reach for most people.
TRust-BO was built by a high school student to solve a concrete problem: optimizing aerodynamics for a Formula Student car, on a laptop, without a GPU.
The goal is to make Bayesian optimization fast enough to run anywhere, simple enough to use without a PhD, and accurate enough to matter in real engineering work.

TRust-BO is a **Trust Region Bayesian Optimization engine** written in Rust, exposed to Python via PyO3.
No GPU required. No cloud required. `pip install trust-bo` and run locally.

**What it does, precisely:** on high-dimensional (≥50D) *or* noisy / constrained problems
(such as real CFD), TRust-BO runs **5–10× faster than BoTorch TuRBO while reaching equal or
better quality**. On low-dimensional, smooth, small-budget problems a GP-based method
(BoTorch, HEBO) is usually a better choice — TRust-BO does not claim to be universally best.
See [docs/PERFORMANCE_ASSESSMENT.md](docs/PERFORMANCE_ASSESSMENT.md) for the honest, data-backed breakdown.

> **Benchmark scope (read before comparing):** results below are measured against
> **BoTorch TuRBO, CMA-ES, Random Search, and NSGA-II**. A comparison against **SAASBO**
> (a strong high-dimensional BO baseline) is **future work**. CFD benchmarks use **3 seeds**
> and multi-objective uses **2 seeds**; more seeds are planned for statistical rigor.

---

## How it compares

| Feature | **TRust-BO** | BoTorch | HEBO | Optuna |
|---|:---:|:---:|:---:|:---:|
| GPU required | ✗ | optional | ✗ | ✗ |
| CPU-optimized core | ✓ (Rust) | △ | △ | ✓ |
| High-dimensional (50D+) | ✓ | ✓ | △ | △ |
| Minimal ask/tell API | ✓ | △ | △ | ✓ |
| CFD workflow focus | ✓ | ✗ | ✗ | ✗ |

> BoTorch and Optuna can also be used for CFD-driven optimization. TRust-BO is
> specifically designed around a lightweight, CPU-first workflow for students and
> small engineering teams.

---

## Installation

```bash
pip install trust-bo
```

Prebuilt wheels (abi3, Python ≥3.9) are published for Linux, macOS, and Windows —
no Rust toolchain needed for a normal install.

To build from source instead (requires a [Rust toolchain](https://rustup.rs/)):

```bash
git clone https://github.com/K092203/TRust-BO
cd TRust-BO
python -m venv .venv && source .venv/bin/activate
pip install .
```

For development, use [maturin](https://github.com/PyO3/maturin) for fast rebuilds: `pip install maturin && maturin develop --release`.

---

## Usage

```python
from trust_bo import TRustBOEngine, Float

# 1. Define the search space
space = [Float(f"x{i}", -5.0, 5.0) for i in range(10)]

# 2. Create the engine
engine = TRustBOEngine(space=space, direction="minimize", seed=42)

# 3. Ask → evaluate → tell
for _ in range(20):                          # 20 rounds × batch_size=10
    candidates = engine.ask(batch_size=10)   # suggest next points
    results = [
        {"value": your_cfd_solver(c), "feasible": True}
        for c in candidates
    ]
    engine.tell(candidates, results)         # feed results back

# 4. Get the best result
print(engine.best())
# {'parameters': {'x0': 0.12, ...}, 'objective_values': [3.47]}
```

### With constraints

```python
engine.tell(candidates, [
    {"value": solver(c), "feasible": constraint_ok(c)}
    for c in candidates
])
```

### Multi-objective (Pareto)

Optimize several objectives at once via `MultiObjectiveEngine`. `method="ehvi"` uses a
closed-form 2-objective Expected Hypervolume Improvement implemented in Rust;
`method="chebyshev"` uses scalarization (any number of objectives).

```python
from trust_bo import MultiObjectiveEngine, Float

space = [Float(f"x{i}", 0.0, 1.0) for i in range(20)]
engine = MultiObjectiveEngine(
    space=space, directions=["maximize", "minimize"],  # e.g. Cl ↑, Cd ↓
    method="ehvi", seed=0,
)
for _ in range(15):
    cands = engine.ask(batch_size=4)
    results = [{"values": [cl(c), cd(c)], "feasible": True} for c in cands]
    engine.tell(cands, results)

print(engine.pareto_front())                 # non-dominated designs
print(engine.hypervolume(ref=[0.0, 0.05]))   # quality metric
```

### Save and resume

```python
engine.save("study.zip")
engine = TRustBOEngine.load("study.zip")
```

### Use as an Optuna sampler

Use TRust-BO as a drop-in [Optuna](https://optuna.org/) sampler. Requires `pip install optuna`.

```python
import optuna
from trust_bo.integrations.optuna import TrustBoOptunaSampler

study = optuna.create_study(direction="minimize", sampler=TrustBoOptunaSampler(seed=42))

def objective(trial):
    x = [trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(10)]
    return sum(v**2 for v in x)

study.optimize(objective, n_trials=100)
print(study.best_value)
```

---

## Benchmarks

![TRust-BO vs BoTorch TuRBO on Ackley 100D: better optimum, 5.7× faster, CPU-only](docs/assets/benchmark_ackley100d.png)

Full data and methodology: [docs/BENCHMARK.md](docs/BENCHMARK.md) and
[docs/PERFORMANCE_ASSESSMENT.md](docs/PERFORMANCE_ASSESSMENT.md).
Compared against **BoTorch TuRBO, CMA-ES, Random Search, NSGA-II**; SAASBO is future work.

### Synthetic high-dimensional (the strong case)

Ackley / Rastrigin / Levy at 50D and 100D, budget 100–500, `enable_phase2=True`.
Lower is better. Quality = median best over seeds; speed = wall-clock per run.

| Condition | TRust-BO | BoTorch TuRBO | Quality | Speed |
|---|:---:|:---:|:---:|:---:|
| Ackley 50D, b=300 | **5.96** | 7.25 | TRust | **8.9×** |
| Levy 50D, b=300 | **152.4** | 194.5 | TRust | **10.7×** |
| Ackley 100D, b=300 | **7.13** | 8.51 | TRust | **5.7×** |
| Levy 100D, b=300 | **284.1** | 723.2 | TRust | **5.7×** |
| Ackley 50D, b=500 | **4.64** | 6.38 | TRust | — |

TRust-BO wins **16 of 18** mid-budget conditions and **5 of 5** at budget=500, while running
**5–10× faster**. The two losses are Rastrigin at budget=100 (small-budget GP edge).

### Real CFD — airfoil shape optimization

**H-1 (NeuralFoil, 16D CST, maximize Cl/Cd, 10 seeds):** clean surrogate, well-posed.

| Method | median Cl/Cd | best |
|---|:---:|:---:|
| BoTorch TuRBO | **241.4** | 245.4 |
| **TRust-BO+P2** | 227.9 | **267.4** |
| CMA-ES | 223.3 | 265.5 |
| Random | 148.7 | 161.1 |

At 16D smooth, BoTorch leads on median; TRust-BO reaches the single best design.

> Measured with `acquisition="ei"` (the engine default at the time). The current default is
> `"ts"`, which underperforms `"ei"` on this same smooth CST 16D Cl/Cd objective in a separate
> A/B (see [docs/BENCHMARK.md §15](docs/BENCHMARK.md), different seeds/budget than the table
> above, so the exact 227.9 is not expected to reproduce under today's default) — pass
> `config={"acquisition": "ei"}` explicitly for CFD-shaped objectives.

**H-2 (SU2 RANS, real Navier-Stokes, 16D, 3 seeds):** noisy, mesh-constrained.

| Method | median Cl/Cd | seeds in physical range |
|---|:---:|:---:|
| **TRust-BO+P2** | **171.6** | **3 / 3** |
| BoTorch TuRBO | 126.1 | 3 / 3 |
| CMA-ES | 774.6 ⚠ | 1 / 3 |
| Random | 316.1 ⚠ | 1 / 3 |

On the harder, noisier RANS problem TRust-BO leads (+36%) and is the most stable.

> ⚠ **Note on the SU2 (H-2) example:** the ⚠ values above were measured before fail-fast
> geometric validation (minimum thickness/area, self-intersection) and a `min_cd` floor were
> added to the SU2 pipeline — they are historical, not reproducible against the current
> feasibility check (see [Known limitations](#known-limitations)).
> **For a clean, ready-to-use CFD example, prefer the NeuralFoil (H-1) pipeline.**

### Multi-objective (Cl ↑ and Cd ↓ simultaneously)

EHVI (closed-form 2-objective expected hypervolume improvement, in Rust) vs Chebyshev
scalarization on SU2 (budget=60, **2 seeds**): EHVI median hypervolume **0.0239 vs 0.0165
(+45%)**. Chebyshev produces a more diverse Pareto front. See [docs/BENCHMARK.md](docs/BENCHMARK.md).

### Native Phase 2

A pure-Rust Matern 5/2 micro-GP fits the *residuals* of the MLP ensemble near the best
point, refining the endgame. One flag, no sklearn, no extra deps:

```python
engine = TRustBOEngine(space=space, config={"enable_phase2": True})
```

> **When to use TRust-BO:** high-dimensional (50D+) *or* noisy/constrained problems,
> moderate-to-large budget (100–1000), and anywhere wall-clock per BO round matters —
> such as CFD where each evaluation already costs minutes to hours.

---

## How it works

TRust-BO implements **TuRBO** (Trust Region Bayesian Optimization) with a custom surrogate:

```
Cold start  →  Halton quasi-random sampling (n_init points)
Warm path   →  MLP Bootstrap Ensemble surrogate (5 members)
             + Cross-Entropy Method (CEM) within Trust Region
             + Trust Region dynamics (expand on success, shrink on failure)
```

Key design choices:
- **Rust core** — the inner loop (surrogate training, CEM, TR management) runs in compiled Rust via PyO3, keeping CPU usage low
- **Warm start** — surrogate weights are serialized between rounds, cutting training time by ~41%
- **Neutral center stability** — TR center only moves on ≥1% relative improvement, preventing instability from minor fluctuations
- **No GPU** — uses `burn` with the `ndarray` backend; a modern laptop CPU is sufficient

---

## Roadmap

### Current (v0.3.0)
- [x] Single Trust Region (exploitation-focused) + TuRBO-M multi-TR (experimental)
- [x] MLP Bootstrap Ensemble surrogate with warm start
- [x] Constraint handling (feasibility surrogate) + fail-fast geometric shape constraints for CFD (minimum thickness/area, pre-mesh)
- [x] 128-test suite (84 Python + 44 Rust), CPU-only, PyO3 Python bindings
- [x] Benchmark vs BoTorch TuRBO / CMA-ES / HEBO / Random / NSGA-II
- [x] Native Phase 2 (Rust Tandem Residual-GP): +32% at 50D, +55% at 10D, zero extra deps
- [x] Async parallel / rolling evaluation (SLURM-ready) for expensive solvers
- [x] Multi-fidelity cascade (LF→HF, 70% fewer high-fidelity evaluations on NeuralFoil)
- [x] Real CFD airfoil optimization — NeuralFoil (H-1) **and** SU2 RANS (H-2) pipelines
- [x] Multi-objective: Chebyshev scalarization + closed-form 2-objective EHVI (Rust)
- [x] Optuna sampler integration
- [x] PyPI release (prebuilt abi3 wheels for Linux/macOS/Windows)

### Planned
- [ ] SAASBO comparison + more benchmark seeds (statistical rigor)
- [ ] Closed-form EHVI beyond 2 objectives (Chebyshev scalarization already supports any objective count today)
- [ ] Multi-TR (TuRBO-M) — deprioritized; single TR is more stable at CFD-scale budgets
- [ ] Research write-up on lightweight BO for engineering design

---

## FAQ

### Is TRust-BO a CFD solver?
No. TRust-BO is an optimizer that works *with* external CFD solvers. It does not
solve Navier-Stokes equations. It decides which design candidates to evaluate next,
reducing the total number of expensive CFD runs.

### Why not just use BoTorch?
BoTorch is excellent and much more flexible. TRust-BO is not a replacement — it is a
smaller CPU-first engine with a simple ask/tell API, designed around lightweight
CFD-driven workflows for students and small teams.

### Has it been validated on real CFD?
Yes, on airfoil shape optimization. Two pipelines are included: **NeuralFoil** (H-1, a fast
learned aerodynamics surrogate) and **SU2 RANS** (H-2, real steady Navier-Stokes, Ma=0.3,
Re=3×10⁶, SA turbulence). See the Benchmarks section and [docs/BENCHMARK.md](docs/BENCHMARK.md).
Validation so far uses 3 seeds for SU2; more seeds and a SAASBO comparison are planned.

### Who is this for?
Students, Formula Student teams, and small engineering teams interested in CFD-driven
design optimization without HPC or GPUs.

### What is the relationship to BoTorch / HEBO / Optuna?
These are all excellent optimizers. TRust-BO does not aim to replace them. It focuses
on a specific gap: a CPU-only, lightweight optimizer packaged around CFD-driven
engineering design workflows.

---

## Known limitations

- **Surrogate accuracy vs GP:** The MLP bootstrap ensemble trades uncertainty calibration for speed. On low-dimensional, smooth problems with small budgets, GP-based methods (BoTorch, HEBO) typically do better (confirmed on the 16D NeuralFoil benchmark). TRust-BO's advantage is at 50D+ or on noisy/constrained problems.
- **Benchmark seeds:** synthetic results use 10 seeds, but real CFD (SU2) uses 3 seeds and multi-objective uses 2 seeds — statistically thin. More seeds are planned.
- **No SAASBO comparison yet:** the strong high-dimensional BO baseline SAASBO has not been benchmarked against (environment constraints). Claims are limited to vs BoTorch TuRBO / CMA-ES / Random / NSGA-II.
- **CFD feasibility has known gaps:** the SU2 (H-2) pipeline now fail-fasts on invalid geometry (minimum thickness/area, self-intersection) before meshing and rejects non-physical `Cd` below a floor threshold, closing the main artifact class seen earlier (see [docs/BENCHMARK.md](docs/BENCHMARK.md) §20.3). Some residual risk remains on the coarser 3-D front-wing mesh used for a from-scratch
FSAE front-wing validation pipeline (65-D, SU2 `INC_RANS`) — see
[docs/BENCHMARK.md §24](docs/BENCHMARK.md#24-2026-07-16-3dフロントウィング最適化パイプライン実用検証-成功初の3d-cfd実行).
- **Multi-objective is 2-objective for EHVI:** the closed-form EHVI is 2-objective; use Chebyshev scalarization for 3+ objectives.
- **Warm-start weight transfer:** surrogate weights are serialized as hex strings (~1 MB/round). Functional but inefficient; a binary transfer mechanism is planned.
- **Multi-TR (`n_trs > 1`) is experimental:** implemented and tested, but deprioritized for CFD-scale budgets where single-TR is more stable.

---

## Development log

The full development history — design decisions, phase-by-phase experiments, and benchmark notes — is kept in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) (Japanese).

---

## Contributing

Contributions are welcome. This is a one-person project so far, and any help — bug reports, benchmark results on real problems, documentation, or code — is genuinely appreciated.

If you use TRust-BO for a CFD problem and get results (good or bad), please open an issue and share them. Real-world feedback is the most valuable thing at this stage.

---

<!--
GitHub Topics to set manually in repository settings:
bayesian-optimization
cfd
aerodynamics
engineering-design
rust
python
pyo3
trust-region
optimization
formula-student
black-box-optimization
-->

## License

MIT © 2026 Kotaro Ozawa
