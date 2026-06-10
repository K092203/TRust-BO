# TRust-BO

**Bayesian optimization that runs on the hardware you already have.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-83%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)]()
[![Rust](https://img.shields.io/badge/built%20with-Rust-orange)]()

---

## Why this exists

The best design should not require the best hardware.

CFD optimization has long been gated by HPC clusters, workstations, and institutional budgets — tools that are excellent, but out of reach for most people.
TRust-BO was built by a high school student to solve a concrete problem: optimizing aerodynamics for a Formula Student car, on a laptop, without a GPU.
The goal is to make Bayesian optimization fast enough to run anywhere, simple enough to use without a PhD, and accurate enough to matter in real engineering work.

TRust-BO is a **Trust Region Bayesian Optimization engine** written in Rust, exposed to Python via PyO3.
No GPU required. No cloud required. Just `pip install` and go.

---

## How it compares

| Feature | **TRust-BO** | BoTorch | HEBO | Optuna |
|---|:---:|:---:|:---:|:---:|
| GPU required | ✗ | optional | ✗ | ✗ |
| CPU-optimized core | ✓ (Rust) | ✗ (Python/PyTorch) | ✗ | ✗ |
| High-dimensional (50D+) | ✓ (Trust Region) | ✓ (TuRBO extension) | △ | △ |
| API complexity | minimal | high | moderate | minimal |
| Designed for CFD workflows | ✓ | ✗ | ✗ | ✗ |

*BoTorch is excellent and production-grade; TRust-BO trades flexibility for simplicity and zero GPU dependency.*

---

## Installation

```bash
pip install trust-bo
```

> **Note:** The package is currently in active development. For the latest version, build from source:
> ```bash
> git clone https://github.com/K092203/TRust-BO
> cd TRust-BO
> pip install maturin
> maturin develop --release
> ```

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

### Save and resume

```python
engine.save("study.zip")
engine = TRustBOEngine.load("study.zip")
```

---

## Benchmark

All results on Ackley minimization, 5 seeds (0–4), batch=4.
Lower is better. `—` = too slow to run (>3 min/trial).

### Setting B — CFD-scale budget (budget=50)

| Method | Ackley 10D | Ackley 50D | Time / run (50D) |
|---|:---:|:---:|:---:|
| HEBO | **4.71** | 9.35 | ~22 s |
| BoTorch TuRBO-1 | 5.89 | 8.83 | ~11 s |
| **TRust-BO** | 7.32 | **8.85** | **2.5 s** |
| Random Search | 7.85 | 9.57 | ~0 s |

At 50D with a small budget, TRust-BO matches BoTorch while running **4× faster**.
At 10D, small-budget GP-based methods have the edge — Trust Region dynamics need more rounds to warm up.

### Setting A — large budget (budget=500)

| Method | Ackley 50D | Time / run (50D) |
|---|:---:|:---:|
| **TRust-BO + native Phase 2** | **5.07** | **29 s** |
| BoTorch TuRBO-1 | 6.38 | ~254 s |
| TandemEngine v2 (sklearn, deprecated) | 7.33 | ~45 s |
| **TRust-BO** | 7.40 | 53 s |
| Random Search | 9.02 | ~0 s |
| HEBO | — | too slow |

**Native Phase 2** (Tandem Residual-GP): a pure-Rust Matern 5/2 micro-GP fits the
*residuals* of the MLP ensemble near the best point, refining the endgame.
Enable with one flag — no sklearn, no extra deps:

```python
engine = TRustBOEngine(space=space, config={"enable_phase2": True})
```

Improves on plain TRust-BO by **+32% at 50D** and **+55% at 10D** (3-seed median),
while also being the fastest method measured.

> **When to use TRust-BO:** high-dimensional problems (50D+), moderate-to-large budget (100–1000), and anywhere wall-clock time per BO round matters — such as CFD workflows where each evaluation already costs hours.

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

### Current (high school phase)
- [x] Single Trust Region (exploitation-focused)
- [x] MLP Bootstrap Ensemble surrogate with warm start
- [x] Constraint handling (feasibility surrogate)
- [x] 47-test suite, CPU-only, PyO3 Python bindings
- [x] Benchmark vs BoTorch / HEBO / Random (Setting A/B complete)
- [x] OSS release prep — MIT license, class name unified, known limitations documented
- [x] Native Phase 2 (Rust Tandem Residual-GP): +32% at 50D, +55% at 10D, zero extra deps
- [x] Mock CFD pipeline (NACA / F1 wing) — drop-in ready for real OpenFOAM
- [ ] Multi-TR (TuRBO-M) — deprioritized; single TR outperforms on CFD-scale budgets
- [ ] OpenFOAM integration + real airfoil optimization
- [ ] PyPI release

### After starting university
- [ ] Research paper on lightweight BO for engineering design
- [ ] End-to-end aerodynamics optimization pipeline (OpenFOAM / SU2)

---

## Known limitations

- **Surrogate accuracy vs GP:** The MLP bootstrap ensemble trades uncertainty calibration for speed. On low-dimensional problems with large budgets, GP-based methods (BoTorch, HEBO) will typically achieve better results. TRust-BO's advantage is speed and scalability to 50D+.
- **Warm-start weight transfer:** Surrogate weights are serialized as hex strings (~1 MB/round) between optimization rounds. This is functional but inefficient; a binary transfer mechanism is planned.
- **Single-objective only:** Multi-objective optimization (Pareto front) is not yet supported.
- **Multi-TR (`n_trs > 1`) is experimental:** Implemented and tested, but deprioritized for CFD-scale budgets where single-TR is more stable.

---

## Contributing

Contributions are welcome. This is a one-person project so far, and any help — bug reports, benchmark results on real problems, documentation, or code — is genuinely appreciated.

If you use TRust-BO for a CFD problem and get results (good or bad), please open an issue and share them. Real-world feedback is the most valuable thing at this stage.

---

## License

MIT © 2026 Kotaro Ozawa
