"""A/B benchmark for the Phase 2 length-scale prior.

Compares the default Phase 2 micro-GP with ``phase2_ls_prior=False`` against
the dimension-aware length-scale prior enabled by ``phase2_ls_prior=True``.
Each completed run is appended to a CSV, so rerunning this script resumes from
the remaining arm/problem/dimension/noise/seed combinations.

Set ``SMOKE=1`` for the small 50D Ackley/Rastrigin smoke configuration.
"""

from __future__ import annotations

import csv
import math
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse the canonical Ackley/Rastrigin/Levy implementations and their domains
# from the mid-budget benchmark. Rosenbrock is added here for this A/B suite.
from midbudget_benchmark import ackley, levy, rastrigin


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


PROBLEMS = {
    "ackley": (-5.0, 5.0, ackley),
    "rastrigin": (-5.12, 5.12, rastrigin),
    "rosenbrock": (-5.0, 10.0, rosenbrock),
    "levy": (-10.0, 10.0, levy),
}
ARMS = {"baseline": False, "lsprior": True}
OPTIMUM = {name: 0.0 for name in PROBLEMS}

DIMS = [50, 100]
NOISES = [0.0, 0.05]
SEEDS = range(8)
BUDGET = 250
N_INIT = 10  # Keep this aligned with midbudget_benchmark.py.
BATCH_SIZE = 4
CSV_PATH = Path(os.environ.get("CSV", "lsprior_ab_results.csv"))

if os.environ.get("SMOKE"):
    DIMS = [50]
    NOISES = [0.0, 0.05]
    SEEDS = range(2)
    BUDGET = 60
    PROBLEMS = {name: PROBLEMS[name] for name in ("ackley", "rastrigin")}
    CSV_PATH = Path(os.environ.get("CSV", "lsprior_ab_results_smoke.csv"))


def csv_write_header() -> None:
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["arm", "problem", "dim", "noise", "seed", "best_true_value", "total_seconds", "final_phase"]
        )


def csv_append(
    arm: str, problem: str, dim: int, noise: float, seed: int, best_true: float, elapsed: float,
    final_phase: str,
) -> None:
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [arm, problem, dim, noise, seed, f"{best_true:.12g}", f"{elapsed:.2f}", final_phase]
        )


def run_trust_bo(
    fn, lb: float, ub: float, dim: int, noise: float, seed: int, ls_prior: bool
) -> tuple[float, float]:
    """Optimize noisy observations while retaining the best noise-free value."""
    from trust_bo import Float, TRustBOEngine

    space = [Float(f"x{i}", lb, ub) for i in range(dim)]
    engine = TRustBOEngine(
        space=space,
        direction="minimize",
        seed=seed,
        config={
            "n_init": N_INIT,
            "enable_phase2": True,
            "phase2_ls_prior": ls_prior,
        },
    )
    noise_rng = np.random.default_rng(seed)
    best_true = float("inf")
    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < BUDGET:
        batch = min(BATCH_SIZE, BUDGET - evaluated)
        cands = engine.ask(batch_size=batch)
        results = []
        for cand in cands:
            x = np.array([cand[f"x{i}"] for i in range(dim)])
            true_value = float(fn(x))
            best_true = min(best_true, true_value)
            observed_value = true_value * (1.0 + noise * noise_rng.normal())
            results.append({"value": observed_value, "feasible": True})
        engine.tell(cands, results)
        evaluated += len(cands)

    return best_true, time.perf_counter() - t0, engine._phase


def run_all() -> None:
    from bench_resume import is_done, resume_or_init

    done_keys = resume_or_init(
        CSV_PATH,
        ("arm", "problem", "dim", "noise", "seed"),
        csv_write_header,
    )
    total = len(ARMS) * len(PROBLEMS) * len(DIMS) * len(NOISES) * len(SEEDS)
    count = 0
    for arm, ls_prior in ARMS.items():
        for problem, (lb, ub, fn) in PROBLEMS.items():
            for dim in DIMS:
                for noise in NOISES:
                    for seed in SEEDS:
                        count += 1
                        tag = (
                            f"[{count}/{total}] {arm:<8s} | {problem:<10s} | "
                            f"{dim:3d}D | noise={noise:.0%} | seed={seed}"
                        )
                        if is_done(done_keys, arm, problem, dim, noise, seed):
                            print(f"{tag} ... [skip]")
                            continue
                        print(f"{tag} ... ", end="", flush=True)
                        try:
                            best_true, elapsed, final_phase = run_trust_bo(
                                fn, lb, ub, dim, noise, seed, ls_prior
                            )
                            csv_append(arm, problem, dim, noise, seed, best_true, elapsed,
                                       final_phase)
                            print(f"best_true={best_true:.6g}  phase={final_phase}  ({elapsed:.1f}s)")
                        except Exception:
                            tb = traceback.format_exc().replace("\n", " | ")
                            print(f"ERROR: {tb[:160]}", flush=True)


def print_summary() -> None:
    """Print baseline/lsprior geometric-mean final-regret ratios by condition."""
    values: dict[tuple[str, int, float, int], dict[str, float]] = defaultdict(dict)
    if not CSV_PATH.exists():
        return
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["problem"], int(row["dim"]), float(row["noise"]), int(row["seed"]))
                values[key][row["arm"]] = float(row["best_true_value"])
            except (KeyError, ValueError):
                continue

    grouped: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for (problem, dim, noise, _seed), arms in values.items():
        if "baseline" in arms and "lsprior" in arms:
            baseline_regret = max(arms["baseline"] - OPTIMUM[problem], 1e-12)
            lsprior_regret = max(arms["lsprior"] - OPTIMUM[problem], 1e-12)
            grouped[(problem, dim, noise)].append(baseline_regret / lsprior_regret)

    print("\n" + "=" * 72)
    print("  Final-regret geometric-mean ratio: baseline / lsprior (>1 favors lsprior)")
    print("=" * 72)
    print(f"{'Problem':<12} {'dim':>4} {'noise':>7} {'GM ratio':>12} {'paired seeds':>13}")
    print("-" * 72)
    for problem in PROBLEMS:
        for dim in DIMS:
            for noise in NOISES:
                ratios = grouped.get((problem, dim, noise), [])
                if ratios:
                    geo_mean = math.exp(float(np.mean(np.log(ratios))))
                    print(
                        f"{problem:<12} {dim:>4} {noise:>6.0%} "
                        f"{geo_mean:>12.4g} {len(ratios):>13}"
                    )
    print("=" * 72)


if __name__ == "__main__":
    print("=" * 72)
    print("  Phase 2 length-scale-prior A/B benchmark")
    print(f"  problems={', '.join(PROBLEMS)}  dims={DIMS}  noise={NOISES}")
    print(f"  budget={BUDGET}  n_init={N_INIT}  batch={BATCH_SIZE}  seeds={list(SEEDS)}")
    print("=" * 72)
    run_all()
    print_summary()
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
