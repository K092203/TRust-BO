"""A/B benchmark for the ensemble sigma calibration (config: sigma_calibration).

Arms (acquisition="ts" -- sigma feeds mu+z*sigma directly):
  base  -- sigma_calibration=False (default)
  cal   -- sigma_calibration=True  (temperature-scaled sigma, shrink forbidden)

Problems/dims/noise/seeds/budget mirror mix_ab_benchmark.py.
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

from midbudget_benchmark import ackley, levy, rastrigin
from lsprior_ab_benchmark import rosenbrock

PROBLEMS = {
    "ackley": (-5.0, 5.0, ackley),
    "rastrigin": (-5.12, 5.12, rastrigin),
    "rosenbrock": (-5.0, 10.0, rosenbrock),
    "levy": (-10.0, 10.0, levy),
}
ARMS: dict[str, dict] = {
    "base": {"acquisition": "ts"},
    "cal": {"acquisition": "ts", "sigma_calibration": True},
}
OPTIMUM = {name: 0.0 for name in PROBLEMS}

DIMS = [50, 100]
NOISES = [0.0, 0.05]
SEEDS = range(8)
BUDGET = 250
N_INIT = 10
BATCH_SIZE = 4
CSV_PATH = Path(os.environ.get("CSV", "cal_ab_results.csv"))

if os.environ.get("SMOKE"):
    DIMS = [50]
    SEEDS = range(2)
    BUDGET = 60
    PROBLEMS = {name: PROBLEMS[name] for name in ("ackley", "rosenbrock")}
    CSV_PATH = Path(os.environ.get("CSV", "cal_ab_results_smoke.csv"))


def csv_write_header() -> None:
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["arm", "problem", "dim", "noise", "seed", "best_true_value", "total_seconds"])


def run_one(fn, lb, ub, dim, noise, seed, arm_config) -> tuple[float, float]:
    from trust_bo import Float, TRustBOEngine

    space = [Float(f"x{i}", lb, ub) for i in range(dim)]
    engine = TRustBOEngine(
        space=space, direction="minimize", seed=seed,
        config={"n_init": N_INIT, **arm_config},
    )
    noise_rng = np.random.default_rng(seed)
    best_true = float("inf")
    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < BUDGET:
        b = min(BATCH_SIZE, BUDGET - evaluated)
        cands = engine.ask(batch_size=b)
        results = []
        for cand in cands:
            x = np.array([cand[f"x{i}"] for i in range(dim)])
            tv = float(fn(x))
            best_true = min(best_true, tv)
            results.append({"value": tv * (1.0 + noise * noise_rng.normal()),
                            "feasible": True})
        engine.tell(cands, results)
        evaluated += len(cands)
    return best_true, time.perf_counter() - t0


def run_all() -> None:
    from bench_resume import is_done, resume_or_init

    done = resume_or_init(CSV_PATH, ("arm", "problem", "dim", "noise", "seed"),
                          csv_write_header)
    total = len(ARMS) * len(PROBLEMS) * len(DIMS) * len(NOISES) * len(SEEDS)
    count = 0
    for arm, arm_config in ARMS.items():
        for problem, (lb, ub, fn) in PROBLEMS.items():
            for dim in DIMS:
                for noise in NOISES:
                    for seed in SEEDS:
                        count += 1
                        tag = (f"[{count}/{total}] {arm:<5s} | {problem:<10s} | "
                               f"{dim:3d}D | noise={noise:.0%} | seed={seed}")
                        if is_done(done, arm, problem, dim, noise, seed):
                            print(f"{tag} ... [skip]")
                            continue
                        print(f"{tag} ... ", end="", flush=True)
                        try:
                            best, elapsed = run_one(fn, lb, ub, dim, noise, seed, arm_config)
                            with open(CSV_PATH, "a", newline="") as f:
                                csv.writer(f).writerow(
                                    [arm, problem, dim, noise, seed,
                                     f"{best:.12g}", f"{elapsed:.2f}"])
                            print(f"best_true={best:.6g}  ({elapsed:.1f}s)")
                        except Exception:
                            tb = traceback.format_exc().replace("\n", " | ")
                            print(f"ERROR: {tb[:160]}", flush=True)


def print_summary() -> None:
    values: dict[tuple, dict[str, float]] = defaultdict(dict)
    if not CSV_PATH.exists():
        return
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["problem"], int(row["dim"]), float(row["noise"]),
                       int(row["seed"]))
                values[key][row["arm"]] = float(row["best_true_value"])
            except (KeyError, ValueError):
                continue

    grouped: dict[str, list[float]] = defaultdict(list)
    for (problem, _d, _n, _s), arms in values.items():
        if "base" in arms and "cal" in arms:
            rb = max(arms["base"] - OPTIMUM[problem], 1e-12)
            rc = max(arms["cal"] - OPTIMUM[problem], 1e-12)
            grouped[problem].append(rb / rc)

    print("\n" + "=" * 64)
    print("  Final-regret GM ratio base/cal (>1 favors calibration)")
    print("=" * 64)
    allr = []
    for problem in PROBLEMS:
        rs = grouped.get(problem, [])
        if rs:
            gm = math.exp(float(np.mean(np.log(rs))))
            allr.extend(rs)
            print(f"{problem:<12}: GM={gm:.4f} (n={len(rs)})")
    if allr:
        gm = math.exp(float(np.mean(np.log(allr))))
        wins = sum(1 for r in allr if r > 1.0)
        print("-" * 64)
        print(f"  OVERALL GM={gm:.4f}  cal wins {wins}/{len(allr)}")
    print("=" * 64)


if __name__ == "__main__":
    print(f"sigma-calibration A/B: arms={list(ARMS)} budget={BUDGET} dims={DIMS} seeds={list(SEEDS)}")
    run_all()
    print_summary()
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
