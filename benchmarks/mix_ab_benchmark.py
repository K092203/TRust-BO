"""A/B benchmark for acquisition "ts_ei" and phase2_early_frac.

Four arms sharing enable_phase2=True:
  ts       -- current default acquisition
  ei       -- expected-improvement acquisition
  ts_ei    -- mixed batch: first ceil(b/2) by EI, rest by TS (new in this branch)
  ts_early -- "ts" plus phase2_early_frac=0.25 (early Phase 2 entry)

Problems, dimensions, noise, seeds, and budget mirror lsprior_ab_benchmark.py.
Each completed run is appended to a CSV, so rerunning resumes from the
remaining combinations. Set ``SMOKE=1`` for the small configuration.
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
    "ts": {"acquisition": "ts"},
    "ei": {"acquisition": "ei"},
    "ts_ei": {"acquisition": "ts_ei"},
    "ts_early": {"acquisition": "ts", "phase2_early_frac": 0.25},
}
OPTIMUM = {name: 0.0 for name in PROBLEMS}

DIMS = [50, 100]
NOISES = [0.0, 0.05]
SEEDS = range(8)
BUDGET = 250
N_INIT = 10  # lsprior_ab_benchmark.py と同じ
BATCH_SIZE = 4
CSV_PATH = Path(os.environ.get("CSV", "mix_ab_results.csv"))

if os.environ.get("SMOKE"):
    DIMS = [50]
    NOISES = [0.0, 0.05]
    SEEDS = range(2)
    BUDGET = 60
    PROBLEMS = {name: PROBLEMS[name] for name in ("ackley", "rastrigin")}
    CSV_PATH = Path(os.environ.get("CSV", "mix_ab_results_smoke.csv"))


def csv_write_header() -> None:
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["arm", "problem", "dim", "noise", "seed", "best_true_value",
             "total_seconds", "final_phase"]
        )


def csv_append(arm, problem, dim, noise, seed, best_true, elapsed, final_phase) -> None:
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [arm, problem, dim, noise, seed, f"{best_true:.12g}",
             f"{elapsed:.2f}", final_phase]
        )


def run_trust_bo(fn, lb, ub, dim, noise, seed, arm_config) -> tuple[float, float, str]:
    from trust_bo import Float, TRustBOEngine

    space = [Float(f"x{i}", lb, ub) for i in range(dim)]
    engine = TRustBOEngine(
        space=space,
        direction="minimize",
        seed=seed,
        config={"n_init": N_INIT, "enable_phase2": True, **arm_config},
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
            observed = true_value * (1.0 + noise * noise_rng.normal())
            results.append({"value": observed, "feasible": True})
        engine.tell(cands, results)
        evaluated += len(cands)

    return best_true, time.perf_counter() - t0, engine._phase


def run_all() -> None:
    from bench_resume import is_done, resume_or_init

    done_keys = resume_or_init(
        CSV_PATH, ("arm", "problem", "dim", "noise", "seed"), csv_write_header
    )
    total = len(ARMS) * len(PROBLEMS) * len(DIMS) * len(NOISES) * len(SEEDS)
    count = 0
    for arm, arm_config in ARMS.items():
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
                            best, elapsed, phase = run_trust_bo(
                                fn, lb, ub, dim, noise, seed, arm_config
                            )
                            csv_append(arm, problem, dim, noise, seed, best,
                                       elapsed, phase)
                            print(f"best_true={best:.6g}  phase={phase}  ({elapsed:.1f}s)")
                        except Exception:
                            tb = traceback.format_exc().replace("\n", " | ")
                            print(f"ERROR: {tb[:160]}", flush=True)


def print_summary() -> None:
    """アームごとのリグレット幾何平均比 (基準 "ts"、>1 で当該アーム優位) を表示。"""
    values: dict[tuple, dict[str, float]] = defaultdict(dict)
    phases = defaultdict(lambda: defaultdict(int))
    if not CSV_PATH.exists():
        return
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["problem"], int(row["dim"]), float(row["noise"]),
                       int(row["seed"]))
                values[key][row["arm"]] = float(row["best_true_value"])
                phases[row["arm"]][row.get("final_phase", "")] += 1
            except (KeyError, ValueError):
                continue

    other_arms = [a for a in ARMS if a != "ts"]
    print("\n" + "=" * 78)
    print('  Final-regret geometric-mean ratio vs baseline "ts" (>1 favors that arm)')
    print("=" * 78)
    header = f"{'Problem':<12} {'dim':>4} {'noise':>7}"
    for a in other_arms:
        header += f" {a:>10}"
    print(header)
    print("-" * 78)
    overall: dict[str, list[float]] = defaultdict(list)
    for problem in PROBLEMS:
        for dim in DIMS:
            for noise in NOISES:
                cells = []
                for arm in other_arms:
                    ratios = []
                    for (p, d, n, _s), arms in values.items():
                        if p == problem and d == dim and n == noise \
                                and "ts" in arms and arm in arms:
                            r_ts = max(arms["ts"] - OPTIMUM[p], 1e-12)
                            r_a = max(arms[arm] - OPTIMUM[p], 1e-12)
                            ratios.append(r_ts / r_a)
                    if ratios:
                        gm = math.exp(float(np.mean(np.log(ratios))))
                        overall[arm].extend(ratios)
                        cells.append(f"{gm:>10.4g}")
                    else:
                        cells.append(f"{'—':>10}")
                print(f"{problem:<12} {dim:>4} {noise:>6.0%} " + " ".join(cells))
    print("-" * 78)
    cells = []
    for arm in other_arms:
        if overall[arm]:
            gm = math.exp(float(np.mean(np.log(overall[arm]))))
            wins = sum(1 for r in overall[arm] if r > 1.0)
            cells.append(f"{arm}: GM={gm:.4f} wins={wins}/{len(overall[arm])}")
    print("  OVERALL  " + "   ".join(cells))
    print("  final_phase counts: " + str({a: dict(c) for a, c in phases.items()}))
    print("=" * 78)


if __name__ == "__main__":
    print("=" * 78)
    print("  Acquisition-mix / Phase2-early A/B benchmark")
    print(f"  arms={list(ARMS)}  problems={list(PROBLEMS)}  dims={DIMS}")
    print(f"  noise={NOISES}  budget={BUDGET}  batch={BATCH_SIZE}  seeds={list(SEEDS)}")
    print("=" * 78)
    run_all()
    print_summary()
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
