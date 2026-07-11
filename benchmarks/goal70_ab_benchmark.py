"""Goal-70 A/B benchmark: can budget 75 match the base budget-250 quality?

The four arms differ in acquisition/configuration and budget.  Completed runs
are appended to a CSV and are skipped on later invocations; set ``SMOKE=1``
for the 50D Ackley/Rastrigin wiring check.
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
# (TRust-BO config, evaluation budget).  base250 is the quality reference.
ARMS: dict[str, tuple[dict, int]] = {
    "base250": (
        {"acquisition": "ei", "enable_phase2": True, "phase2_early_frac": 0.25},
        250,
    ),
    "ts75": ({"acquisition": "ts", "enable_phase2": True}, 75),
    "ei75": (
        {"acquisition": "ei", "enable_phase2": True, "phase2_early_frac": 0.25},
        75,
    ),
    "raasp75": (
        {
            "acquisition": "ei",
            "enable_phase2": True,
            "phase2_early_frac": 0.25,
            "cem_dim_mask": True,
        },
        75,
    ),
}
OPTIMUM = {name: 0.0 for name in PROBLEMS}

DIMS = [50, 100]
NOISES = [0.0, 0.05]
SEEDS = range(8)
N_INIT = 10
BATCH_SIZE = 4
CSV_PATH = Path(os.environ.get("CSV", "goal70_ab_results.csv"))

if os.environ.get("SMOKE"):
    DIMS = [50]
    SEEDS = range(2)
    PROBLEMS = {name: PROBLEMS[name] for name in ("ackley", "rastrigin")}
    ARMS = {
        name: (config, 100 if name == "base250" else 30)
        for name, (config, _budget) in ARMS.items()
    }
    CSV_PATH = Path(os.environ.get("CSV", "goal70_ab_results_smoke.csv"))


def csv_write_header() -> None:
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "arm", "problem", "dim", "noise", "seed", "best_true_value",
                "total_seconds", "final_phase",
            ]
        )


def csv_append(arm, problem, dim, noise, seed, best_true, elapsed, final_phase) -> None:
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                arm, problem, dim, noise, seed, f"{best_true:.12g}",
                f"{elapsed:.2f}", final_phase,
            ]
        )


def run_trust_bo(fn, lb, ub, dim, noise, seed, arm_config, budget) -> tuple[float, float, str]:
    from trust_bo import Float, TRustBOEngine

    space = [Float(f"x{i}", lb, ub) for i in range(dim)]
    engine = TRustBOEngine(
        space=space,
        direction="minimize",
        seed=seed,
        config={"n_init": N_INIT, **arm_config},
    )
    noise_rng = np.random.default_rng(seed)
    best_true = float("inf")
    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)
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
    for arm, (arm_config, budget) in ARMS.items():
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
                                fn, lb, ub, dim, noise, seed, arm_config, budget
                            )
                            csv_append(arm, problem, dim, noise, seed, best, elapsed, phase)
                            print(f"best_true={best:.6g}  phase={phase}  ({elapsed:.1f}s)")
                        except Exception:
                            tb = traceback.format_exc().replace("\n", " | ")
                            print(f"ERROR: {tb[:160]}", flush=True)


def _geometric_mean(ratios: list[float]) -> float:
    return math.exp(float(np.mean(np.log(ratios))))


def print_summary() -> None:
    """Print base250/arm final-regret ratios; ratios above one meet Goal-70."""
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

    target_arms = ("ts75", "ei75", "raasp75")
    by_problem: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[str, list[float]] = defaultdict(list)
    for (problem, _dim, _noise, _seed), arms in values.items():
        if "base250" not in arms:
            continue
        base_regret = max(arms["base250"] - OPTIMUM[problem], 1e-12)
        for arm in target_arms:
            if arm in arms:
                ratio = base_regret / max(arms[arm] - OPTIMUM[problem], 1e-12)
                by_problem[problem][arm].append(ratio)
                overall[arm].append(ratio)

    print("\n" + "=" * 90)
    print("  Goal-70: final-regret GM ratio base250 / arm (>1 means arm meets/exceeds base250)")
    print("=" * 90)
    print(f"{'Problem':<12}" + "".join(f" {arm:>23}" for arm in target_arms))
    print(f"{'':<12}" + "".join(" GM      achieved".rjust(24) for _ in target_arms))
    print("-" * 90)
    for problem in PROBLEMS:
        cells = []
        for arm in target_arms:
            ratios = by_problem[problem][arm]
            if ratios:
                achieved = sum(ratio > 1.0 for ratio in ratios)
                cells.append(f"{_geometric_mean(ratios):8.4f} {achieved:3d}/{len(ratios):<3d} ({achieved / len(ratios):5.1%})")
            else:
                cells.append("       —              —")
        print(f"{problem:<12}" + " ".join(cells))
    print("-" * 90)
    cells = []
    for arm in target_arms:
        ratios = overall[arm]
        if ratios:
            achieved = sum(ratio > 1.0 for ratio in ratios)
            cells.append(f"{_geometric_mean(ratios):8.4f} {achieved:3d}/{len(ratios):<3d} ({achieved / len(ratios):5.1%})")
        else:
            cells.append("       —              —")
    print(f"{'OVERALL':<12}" + " ".join(cells))
    print("=" * 90)


if __name__ == "__main__":
    print("=" * 78)
    print("  Goal-70 A/B benchmark: budget-75 quality vs base budget-250")
    print(f"  arms/budgets={[(name, budget) for name, (_, budget) in ARMS.items()]}")
    print(f"  problems={list(PROBLEMS)} dims={DIMS} noise={NOISES}")
    print(f"  n_init={N_INIT} batch={BATCH_SIZE} seeds={list(SEEDS)}")
    print("=" * 78)
    run_all()
    print_summary()
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
