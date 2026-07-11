"""A/B benchmark of Thompson sampling versus expected improvement on NeuralFoil.

The CST design space and the NeuralFoil evaluator are reused from
``cfd_neuralfoil_benchmark.py``. Results are appended after every completed
run and automatically resumed from the CSV on a later invocation.

Set ``SMOKE=1`` to run two seeds with a budget of 40.
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

# This script lives beside the source benchmark, so its directory is on
# sys.path when invoked as ``python benchmarks/cfd_ts_ab_benchmark.py``.
from cfd_neuralfoil_benchmark import (
    LOWER_LB,
    LOWER_UB,
    N_LOWER,
    N_UPPER,
    UPPER_LB,
    UPPER_UB,
    evaluate_cst,
)


ARMS = tuple(os.environ.get("ARMS", "ts,ei").split(","))
NOISES = [0.0, 0.05]
SEEDS = range(8)
BUDGET = 200
N_INIT = 10
BATCH_SIZE = 4
CSV_PATH = Path(os.environ.get("CSV", "cfd_ts_ab_results.csv"))

if os.environ.get("SMOKE"):
    SEEDS = range(2)
    BUDGET = 40
    CSV_PATH = Path(os.environ.get("CSV", "cfd_ts_ab_results_smoke.csv"))


def csv_write_header() -> None:
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["arm", "noise", "seed", "best_true_clcd", "total_seconds"])


def csv_append(arm: str, noise: float, seed: int, best_true: float, elapsed: float) -> None:
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([arm, noise, seed, f"{best_true:.12g}", f"{elapsed:.2f}"])


def run_trust_bo(arm: str, noise: float, seed: int) -> tuple[float, float]:
    """Run one acquisition arm, feeding noise to BO and retaining true best Cl/Cd."""
    from trust_bo import Float, TRustBOEngine

    space = [Float(f"u{i}", float(UPPER_LB[i]), float(UPPER_UB[i])) for i in range(N_UPPER)]
    space += [Float(f"l{i}", float(LOWER_LB[i]), float(LOWER_UB[i])) for i in range(N_LOWER)]
    engine = TRustBOEngine(
        space=space,
        direction="maximize",
        seed=seed,
        config={
            "n_init": N_INIT,
            "enable_phase2": True,
            "acquisition": arm,
        },
    )
    noise_rng = np.random.default_rng(seed)
    best_true = 0.0
    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < BUDGET:
        batch = min(BATCH_SIZE, BUDGET - evaluated)
        cands = engine.ask(batch_size=batch)
        results = []
        for cand in cands:
            params = np.array(
                [cand[f"u{i}"] for i in range(N_UPPER)]
                + [cand[f"l{i}"] for i in range(N_LOWER)]
            )
            true_value, feasible = evaluate_cst(params)
            if feasible:
                best_true = max(best_true, true_value)
            observed_value = true_value * (1.0 + noise * noise_rng.normal())
            results.append({"value": observed_value, "feasible": feasible})
        engine.tell(cands, results)
        evaluated += len(cands)

    return best_true, time.perf_counter() - t0


def run_all() -> None:
    from bench_resume import is_done, resume_or_init

    done_keys = resume_or_init(CSV_PATH, ("arm", "noise", "seed"), csv_write_header)
    total = len(ARMS) * len(NOISES) * len(SEEDS)
    count = 0
    for arm in ARMS:
        for noise in NOISES:
            for seed in SEEDS:
                count += 1
                tag = f"[{count}/{total}] {arm:<2s} | noise={noise:.0%} | seed={seed}"
                if is_done(done_keys, arm, noise, seed):
                    print(f"{tag} ... [skip]")
                    continue
                print(f"{tag} ... ", end="", flush=True)
                try:
                    best_true, elapsed = run_trust_bo(arm, noise, seed)
                    csv_append(arm, noise, seed, best_true, elapsed)
                    print(f"best_true_Cl/Cd={best_true:.4f}  ({elapsed:.1f}s)")
                except Exception:
                    tb = traceback.format_exc().replace("\n", " | ")
                    print(f"ERROR: {tb[:160]}", flush=True)


def print_summary() -> None:
    """Print per-arm summary statistics and paired TS/EI geometric-mean ratios."""
    values: dict[tuple[float, int], dict[str, float]] = defaultdict(dict)
    if not CSV_PATH.exists():
        return
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                values[(float(row["noise"]), int(row["seed"]))][row["arm"]] = float(
                    row["best_true_clcd"]
                )
            except (KeyError, ValueError):
                continue

    print("\n" + "=" * 72)
    print("  NeuralFoil acquisition A/B (true Cl/Cd; larger is better)")
    print("=" * 72)
    print(f"{'noise':>7} {'arm':<4} {'mean':>12} {'median':>12} {'N':>4}")
    print("-" * 72)
    for noise in NOISES:
        for arm in ARMS:
            arm_values = [arms[arm] for (row_noise, _), arms in values.items()
                          if row_noise == noise and arm in arms]
            if arm_values:
                print(
                    f"{noise:>6.0%} {arm:<4} {np.mean(arm_values):>12.4f} "
                    f"{np.median(arm_values):>12.4f} {len(arm_values):>4}"
                )

    print("\nPaired seed geometric-mean ratio: ts / ei (>1 favors ts)")
    print(f"{'noise':>7} {'GM ratio':>12} {'paired seeds':>13}")
    print("-" * 36)
    for noise in NOISES:
        ratios = [
            max(arms["ts"], 1e-12) / max(arms["ei"], 1e-12)
            for (row_noise, _), arms in values.items()
            if row_noise == noise and "ts" in arms and "ei" in arms
        ]
        if ratios:
            geo_mean = math.exp(float(np.mean(np.log(ratios))))
            print(f"{noise:>6.0%} {geo_mean:>12.4g} {len(ratios):>13}")
    print("=" * 72)


if __name__ == "__main__":
    print("=" * 72)
    print("  NeuralFoil acquisition A/B: Thompson sampling vs expected improvement")
    print(f"  budget={BUDGET}  n_init={N_INIT}  batch={BATCH_SIZE}  seeds={list(SEEDS)}")
    print(f"  noise={NOISES}")
    print("=" * 72)
    run_all()
    print_summary()
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
