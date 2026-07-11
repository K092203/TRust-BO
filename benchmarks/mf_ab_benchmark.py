"""CST/NeuralFoil multi-fidelity cascade A/B benchmark.

Arms can be narrowed with ``ARMS=hf30,casc30``.  ``SMOKE=1`` runs seeds 0
and 1, halves the HF budgets, and reduces the cascade LF budget to 100.
The benchmark is resumable: each completed arm/seed is appended to its CSV.
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


N_UPPER = 8
N_LOWER = 8
ALPHA = 4.0
RE = 3e6
UPPER_LB = np.full(N_UPPER, 0.04)
UPPER_UB = np.full(N_UPPER, 0.45)
LOWER_LB = np.full(N_LOWER, -0.40)
LOWER_UB = np.full(N_LOWER, 0.15)

DEFAULT_ARMS = ("hf30", "hf50", "hf100", "casc30", "casc50")
HF_BUDGETS = {"hf30": 30, "hf50": 50, "hf100": 100, "casc30": 30, "casc50": 50}
ARMS = tuple(name.strip() for name in os.environ.get("ARMS", ",".join(DEFAULT_ARMS)).split(",")
             if name.strip())
UNKNOWN_ARMS = set(ARMS) - set(HF_BUDGETS)
if UNKNOWN_ARMS:
    raise ValueError(f"unknown ARMS: {', '.join(sorted(UNKNOWN_ARMS))}")

SEEDS = range(8)
LF_BUDGET = 300
BATCH_SIZE = 4
TOP_K = 8
CONFIG = {"acquisition": "ei", "enable_phase2": True}
CSV_PATH = Path(os.environ.get("CSV", "mf_ab_results.csv"))

if os.environ.get("SMOKE"):
    SEEDS = range(2)
    HF_BUDGETS = {arm: budget // 2 for arm, budget in HF_BUDGETS.items()}
    LF_BUDGET = 100
    CSV_PATH = Path(os.environ.get("CSV", "mf_ab_results_smoke.csv"))


def make_space():
    from trust_bo import Float

    return (
        [Float(f"u{i}", float(UPPER_LB[i]), float(UPPER_UB[i])) for i in range(N_UPPER)]
        + [Float(f"l{i}", float(LOWER_LB[i]), float(LOWER_UB[i])) for i in range(N_LOWER)]
    )


def evaluate_cst(candidate: dict, model_size: str) -> dict[str, float | bool]:
    """Evaluate a raw CST parameter dict with the requested NeuralFoil model."""
    import aerosandbox as asb
    import neuralfoil as nf

    params = np.array(
        [candidate[f"u{i}"] for i in range(N_UPPER)]
        + [candidate[f"l{i}"] for i in range(N_LOWER)],
        dtype=float,
    )
    try:
        airfoil = asb.KulfanAirfoil(
            upper_weights=params[:N_UPPER],
            lower_weights=params[N_UPPER:],
            leading_edge_weight=0.0,
            TE_thickness=0.0,
        )
        result = nf.get_aero_from_airfoil(
            airfoil=airfoil, alpha=ALPHA, Re=RE, model_size=model_size
        )
        cl = float(result["CL"].item())
        cd = float(result["CD"].item())
        confidence = float(result["analysis_confidence"].item())
        if confidence < 0.5 or cd <= 0.0 or cl <= 0.0:
            return {"value": 0.0, "feasible": False}
        return {"value": cl / cd, "feasible": True}
    except Exception:
        return {"value": 0.0, "feasible": False}


def run_hf(seed: int, budget: int) -> tuple[float, int, int, float]:
    from trust_bo import TRustBOEngine

    engine = TRustBOEngine(
        space=make_space(), direction="maximize", seed=seed, config=CONFIG
    )
    started = time.perf_counter()
    evaluated = 0
    while evaluated < budget:
        candidates = engine.ask(batch_size=min(BATCH_SIZE, budget - evaluated))
        if not candidates:
            raise RuntimeError("TRustBOEngine.ask() returned no candidates")
        engine.tell(candidates, [evaluate_cst(c, "xxxlarge") for c in candidates])
        evaluated += len(candidates)
    best = engine.best()
    return (best["objective_values"][0] if best else 0.0, evaluated, 0,
            time.perf_counter() - started)


def run_cascade(seed: int, budget: int) -> tuple[float, int, int, float]:
    from trust_bo import CascadeMFEngine

    cascade = CascadeMFEngine(
        space=make_space(), direction="maximize", seed=seed, config=CONFIG,
        lf_budget=LF_BUDGET, top_k=TOP_K, batch_size=BATCH_SIZE,
    )
    started = time.perf_counter()
    result = cascade.run(
        lambda candidate: evaluate_cst(candidate, "xsmall"),
        lambda candidate: evaluate_cst(candidate, "xxxlarge"),
        hf_budget=budget,
    )
    best = result["best"]
    return (best["objective_values"][0] if best else 0.0, result["hf_evals"],
            result["lf_evals"], time.perf_counter() - started)


def csv_write_header() -> None:
    with open(CSV_PATH, "w", newline="") as handle:
        csv.writer(handle).writerow(
            ["arm", "seed", "best_hf_clcd", "hf_evals", "lf_evals", "total_seconds"]
        )


def run_all() -> None:
    from bench_resume import is_done, resume_or_init

    done = resume_or_init(CSV_PATH, ("arm", "seed"), csv_write_header)
    total = len(ARMS) * len(SEEDS)
    for index, (arm, seed) in enumerate(
        ((arm, seed) for arm in ARMS for seed in SEEDS), start=1
    ):
        tag = f"[{index}/{total}] {arm:<6s} | seed={seed}"
        if is_done(done, arm, seed):
            print(f"{tag} ... [skip]")
            continue
        print(f"{tag} ... ", end="", flush=True)
        try:
            runner = run_cascade if arm.startswith("casc") else run_hf
            best, hf_evals, lf_evals, elapsed = runner(seed, HF_BUDGETS[arm])
            with open(CSV_PATH, "a", newline="") as handle:
                csv.writer(handle).writerow(
                    [arm, seed, f"{best:.12g}", hf_evals, lf_evals, f"{elapsed:.2f}"]
                )
            print(f"best_hf_clcd={best:.6g}  hf={hf_evals} lf={lf_evals} ({elapsed:.1f}s)")
        except Exception:
            trace = traceback.format_exc().replace("\n", " | ")
            print(f"ERROR: {trace[:180]}", flush=True)


def print_summary() -> None:
    values: dict[str, list[float]] = defaultdict(list)
    paired: dict[int, dict[str, float]] = defaultdict(dict)
    if not CSV_PATH.exists():
        return
    with open(CSV_PATH, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                arm, seed = row["arm"], int(row["seed"])
                value = float(row["best_hf_clcd"])
            except (KeyError, ValueError):
                continue
            values[arm].append(value)
            paired[seed][arm] = value

    print("\n" + "=" * 68)
    print("  Best HF Cl/Cd by arm")
    print("=" * 68)
    for arm in ARMS:
        series = values.get(arm, [])
        if series:
            print(f"  {arm:<6s} mean={np.mean(series):.6g}  median={np.median(series):.6g}  n={len(series)}")

    for cascade in ("casc30", "casc50"):
        ratios = [arms[cascade] / arms["hf100"] for arms in paired.values()
                  if cascade in arms and "hf100" in arms
                  and arms[cascade] > 0.0 and arms["hf100"] > 0.0]
        if not ratios:
            continue
        geometric_mean = math.exp(float(np.mean(np.log(ratios))))
        cascade_wins = sum(ratio > 1.0 for ratio in ratios)
        verdict = "cascade wins" if geometric_mean > 1.0 else "hf100 wins"
        print(
            f"  {cascade} vs hf100: GM ratio={geometric_mean:.4f} "
            f"({verdict}; cascade wins {cascade_wins}/{len(ratios)})"
        )
    print("=" * 68)


if __name__ == "__main__":
    print(f"MF cascade A/B: arms={list(ARMS)} seeds={list(SEEDS)} SMOKE={bool(os.environ.get('SMOKE'))}")
    run_all()
    print_summary()
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
