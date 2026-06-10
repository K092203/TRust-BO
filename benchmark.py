"""
benchmark.py — TRust-BO vs BoTorch (TuRBO-1) vs HEBO vs Random Search

Problems  : Ackley 10D, Ackley 50D  (minimization, [-5, 5]^D)
Budget    : 200 evaluations, batch_size=4
Seeds     : 0, 1, 2, 3, 4
Output    : results.csv + summary table
"""

from __future__ import annotations

import csv
import math
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np

# ── 問題定義 ──────────────────────────────────────────────────────────────────

def ackley(x: np.ndarray) -> float:
    d = len(x)
    return (
        -20.0 * math.exp(-0.2 * math.sqrt(np.sum(x**2) / d))
        - math.exp(np.sum(np.cos(2 * math.pi * x)) / d)
        + 20.0 + math.e
    )

PROBLEMS = {
    "Ackley_10D": {"n_dims": 10, "lb": -5.0, "ub": 5.0, "fn": ackley},
    "Ackley_50D": {"n_dims": 50, "lb": -5.0, "ub": 5.0, "fn": ackley},
}

BUDGET     = 200
BATCH_SIZE = 4
SEEDS      = [0, 1, 2, 3, 4]
CSV_PATH   = Path("results.csv")

# ── CSV 管理 ──────────────────────────────────────────────────────────────────

def csv_write_header():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "problem", "seed", "best_value", "time_seconds"])

def csv_append(method, problem, seed, best_value, elapsed):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([method, problem, seed, best_value, f"{elapsed:.2f}"])

# ── TRust-BO ──────────────────────────────────────────────────────────────────

def run_trust_bo(prob_name: str, prob: dict, seed: int) -> tuple[float, float]:
    from trust_bo import TRustBOEngine, Float

    space = [Float(f"x{i}", float(prob["lb"]), float(prob["ub"]))
             for i in range(prob["n_dims"])]
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed)
    fn = prob["fn"]

    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < BUDGET:
        batch = min(BATCH_SIZE, BUDGET - evaluated)
        cands = engine.ask(batch_size=batch)
        results = []
        for c in cands:
            x = np.array([c[f"x{i}"] for i in range(prob["n_dims"])])
            results.append({"value": fn(x), "feasible": True})
        engine.tell(cands, results)
        evaluated += batch

    best = engine.best()
    best_val = best["objective_values"][0] if best else float("inf")
    return best_val, time.perf_counter() - t0

# ── BoTorch TuRBO-1 ───────────────────────────────────────────────────────────

def run_botorch_turbo(prob_name: str, prob: dict, seed: int) -> tuple[float, float]:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.generation import MaxPosteriorSampling
    from botorch.utils.transforms import normalize, unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_dims    = prob["n_dims"]
    lb, ub    = float(prob["lb"]), float(prob["ub"])
    fn        = prob["fn"]
    device    = torch.device("cpu")
    dtype     = torch.double

    # TuRBO state
    l_init    = 0.8
    l_min     = 0.5 ** 7
    l_max     = 1.6
    tau_succ  = 3
    tau_fail  = max(n_dims, 5)
    succ_cnt  = 0
    fail_cnt  = 0
    side_len  = l_init

    n_init = max(10, min(2 * (n_dims + 1), 50))
    bounds_cpu = torch.tensor([[lb] * n_dims, [ub] * n_dims], dtype=dtype)

    t0 = time.perf_counter()

    # Cold start (Sobol)
    rng = np.random.default_rng(seed)
    X_init = rng.uniform(lb, ub, size=(n_init, n_dims))
    Y_init = np.array([fn(x) for x in X_init])
    X_all = X_init.copy()
    Y_all = Y_init.copy()

    best_idx  = int(np.argmin(Y_all))
    best_x    = X_all[best_idx].copy()
    best_y    = float(Y_all[best_idx])

    evaluated = n_init

    while evaluated < BUDGET:
        batch = min(BATCH_SIZE, BUDGET - evaluated)

        # TR bounds
        tr_lb = np.clip(best_x - side_len / 2 * (ub - lb), lb, ub)
        tr_ub = np.clip(best_x + side_len / 2 * (ub - lb), lb, ub)

        # GP on normalized data
        X_t = torch.tensor(X_all, dtype=dtype)
        Y_t = torch.tensor(-Y_all, dtype=dtype).unsqueeze(-1)  # maximize
        X_norm = normalize(X_t, bounds_cpu)

        train_x = X_norm
        train_y = (Y_t - Y_t.mean()) / (Y_t.std() + 1e-8)

        gp = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()

        # Thompson sampling inside TR
        # X_cand shape: (n_cands, n_dims) — BoTorch MaxPosteriorSampling の標準入力形式
        # 出力 shape: (batch, n_dims)
        n_cands = min(100 * n_dims, 5000)
        cand_x_np = np.random.default_rng(seed + evaluated).uniform(
            tr_lb, tr_ub, (n_cands, n_dims)
        )
        cand_norm = normalize(
            torch.tensor(np.array(cand_x_np), dtype=dtype), bounds_cpu
        )  # (n_cands, n_dims)

        sampler = MaxPosteriorSampling(model=gp, replacement=False)
        with torch.no_grad():
            X_next = sampler(cand_norm, num_samples=batch)  # (batch, n_dims) normalized

        new_x_np = unnormalize(X_next, bounds_cpu).numpy()  # (batch, n_dims) in [lb, ub]
        new_x_np = np.atleast_2d(new_x_np)

        new_y = np.array([fn(x) for x in new_x_np])
        X_all = np.vstack([X_all, new_x_np])
        Y_all = np.concatenate([Y_all, new_y])
        evaluated += batch

        # TR update
        prev_best = best_y
        new_best_idx = int(np.argmin(new_y))
        if new_y[new_best_idx] < best_y:
            best_y = float(new_y[new_best_idx])
            best_x = new_x_np[new_best_idx].copy()

        if best_y < prev_best:
            succ_cnt += 1
            fail_cnt  = 0
        else:
            fail_cnt += 1
            succ_cnt  = 0

        if succ_cnt >= tau_succ:
            side_len = min(side_len * 2, l_max)
            succ_cnt = 0
        if fail_cnt >= tau_fail:
            side_len = max(side_len / 2, l_min)
            fail_cnt = 0
            # restart from best known
            best_idx = int(np.argmin(Y_all))
            best_x   = X_all[best_idx].copy()
            best_y   = float(Y_all[best_idx])

    return float(np.min(Y_all)), time.perf_counter() - t0

# ── HEBO ──────────────────────────────────────────────────────────────────────

def run_hebo(prob_name: str, prob: dict, seed: int) -> tuple[float, float]:
    import pandas as pd
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO

    n_dims = prob["n_dims"]
    lb, ub = float(prob["lb"]), float(prob["ub"])
    fn     = prob["fn"]

    space_params = [{"name": f"x{i}", "type": "num", "lb": lb, "ub": ub}
                    for i in range(n_dims)]
    space = DesignSpace().parse(space_params)
    opt   = HEBO(space, rand_sample=max(10, min(2 * (n_dims + 1), 50)),
                 scramble_seed=seed)

    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < BUDGET:
        batch = min(BATCH_SIZE, BUDGET - evaluated)
        rec = opt.suggest(n_suggestions=batch)
        y   = np.array([[fn(rec.iloc[i][[f"x{j}" for j in range(n_dims)]].values)]
                        for i in range(len(rec))])
        opt.observe(rec, y)
        evaluated += batch

    best_y = float(opt.y.min())
    return best_y, time.perf_counter() - t0

# ── Random Search ─────────────────────────────────────────────────────────────

def run_random(prob_name: str, prob: dict, seed: int) -> tuple[float, float]:
    n_dims = prob["n_dims"]
    lb, ub = float(prob["lb"]), float(prob["ub"])
    fn     = prob["fn"]
    rng    = np.random.default_rng(seed)

    t0 = time.perf_counter()
    best_y = float("inf")
    evaluated = 0
    while evaluated < BUDGET:
        batch = min(BATCH_SIZE, BUDGET - evaluated)
        xs = rng.uniform(lb, ub, (batch, n_dims))
        for x in xs:
            y = fn(x)
            if y < best_y:
                best_y = y
        evaluated += batch

    return best_y, time.perf_counter() - t0

# ── ランナー ──────────────────────────────────────────────────────────────────

RUNNERS = {
    "TRust-BO": run_trust_bo,
    "BoTorch_TuRBO": run_botorch_turbo,
    "HEBO": run_hebo,
    "Random": run_random,
}

def run_all():
    csv_write_header()
    total = len(RUNNERS) * len(PROBLEMS) * len(SEEDS)
    done  = 0

    for method, runner in RUNNERS.items():
        for prob_name, prob in PROBLEMS.items():
            for seed in SEEDS:
                done += 1
                tag = f"[{done}/{total}] {method:15s} | {prob_name:12s} | seed={seed}"
                print(f"{tag} ... ", end="", flush=True)
                try:
                    best_val, elapsed = runner(prob_name, prob, seed)
                    csv_append(method, prob_name, seed, f"{best_val:.6f}", elapsed)
                    print(f"best={best_val:.4f}  ({elapsed:.1f}s)")
                except Exception:
                    tb = traceback.format_exc().replace("\n", " | ")
                    csv_append(method, prob_name, seed, "error", 0.0)
                    print(f"ERROR: {tb[:120]}")

# ── 集計表示 ──────────────────────────────────────────────────────────────────

def print_summary():
    import csv as _csv

    rows: list[dict] = []
    with open(CSV_PATH, newline="") as f:
        for r in _csv.DictReader(f):
            try:
                rows.append({**r, "best_value": float(r["best_value"])})
            except ValueError:
                pass  # errorレコードをスキップ

    # (method, problem) ごとに集計
    from collections import defaultdict
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["problem"])].append(r["best_value"])

    methods  = list(RUNNERS.keys())
    problems = list(PROBLEMS.keys())

    header = f"{'Method':<18} {'Problem':<14} {'Median':>10} {'Mean':>10} {'Std':>10}  {'N':>4}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for prob_name in problems:
        for method in methods:
            vals = groups.get((method, prob_name), [])
            if not vals:
                print(f"{method:<18} {prob_name:<14} {'N/A':>10} {'N/A':>10} {'N/A':>10}  {0:>4}")
                continue
            arr = np.array(vals)
            print(f"{method:<18} {prob_name:<14} "
                  f"{np.median(arr):>10.4f} {np.mean(arr):>10.4f} {np.std(arr):>10.4f}  {len(arr):>4}")
        print()

    print(sep)

# ── エントリポイント ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  TRust-BO Benchmark")
    print(f"  problems : {', '.join(PROBLEMS)}")
    print(f"  methods  : {', '.join(RUNNERS)}")
    print(f"  budget   : {BUDGET}  batch={BATCH_SIZE}  seeds={SEEDS}")
    print("=" * 60)

    run_all()

    print("\n結果を集計中...")
    print_summary()
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
