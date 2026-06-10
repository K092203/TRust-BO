"""
benchmark_v2.py — TRust-BO vs BoTorch (TuRBO-1) vs HEBO vs Random Search

設定A（高次元・大budget）:
  問題  : Ackley 50D, Ackley 100D
  budget: 500, batch_size=4, seeds=5

設定B（CFDスケール・小budget）:
  問題  : Ackley 10D, Ackley 50D
  budget: 50, batch_size=4, seeds=5

共通: seeds=[0,1,2,3,4]
出力 : results_v2.csv + サマリー表示
"""

from __future__ import annotations

import csv
import math
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

SEEDS    = [0, 1, 2, 3, 4]
CSV_PATH = Path("results_v2.csv")

# 設定ごとに (problem_name, n_dims, budget) を定義
SETTING_A = [
    ("Ackley_50D",  50,  500),
    ("Ackley_100D", 100, 500),
]
SETTING_B = [
    ("Ackley_10D",  10,  50),
    ("Ackley_50D",  50,  50),
]

BATCH_SIZE = 4

# ── n_init: 小 budget でも warm phase が最低 10 ラウンド確保できるよう調整 ──────

def calc_n_init(n_dims: int, budget: int) -> int:
    default = max(10, min(2 * (n_dims + 1), 50))
    # warm phase が最低 10 ラウンド × batch_size 分は残るよう cap
    max_cold = max(10, budget - 10 * BATCH_SIZE)
    return min(default, max_cold)

# ── CSV 管理 ──────────────────────────────────────────────────────────────────

def load_done() -> set[tuple]:
    """既存 CSV から完了済み (setting, method, problem, seed) を返す。"""
    done: set[tuple] = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["setting"], r["method"], r["problem"], r["seed"]))
    return done

def csv_write_header():
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["setting", "method", "problem", "seed", "best_value", "time_seconds"]
        )

def csv_append(setting, method, problem, seed, best_value, elapsed):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [setting, method, problem, seed, best_value, f"{elapsed:.2f}"]
        )

# ── TRust-BO ──────────────────────────────────────────────────────────────────

def run_trust_bo(n_dims: int, budget: int, seed: int) -> tuple[float, float]:
    from trust_bo import TRustBOEngine, Float

    n_init = calc_n_init(n_dims, budget)
    space  = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    engine = TRustBOEngine(
        space=space, direction="minimize", seed=seed,
        config={"n_init": n_init},
    )

    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)
        cands = engine.ask(batch_size=batch)
        engine.tell(cands, [
            {"value": ackley(np.array([c[f"x{i}"] for i in range(n_dims)])),
             "feasible": True}
            for c in cands
        ])
        evaluated += batch

    best = engine.best()
    return (best["objective_values"][0] if best else float("inf"),
            time.perf_counter() - t0)

# ── BoTorch TuRBO-1 ───────────────────────────────────────────────────────────

def run_botorch_turbo(n_dims: int, budget: int, seed: int) -> tuple[float, float]:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.generation import MaxPosteriorSampling
    from botorch.utils.transforms import normalize, unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.manual_seed(seed)
    lb, ub   = -5.0, 5.0
    dtype    = torch.double
    bounds   = torch.tensor([[lb] * n_dims, [ub] * n_dims], dtype=dtype)

    l_init   = 0.8
    l_min    = 0.5 ** 7
    l_max    = 1.6
    tau_succ = 3
    tau_fail = max(n_dims, 5)
    succ_cnt = fail_cnt = 0
    side_len = l_init

    n_init = calc_n_init(n_dims, budget)
    t0     = time.perf_counter()

    rng    = np.random.default_rng(seed)
    X_all  = rng.uniform(lb, ub, (n_init, n_dims))
    Y_all  = np.array([ackley(x) for x in X_all])

    best_idx = int(np.argmin(Y_all))
    best_x   = X_all[best_idx].copy()
    best_y   = float(Y_all[best_idx])
    evaluated = n_init

    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)

        tr_lb = np.clip(best_x - side_len / 2 * (ub - lb), lb, ub)
        tr_ub = np.clip(best_x + side_len / 2 * (ub - lb), lb, ub)

        X_t   = torch.tensor(X_all, dtype=dtype)
        Y_t   = torch.tensor(-Y_all, dtype=dtype).unsqueeze(-1)
        X_norm = normalize(X_t, bounds)
        train_y = (Y_t - Y_t.mean()) / (Y_t.std() + 1e-8)

        gp  = SingleTaskGP(X_norm, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()

        n_cands    = min(100 * n_dims, 5000)
        cand_np    = np.random.default_rng(seed + evaluated).uniform(tr_lb, tr_ub, (n_cands, n_dims))
        cand_norm  = normalize(torch.tensor(np.array(cand_np), dtype=dtype), bounds)

        sampler = MaxPosteriorSampling(model=gp, replacement=False)
        with torch.no_grad():
            X_next = sampler(cand_norm, num_samples=batch)   # (batch, n_dims)

        new_x = unnormalize(X_next, bounds).numpy()
        new_x = np.atleast_2d(new_x)
        new_y = np.array([ackley(x) for x in new_x])

        X_all = np.vstack([X_all, new_x])
        Y_all = np.concatenate([Y_all, new_y])
        evaluated += batch

        prev_best = best_y
        nb_idx = int(np.argmin(new_y))
        if new_y[nb_idx] < best_y:
            best_y = float(new_y[nb_idx])
            best_x = new_x[nb_idx].copy()

        if best_y < prev_best:
            succ_cnt += 1; fail_cnt = 0
        else:
            fail_cnt += 1; succ_cnt = 0

        if succ_cnt >= tau_succ:
            side_len = min(side_len * 2, l_max); succ_cnt = 0
        if fail_cnt >= tau_fail:
            side_len = max(side_len / 2, l_min); fail_cnt = 0
            best_idx = int(np.argmin(Y_all))
            best_x   = X_all[best_idx].copy()
            best_y   = float(Y_all[best_idx])

    return float(np.min(Y_all)), time.perf_counter() - t0

# ── HEBO ──────────────────────────────────────────────────────────────────────

def run_hebo(n_dims: int, budget: int, seed: int) -> tuple[float, float]:
    import pandas as pd
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO

    n_init = calc_n_init(n_dims, budget)
    space_params = [{"name": f"x{i}", "type": "num", "lb": -5.0, "ub": 5.0}
                    for i in range(n_dims)]
    space = DesignSpace().parse(space_params)
    opt   = HEBO(space, rand_sample=n_init, scramble_seed=seed)

    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)
        rec = opt.suggest(n_suggestions=batch)
        y   = np.array([[ackley(rec.iloc[i][[f"x{j}" for j in range(n_dims)]].values)]
                        for i in range(len(rec))])
        opt.observe(rec, y)
        evaluated += batch

    return float(opt.y.min()), time.perf_counter() - t0

# ── Random Search ─────────────────────────────────────────────────────────────

def run_random(n_dims: int, budget: int, seed: int) -> tuple[float, float]:
    rng    = np.random.default_rng(seed)
    t0     = time.perf_counter()
    best_y = float("inf")
    evaluated = 0
    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)
        for x in rng.uniform(-5.0, 5.0, (batch, n_dims)):
            y = ackley(x)
            if y < best_y:
                best_y = y
        evaluated += batch
    return best_y, time.perf_counter() - t0

# ── ランナー辞書 ──────────────────────────────────────────────────────────────

RUNNERS = {
    "TRust-BO":     run_trust_bo,
    "BoTorch_TuRBO": run_botorch_turbo,
    "HEBO":         run_hebo,
    "Random":       run_random,
}

# ── 実験ループ ────────────────────────────────────────────────────────────────

BOTORCH_MAX_DIMS = 50   # 100D は ~600s/run のためスキップ
HEBO_MAX_BUDGET  = 200  # budget>=500 で BoTorch 並みに遅いためスキップ

def run_setting(setting_name: str, problems: list[tuple[str, int, int]],
                completed: set[tuple]):
    total = len(RUNNERS) * len(problems) * len(SEEDS)
    done  = 0
    print(f"\n{'='*62}")
    print(f"  Setting {setting_name}")
    for prob_name, n_dims, budget in problems:
        print(f"  {prob_name} | budget={budget} | n_init={calc_n_init(n_dims, budget)}")
    print(f"  methods: {', '.join(RUNNERS)}")
    print(f"  seeds  : {SEEDS}")
    print(f"{'='*62}")

    for method, runner in RUNNERS.items():
        for prob_name, n_dims, budget in problems:
            for seed in SEEDS:
                done += 1
                key = (setting_name, method, prob_name, str(seed))
                tag = f"[{done}/{total}] {method:15s} | {prob_name:12s} | seed={seed}"

                if key in completed:
                    print(f"{tag} ... SKIP (already done)")
                    continue

                # BoTorch は高次元で非現実的に遅いためスキップ
                if method == "BoTorch_TuRBO" and n_dims > BOTORCH_MAX_DIMS:
                    print(f"{tag} ... SKIP (BoTorch {n_dims}D too slow)")
                    csv_append(setting_name, method, prob_name, seed, "too_slow", 0.0)
                    continue

                # HEBO は budget>=500 で BoTorch 並みに遅いためスキップ
                if method == "HEBO" and budget > HEBO_MAX_BUDGET:
                    print(f"{tag} ... SKIP (HEBO budget={budget} too slow)")
                    csv_append(setting_name, method, prob_name, seed, "too_slow", 0.0)
                    continue

                print(f"{tag} ... ", end="", flush=True)
                try:
                    best_val, elapsed = runner(n_dims, budget, seed)
                    csv_append(setting_name, method, prob_name, seed,
                               f"{best_val:.6f}", elapsed)
                    print(f"best={best_val:.4f}  ({elapsed:.1f}s)")
                except Exception:
                    tb = traceback.format_exc().replace("\n", " | ")
                    csv_append(setting_name, method, prob_name, seed, "error", 0.0)
                    print(f"ERROR: {tb[:120]}")

# ── 集計表示 ──────────────────────────────────────────────────────────────────

def print_summary():
    from collections import defaultdict

    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({**r, "best_value": float(r["best_value"])})
            except ValueError:
                pass

    methods  = list(RUNNERS.keys())
    settings = {"A": SETTING_A, "B": SETTING_B}

    for setting_name, problems in settings.items():
        prob_names = [p[0] for p in problems]
        budgets    = {p[0]: p[2] for p in problems}

        print(f"\n{'─'*72}")
        print(f"  Setting {setting_name} (budget={budgets[prob_names[0]]})")
        print(f"{'─'*72}")
        hdr = f"  {'Method':<18} {'Problem':<14} {'Min':>8} {'Median':>8} {'Mean':>8} {'Std':>7}  N"
        print(hdr)
        print(f"  {'-'*66}")

        groups: dict[tuple, list[float]] = defaultdict(list)
        for r in rows:
            if r["setting"] == setting_name:
                groups[(r["method"], r["problem"])].append(r["best_value"])

        for prob_name in prob_names:
            for method in methods:
                vals = groups.get((method, prob_name), [])
                if not vals:
                    print(f"  {method:<18} {prob_name:<14} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>7}  0")
                    continue
                arr = np.array(vals)
                print(f"  {method:<18} {prob_name:<14} "
                      f"{np.min(arr):>8.4f} {np.median(arr):>8.4f} "
                      f"{np.mean(arr):>8.4f} {np.std(arr):>7.4f}  {len(arr)}")
            print()

    print(f"{'─'*72}")

# ── エントリポイント ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("TRust-BO Benchmark v2")
    print(f"CSV: {CSV_PATH.resolve()}")

    completed = load_done()
    if completed:
        print(f"Resume: {len(completed)} rows already done, skipping.")
    else:
        csv_write_header()

    run_setting("B", SETTING_B, completed)
    run_setting("A", SETTING_A, completed)

    print("\n結果を集計中...")
    print_summary()
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
