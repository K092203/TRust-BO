"""
midbudget_benchmark.py — クロスオーバー特定用中間 budget ベンチマーク。

既存データ:
  cfd_scale_results.csv  : budget=20/30/50,  50D,       n_init=10, 0.1s遅延
  large_budget_results.csv: budget=500,       50/100/200D, n_init=50, 遅延なし

本スクリプトはその間 (budget=100/200/300) を n_init=10・遅延なしで埋め、
クロスオーバーポイント特定を可能にする。

Problems  : Ackley / Rastrigin / Levy  (50D・100D)
Budget    : 100 / 200 / 300  (batch_size=4, n_init=10)
Seeds     : TRust-BO+P2 / Random = 10,  BoTorch / HEBO = 3
            HEBO は 100D を除外 (GP が O(n³) で非現実的)
Delay     : なし
Output    : midbudget_results.csv
            (method,problem,dim,budget,seed,best_value,total_seconds)

環境変数で上書き可:
  DIMS=50 BUDGETS=100 NSEED_FAST=1 NSEED_SLOW=1 SMOKE=1
"""

from __future__ import annotations

import csv
import contextlib
import math
import os
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


def rastrigin(x: np.ndarray) -> float:
    d = len(x)
    return float(10.0 * d + np.sum(x**2 - 10.0 * np.cos(2 * math.pi * x)))


def levy(x: np.ndarray) -> float:
    w = 1.0 + (x - 1.0) / 4.0
    term1 = math.sin(math.pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2 * math.pi * w[-1]) ** 2)
    wm = w[:-1]
    term2 = np.sum((wm - 1.0) ** 2 * (1.0 + 10.0 * np.sin(math.pi * wm + 1.0) ** 2))
    return float(term1 + term2 + term3)


PROB_DOMAIN = {
    "Ackley":    (-5.0,  5.0,  ackley),
    "Rastrigin": (-5.12, 5.12, rastrigin),
    "Levy":      (-10.0, 10.0, levy),
}

# ── 設定 ──────────────────────────────────────────────────────────────────────

DIMS         = [int(x) for x in os.environ.get("DIMS",    "50,100").split(",")]
BUDGETS      = [int(x) for x in os.environ.get("BUDGETS", "100,200,300").split(",")]
PROBLEMS     = os.environ.get("PROBLEMS", "Ackley,Rastrigin,Levy").split(",")
METHODS      = os.environ.get("METHODS",  "TRust-BO+P2,Random,BoTorch_TuRBO,HEBO").split(",")
N_INIT       = 10
BATCH_SIZE   = 4
N_SEED_FAST  = int(os.environ.get("NSEED_FAST", "10"))
N_SEED_SLOW  = int(os.environ.get("NSEED_SLOW", "3"))
HEBO_MAX_DIM = 50
CSV_PATH     = Path(os.environ.get("CSV", "midbudget_results.csv"))

SLOW_METHODS = {"BoTorch_TuRBO", "HEBO"}

if os.environ.get("SMOKE"):
    DIMS     = [50]
    BUDGETS  = [100]
    PROBLEMS = ["Ackley"]
    N_SEED_FAST = 1
    N_SEED_SLOW = 1
    CSV_PATH = Path("midbudget_results_smoke.csv")


SAASBO_MAX_DIM = int(os.environ.get("SAASBO_MAX_DIM", "100"))
SLOW_METHODS = {"BoTorch_TuRBO", "HEBO", "SAASBO"}


def n_seed(method: str, dim: int) -> int:
    if method == "HEBO" and dim > HEBO_MAX_DIM:
        return 0
    if method == "SAASBO" and dim > SAASBO_MAX_DIM:
        return 0
    return N_SEED_SLOW if method in SLOW_METHODS else N_SEED_FAST


# ── CSV ───────────────────────────────────────────────────────────────────────

def csv_write_header():
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["method", "problem", "dim", "budget", "seed", "best_value", "total_seconds"])


def csv_load_done() -> set[tuple]:
    """既完了の (method, problem, dim, budget, seed) を返す。"""
    done: set[tuple] = set()
    if not CSV_PATH.exists():
        return done
    try:
        with open(CSV_PATH, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["method"], r["problem"], int(r["dim"]),
                          int(r["budget"]), int(r["seed"])))
    except Exception:
        pass
    return done


def csv_append(method, problem, dim, budget, seed, best_value, elapsed):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [method, problem, dim, budget, seed, best_value, f"{elapsed:.2f}"])


# ── TRust-BO + Phase 2 ────────────────────────────────────────────────────────

def run_trust_bo(fn, lb, ub, dim, budget, seed) -> tuple[float, float]:
    from trust_bo import TRustBOEngine, Float

    space = [Float(f"x{i}", lb, ub) for i in range(dim)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed,
                           config={"n_init": N_INIT, "enable_phase2": True})

    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)
        cands = engine.ask(batch_size=batch)
        results = [{"value": fn(np.array([c[f"x{i}"] for i in range(dim)])),
                    "feasible": True} for c in cands]
        engine.tell(cands, results)
        evaluated += batch

    best = engine.best()
    return (best["objective_values"][0] if best else float("inf")), time.perf_counter() - t0


# ── BoTorch TuRBO-1 ───────────────────────────────────────────────────────────

def run_botorch_turbo(fn, lb, ub, dim, budget, seed) -> tuple[float, float]:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.generation import MaxPosteriorSampling
    from botorch.utils.transforms import normalize, unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.double

    l_init, l_min, l_max = 0.8, 0.5 ** 7, 1.6
    tau_succ, tau_fail = 3, max(dim, 5)
    succ_cnt = fail_cnt = 0
    side_len = l_init

    n_init = min(N_INIT, budget)
    bounds = torch.tensor([[lb] * dim, [ub] * dim], dtype=dtype)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    X_all = rng.uniform(lb, ub, size=(n_init, dim))
    Y_all = np.array([fn(x) for x in X_all])
    best_idx = int(np.argmin(Y_all))
    best_x, best_y = X_all[best_idx].copy(), float(Y_all[best_idx])
    evaluated = n_init

    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)
        tr_lb = np.clip(best_x - side_len / 2 * (ub - lb), lb, ub)
        tr_ub = np.clip(best_x + side_len / 2 * (ub - lb), lb, ub)

        X_t = torch.tensor(X_all, dtype=dtype)
        Y_t = torch.tensor(-Y_all, dtype=dtype).unsqueeze(-1)
        X_norm = normalize(X_t, bounds)
        train_y = (Y_t - Y_t.mean()) / (Y_t.std() + 1e-8)

        gp = SingleTaskGP(X_norm, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()

        n_cands = min(100 * dim, 5000)
        cand_np = np.random.default_rng(seed + evaluated).uniform(tr_lb, tr_ub, (n_cands, dim))
        cand_norm = normalize(torch.tensor(np.array(cand_np), dtype=dtype), bounds)
        sampler = MaxPosteriorSampling(model=gp, replacement=False)
        with torch.no_grad():
            X_next = sampler(cand_norm, num_samples=batch)

        new_x = np.atleast_2d(unnormalize(X_next, bounds).numpy())
        new_y = np.array([fn(x) for x in new_x])
        X_all = np.vstack([X_all, new_x])
        Y_all = np.concatenate([Y_all, new_y])
        evaluated += batch

        prev_best = best_y
        nb = int(np.argmin(new_y))
        if new_y[nb] < best_y:
            best_y, best_x = float(new_y[nb]), new_x[nb].copy()
        if best_y < prev_best:
            succ_cnt += 1; fail_cnt = 0
        else:
            fail_cnt += 1; succ_cnt = 0
        if succ_cnt >= tau_succ:
            side_len = min(side_len * 2, l_max); succ_cnt = 0
        if fail_cnt >= tau_fail:
            side_len = max(side_len / 2, l_min); fail_cnt = 0
            best_idx = int(np.argmin(Y_all))
            best_x, best_y = X_all[best_idx].copy(), float(Y_all[best_idx])

    return float(np.min(Y_all)), time.perf_counter() - t0


# ── HEBO ──────────────────────────────────────────────────────────────────────

def run_hebo(fn, lb, ub, dim, budget, seed) -> tuple[float, float]:
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO

    space_params = [{"name": f"x{i}", "type": "num", "lb": lb, "ub": ub}
                    for i in range(dim)]
    space = DesignSpace().parse(space_params)
    opt = HEBO(space, rand_sample=min(N_INIT, budget), scramble_seed=seed)

    t0 = time.perf_counter()
    evaluated = 0
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        while evaluated < budget:
            batch = min(BATCH_SIZE, budget - evaluated)
            rec = opt.suggest(n_suggestions=batch)
            y = np.array([[fn(rec.iloc[i][[f"x{j}" for j in range(dim)]].values)]
                          for i in range(len(rec))])
            opt.observe(rec, y)
            evaluated += batch

    return float(opt.y.min()), time.perf_counter() - t0


# ── Random ────────────────────────────────────────────────────────────────────

def run_random(fn, lb, ub, dim, budget, seed) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best_y = float("inf")
    evaluated = 0
    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)
        for x in rng.uniform(lb, ub, (batch, dim)):
            best_y = min(best_y, fn(x))
        evaluated += batch
    return best_y, time.perf_counter() - t0


def run_saasbo(fn, lb, ub, dim, budget, seed) -> tuple[float, float]:
    # JAX/NumPyro(SAASBO の MCMC バックエンド)が WSL の限られた RAM で OOM し
    # WSL ごとクラッシュするのを防ぐ。メモリの事前確保を無効化し CPU 実行を強制。
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    import torch
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.fit import fit_fully_bayesian_model_nuts
    from botorch.generation import MaxPosteriorSampling
    from botorch.utils.transforms import normalize, unnormalize

    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.double
    bounds = torch.tensor([[lb] * dim, [ub] * dim], dtype=dtype)
    n_init = min(N_INIT, budget)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    X_all = rng.uniform(lb, ub, size=(n_init, dim))
    Y_all = np.array([fn(x) for x in X_all])
    evaluated = n_init

    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)
        X_t = normalize(torch.tensor(X_all, dtype=dtype), bounds)
        Y_t = torch.tensor(-Y_all, dtype=dtype).unsqueeze(-1)
        Y_t = (Y_t - Y_t.mean()) / (Y_t.std() + 1e-8)

        gp = SaasFullyBayesianSingleTaskGP(X_t, Y_t)
        fit_fully_bayesian_model_nuts(
            gp, warmup_steps=64, num_samples=32, thinning=4,
            disable_progbar=True,
        )
        gp.eval()

        n_cands = min(100 * dim, 5000)
        cand_np = np.random.default_rng(seed + evaluated).uniform(lb, ub, (n_cands, dim))
        cand_norm = normalize(torch.tensor(cand_np, dtype=dtype), bounds)
        sampler = MaxPosteriorSampling(model=gp, replacement=False)
        with torch.no_grad():
            X_next = sampler(cand_norm, num_samples=batch)

        new_x = unnormalize(X_next, bounds).numpy()
        new_y = np.array([fn(x) for x in new_x])
        X_all = np.vstack([X_all, new_x])
        Y_all = np.concatenate([Y_all, new_y])
        evaluated += batch

    return float(np.min(Y_all)), time.perf_counter() - t0


RUNNERS = {
    "TRust-BO+P2":   run_trust_bo,
    "BoTorch_TuRBO": run_botorch_turbo,
    "SAASBO":        run_saasbo,
    "HEBO":          run_hebo,
    "Random":        run_random,
}


# ── ランナー ──────────────────────────────────────────────────────────────────

def run_all():
    resume = CSV_PATH.exists()
    if resume:
        completed = csv_load_done()
        print(f"  [resume] CSV found — {len(completed)} run(s) already done, skipping")
    else:
        csv_write_header()
        completed = set()
    total = sum(
        n_seed(m, d) * len(PROBLEMS) * len(BUDGETS)
        for m in METHODS for d in DIMS
    )
    done = 0
    for method in METHODS:
        runner = RUNNERS.get(method)
        if runner is None:
            print(f"[skip] unknown method: {method}")
            continue
        for dim in DIMS:
            seeds = range(n_seed(method, dim))
            if not seeds:
                continue
            for pname in PROBLEMS:
                lb, ub, fn = PROB_DOMAIN[pname]
                for budget in BUDGETS:
                    for seed in seeds:
                        done += 1
                        key = (method, pname, dim, budget, seed)
                        tag = (f"[{done}/{total}] {method:14s} | {pname:10s} "
                               f"| {dim:3d}D | b={budget:3d} | seed={seed}")
                        if key in completed:
                            print(f"{tag} ... [skip]")
                            continue
                        print(f"{tag} ... ", end="", flush=True)
                        try:
                            best, elapsed = runner(fn, lb, ub, dim, budget, seed)
                            csv_append(method, pname, dim, budget, seed, f"{best:.6f}", elapsed)
                            print(f"best={best:.4f}  ({elapsed:.1f}s)", flush=True)
                        except Exception:
                            tb = traceback.format_exc().replace("\n", " | ")
                            csv_append(method, pname, dim, budget, seed, "error", 0.0)
                            print(f"ERROR: {tb[:160]}", flush=True)


# ── 集計 ──────────────────────────────────────────────────────────────────────

def print_summary():
    from collections import defaultdict
    g_best, g_time = defaultdict(list), defaultdict(list)
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            try:
                key = (r["problem"], int(r["dim"]), int(r["budget"]), r["method"])
                g_best[key].append(float(r["best_value"]))
                g_time[key].append(float(r["total_seconds"]))
            except ValueError:
                pass

    header = (f"{'Problem':<11} {'dim':>4} {'budget':>6} {'Method':<15} "
              f"{'Median':>10} {'Mean':>10} {'Std':>9} {'Time/run':>9} {'N':>3}")
    sep = "-" * len(header)
    print("\n" + sep + "\n" + header + "\n" + sep)
    for pname in PROBLEMS:
        for dim in DIMS:
            for budget in BUDGETS:
                for method in METHODS:
                    key = (pname, dim, budget, method)
                    vals = g_best.get(key, [])
                    if not vals:
                        continue
                    arr = np.array(vals)
                    t = np.mean(g_time.get(key, [0]))
                    print(f"{pname:<11} {dim:>4} {budget:>6} {method:<15} "
                          f"{np.median(arr):>10.4f} {np.mean(arr):>10.4f} "
                          f"{np.std(arr):>9.4f} {t:>8.1f}s {len(arr):>3}")
                print()
    print(sep)


if __name__ == "__main__":
    print("=" * 70)
    print("  Mid-budget / Crossover Benchmark")
    print(f"  problems : {', '.join(PROBLEMS)}")
    print(f"  methods  : {', '.join(METHODS)}")
    print(f"  dims     : {DIMS}  budgets={BUDGETS}  batch={BATCH_SIZE}  n_init={N_INIT}")
    print(f"  seeds    : fast={N_SEED_FAST}  slow={N_SEED_SLOW}  HEBO_MAX_DIM={HEBO_MAX_DIM}")
    print("=" * 70)
    run_all()
    print("\n集計中...")
    print_summary()
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
