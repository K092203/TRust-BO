"""
cfd_scale_benchmark.py — CFD-scale (tiny budget) comparison.

Problems  : Ackley 50D / Rastrigin 50D / Levy 50D  (minimization)
Budget    : 20, 30, 50   (batch_size=4)
Seeds     : 0..9 (10)
Methods   : TRust-BO+P2 / BoTorch TuRBO-1 / HEBO / Random
Delay     : 0.1 s artificial latency per evaluation (applied to ALL methods)
Output    : cfd_scale_results.csv  (method,problem,budget,seed,best_value,total_seconds)

n_init = 10 is fixed and applied uniformly to every surrogate method. At 50D the
engine default would be 50, which would make every tiny-budget run pure random
cold-start. Phase 2 keeps its default guard (3*n_init = 30), so it naturally
engages only near budget=50.

Set SMOKE=1 to run a tiny subset (1 problem, budget=20, 1 seed) for wiring checks.
"""

from __future__ import annotations

import csv
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


PROBLEMS = {
    "Ackley_50D":    {"n_dims": 50, "lb": -5.0,  "ub": 5.0,  "fn": ackley},
    "Rastrigin_50D": {"n_dims": 50, "lb": -5.12, "ub": 5.12, "fn": rastrigin},
    "Levy_50D":      {"n_dims": 50, "lb": -10.0, "ub": 10.0, "fn": levy},
}

BUDGETS    = [20, 30, 50]
SEEDS      = list(range(10))
BATCH_SIZE = 4
N_INIT     = 10
DELAY_SEC  = 0.1
CSV_PATH   = Path("cfd_scale_results.csv")

if os.environ.get("SMOKE"):
    PROBLEMS = {"Ackley_50D": PROBLEMS["Ackley_50D"]}
    BUDGETS  = [20]
    SEEDS    = [0]
    CSV_PATH = Path("cfd_scale_results_smoke.csv")


def make_delayed(fn):
    """各評価に 0.1s の人工遅延を付与（全手法に同一適用）。"""
    def _f(x):
        time.sleep(DELAY_SEC)
        return fn(x)
    return _f


# ── CSV 管理 ──────────────────────────────────────────────────────────────────

def csv_write_header():
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["method", "problem", "budget", "seed", "best_value", "total_seconds"]
        )


def csv_append(method, problem, budget, seed, best_value, elapsed):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([method, problem, budget, seed, best_value, f"{elapsed:.2f}"])


# ── TRust-BO + Phase 2 ────────────────────────────────────────────────────────

def run_trust_bo(prob: dict, budget: int, seed: int) -> tuple[float, float]:
    from trust_bo import TRustBOEngine, Float

    n_dims = prob["n_dims"]
    space = [Float(f"x{i}", float(prob["lb"]), float(prob["ub"])) for i in range(n_dims)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed,
                           config={"n_init": N_INIT, "enable_phase2": True})
    fn = make_delayed(prob["fn"])

    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)
        cands = engine.ask(batch_size=batch)
        results = []
        for c in cands:
            x = np.array([c[f"x{i}"] for i in range(n_dims)])
            results.append({"value": fn(x), "feasible": True})
        engine.tell(cands, results)
        evaluated += batch

    best = engine.best()
    best_val = best["objective_values"][0] if best else float("inf")
    return best_val, time.perf_counter() - t0


# ── BoTorch TuRBO-1 ───────────────────────────────────────────────────────────

def run_botorch_turbo(prob: dict, budget: int, seed: int) -> tuple[float, float]:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.generation import MaxPosteriorSampling
    from botorch.utils.transforms import normalize, unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_dims = prob["n_dims"]
    lb, ub = float(prob["lb"]), float(prob["ub"])
    fn     = make_delayed(prob["fn"])
    dtype  = torch.double

    l_init, l_min, l_max = 0.8, 0.5 ** 7, 1.6
    tau_succ, tau_fail = 3, max(n_dims, 5)
    succ_cnt = fail_cnt = 0
    side_len = l_init

    n_init = min(N_INIT, budget)
    bounds_cpu = torch.tensor([[lb] * n_dims, [ub] * n_dims], dtype=dtype)

    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)
    X_all = rng.uniform(lb, ub, size=(n_init, n_dims))
    Y_all = np.array([fn(x) for x in X_all])

    best_idx = int(np.argmin(Y_all))
    best_x   = X_all[best_idx].copy()
    best_y   = float(Y_all[best_idx])
    evaluated = n_init

    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)

        tr_lb = np.clip(best_x - side_len / 2 * (ub - lb), lb, ub)
        tr_ub = np.clip(best_x + side_len / 2 * (ub - lb), lb, ub)

        X_t = torch.tensor(X_all, dtype=dtype)
        Y_t = torch.tensor(-Y_all, dtype=dtype).unsqueeze(-1)  # maximize
        X_norm = normalize(X_t, bounds_cpu)
        train_y = (Y_t - Y_t.mean()) / (Y_t.std() + 1e-8)

        gp = SingleTaskGP(X_norm, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()

        n_cands = min(100 * n_dims, 5000)
        cand_x_np = np.random.default_rng(seed + evaluated).uniform(
            tr_lb, tr_ub, (n_cands, n_dims))
        cand_norm = normalize(torch.tensor(np.array(cand_x_np), dtype=dtype), bounds_cpu)

        sampler = MaxPosteriorSampling(model=gp, replacement=False)
        with torch.no_grad():
            X_next = sampler(cand_norm, num_samples=batch)

        new_x_np = np.atleast_2d(unnormalize(X_next, bounds_cpu).numpy())
        new_y = np.array([fn(x) for x in new_x_np])
        X_all = np.vstack([X_all, new_x_np])
        Y_all = np.concatenate([Y_all, new_y])
        evaluated += batch

        prev_best = best_y
        nb = int(np.argmin(new_y))
        if new_y[nb] < best_y:
            best_y = float(new_y[nb])
            best_x = new_x_np[nb].copy()

        if best_y < prev_best:
            succ_cnt += 1; fail_cnt = 0
        else:
            fail_cnt += 1; succ_cnt = 0
        if succ_cnt >= tau_succ:
            side_len = min(side_len * 2, l_max); succ_cnt = 0
        if fail_cnt >= tau_fail:
            side_len = max(side_len / 2, l_min); fail_cnt = 0
            best_idx = int(np.argmin(Y_all))
            best_x = X_all[best_idx].copy(); best_y = float(Y_all[best_idx])

    return float(np.min(Y_all)), time.perf_counter() - t0


# ── HEBO ──────────────────────────────────────────────────────────────────────

def run_hebo(prob: dict, budget: int, seed: int) -> tuple[float, float]:
    import contextlib
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO

    n_dims = prob["n_dims"]
    lb, ub = float(prob["lb"]), float(prob["ub"])
    fn     = make_delayed(prob["fn"])

    space_params = [{"name": f"x{i}", "type": "num", "lb": lb, "ub": ub}
                    for i in range(n_dims)]
    space = DesignSpace().parse(space_params)
    opt = HEBO(space, rand_sample=min(N_INIT, budget), scramble_seed=seed)

    t0 = time.perf_counter()
    evaluated = 0
    # HEBO/GPy が "jitter = ..." を stdout に大量出力するため devnull に退避
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        while evaluated < budget:
            batch = min(BATCH_SIZE, budget - evaluated)
            rec = opt.suggest(n_suggestions=batch)
            y = np.array([[fn(rec.iloc[i][[f"x{j}" for j in range(n_dims)]].values)]
                          for i in range(len(rec))])
            opt.observe(rec, y)
            evaluated += batch

    return float(opt.y.min()), time.perf_counter() - t0


# ── Random Search ─────────────────────────────────────────────────────────────

def run_random(prob: dict, budget: int, seed: int) -> tuple[float, float]:
    n_dims = prob["n_dims"]
    lb, ub = float(prob["lb"]), float(prob["ub"])
    fn     = make_delayed(prob["fn"])
    rng    = np.random.default_rng(seed)

    t0 = time.perf_counter()
    best_y = float("inf")
    evaluated = 0
    while evaluated < budget:
        batch = min(BATCH_SIZE, budget - evaluated)
        for x in rng.uniform(lb, ub, (batch, n_dims)):
            best_y = min(best_y, fn(x))
        evaluated += batch
    return best_y, time.perf_counter() - t0


RUNNERS = {
    "TRust-BO+P2":   run_trust_bo,
    "BoTorch_TuRBO": run_botorch_turbo,
    "HEBO":          run_hebo,
    "Random":        run_random,
}


# ── ランナー ──────────────────────────────────────────────────────────────────

def run_all():
    from bench_resume import is_done, resume_or_init
    done_keys = resume_or_init(
        CSV_PATH, ("method", "problem", "budget", "seed"), csv_write_header)
    total = len(RUNNERS) * len(PROBLEMS) * len(BUDGETS) * len(SEEDS)
    done = 0
    for method, runner in RUNNERS.items():
        for prob_name, prob in PROBLEMS.items():
            for budget in BUDGETS:
                for seed in SEEDS:
                    done += 1
                    tag = (f"[{done}/{total}] {method:14s} | {prob_name:13s} "
                           f"| b={budget:2d} | seed={seed}")
                    if is_done(done_keys, method, prob_name, budget, seed):
                        print(f"{tag} ... [skip]")
                        continue
                    print(f"{tag} ... ", end="", flush=True)
                    try:
                        best_val, elapsed = runner(prob, budget, seed)
                        csv_append(method, prob_name, budget, seed, f"{best_val:.6f}", elapsed)
                        print(f"best={best_val:.4f}  ({elapsed:.1f}s)", flush=True)
                    except Exception:
                        tb = traceback.format_exc().replace("\n", " | ")
                        csv_append(method, prob_name, budget, seed, "error", 0.0)
                        print(f"ERROR: {tb[:160]}", flush=True)


# ── 集計表示 ──────────────────────────────────────────────────────────────────

def print_summary():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({**r, "best_value": float(r["best_value"]),
                             "total_seconds": float(r["total_seconds"])})
            except ValueError:
                pass

    from collections import defaultdict
    g_best = defaultdict(list)
    g_time = defaultdict(list)
    for r in rows:
        key = (r["problem"], int(r["budget"]), r["method"])
        g_best[key].append(r["best_value"])
        g_time[key].append(r["total_seconds"])

    header = (f"{'Problem':<14} {'budget':>6} {'Method':<15} "
             f"{'Median':>10} {'Mean':>10} {'Std':>9} {'Time/run':>9} {'N':>3}")
    sep = "-" * len(header)
    print("\n" + sep + "\n" + header + "\n" + sep)
    for prob_name in PROBLEMS:
        for budget in BUDGETS:
            for method in RUNNERS:
                key = (prob_name, budget, method)
                vals = g_best.get(key, [])
                if not vals:
                    continue
                arr = np.array(vals)
                t = np.mean(g_time.get(key, [0]))
                print(f"{prob_name:<14} {budget:>6} {method:<15} "
                      f"{np.median(arr):>10.4f} {np.mean(arr):>10.4f} "
                      f"{np.std(arr):>9.4f} {t:>8.1f}s {len(arr):>3}")
            print()
    print(sep)


if __name__ == "__main__":
    print("=" * 64)
    print("  CFD-scale Benchmark")
    print(f"  problems : {', '.join(PROBLEMS)}")
    print(f"  methods  : {', '.join(RUNNERS)}")
    print(f"  budgets  : {BUDGETS}  batch={BATCH_SIZE}  seeds={SEEDS}")
    print(f"  n_init   : {N_INIT}   delay={DELAY_SEC}s/eval")
    print("=" * 64)

    run_all()
    print("\n集計中...")
    print_summary()
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
