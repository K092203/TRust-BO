"""
cfd_neuralfoil_benchmark.py — 翼型形状最適化ベンチマーク（Phase H-1）

NeuralFoil（XFOIL の ML サロゲート）を CFD の代替として使い、
TRust-BO+P2 / BoTorch_TuRBO / CMA-ES / Random を比較する。

問題設定:
  パラメータ化: CST (Kulfan) — 上面 8 係数 + 下面 8 係数 = 16D
  目的        : Cl/Cd 最大化（α=4°, Re=3×10^6）
  制約        : analysis_confidence > 0.5（物理的に非現実的な形状を除外）
  Budget      : 200 / Seeds: TRust-BO・Random=10, BoTorch=3, CMA-ES=5
  出力        : neuralfoil_benchmark_results.csv

設計空間（NACA 4 桁系から導出した実用的な範囲）:
  upper weights: [0.04, 0.45] × 8
  lower weights: [-0.40, 0.15] × 8

環境変数で上書き可:
  BUDGET=200 METHODS=TRust-BO+P2,Random SMOKE=1
"""

from __future__ import annotations

import contextlib
import csv
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np

# ── 設定 ──────────────────────────────────────────────────────────────────────

N_UPPER = 8
N_LOWER = 8
DIM     = N_UPPER + N_LOWER  # 16

ALPHA   = 4.0     # 迎角 [deg]
RE      = 3e6     # Reynolds 数

# CST 係数の探索範囲
UPPER_LB = np.full(N_UPPER, 0.04)
UPPER_UB = np.full(N_UPPER, 0.45)
LOWER_LB = np.full(N_LOWER, -0.40)
LOWER_UB = np.full(N_LOWER, 0.15)

LB = np.concatenate([UPPER_LB, LOWER_LB])
UB = np.concatenate([UPPER_UB, LOWER_UB])

BUDGET       = int(os.environ.get("BUDGET",     "200"))
N_INIT       = 10
BATCH_SIZE   = 4
N_SEED_FAST  = int(os.environ.get("NSEED_FAST", "10"))   # TRust-BO, Random
N_SEED_SLOW  = int(os.environ.get("NSEED_SLOW", "3"))    # BoTorch
N_SEED_CMA   = int(os.environ.get("NSEED_CMA",  "5"))    # CMA-ES
METHODS      = os.environ.get("METHODS", "TRust-BO+P2,Random,BoTorch_TuRBO,CMA-ES").split(",")
CSV_PATH     = Path(os.environ.get("CSV", "neuralfoil_benchmark_results.csv"))

if os.environ.get("SMOKE"):
    BUDGET      = 40
    N_SEED_FAST = 2
    N_SEED_SLOW = 1
    N_SEED_CMA  = 1
    CSV_PATH    = Path("neuralfoil_benchmark_smoke.csv")


# ── 評価関数 ──────────────────────────────────────────────────────────────────

def evaluate_cst(params: np.ndarray) -> tuple[float, bool]:
    """
    CST 係数 (16D) → (Cl/Cd, feasible)。

    Returns
    -------
    value    : Cl/Cd（最大化目標）。infeasible の場合は 0.0
    feasible : analysis_confidence > 0.5 かつ Cd > 0 かつ Cl > 0
    """
    import aerosandbox as asb
    import neuralfoil as nf

    upper = params[:N_UPPER]
    lower = params[N_UPPER:]

    try:
        af = asb.KulfanAirfoil(
            upper_weights=upper,
            lower_weights=lower,
            leading_edge_weight=0.0,
            TE_thickness=0.0,
        )
        result = nf.get_aero_from_airfoil(airfoil=af, alpha=ALPHA, Re=RE)
        cl   = float(result["CL"].item())
        cd   = float(result["CD"].item())
        conf = float(result["analysis_confidence"].item())

        if conf < 0.5 or cd <= 0 or cl <= 0:
            return 0.0, False
        return cl / cd, True

    except Exception:
        return 0.0, False


# ── CSV ───────────────────────────────────────────────────────────────────────

def csv_write_header():
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["method", "budget", "seed", "best_clcd", "total_seconds"])


def csv_append(method, budget, seed, best_clcd, elapsed):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [method, budget, seed, f"{best_clcd:.6f}", f"{elapsed:.2f}"])


# ── ランナー ──────────────────────────────────────────────────────────────────

def run_trust_bo(seed: int) -> tuple[float, float]:
    from trust_bo import TRustBOEngine, Float

    space = [Float(f"u{i}", float(UPPER_LB[i]), float(UPPER_UB[i])) for i in range(N_UPPER)]
    space += [Float(f"l{i}", float(LOWER_LB[i]), float(LOWER_UB[i])) for i in range(N_LOWER)]

    engine = TRustBOEngine(
        space=space, direction="maximize", seed=seed,
        config={"n_init": N_INIT, "enable_phase2": True, "batch_size": BATCH_SIZE},
    )

    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < BUDGET:
        b = min(BATCH_SIZE, BUDGET - evaluated)
        cands = engine.ask(batch_size=b)
        results = []
        for c in cands:
            params = np.array(
                [c[f"u{i}"] for i in range(N_UPPER)] +
                [c[f"l{i}"] for i in range(N_LOWER)]
            )
            val, feas = evaluate_cst(params)
            results.append({"value": val, "feasible": feas})
        engine.tell(cands, results)
        evaluated += b

    best = engine.best()
    best_val = best["objective_values"][0] if best else 0.0
    return best_val, time.perf_counter() - t0


def run_botorch(seed: int) -> tuple[float, float]:
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
    tau_succ, tau_fail = 3, max(DIM, 5)
    succ_cnt = fail_cnt = 0
    side_len = l_init

    bounds = torch.tensor([LB, UB], dtype=dtype)
    n_init = min(N_INIT, BUDGET)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    X_all = rng.uniform(LB, UB, size=(n_init, DIM))
    Y_raw = []
    for x in X_all:
        v, feas = evaluate_cst(x)
        Y_raw.append(v if feas else 0.0)
    Y_all = np.array(Y_raw)
    best_idx = int(np.argmax(Y_all))
    best_x, best_y = X_all[best_idx].copy(), float(Y_all[best_idx])
    evaluated = n_init

    while evaluated < BUDGET:
        batch = min(BATCH_SIZE, BUDGET - evaluated)
        tr_lb = np.clip(best_x - side_len / 2 * (UB - LB), LB, UB)
        tr_ub = np.clip(best_x + side_len / 2 * (UB - LB), LB, UB)

        X_t = torch.tensor(X_all, dtype=dtype)
        Y_t = torch.tensor(Y_all, dtype=dtype).unsqueeze(-1)
        X_norm = normalize(X_t, bounds)
        train_y = (Y_t - Y_t.mean()) / (Y_t.std() + 1e-8)

        gp = SingleTaskGP(X_norm, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()

        n_cands = min(100 * DIM, 5000)
        cand_np = np.random.default_rng(seed + evaluated).uniform(tr_lb, tr_ub, (n_cands, DIM))
        cand_norm = normalize(torch.tensor(np.array(cand_np), dtype=dtype), bounds)
        sampler = MaxPosteriorSampling(model=gp, replacement=False)
        with torch.no_grad():
            X_next = sampler(cand_norm, num_samples=batch)

        new_x = np.atleast_2d(unnormalize(X_next, bounds).numpy())
        new_y_raw = []
        for x in new_x:
            v, feas = evaluate_cst(x)
            new_y_raw.append(v if feas else 0.0)
        new_y = np.array(new_y_raw)

        X_all = np.vstack([X_all, new_x])
        Y_all = np.concatenate([Y_all, new_y])
        evaluated += batch

        prev_best = best_y
        nb = int(np.argmax(new_y))
        if new_y[nb] > best_y:
            best_y, best_x = float(new_y[nb]), new_x[nb].copy()
        if best_y > prev_best:
            succ_cnt += 1; fail_cnt = 0
        else:
            fail_cnt += 1; succ_cnt = 0
        if succ_cnt >= tau_succ:
            side_len = min(side_len * 2, l_max); succ_cnt = 0
        if fail_cnt >= tau_fail:
            side_len = max(side_len / 2, l_min); fail_cnt = 0
            best_idx = int(np.argmax(Y_all))
            best_x, best_y = X_all[best_idx].copy(), float(Y_all[best_idx])

    return float(np.max(Y_all)), time.perf_counter() - t0


def run_cma_es(seed: int) -> tuple[float, float]:
    import cma

    x0 = (LB + UB) / 2.0
    sigma0 = float(np.mean(UB - LB) / 6.0)
    bounds_cma = [LB.tolist(), UB.tolist()]

    t0 = time.perf_counter()
    evaluated = 0
    best_val = 0.0

    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {"seed": seed, "bounds": bounds_cma, "maxfevals": BUDGET,
         "popsize": BATCH_SIZE, "verbose": -9},
    )
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        while not es.stop() and evaluated < BUDGET:
            solutions = es.ask()
            fitnesses = []
            for x in solutions:
                x_clipped = np.clip(x, LB, UB)
                v, feas = evaluate_cst(x_clipped)
                fitness = -v if feas else 0.0  # CMA minimizes
                fitnesses.append(fitness)
                if feas and v > best_val:
                    best_val = v
                evaluated += 1
            es.tell(solutions, fitnesses)

    return best_val, time.perf_counter() - t0


def run_random(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best_val = 0.0
    evaluated = 0
    while evaluated < BUDGET:
        b = min(BATCH_SIZE, BUDGET - evaluated)
        for _ in range(b):
            x = rng.uniform(LB, UB)
            v, feas = evaluate_cst(x)
            if feas and v > best_val:
                best_val = v
        evaluated += b
    return best_val, time.perf_counter() - t0


RUNNERS = {
    "TRust-BO+P2":   run_trust_bo,
    "BoTorch_TuRBO": run_botorch,
    "CMA-ES":        run_cma_es,
    "Random":        run_random,
}

N_SEEDS = {
    "TRust-BO+P2":   N_SEED_FAST,
    "BoTorch_TuRBO": N_SEED_SLOW,
    "CMA-ES":        N_SEED_CMA,
    "Random":        N_SEED_FAST,
}


# ── メイン ────────────────────────────────────────────────────────────────────

def run_all():
    from bench_resume import is_done, resume_or_init
    done_keys = resume_or_init(
        CSV_PATH, ("method", "seed"), csv_write_header)
    total = sum(N_SEEDS[m] for m in METHODS if m in RUNNERS)
    done = 0
    for method in METHODS:
        if method not in RUNNERS:
            print(f"[skip] unknown method: {method}")
            continue
        runner = RUNNERS[method]
        n = N_SEEDS[method]
        for seed in range(n):
            done += 1
            tag = f"[{done}/{total}] {method:<15s} | budget={BUDGET} | seed={seed}"
            if is_done(done_keys, method, seed):
                print(f"{tag} ... [skip]")
                continue
            print(f"{tag} ... ", end="", flush=True)
            try:
                best, elapsed = runner(seed)
                csv_append(method, BUDGET, seed, best, elapsed)
                print(f"best_Cl/Cd={best:.2f}  ({elapsed:.1f}s)", flush=True)
            except Exception:
                tb = traceback.format_exc().replace("\n", " | ")
                csv_append(method, BUDGET, seed, 0.0, 0.0)
                print(f"ERROR: {tb[:160]}", flush=True)


def print_summary():
    from collections import defaultdict
    g = defaultdict(list)
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            try:
                g[r["method"]].append(float(r["best_clcd"]))
            except ValueError:
                pass

    print("\n" + "=" * 55)
    print(f"  翼型最適化ベンチ（Cl/Cd最大化, α={ALPHA}°, Re={RE:.0e}）")
    print(f"  budget={BUDGET}, dim={DIM}D (CST)")
    print("=" * 55)
    print(f"{'Method':<15} {'median':>8} {'mean':>8} {'std':>7} {'best':>8} {'N':>3}")
    print("-" * 55)
    for method in METHODS:
        vals = g.get(method, [])
        if not vals:
            continue
        arr = np.array(vals)
        print(f"{method:<15} {np.median(arr):>8.2f} {np.mean(arr):>8.2f} "
              f"{np.std(arr):>7.2f} {np.max(arr):>8.2f} {len(arr):>3}")
    print("=" * 55)

    # NACA2412 参照値を表示
    ref_val, ref_feas = evaluate_cst(
        np.concatenate([
            np.array([0.1769, 0.1832, 0.2263, 0.1896, 0.2013, 0.2003, 0.2013, 0.2085]),
            np.array([-0.1701, -0.1208, -0.1204, -0.0659, -0.1285, -0.0523, -0.0909, -0.0704]),
        ])
    )
    print(f"\n参照値 NACA2412: Cl/Cd = {ref_val:.2f}")
    print(f"CSV saved to: {CSV_PATH.resolve()}")


if __name__ == "__main__":
    print("=" * 55)
    print("  NeuralFoil 翼型最適化ベンチマーク (Phase H-1)")
    print(f"  methods : {', '.join(METHODS)}")
    print(f"  dim={DIM}D  budget={BUDGET}  n_init={N_INIT}  batch={BATCH_SIZE}")
    print(f"  seeds   : fast={N_SEED_FAST}  slow={N_SEED_SLOW}  cma={N_SEED_CMA}")
    print("=" * 55)
    run_all()
    print_summary()
