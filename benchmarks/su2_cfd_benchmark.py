"""
su2_cfd_benchmark.py — Phase H-2-6/7: 実 CFD (SU2 RANS) 翼型最適化ベンチマーク

実 RANS ソルバー(SU2、Ma=0.3 Re=3e6 SA)を目的関数に、
TRust-BO+P2 / BoTorch_TuRBO / CMA-ES / Random を比較する。

問題設定(H-1 と同一の探索空間):
  パラメータ化: CST 上面 8 + 下面 8 = 16D
  目的        : Cl/Cd 最大化(α=2°)
  制約        : メッシュ生成失敗・発散・非物理値は feasible=False
  評価        : 1 形状あたり SU2 RANS ~30-50s。バッチ内は並列実行。

NeuralFoil(H-1)との違いは「実 RANS ソルバー」である点のみ。
全手法が同一の SU2 評価関数・同一の並列度を共有するため公平。

環境変数:
  BUDGET=100 SEEDS=3 BATCH=8 WORKERS=8 NTHREAD=2 ITER=4000 AOA=2.0
  METHODS=TRust-BO+P2,Random,BoTorch_TuRBO,CMA-ES
  SMOKE=1  (budget=16, 1 seed, 短時間)

本番実行(数時間規模、バックグラウンド推奨):
  cd /home/k0903/trm-engine
  .venv/bin/python benchmarks/su2_cfd_benchmark.py
"""

from __future__ import annotations

import contextlib
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "su2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from su2_runner import SU2Settings, run_cst  # noqa: E402

# ── 設定 ──────────────────────────────────────────────────────────────────────

N_UPPER = 8
N_LOWER = 8
DIM = N_UPPER + N_LOWER

# 翼型らしい設計空間(全 seed で十分な実行可能率、かつ一部 infeasible を残す)。
# H-1 の広い箱は strict なメッシュ妥当性には feasibility が低く seed 依存が強いため、
# SU2 用にやや絞った範囲を用いる(§BENCHMARK 参照)。
UPPER_LB = np.full(N_UPPER, 0.05)
UPPER_UB = np.full(N_UPPER, 0.35)
LOWER_LB = np.full(N_LOWER, -0.35)
LOWER_UB = np.full(N_LOWER, 0.05)
LB = np.concatenate([UPPER_LB, LOWER_LB])
UB = np.concatenate([UPPER_UB, LOWER_UB])

AOA       = float(os.environ.get("AOA", "2.0"))
BUDGET    = int(os.environ.get("BUDGET", "100"))
N_INIT    = int(os.environ.get("NINIT", "12"))
BATCH     = int(os.environ.get("BATCH", "8"))
WORKERS   = int(os.environ.get("WORKERS", "8"))
NTHREAD   = int(os.environ.get("NTHREAD", "2"))
ITER      = int(os.environ.get("ITER", "4000"))
N_SEED    = int(os.environ.get("SEEDS", "3"))
METHODS   = os.environ.get("METHODS",
                           "TRust-BO+P2,Random,BoTorch_TuRBO,CMA-ES").split(",")
CSV_PATH  = Path(os.environ.get("CSV", "su2_benchmark_results.csv"))

if os.environ.get("SMOKE"):
    BUDGET = 16
    N_SEED = 1
    BATCH = 8
    ITER = 1500
    CSV_PATH = Path("su2_benchmark_smoke.csv")

_SETTINGS = SU2Settings(aoa=AOA, n_threads=NTHREAD, max_iter=ITER)
_EXECUTOR: ThreadPoolExecutor | None = None


# ── 並列 SU2 評価 ─────────────────────────────────────────────────────────────

def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=WORKERS)
    return _EXECUTOR


def evaluate_one(params: np.ndarray) -> tuple[float, bool]:
    """CST 16D → (Cl/Cd, feasible)。SU2 RANS 1 回。"""
    wu = params[:N_UPPER]
    wl = params[N_UPPER:]
    cl, cd, feasible, _ = run_cst(wu, wl, settings=_SETTINGS)
    if not feasible or cd <= 0:
        return 0.0, False
    return cl / cd, True


def evaluate_batch(X: list[np.ndarray]) -> list[tuple[float, bool]]:
    """複数形状を WORKERS 並列で評価する(バッチ内並列)。"""
    return list(_executor().map(evaluate_one, X))


# ── CSV ───────────────────────────────────────────────────────────────────────

def csv_write_header():
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["method", "budget", "seed", "best_clcd", "n_feasible", "total_seconds"])


def csv_load_done() -> set[tuple[str, int]]:
    """既完了の (method, seed) を返す。CSV がなければ空集合。"""
    done: set[tuple[str, int]] = set()
    if not CSV_PATH.exists():
        return done
    try:
        with open(CSV_PATH, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["method"], int(r["seed"])))
    except Exception:
        pass
    return done


def csv_append(method, seed, best, n_feas, elapsed):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [method, BUDGET, seed, f"{best:.6f}", n_feas, f"{elapsed:.1f}"])


# ── ランナー(全手法でバッチ並列評価を共有)──────────────────────────────────

def run_trust_bo(seed: int) -> tuple[float, int, float]:
    from trust_bo import Float, TRustBOEngine
    space = [Float(f"u{i}", float(UPPER_LB[i]), float(UPPER_UB[i])) for i in range(N_UPPER)]
    space += [Float(f"l{i}", float(LOWER_LB[i]), float(LOWER_UB[i])) for i in range(N_LOWER)]
    engine = TRustBOEngine(
        space=space, direction="maximize", seed=seed,
        config={"n_init": N_INIT, "enable_phase2": True, "batch_size": BATCH},
    )
    t0 = time.perf_counter()
    evaluated = n_feas = 0
    while evaluated < BUDGET:
        b = min(BATCH, BUDGET - evaluated)
        cands = engine.ask(batch_size=b)
        X = [np.array([c[f"u{i}"] for i in range(N_UPPER)]
                      + [c[f"l{i}"] for i in range(N_LOWER)]) for c in cands]
        res = evaluate_batch(X)
        engine.tell(cands, [{"value": v, "feasible": fz} for v, fz in res])
        n_feas += sum(int(fz) for _, fz in res)
        evaluated += b
    best = engine.best()
    return (best["objective_values"][0] if best else 0.0), n_feas, time.perf_counter() - t0


def run_random(seed: int) -> tuple[float, int, float]:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best = 0.0
    evaluated = n_feas = 0
    while evaluated < BUDGET:
        b = min(BATCH, BUDGET - evaluated)
        X = [rng.uniform(LB, UB) for _ in range(b)]
        for v, fz in evaluate_batch(X):
            if fz:
                n_feas += 1
                best = max(best, v)
        evaluated += b
    return best, n_feas, time.perf_counter() - t0


def run_botorch(seed: int) -> tuple[float, int, float]:
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.generation import MaxPosteriorSampling
    from botorch.models import SingleTaskGP
    from botorch.utils.transforms import normalize, unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.double
    l_init, l_min, l_max = 0.8, 0.5 ** 7, 1.6
    tau_succ, tau_fail = 3, max(DIM, 5)
    succ = fail = 0
    side = l_init
    bounds = torch.tensor([LB, UB], dtype=dtype)
    n_init = min(N_INIT, BUDGET)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    X_all = rng.uniform(LB, UB, size=(n_init, DIM))
    res = evaluate_batch([x for x in X_all])
    Y_all = np.array([v if fz else 0.0 for v, fz in res])
    n_feas = sum(int(fz) for _, fz in res)
    bi = int(np.argmax(Y_all))
    best_x, best_y = X_all[bi].copy(), float(Y_all[bi])
    evaluated = n_init

    while evaluated < BUDGET:
        batch = min(BATCH, BUDGET - evaluated)
        tr_lb = np.clip(best_x - side / 2 * (UB - LB), LB, UB)
        tr_ub = np.clip(best_x + side / 2 * (UB - LB), LB, UB)
        X_t = torch.tensor(X_all, dtype=dtype)
        Y_t = torch.tensor(Y_all, dtype=dtype).unsqueeze(-1)
        gp = SingleTaskGP(normalize(X_t, bounds), (Y_t - Y_t.mean()) / (Y_t.std() + 1e-8))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        gp.eval()
        n_c = min(100 * DIM, 5000)
        cand = np.random.default_rng(seed + evaluated).uniform(tr_lb, tr_ub, (n_c, DIM))
        with torch.no_grad():
            X_next = MaxPosteriorSampling(model=gp, replacement=False)(
                normalize(torch.tensor(cand, dtype=dtype), bounds), num_samples=batch)
        new_x = np.atleast_2d(unnormalize(X_next, bounds).numpy())
        res = evaluate_batch([x for x in new_x])
        new_y = np.array([v if fz else 0.0 for v, fz in res])
        n_feas += sum(int(fz) for _, fz in res)
        X_all = np.vstack([X_all, new_x])
        Y_all = np.concatenate([Y_all, new_y])
        evaluated += batch
        prev = best_y
        nb = int(np.argmax(new_y))
        if new_y[nb] > best_y:
            best_y, best_x = float(new_y[nb]), new_x[nb].copy()
        if best_y > prev:
            succ += 1; fail = 0
        else:
            fail += 1; succ = 0
        if succ >= tau_succ:
            side = min(side * 2, l_max); succ = 0
        if fail >= tau_fail:
            side = max(side / 2, l_min); fail = 0
            bi = int(np.argmax(Y_all)); best_x, best_y = X_all[bi].copy(), float(Y_all[bi])
    return float(np.max(Y_all)), n_feas, time.perf_counter() - t0


def run_cma_es(seed: int) -> tuple[float, int, float]:
    import cma
    x0 = (LB + UB) / 2.0
    sigma0 = float(np.mean(UB - LB) / 6.0)
    t0 = time.perf_counter()
    evaluated = n_feas = 0
    best = 0.0
    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {"seed": seed, "bounds": [LB.tolist(), UB.tolist()],
         "maxfevals": BUDGET, "popsize": BATCH, "verbose": -9},
    )
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        while not es.stop() and evaluated < BUDGET:
            sols = es.ask()
            X = [np.clip(x, LB, UB) for x in sols]
            res = evaluate_batch(X)
            fits = []
            for v, fz in res:
                fits.append(-v if fz else 0.0)
                if fz:
                    n_feas += 1
                    best = max(best, v)
                evaluated += 1
            es.tell(sols, fits)
    return best, n_feas, time.perf_counter() - t0


RUNNERS = {
    "TRust-BO+P2": run_trust_bo,
    "BoTorch_TuRBO": run_botorch,
    "CMA-ES": run_cma_es,
    "Random": run_random,
}


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  SU2 RANS 翼型最適化ベンチマーク (Phase H-2)")
    print(f"  Ma=0.3 Re=3e6 SA, α={AOA}°, dim={DIM}D CST")
    print(f"  budget={BUDGET} batch={BATCH} workers={WORKERS} threads/job={NTHREAD} "
          f"iter={ITER} seeds={N_SEED}")
    print(f"  methods: {', '.join(METHODS)}")
    print("=" * 62)
    resume = CSV_PATH.exists()
    if resume:
        done = csv_load_done()
        print(f"  [resume] CSV found — skipping {len(done)} completed run(s)")
    else:
        csv_write_header()
        done = set()
    for method in METHODS:
        if method not in RUNNERS:
            print(f"[skip] unknown method: {method}")
            continue
        for seed in range(N_SEED):
            if (method, seed) in done:
                print(f"  {method:<14s} seed={seed} ... [already done, skip]")
                continue
            print(f"  {method:<14s} seed={seed} ... ", end="", flush=True)
            try:
                best, n_feas, elapsed = RUNNERS[method](seed)
                csv_append(method, seed, best, n_feas, elapsed)
                print(f"best Cl/Cd={best:.2f}  feasible={n_feas}/{BUDGET}  "
                      f"({elapsed/60:.1f}min)", flush=True)
            except Exception:
                tb = traceback.format_exc().replace("\n", " | ")
                csv_append(method, seed, 0.0, 0, 0.0)
                print(f"ERROR: {tb[:200]}", flush=True)
    print_summary()


def print_summary():
    from collections import defaultdict
    g = defaultdict(list)
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            with contextlib.suppress(ValueError):
                g[r["method"]].append(float(r["best_clcd"]))
    print("\n" + "=" * 62)
    print(f"  結果サマリ (Cl/Cd 最大化, budget={BUDGET})")
    print("=" * 62)
    print(f"{'Method':<15}{'median':>9}{'mean':>9}{'best':>9}{'N':>4}")
    for method in METHODS:
        v = g.get(method, [])
        if v:
            a = np.array(v)
            print(f"{method:<15}{np.median(a):>9.2f}{np.mean(a):>9.2f}{np.max(a):>9.2f}{len(a):>4}")
    print("=" * 62)
    print(f"CSV: {CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
