"""
su2_mo_benchmark.py — Phase K-2-8: 実 CFD 多目的最適化（Cl 最大化 + Cd 最小化）

SU2 RANS を目的関数に EHVI (K-2) と Chebyshev (K-1) を比較する。
単目的 H-2 との違いは Cl/Cd の比ではなく Cl と Cd を別々の目的として扱う点。

目的:
  f1 = -Cl   (minimize = Cl 最大化)
  f2 =  Cd   (minimize)

評価は 16D CST → SU2 RANS。infeasible 形状は Pareto に含まない。

環境変数:
  BUDGET=60 SEEDS=2 BATCH=8 WORKERS=8 NTHREAD=2 ITER=2500 AOA=2.0
  METHODS=EHVI,Chebyshev
  CSV=su2_mo_results.csv
  SMOKE=1  (budget=16, 1 seed)

実行:
  cd /home/k0903/trm-engine
  .venv/bin/python benchmarks/su2_mo_benchmark.py
"""

from __future__ import annotations

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

# ── 設定 ─────────────────────────────────────────────────────────────────────

N_UPPER = 8
N_LOWER = 8
DIM = N_UPPER + N_LOWER

UPPER_LB = np.full(N_UPPER, 0.05)
UPPER_UB = np.full(N_UPPER, 0.35)
LOWER_LB = np.full(N_LOWER, -0.35)
LOWER_UB = np.full(N_LOWER, 0.05)
LB = np.concatenate([UPPER_LB, LOWER_LB])
UB = np.concatenate([UPPER_UB, LOWER_UB])

AOA     = float(os.environ.get("AOA", "2.0"))
BUDGET  = int(os.environ.get("BUDGET", "60"))
N_INIT  = int(os.environ.get("NINIT", "12"))
BATCH   = int(os.environ.get("BATCH", "8"))
WORKERS = int(os.environ.get("WORKERS", "8"))
NTHREAD = int(os.environ.get("NTHREAD", "2"))
ITER    = int(os.environ.get("ITER", "2500"))
N_SEED  = int(os.environ.get("SEEDS", "2"))
METHODS = os.environ.get("METHODS", "EHVI,Chebyshev").split(",")
CSV_PATH = Path(os.environ.get("CSV", "su2_mo_results.csv"))

# 超体積の参照点: (-Cl=0, Cd=0.05) = Cl=0 かつ Cd=5% は現実翼型より必ず悪い
HV_REF = np.array([0.0, 0.05])

if os.environ.get("SMOKE"):
    BUDGET = 16
    N_SEED = 1
    ITER = 1500
    CSV_PATH = Path("su2_mo_smoke.csv")

_SETTINGS = SU2Settings(aoa=AOA, n_threads=NTHREAD, max_iter=ITER)
_EXECUTOR_OBJ: ThreadPoolExecutor | None = None


# ── SU2 評価 ─────────────────────────────────────────────────────────────────

def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR_OBJ
    if _EXECUTOR_OBJ is None:
        _EXECUTOR_OBJ = ThreadPoolExecutor(max_workers=WORKERS)
    return _EXECUTOR_OBJ


def evaluate_one_mo(params: np.ndarray) -> tuple[float, float, bool]:
    """CST 16D → (-Cl, Cd, feasible)。"""
    wu, wl = params[:N_UPPER], params[N_UPPER:]
    cl, cd, feasible, _ = run_cst(wu, wl, settings=_SETTINGS)
    if not feasible or cd <= 1e-6 or not (np.isfinite(cl) and np.isfinite(cd)):
        return 0.0, 0.0, False
    return -cl, cd, True


def evaluate_batch_mo(X: list[np.ndarray]) -> list[tuple[float, float, bool]]:
    return list(_get_executor().map(evaluate_one_mo, X))


# ── CSV ──────────────────────────────────────────────────────────────────────

HEADER = ["method", "seed", "budget", "n_feasible", "hv", "total_seconds",
          "front_neg_cl", "front_cd"]


def csv_init():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)


def csv_load_done() -> set[tuple[str, int]]:
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


def csv_append(method: str, seed: int, n_feas: int, hv: float,
               elapsed: float, front: np.ndarray):
    neg_cl_str = ";".join(f"{v:.6f}" for v in front[:, 0])
    cd_str     = ";".join(f"{v:.6f}" for v in front[:, 1])
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [method, seed, BUDGET, n_feas, f"{hv:.6f}", f"{elapsed:.1f}",
             neg_cl_str, cd_str])


# ── 共通ユーティリティ ────────────────────────────────────────────────────────

def params_from_candidate(c: dict) -> np.ndarray:
    return np.array([c[f"u{i}"] for i in range(N_UPPER)]
                    + [c[f"l{i}"] for i in range(N_LOWER)])


def make_space():
    from trust_bo import Float
    space = [Float(f"u{i}", float(UPPER_LB[i]), float(UPPER_UB[i]))
             for i in range(N_UPPER)]
    space += [Float(f"l{i}", float(LOWER_LB[i]), float(LOWER_UB[i]))
              for i in range(N_LOWER)]
    return space


def pareto_front_2d(costs: np.ndarray) -> np.ndarray:
    """非支配解を返す（最小化、costs shape=(n,2)）。"""
    if len(costs) == 0:
        return np.empty((0, 2))
    idx = np.argsort(costs[:, 0])
    costs_s = costs[idx]
    front = [costs_s[0]]
    best_cd = costs_s[0, 1]
    for pt in costs_s[1:]:
        if pt[1] < best_cd:
            front.append(pt)
            best_cd = pt[1]
    return np.array(front)


def hypervolume_2d(front: np.ndarray, ref: np.ndarray) -> float:
    """2D 超体積（最小化）を掃引線で計算。"""
    if len(front) == 0:
        return 0.0
    pts = front[front[:, 0] < ref[0]]
    pts = pts[pts[:, 1] < ref[1]]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_cd = ref[1]
    for pt in pts:
        hv += (ref[0] - pt[0]) * (prev_cd - pt[1])
        prev_cd = pt[1]
    return hv


# ── EHVI ランナー ─────────────────────────────────────────────────────────────

def run_ehvi(seed: int) -> tuple[int, float, float, np.ndarray]:
    from trust_bo import MultiObjectiveEngine
    engine = MultiObjectiveEngine(
        space=make_space(),
        directions=["minimize", "minimize"],
        seed=seed,
        method="ehvi",
        config={"n_init": N_INIT, "enable_phase2": True, "batch_size": BATCH},
    )
    t0 = time.perf_counter()
    evaluated = n_feas = 0
    all_costs: list[np.ndarray] = []

    while evaluated < BUDGET:
        b = min(BATCH, BUDGET - evaluated)
        cands = engine.ask(batch_size=b)
        X = [params_from_candidate(c) for c in cands]
        res = evaluate_batch_mo(X)
        results = []
        for neg_cl, cd, fz in res:
            if fz:
                n_feas += 1
                all_costs.append(np.array([neg_cl, cd]))
            results.append({"values": [neg_cl, cd], "feasible": fz})
        engine.tell(cands, results)
        evaluated += b

    costs = np.array(all_costs) if all_costs else np.empty((0, 2))
    front = pareto_front_2d(costs)
    hv = hypervolume_2d(front, HV_REF)
    return n_feas, hv, time.perf_counter() - t0, front


# ── Chebyshev ランナー ────────────────────────────────────────────────────────

def run_chebyshev(seed: int) -> tuple[int, float, float, np.ndarray]:
    from trust_bo import MultiObjectiveEngine
    engine = MultiObjectiveEngine(
        space=make_space(),
        directions=["minimize", "minimize"],
        seed=seed,
        method="chebyshev",
        config={"n_init": N_INIT, "enable_phase2": True, "batch_size": BATCH},
    )
    t0 = time.perf_counter()
    evaluated = n_feas = 0
    all_costs: list[np.ndarray] = []

    while evaluated < BUDGET:
        b = min(BATCH, BUDGET - evaluated)
        cands = engine.ask(batch_size=b)
        X = [params_from_candidate(c) for c in cands]
        res = evaluate_batch_mo(X)
        results = []
        for neg_cl, cd, fz in res:
            if fz:
                n_feas += 1
                all_costs.append(np.array([neg_cl, cd]))
            results.append({"values": [neg_cl, cd], "feasible": fz})
        engine.tell(cands, results)
        evaluated += b

    costs = np.array(all_costs) if all_costs else np.empty((0, 2))
    front = pareto_front_2d(costs)
    hv = hypervolume_2d(front, HV_REF)
    return n_feas, hv, time.perf_counter() - t0, front


RUNNERS = {"EHVI": run_ehvi, "Chebyshev": run_chebyshev}


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  SU2 RANS 多目的翼型最適化ベンチマーク (Phase K-2-8)")
    print(f"  Ma=0.3 Re=3e6 SA, α={AOA}°, 目的: max Cl + min Cd")
    print(f"  dim={DIM}D CST, budget={BUDGET}, batch={BATCH}, "
          f"workers={WORKERS}, threads/job={NTHREAD}")
    print(f"  HV参照点: (-Cl={HV_REF[0]}, Cd={HV_REF[1]})")
    print(f"  seeds={N_SEED}, methods: {', '.join(METHODS)}")
    print("=" * 62)

    csv_init()
    done = csv_load_done()
    if done:
        print(f"  [resume] {len(done)} run(s) already done, skipping")

    for method in METHODS:
        if method not in RUNNERS:
            print(f"[skip] unknown method: {method}")
            continue
        for seed in range(N_SEED):
            if (method, seed) in done:
                print(f"  {method:<12s} seed={seed} ... [already done, skip]")
                continue
            print(f"  {method:<12s} seed={seed} ... ", end="", flush=True)
            try:
                n_feas, hv, elapsed, front = RUNNERS[method](seed)
                csv_append(method, seed, n_feas, hv, elapsed, front)
                print(f"HV={hv:.4f}  front={len(front)}pts  "
                      f"feasible={n_feas}/{BUDGET}  ({elapsed/60:.1f}min)")
            except Exception:
                tb = traceback.format_exc().replace("\n", " | ")
                print(f"ERROR: {tb[:200]}")

    print_summary()


def print_summary():
    from collections import defaultdict
    g: dict[str, list[float]] = defaultdict(list)
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            g[r["method"]].append(float(r["hv"]))
    print()
    print("=" * 62)
    print(f"  結果サマリ (Hypervolume, ref={HV_REF.tolist()}, budget={BUDGET})")
    print("=" * 62)
    print(f"{'Method':<14s} {'median HV':>10} {'mean HV':>10} {'best HV':>10}   N")
    for method in METHODS:
        vs = g.get(method, [])
        if not vs:
            continue
        print(f"{method:<14s} {np.median(vs):>10.4f} {np.mean(vs):>10.4f} "
              f"{max(vs):>10.4f}   {len(vs)}")
    print("=" * 62)
    print(f"CSV: {CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
