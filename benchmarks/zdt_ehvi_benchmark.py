"""
zdt_ehvi_benchmark.py — Phase K-2 検証ベンチマーク (2D EHVI)

2D EHVI (Rust コア) を以下と比較する:
  - TRust-BO EHVI       (K-2、Rust propose_mo)
  - TRust-BO Chebyshev  (K-1、ask-time re-scalarization)
  - NSGA-II             (pymoo、進化的多目的の標準手法)
  - Random              (一様乱数)

2 つの問題系で評価する:
  (A) 低次元の滑らかな ZDT (n_var=5): NSGA-II が得意なゾーン。
      MLP アンサンブルの不確実性が低次元では校正不足で、EHVI は不利になりやすい。
  (B) 高次元 2-sphere (D=10/20/30): 本エンジンの設計ターゲット。
      予算制約下では NSGA-II の集団探索が次元の呪いで失速し、EHVI が有利になる。

指標: 評価済み全点の累積 Pareto フロントの超体積 (固定参照点)。
      手法間で完全に同一の hypervolume_2d + 参照点を用いる。

合格基準 (K-2):
  - EHVI が全問題で K-1 Chebyshev を上回る（エンジン内比較）。
  - 高次元 (D>=20) で EHVI が NSGA-II 以上（設計テーゼ）。

使い方:
  cd /home/k0903/trm-engine
  SMOKE=1 .venv/bin/python benchmarks/zdt_ehvi_benchmark.py   # ~1分
  .venv/bin/python benchmarks/zdt_ehvi_benchmark.py            # ~8分
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "python"))

from trust_bo import Float, MultiObjectiveEngine, hypervolume_2d
from trust_bo.multiobjective import _pareto_mask

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.problems import get_problem

# ── 設定 ──────────────────────────────────────────────────────────────────────

SMOKE   = bool(os.environ.get("SMOKE"))
BUDGET  = 40 if SMOKE else 120
BATCH   = 4
SEEDS   = [0] if SMOKE else [0, 1, 2]
EPOCHS  = 120 if SMOKE else 200
N_INIT  = 10
METHODS = ["EHVI", "Chebyshev", "NSGA-II", "Random"]


# ── 問題定義 ──────────────────────────────────────────────────────────────────

@dataclass
class MoProblem:
    name: str
    n_var: int
    ref: np.ndarray
    family: str                          # "lowdim" | "highdim"
    eval_fn: Callable[[np.ndarray], tuple[float, float]]   # x(D,) → (f1,f2)
    pymoo_factory: Callable[[], Problem]  # NSGA-II 用 pymoo 問題


def _zdt_eval(name: str, n_var: int):
    p = get_problem(name, n_var=n_var)
    def f(x: np.ndarray) -> tuple[float, float]:
        F = p.evaluate(x.reshape(1, -1))
        return float(F[0, 0]), float(F[0, 1])
    return f


class _RecordingProblem(Problem):
    """評価のたびに F をアーカイブへ追記する pymoo ラッパー。"""

    def __init__(self, n_var, evaluate_batch, archive):
        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0)
        self._evaluate_batch = evaluate_batch
        self._archive = archive

    def _evaluate(self, X, out, *args, **kwargs):
        F = self._evaluate_batch(X)
        out["F"] = F
        for row in np.atleast_2d(F):
            self._archive.append([float(row[0]), float(row[1])])


def _zdt_batch(name: str, n_var: int):
    p = get_problem(name, n_var=n_var)
    return lambda X: p.evaluate(X)


def _sphere2_batch(X: np.ndarray) -> np.ndarray:
    """2-sphere 双目的: f1=mean((x-0.2)^2), f2=mean((x-0.8)^2)。"""
    f1 = np.mean((X - 0.2) ** 2, axis=1)
    f2 = np.mean((X - 0.8) ** 2, axis=1)
    return np.column_stack([f1, f2])


def build_problems() -> list[MoProblem]:
    probs: list[MoProblem] = []
    # (A) 低次元 ZDT (n_var=5)、参照点は到達領域を覆う緩めの値
    zdt_names = ["zdt1"] if SMOKE else ["zdt1", "zdt2", "zdt3"]
    for nm in zdt_names:
        batch = _zdt_batch(nm, 5)
        probs.append(MoProblem(
            name=f"{nm}_5d", n_var=5, ref=np.array([1.1, 2.0]), family="lowdim",
            eval_fn=_zdt_eval(nm, 5),
            pymoo_factory=(lambda archive, bf=batch: _RecordingProblem(5, bf, archive)),
        ))
    # (B) 高次元 2-sphere、参照点 (0.4, 0.4)
    dims = [20] if SMOKE else [10, 20, 30]
    for d in dims:
        probs.append(MoProblem(
            name=f"sphere2_{d}d", n_var=d, ref=np.array([0.4, 0.4]), family="highdim",
            eval_fn=(lambda x: (float(np.mean((x - 0.2) ** 2)), float(np.mean((x - 0.8) ** 2)))),
            pymoo_factory=(lambda archive, nv=d: _RecordingProblem(nv, _sphere2_batch, archive)),
        ))
    return probs


# ── 共通ユーティリティ ────────────────────────────────────────────────────────

def hv_of(archive: list, ref: np.ndarray) -> float:
    if not archive:
        return 0.0
    costs = np.asarray(archive, dtype=float)
    mask = _pareto_mask(costs)
    return hypervolume_2d(costs[mask], ref)


# ── 各手法 ────────────────────────────────────────────────────────────────────

def run_trustbo(method: str, prob: MoProblem, seed: int) -> tuple[list, float]:
    space = [Float(f"x{i}", 0.0, 1.0) for i in range(prob.n_var)]
    engine = MultiObjectiveEngine(
        space=space, directions=["minimize", "minimize"],
        seed=seed, method=method, config={"n_init": N_INIT, "epochs": EPOCHS},
    )
    archive: list = []
    evaluated = 0
    t0 = time.perf_counter()
    while evaluated < BUDGET:
        b = min(BATCH, BUDGET - evaluated)
        cands = engine.ask(batch_size=b)
        results = []
        for c in cands:
            x = np.array([c[f"x{i}"] for i in range(prob.n_var)])
            f1, f2 = prob.eval_fn(x)
            archive.append([f1, f2])
            results.append({"values": [f1, f2], "feasible": True})
        engine.tell(cands, results)
        evaluated += b
    return archive, time.perf_counter() - t0


def run_random(prob: MoProblem, seed: int) -> tuple[list, float]:
    rng = np.random.default_rng(seed)
    archive: list = []
    t0 = time.perf_counter()
    for _ in range(BUDGET):
        x = rng.random(prob.n_var)
        f1, f2 = prob.eval_fn(x)
        archive.append([f1, f2])
    return archive, time.perf_counter() - t0


def run_nsga2(prob: MoProblem, seed: int) -> tuple[list, float]:
    archive: list = []
    rec = prob.pymoo_factory(archive)
    pop = max(8, min(20, BUDGET // 5))
    t0 = time.perf_counter()
    minimize(rec, NSGA2(pop_size=pop), ("n_evals", BUDGET), seed=seed, verbose=False)
    return archive[:BUDGET], time.perf_counter() - t0


def run_one(method: str, prob: MoProblem, seed: int):
    if method == "EHVI":
        return run_trustbo("ehvi", prob, seed)
    if method == "Chebyshev":
        return run_trustbo("chebyshev", prob, seed)
    if method == "NSGA-II":
        return run_nsga2(prob, seed)
    if method == "Random":
        return run_random(prob, seed)
    raise ValueError(method)


# ── エントリポイント ──────────────────────────────────────────────────────────

def main() -> int:
    problems = build_problems()
    print("=" * 74)
    print("  Phase K-2 検証: 2D EHVI vs NSGA-II vs Chebyshev(K-1) vs Random")
    print(f"  budget={BUDGET}  batch={BATCH}  seeds={SEEDS}  epochs={EPOCHS}  SMOKE={SMOKE}")
    print("=" * 74)

    summary: dict[str, dict[str, tuple[float, float]]] = {}
    families: dict[str, str] = {}
    cheby_pass = []
    highdim_pass = []

    for prob in problems:
        families[prob.name] = prob.family
        print(f"\n[{prob.name}]  family={prob.family}  n_var={prob.n_var}  ref={prob.ref.tolist()}")
        hv_acc: dict[str, list] = {m: [] for m in METHODS}
        t_acc: dict[str, list] = {m: [] for m in METHODS}
        for seed in SEEDS:
            row = []
            for m in METHODS:
                archive, elapsed = run_one(m, prob, seed)
                hv = hv_of(archive, prob.ref)
                hv_acc[m].append(hv)
                t_acc[m].append(elapsed)
                row.append(f"{m}={hv:.4f}({elapsed:.0f}s)")
            print(f"  seed={seed}  " + "  ".join(row))

        summary[prob.name] = {
            m: (float(np.median(hv_acc[m])), float(np.mean(t_acc[m]))) for m in METHODS
        }
        ehvi = summary[prob.name]["EHVI"][0]
        cheby = summary[prob.name]["Chebyshev"][0]
        nsga = summary[prob.name]["NSGA-II"][0]
        cheby_pass.append(ehvi >= cheby)
        if prob.family == "highdim" and prob.n_var >= 20:
            highdim_pass.append(ehvi >= nsga)
        print(f"  → median HV: EHVI={ehvi:.4f}  NSGA-II={nsga:.4f}  Chebyshev={cheby:.4f}")

    # サマリ表
    print("\n" + "=" * 74)
    print("  median HV サマリ (higher is better)")
    print("=" * 74)
    print(f"  {'problem':<14}" + "".join(f"{m:>13}" for m in METHODS))
    for name in summary:
        cells = "".join(f"{summary[name][m][0]:>13.4f}" for m in METHODS)
        print(f"  {name:<14}{cells}")

    print("\n  平均実行時間 (s/run)")
    print(f"  {'problem':<14}" + "".join(f"{m:>13}" for m in METHODS))
    for name in summary:
        cells = "".join(f"{summary[name][m][1]:>13.1f}" for m in METHODS)
        print(f"  {name:<14}{cells}")

    ok_cheby = all(cheby_pass)
    ok_highdim = all(highdim_pass) if highdim_pass else True
    print("\n" + "=" * 74)
    print(f"  判定1 EHVI >= Chebyshev (全問題):       {'PASS' if ok_cheby else 'FAIL'}")
    print(f"  判定2 EHVI >= NSGA-II (高次元 D>=20):    {'PASS' if ok_highdim else 'FAIL'}")
    print(f"\n  総合: {'ALL PASSED' if (ok_cheby and ok_highdim) else 'SOME FAILED'}")
    print("=" * 74)
    return 0 if (ok_cheby and ok_highdim) else 1


if __name__ == "__main__":
    sys.exit(main())
