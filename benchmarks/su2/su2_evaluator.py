"""
su2_evaluator.py — Phase H-2-5: SU2 ローカル並列評価器 (JobEvaluator)

Phase G の RollingTRustBOEngine に接続する JobEvaluator 実装。
SU2 RANS は subprocess で走る(GIL を解放する)ため、ThreadPoolExecutor で
max_workers 件を並列実行する。SLURM 不要のローカル HPC 相当。

candidate dict は {u0..u{nu-1}, l0..l{nl-1}} の CST 重みを想定(H-1 と同じ)。
目的は Cl/Cd 最大化。メッシュ生成失敗・発散・非物理値は feasible=False。

使い方:
    from su2_evaluator import SU2LocalEvaluator
    from trust_bo import TRustBOEngine, RollingTRustBOEngine, Float

    ev = SU2LocalEvaluator(aoa=2.0, max_workers=8, n_threads_per_job=2)
    rolling = RollingTRustBOEngine(base_engine, ev, max_concurrent=8, poll_interval=2.0)
    rolling.run(budget=100)
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# RollingTRustBOEngine が要求する JobEvaluator 基底
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from trust_bo import JobEvaluator  # noqa: E402

from su2_runner import SU2Settings, run_cst  # noqa: E402

# 最適化探索空間: 上面 8 + 下面 8 = 16D CST。
# 翼型らしい範囲(全 seed で十分な実行可能率、一部 infeasible を残す)。
N_UPPER = 8
N_LOWER = 8
UPPER_LB, UPPER_UB = 0.05, 0.35
LOWER_LB, LOWER_UB = -0.35, 0.05

# infeasible 時に渡す目的値(maximize なので低い値)。engine 側で worst_feasible に
# 置換され主サロゲートには使われないが、意味的に低くしておく。
INFEASIBLE_VALUE = 0.0


def make_space():
    """16D CST 探索空間(trust_bo.Float のリスト)を返す。"""
    from trust_bo import Float
    space = [Float(f"u{i}", UPPER_LB, UPPER_UB) for i in range(N_UPPER)]
    space += [Float(f"l{i}", LOWER_LB, LOWER_UB) for i in range(N_LOWER)]
    return space


def candidate_to_weights(candidate: dict) -> tuple[np.ndarray, np.ndarray]:
    wu = np.array([candidate[f"u{i}"] for i in range(N_UPPER)])
    wl = np.array([candidate[f"l{i}"] for i in range(N_LOWER)])
    return wu, wl


class SU2LocalEvaluator(JobEvaluator):
    """SU2 RANS をローカル並列実行する JobEvaluator。Cl/Cd 最大化。"""

    def __init__(self, aoa: float = 2.0, max_workers: int = 8,
                 n_threads_per_job: int = 2, settings: SU2Settings | None = None,
                 verbose: bool = False) -> None:
        self._aoa = aoa
        self._verbose = verbose
        self._settings = settings or SU2Settings(
            aoa=aoa, n_threads=n_threads_per_job,
        )
        self._settings.aoa = aoa
        self._settings.n_threads = n_threads_per_job
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[str, Future] = {}
        self._counter = 0

    def submit(self, candidate: dict) -> str:
        jid = f"su2-{self._counter:05d}"
        self._counter += 1
        wu, wl = candidate_to_weights(candidate)
        fut = self._executor.submit(self._evaluate, wu, wl)
        self._futures[jid] = fut
        return jid

    def _evaluate(self, wu, wl) -> tuple[float, bool, dict]:
        cl, cd, feasible, info = run_cst(wu, wl, settings=self._settings)
        if not feasible or cd <= 0:
            return INFEASIBLE_VALUE, False, info
        return cl / cd, True, info

    def poll(self) -> list[tuple[str, float, bool]]:
        done: list[tuple[str, float, bool]] = []
        for jid, fut in list(self._futures.items()):
            if not fut.done():
                continue
            self._futures.pop(jid)
            try:
                value, feasible, info = fut.result()
            except Exception as e:  # noqa: BLE001
                if self._verbose:
                    print(f"  [su2 {jid}] exception: {e}")
                done.append((jid, INFEASIBLE_VALUE, False))
                continue
            if self._verbose:
                tag = "OK" if feasible else f"INFEAS({info.get('error')})"
                print(f"  [su2 {jid}] Cl/Cd={value:.2f} {tag} "
                      f"({info.get('elapsed_s', 0):.0f}s)")
            done.append((jid, float(value), bool(feasible)))
        return done

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    # 小規模ローリング実行のスモークテスト
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
    from trust_bo import RollingTRustBOEngine, TRustBOEngine

    budget = int(os.environ.get("BUDGET", "12"))
    mc = int(os.environ.get("CONCURRENT", "4"))
    nt = int(os.environ.get("NTHREAD", "3"))
    iters = int(os.environ.get("ITER", "1500"))

    st = SU2Settings(aoa=2.0, n_threads=nt, max_iter=iters)
    ev = SU2LocalEvaluator(aoa=2.0, max_workers=mc, n_threads_per_job=nt,
                           settings=st, verbose=True)
    base = TRustBOEngine(
        space=make_space(), direction="maximize", seed=0,
        config={"n_init": 8, "enable_phase2": True, "batch_size": mc},
    )
    rolling = RollingTRustBOEngine(
        base_engine=base, evaluator=ev, max_concurrent=mc,
        poll_interval=2.0, verbose=True,
    )
    print(f"=== SU2 rolling smoke: budget={budget} concurrent={mc} "
          f"threads/job={nt} iter={iters} ===")
    best = rolling.run(budget=budget)
    ev.shutdown()
    print("best:", best)
