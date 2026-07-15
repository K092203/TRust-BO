"""goal2_su2_ab_benchmark.py -- Goal 2 の SU2 実測 A/B 専用ハーネス。

候補 6 (diverse CEM starts + joint batch selection)、候補 9 (MADS
poll)、および候補 1 単体を、Phase 2 + EI の同一ベースラインと paired
seed で比較する。既存の su2_cfd_benchmark.py は確立済みの他比較専用の
ため、このファイルからは変更しない。

環境変数:
  BUDGET=100 SEEDS=8 BATCH=8 WORKERS=8 NTHREAD=2 ITER=4000
  ARMS=baseline,c6c9,c6c9c1,c1only
  SMOKE=1  (budget=16, 1 seed, ITER=1500, *_smoke.jsonl へ出力)

実行例:
  SU2_RUN=/home/kotaro/su2/bin SU2_WORK=/home/kotaro/su2/work \
    SMOKE=1 .venv/bin/python benchmarks/goal2_su2_ab_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR / "su2"))
sys.path.insert(0, str(BENCHMARK_DIR.parent / "python"))

from su2_runner import SU2Settings, run_cst  # noqa: E402


# ── CST 探索空間 (su2_cfd_benchmark.py と同一) ───────────────────────────────

N_UPPER = 8
N_LOWER = 8
DIM = N_UPPER + N_LOWER

# 翼型らしい設計空間。既存の SU2 比較と完全に揃え、公平性を保つ。
UPPER_LB = np.full(N_UPPER, 0.05)
UPPER_UB = np.full(N_UPPER, 0.35)
LOWER_LB = np.full(N_LOWER, -0.35)
LOWER_UB = np.full(N_LOWER, 0.05)


# ── 実行設定 ─────────────────────────────────────────────────────────────────

AOA = 2.0
BUDGET = int(os.environ.get("BUDGET", "100"))
N_INIT = int(os.environ.get("NINIT", "12"))
BATCH = int(os.environ.get("BATCH", "8"))
WORKERS = int(os.environ.get("WORKERS", "8"))
NTHREAD = int(os.environ.get("NTHREAD", "2"))
ITER = int(os.environ.get("ITER", "4000"))
N_SEEDS = int(os.environ.get("SEEDS", "8"))
ARMS = tuple(
    arm.strip() for arm in os.environ.get(
        "ARMS", "baseline,c6c9,c6c9c1,c1only"
    ).split(",") if arm.strip()
)
RESULTS_PATH = BENCHMARK_DIR / "goal2_su2_ab_results.jsonl"

if os.environ.get("SMOKE"):
    BUDGET = 16
    N_SEEDS = 1
    BATCH = 8
    ITER = 1500
    RESULTS_PATH = BENCHMARK_DIR / "goal2_su2_ab_results_smoke.jsonl"

CHECKPOINT_EVERY = 10
_SETTINGS = SU2Settings(aoa=AOA, n_threads=NTHREAD, max_iter=ITER)
_EXECUTOR: ThreadPoolExecutor | None = None

# baseline では acquisition を明示する。engine.py のデフォルト ("ts") に依存しない。
BASE_CONFIG = {
    "n_init": N_INIT,
    "batch_size": BATCH,
    "enable_phase2": True,
    "acquisition": "ei",
    "phase2_early_frac": 0.25,
}
ARM_CONFIGS = {
    "baseline": BASE_CONFIG,
    "c6c9": {
        **BASE_CONFIG,
        "cem_diverse_starts": True,
        "joint_batch_select": True,
    },
    "c6c9c1": {
        **BASE_CONFIG,
        "cem_diverse_starts": True,
        "joint_batch_select": True,
        "enable_mads_poll": True,
    },
    "c1only": {
        **BASE_CONFIG,
        "enable_mads_poll": True,
    },
}


# ── 並列 SU2 評価 ─────────────────────────────────────────────────────────────

def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=WORKERS)
    return _EXECUTOR


def evaluate_one(params: np.ndarray) -> tuple[float, bool]:
    """CST 16D を SU2 RANS で評価し、(Cl/Cd, feasible) を返す。"""
    wu = params[:N_UPPER]
    wl = params[N_UPPER:]
    cl, cd, feasible, _ = run_cst(wu, wl, settings=_SETTINGS)
    if not feasible or cd <= 0:
        return 0.0, False
    return cl / cd, True


def evaluate_batch(points: list[np.ndarray]) -> list[tuple[float, bool]]:
    """バッチ内の SU2 評価を WORKERS 本で並列化する。"""
    return list(_executor().map(evaluate_one, points))


# ── JSON Lines の再開・記録 ───────────────────────────────────────────────────

def jsonl_load_done() -> set[tuple[str, int]]:
    """正常完了済みの (arm, seed) を返す。壊れた最終行は再実行対象とする。"""
    done: set[tuple[str, int]] = set()
    if not RESULTS_PATH.exists():
        return done
    with open(RESULTS_PATH) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                arm, seed = row["arm"], int(row["seed"])
                # 必須の完了指標をもつレコードだけを完了として扱う。
                if "final_best" in row and "eval_counts" in row and "best_so_far" in row:
                    done.add((arm, seed))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(f"[resume] ignoring malformed JSONL line {line_number}: {exc}")
    return done


def jsonl_append(record: dict) -> None:
    """1 arm × 1 seed の完了結果を追記する。"""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── TRust-BO の A/B 実行 ──────────────────────────────────────────────────────

def _make_space():
    from trust_bo import Float

    upper = [
        Float(f"u{i}", float(UPPER_LB[i]), float(UPPER_UB[i]))
        for i in range(N_UPPER)
    ]
    lower = [
        Float(f"l{i}", float(LOWER_LB[i]), float(LOWER_UB[i]))
        for i in range(N_LOWER)
    ]
    return upper + lower


def _to_cst(candidate: dict) -> np.ndarray:
    return np.array(
        [candidate[f"u{i}"] for i in range(N_UPPER)]
        + [candidate[f"l{i}"] for i in range(N_LOWER)],
        dtype=float,
    )


def _checkpoint_curve(per_evaluation_best: list[float]) -> tuple[list[int], list[float]]:
    """10 評価ごと（および最終評価）の best-so-far を抽出する。"""
    counts = list(range(CHECKPOINT_EVERY, len(per_evaluation_best) + 1, CHECKPOINT_EVERY))
    if len(per_evaluation_best) and (not counts or counts[-1] != len(per_evaluation_best)):
        counts.append(len(per_evaluation_best))
    return counts, [per_evaluation_best[count - 1] for count in counts]


def run_arm(arm: str, seed: int) -> dict:
    """指定アームを一つ実行し、比較に必要な全軌跡を返す。

    空間・n_init・seed は全 arm で共通であり、cold start 中の候補は Rust の
    Halton(seed) のみから決まる。そのため同一 seed の初期点列は paired である。
    """
    from trust_bo import TRustBOEngine

    config = ARM_CONFIGS[arm]
    engine = TRustBOEngine(
        space=_make_space(), direction="maximize", seed=seed, config=config,
    )
    started = time.perf_counter()
    evaluated = 0
    n_feasible = 0
    best = 0.0
    per_evaluation_best: list[float] = []

    while evaluated < BUDGET:
        batch_size = min(BATCH, BUDGET - evaluated)
        candidates = engine.ask(batch_size=batch_size)
        results = evaluate_batch([_to_cst(candidate) for candidate in candidates])
        engine.tell(candidates, [
            {"value": value, "feasible": feasible} for value, feasible in results
        ])
        for value, feasible in results:
            if feasible:
                n_feasible += 1
                best = max(best, float(value))
            per_evaluation_best.append(best)
        evaluated += batch_size

    eval_counts, best_so_far = _checkpoint_curve(per_evaluation_best)
    elapsed = time.perf_counter() - started
    return {
        "arm": arm,
        "seed": seed,
        "budget": BUDGET,
        "eval_counts": eval_counts,
        "best_so_far": best_so_far,
        "n_feasible": n_feasible,
        "elapsed_seconds": elapsed,
        "final_best": best,
    }


def main() -> None:
    unknown = [arm for arm in ARMS if arm not in ARM_CONFIGS]
    if unknown:
        raise ValueError(f"unknown ARMS entries: {', '.join(unknown)}")
    if not ARMS:
        raise ValueError("ARMS must contain at least one arm")
    if BUDGET <= 0 or BATCH <= 0 or N_SEEDS <= 0:
        raise ValueError("BUDGET, BATCH, and SEEDS must be positive")

    print("=" * 68)
    print("  Goal 2: SU2 RANS candidates 6 + 9 + 1 paired A/B")
    print(f"  Ma=0.3 Re=3e6 SA, alpha={AOA} deg, CST {DIM}D")
    print(f"  budget={BUDGET} batch={BATCH} workers={WORKERS} threads/job={NTHREAD} "
          f"iter={ITER} seeds={N_SEEDS}")
    print(f"  arms: {', '.join(ARMS)}")
    print(f"  output: {RESULTS_PATH}")
    print("=" * 68)

    done = jsonl_load_done()
    if done:
        print(f"  [resume] skipping {len(done)} completed arm/seed run(s)")

    try:
        for arm in ARMS:
            for seed in range(N_SEEDS):
                if (arm, seed) in done:
                    print(f"  {arm:<9s} seed={seed} ... [already done, skip]", flush=True)
                    continue
                print(f"  {arm:<9s} seed={seed} ... ", end="", flush=True)
                try:
                    record = run_arm(arm, seed)
                    jsonl_append(record)
                    print(
                        f"best Cl/Cd={record['final_best']:.2f}  "
                        f"feasible={record['n_feasible']}/{BUDGET}  "
                        f"({record['elapsed_seconds'] / 60:.1f}min)",
                        flush=True,
                    )
                except Exception:
                    # 失敗レコードを完了扱いで書かず、修正後の再開時に再試行できるようにする。
                    print(f"ERROR: {traceback.format_exc().replace(chr(10), ' | ')[:500]}", flush=True)
    finally:
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown(wait=True)


if __name__ == "__main__":
    main()
