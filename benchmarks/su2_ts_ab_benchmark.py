"""SU2 実RANS での acquisition A/B (ts vs ei)。

評価パイプライン(CST 16D → SU2 RANS、WORKERS 並列)と設計空間は
su2_cfd_benchmark.py をそのまま流用する。環境変数も同一
(BUDGET=100 NINIT=12 BATCH=8 WORKERS=8 NTHREAD=2 ITER=4000 AOA=2.0 SEEDS=3)。
追加: ARMS (デフォルト "ts,ei")。SMOKE=1 で BUDGET=16・1 シード。

実行例 (本走は数時間、バックグラウンド推奨):
  SU2_RUN=$HOME/su2/bin SU2_WORK=$HOME/su2/work \
      .venv/bin/python benchmarks/su2_ts_ab_benchmark.py
"""

from __future__ import annotations

import csv
import math
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

# su2_cfd_benchmark の import 時に環境変数(BUDGET等)が読まれる。
from su2_cfd_benchmark import (
    BATCH,
    BUDGET,
    LOWER_LB,
    LOWER_UB,
    N_INIT,
    N_LOWER,
    N_SEED,
    N_UPPER,
    UPPER_LB,
    UPPER_UB,
    evaluate_batch,
)

ARMS = tuple(os.environ.get("ARMS", "ts,ei").split(","))
# アーム名 → engine config 追加分。未登録名は acquisition 文字列としてそのまま扱う
ARM_CONFIGS: dict[str, dict] = {
    "ei_early": {"acquisition": "ei", "phase2_early_frac": 0.25},
    "ts_early": {"acquisition": "ts", "phase2_early_frac": 0.25},
}
SEEDS = range(1 if os.environ.get("SMOKE") else N_SEED)
CSV_PATH = Path(os.environ.get("AB_CSV", "su2_ts_ab_results.csv"))
if os.environ.get("SMOKE"):
    CSV_PATH = Path("su2_ts_ab_results_smoke.csv")

PHYSICAL_MAX = 300.0  # Cl/Cd がこれ以上は非物理アーティファクト扱い (BENCHMARK.md §12)


def csv_write_header() -> None:
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["arm", "budget", "seed", "best_clcd", "n_feasible", "total_seconds"])


def csv_append(arm, seed, best, n_feas, elapsed) -> None:
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [arm, BUDGET, seed, f"{best:.6f}", n_feas, f"{elapsed:.1f}"])


def run_arm(arm: str, seed: int) -> tuple[float, int, float]:
    from trust_bo import Float, TRustBOEngine

    space = [Float(f"u{i}", float(UPPER_LB[i]), float(UPPER_UB[i]))
             for i in range(N_UPPER)]
    space += [Float(f"l{i}", float(LOWER_LB[i]), float(LOWER_UB[i]))
              for i in range(N_LOWER)]
    arm_cfg = ARM_CONFIGS.get(arm, {"acquisition": arm})
    engine = TRustBOEngine(
        space=space, direction="maximize", seed=seed,
        config={"n_init": N_INIT, "enable_phase2": True,
                "batch_size": BATCH, **arm_cfg},
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
    return (best["objective_values"][0] if best else 0.0), n_feas, \
        time.perf_counter() - t0


def run_all() -> None:
    from bench_resume import is_done, resume_or_init

    done_keys = resume_or_init(CSV_PATH, ("arm", "seed"), csv_write_header)
    total = len(ARMS) * len(SEEDS)
    count = 0
    for arm in ARMS:
        for seed in SEEDS:
            count += 1
            tag = f"[{count}/{total}] {arm:<5s} | budget={BUDGET} | seed={seed}"
            if is_done(done_keys, arm, seed):
                print(f"{tag} ... [skip]")
                continue
            print(f"{tag} ... ", end="", flush=True)
            try:
                best, n_feas, elapsed = run_arm(arm, seed)
                csv_append(arm, seed, best, n_feas, elapsed)
                print(f"best Cl/Cd={best:.2f}  feas={n_feas}/{BUDGET}  "
                      f"({elapsed / 60:.1f}min)")
            except Exception:
                tb = traceback.format_exc().replace("\n", " | ")
                print(f"ERROR: {tb[:160]}", flush=True)


def print_summary() -> None:
    values: dict[int, dict[str, float]] = defaultdict(dict)
    if not CSV_PATH.exists():
        return
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                values[int(row["seed"])][row["arm"]] = float(row["best_clcd"])
            except (KeyError, ValueError):
                continue

    print("\n" + "=" * 64)
    print("  SU2 RANS acquisition A/B (best Cl/Cd; larger is better)")
    print("=" * 64)
    print(f"{'arm':<7} {'mean':>10} {'median':>10} {'physical(<300)':>15} {'N':>3}")
    print("-" * 64)
    for arm in ARMS:
        vals = [a[arm] for a in values.values() if arm in a]
        if vals:
            n_phys = sum(1 for v in vals if v < PHYSICAL_MAX)
            print(f"{arm:<7} {np.mean(vals):>10.2f} {np.median(vals):>10.2f} "
                  f"{n_phys:>11}/{len(vals):<3} {len(vals):>3}")

    if "ts" in ARMS and "ei" in ARMS:
        ratios = [max(a["ts"], 1e-12) / max(a["ei"], 1e-12)
                  for a in values.values() if "ts" in a and "ei" in a]
        if ratios:
            gm = math.exp(float(np.mean(np.log(ratios))))
            wins = sum(1 for r in ratios if r > 1.0)
            print(f"\n  paired ts/ei GM ratio = {gm:.4f} "
                  f"(>1 favors ts), ts wins {wins}/{len(ratios)}")
    print("=" * 64)


if __name__ == "__main__":
    print("=" * 64)
    print("  SU2 RANS ts-vs-ei A/B")
    print(f"  arms={ARMS}  budget={BUDGET}  n_init={N_INIT}  batch={BATCH}  "
          f"seeds={list(SEEDS)}")
    print("=" * 64)
    run_all()
    print_summary()
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
