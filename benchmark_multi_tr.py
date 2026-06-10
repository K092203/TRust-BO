#!/usr/bin/env python3
"""
benchmark_multi_tr.py -- TuRBO-M 有効性検証
50D Ackley: n_trs in {1, 2, 3, 5} vs Random
budget=400, seeds=5
"""
import math
import statistics
import time
from trust_bo import Float, TRustBOEngine

N_DIMS     = 50
BUDGET     = 400
BATCH      = 10
SEEDS      = [42, 7, 13, 1, 99]
BOUNDS     = (-5.0, 5.0)
N_TRS_LIST = [1, 2, 3, 5]

BASE_CONFIG = {
    "epochs": 300,
    "ensemble_size": 5,
    "n_cem_samples": 512,
    "n_cem_iters": 15,
    "acquisition": "ei",
    "tau_succ": 3,
    "tau_fail": 5,
    "l_max": 1.0,
    "l_min": 0.0078125,
}


def ackley(x: dict) -> float:
    vals = list(x.values())
    d = len(vals)
    a = -20.0 * math.exp(-0.2 * math.sqrt(sum(v ** 2 for v in vals) / d))
    b = -math.exp(sum(math.cos(2 * math.pi * v) for v in vals) / d)
    return a + b + 20.0 + math.e


def run_trm(n_trs: int, seed: int) -> tuple[float, list[float]]:
    space = [Float(f"x{i}", BOUNDS[0], BOUNDS[1]) for i in range(N_DIMS)]
    cfg = {**BASE_CONFIG, "n_trs": n_trs}
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed, config=cfg)
    best_curve: list[float] = []
    current_best = float("inf")
    for _ in range(BUDGET // BATCH):
        cands = engine.ask(batch_size=BATCH)
        vals = [ackley(c) for c in cands]
        engine.tell(cands, [{"value": v, "feasible": True} for v in vals])
        current_best = min(current_best, min(vals))
        best_curve.append(current_best)
    return engine.best()["objective_values"][0], best_curve


def run_random(seed: int) -> tuple[float, list[float]]:
    """pure random (cold-start only)"""
    space = [Float(f"x{i}", BOUNDS[0], BOUNDS[1]) for i in range(N_DIMS)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed,
                       config={"n_init": BUDGET + 1})
    best = float("inf")
    curve: list[float] = []
    for _ in range(BUDGET // BATCH):
        cands = engine.ask(batch_size=BATCH)
        vals = [ackley(c) for c in cands]
        engine.tell(cands, [{"value": v, "feasible": True} for v in vals])
        best = min(best, min(vals))
        curve.append(best)
    return best, curve


def pct_improvement(a: float, b: float) -> float:
    """(b - a) / |b| * 100.  正 = a が b より良い（最小化なので a < b が良い）"""
    return (b - a) / (abs(b) + 1e-9) * 100


def main():
    print(f"\n{'='*62}")
    print(f"  TuRBO-M Validation  |  50D Ackley  |  Budget={BUDGET}")
    print(f"  Seeds={SEEDS}  |  Optimum=0.0")
    print(f"{'='*62}\n")

    all_results: dict[str, list[float]] = {}

    # ── Random baseline ──────────────────────────────────────────────
    print("[ Random baseline ]")
    rnd_vals: list[float] = []
    t_start = time.perf_counter()
    for seed in SEEDS:
        v, _ = run_random(seed)
        rnd_vals.append(v)
        print(f"  seed={seed:3d}  best={v:.4f}")
    print(f"  → median={statistics.median(rnd_vals):.4f}  "
          f"({time.perf_counter()-t_start:.0f}s)\n")
    all_results["random"] = rnd_vals

    # ── TRM variants ─────────────────────────────────────────────────
    for n_trs in N_TRS_LIST:
        print(f"[ n_trs={n_trs} ]")
        vals: list[float] = []
        t_start = time.perf_counter()
        for seed in SEEDS:
            t0 = time.perf_counter()
            v, _ = run_trm(n_trs, seed)
            vals.append(v)
            elapsed = time.perf_counter() - t0
            print(f"  seed={seed:3d}  best={v:.4f}  ({elapsed:.0f}s)")
        med = statistics.median(vals)
        rnd_med = statistics.median(rnd_vals)
        pct_vs_rnd = pct_improvement(med, rnd_med)
        total = time.perf_counter() - t_start
        print(f"  → median={med:.4f}  vs Random: {pct_vs_rnd:+.1f}%  "
              f"({total:.0f}s total)\n")
        all_results[f"n_trs={n_trs}"] = vals

    # ── Summary ──────────────────────────────────────────────────────
    rnd_med  = statistics.median(all_results["random"])
    med1     = statistics.median(all_results["n_trs=1"])

    print(f"\n{'='*62}")
    print(f"  RESULTS SUMMARY  (50D Ackley, budget={BUDGET}, {len(SEEDS)} seeds)")
    print(f"{'='*62}")
    print(f"  {'Method':<13} {'Median':>8} {'vs Random':>11} {'vs n=1':>9}  "
          f"{'Min':>7}  {'Max':>7}")
    print(f"  {'-'*58}")
    print(f"  {'Random':<13} {rnd_med:>8.4f} {'(baseline)':>11}")

    for n_trs in N_TRS_LIST:
        k    = f"n_trs={n_trs}"
        vals = all_results[k]
        med  = statistics.median(vals)
        pvr  = pct_improvement(med, rnd_med)
        pv1  = pct_improvement(med, med1) if n_trs != 1 else 0.0
        tag  = f"{pv1:+.1f}%" if n_trs != 1 else "(base)"
        print(f"  {k:<13} {med:>8.4f} {pvr:>+10.1f}%  {tag:>9}  "
              f"{min(vals):>7.4f}  {max(vals):>7.4f}")

    print(f"\n  判定基準: n_trs>1 の vs n=1 が +3% 以上 → TuRBO-M 有効")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
