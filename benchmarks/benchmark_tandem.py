"""benchmark_tandem.py — TandemEngine vs TRust-BO vs HEBO vs Random

問題: Ackley 10D/50D/100D
seeds: 0-4
出力: tandem_results.csv
"""
from __future__ import annotations
import csv, math, time, traceback
from pathlib import Path
import numpy as np

CSV_PATH = Path("tandem_results.csv")
SEEDS = [0, 1, 2, 3, 4]
BATCH = 4

PROBLEMS = [
    ("Ackley_10D",  10,  200),
    ("Ackley_50D",  50,  500),
    ("Ackley_100D", 100, 500),
]


def ackley(x: np.ndarray) -> float:
    d = len(x)
    return (-20.0 * math.exp(-0.2 * math.sqrt(np.sum(x**2) / d))
            - math.exp(np.sum(np.cos(2 * math.pi * x)) / d)
            + 20.0 + math.e)


def calc_n_init(n_dims: int, budget: int) -> int:
    default = max(10, min(2 * (n_dims + 1), 50))
    return min(default, max(10, budget - 10 * BATCH))


def load_done() -> set[tuple]:
    done: set[tuple] = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["method"], r["problem"], r["seed"]))
    return done


def csv_init():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["method", "problem", "seed", "best_value", "time_seconds", "phase_switch"])


def csv_append(method, problem, seed, best_value, elapsed, phase_switch="N/A"):
    with open(CSV_PATH, "a", newline="", buffering=1) as f:
        csv.writer(f).writerow(
            [method, problem, seed, f"{best_value:.6f}", f"{elapsed:.2f}", phase_switch])


# ── runners ──────────────────────────────────────────────────────────────────

def run_tandem(n_dims, budget, seed):
    from trust_bo import TandemEngine, Float
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    n_init = calc_n_init(n_dims, budget)
    engine = TandemEngine(space=space, direction="minimize", seed=seed,
                          budget=budget, phase1_ratio=0.8,
                          config={"n_init": n_init})
    t0 = time.perf_counter()
    ev = 0
    phase_switch = -1
    while ev < budget:
        b = min(BATCH, budget - ev)
        cands = engine.ask(batch_size=b)
        engine.tell(cands, [
            {"value": ackley(np.array([c[f"x{i}"] for i in range(n_dims)])),
             "feasible": True} for c in cands])
        ev += len(cands)  # use actual count to avoid divergence with TRustBOEngine init batches
        if engine.phase == 2 and phase_switch < 0:
            phase_switch = ev
    best = engine.best()
    return (best["objective_values"][0] if best else float("inf"),
            time.perf_counter() - t0, phase_switch)


def run_trust_bo(n_dims, budget, seed):
    from trust_bo import TRustBOEngine, Float
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    n_init = calc_n_init(n_dims, budget)
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed,
                           config={"n_init": n_init})
    t0 = time.perf_counter()
    ev = 0
    while ev < budget:
        b = min(BATCH, budget - ev)
        cands = engine.ask(batch_size=b)
        engine.tell(cands, [
            {"value": ackley(np.array([c[f"x{i}"] for i in range(n_dims)])),
             "feasible": True} for c in cands])
        ev += b
    best = engine.best()
    return (best["objective_values"][0] if best else float("inf"),
            time.perf_counter() - t0)


def run_hebo(n_dims, budget, seed):
    import pandas as pd
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
    n_init = calc_n_init(n_dims, budget)
    space_params = [{"name": f"x{i}", "type": "num", "lb": -5.0, "ub": 5.0}
                    for i in range(n_dims)]
    space = DesignSpace().parse(space_params)
    opt = HEBO(space, rand_sample=n_init, scramble_seed=seed)
    t0 = time.perf_counter()
    ev = 0
    while ev < budget:
        b = min(BATCH, budget - ev)
        rec = opt.suggest(n_suggestions=b)
        y = np.array([[ackley(rec.iloc[i][[f"x{j}" for j in range(n_dims)]].values)]
                      for i in range(len(rec))])
        opt.observe(rec, y)
        ev += b
    return float(opt.y.min()), time.perf_counter() - t0


def run_random(n_dims, budget, seed):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best_y = float("inf")
    ev = 0
    while ev < budget:
        b = min(BATCH, budget - ev)
        for x in rng.uniform(-5.0, 5.0, (b, n_dims)):
            best_y = min(best_y, ackley(x))
        ev += b
    return best_y, time.perf_counter() - t0


RUNNERS = {
    "TandemEngine": run_tandem,
    "TRust-BO":     run_trust_bo,
    "HEBO":         run_hebo,
    "Random":       run_random,
}

HEBO_MAX_BUDGET = 200

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_init()
    done = load_done()
    if done:
        print(f"Resume: {len(done)} rows already done.")

    total = len(RUNNERS) * len(PROBLEMS) * len(SEEDS)
    idx = 0

    for prob_name, n_dims, budget in PROBLEMS:
        for method, runner in RUNNERS.items():
            for seed in SEEDS:
                idx += 1
                key = (method, prob_name, str(seed))
                tag = f"[{idx}/{total}] {method:<15} | {prob_name:<12} | seed={seed}"

                if key in done:
                    print(f"{tag} ... SKIP")
                    continue

                if method == "HEBO" and budget > HEBO_MAX_BUDGET:
                    print(f"{tag} ... SKIP (HEBO too slow)")
                    csv_append(method, prob_name, seed, float("nan"), 0.0, "too_slow")
                    continue

                print(f"{tag} ... ", end="", flush=True)
                try:
                    if method == "TandemEngine":
                        best_val, elapsed, ps = runner(n_dims, budget, seed)
                        csv_append(method, prob_name, seed, best_val, elapsed, ps)
                        print(f"best={best_val:.4f}  phase_switch={ps}  ({elapsed:.1f}s)")
                    else:
                        result = runner(n_dims, budget, seed)
                        best_val, elapsed = result[0], result[1]
                        csv_append(method, prob_name, seed, best_val, elapsed)
                        print(f"best={best_val:.4f}  ({elapsed:.1f}s)")
                except Exception:
                    tb = traceback.format_exc().replace("\n", " | ")
                    csv_append(method, prob_name, seed, float("nan"), 0.0, "error")
                    print(f"ERROR: {tb[:100]}")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({**r, "v": float(r["best_value"])})
            except ValueError:
                pass

    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], r["problem"])].append(r["v"])

    print(f"{'Method':<18} {'Problem':<14} {'Median':>8} {'Min':>8} N")
    for prob_name, n_dims, budget in PROBLEMS:
        for method in RUNNERS:
            vals = g.get((method, prob_name), [])
            if not vals:
                print(f"{method:<18} {prob_name:<14} {'N/A':>8}")
            else:
                arr = np.array(vals)
                print(f"{method:<18} {prob_name:<14} {np.median(arr):>8.3f} {np.min(arr):>8.3f} {len(arr)}")
        print()

    print(f"CSV: {CSV_PATH.resolve()}")
