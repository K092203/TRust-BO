"""benchmark_tandem_v2.py — TandemEngine_v2 vs v1 vs TRust-BO vs HEBO vs Random

Usage:
    python benchmark_tandem_v2.py
Output: tandem_v2_results.csv
"""
from __future__ import annotations
import csv, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "python"))

CSV_PATH = Path("tandem_v2_results.csv")
SEEDS = [0, 1, 2, 3, 4]
BATCH = 4
HEBO_MAX_BUDGET = 200

PROBLEMS = [
    ("Ackley_10D",  10,  200),
    ("Ackley_50D",  50,  500),
    ("Ackley_100D", 100, 500),
]


def ackley(x: np.ndarray) -> float:
    n = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e
    )


def calc_n_init(n_dims: int, budget: int) -> int:
    default = max(10, min(2 * (n_dims + 1), 50))
    return min(default, max(10, budget - 10 * BATCH))


# ── CSV helpers ──────────────────────────────────────────────────────────────

def load_done() -> set[tuple]:
    done: set[tuple] = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["method"], r["problem"], r["seed"]))
    return done


def csv_append(method, problem, seed, best_val, elapsed, phase_switch="N/A"):
    with open(CSV_PATH, "a", newline="", buffering=1) as f:
        csv.writer(f).writerow(
            [method, problem, seed, f"{best_val:.6f}", f"{elapsed:.2f}", phase_switch]
        )


# ── runners ──────────────────────────────────────────────────────────────────

def _run_tandem(EngineClass, n_dims, budget, seed, label):
    from trust_bo import Float
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    n_init = calc_n_init(n_dims, budget)
    engine = EngineClass(space=space, direction="minimize", seed=seed,
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
        ev += len(cands)
        if engine.phase == 2 and phase_switch < 0:
            phase_switch = ev
    best = engine.best()
    return (best["objective_values"][0] if best else float("inf"),
            time.perf_counter() - t0, phase_switch)


def run_tandem_v1(n_dims, budget, seed):
    from trust_bo import TandemEngine
    return _run_tandem(TandemEngine, n_dims, budget, seed, "TandemEngine_v1")


def run_tandem_v2(n_dims, budget, seed):
    from trust_bo import TandemEngineV2
    return _run_tandem(TandemEngineV2, n_dims, budget, seed, "TandemEngine_v2")


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
        ev += len(cands)
    best = engine.best()
    return best["objective_values"][0] if best else float("inf"), time.perf_counter() - t0


def run_hebo(n_dims, budget, seed):
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
    space = DesignSpace().parse([
        {"name": f"x{i}", "type": "num", "lb": -5.0, "ub": 5.0}
        for i in range(n_dims)
    ])
    n_init = calc_n_init(n_dims, budget)
    opt = HEBO(space, rand_sample=n_init, scramble_seed=seed)
    t0 = time.perf_counter()
    for _ in range(budget):
        rec = opt.suggest(n_suggestions=1)
        x = np.array([rec[f"x{i}"].values[0] for i in range(n_dims)])
        opt.observe(rec, np.array([[ackley(x)]]))
    return float(opt.y.min()), time.perf_counter() - t0


def run_random(n_dims, budget, seed):
    rng = np.random.default_rng(seed)
    best = float("inf")
    for _ in range(budget):
        x = rng.uniform(-5.0, 5.0, size=n_dims)
        best = min(best, ackley(x))
    return best, 0.0


RUNNERS = {
    "TandemEngine_v2": run_tandem_v2,
    "TandemEngine_v1": run_tandem_v1,
    "TRust-BO":        run_trust_bo,
    "HEBO":            run_hebo,
    "Random":          run_random,
}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", buffering=1) as f:
            csv.writer(f).writerow(
                ["method", "problem", "seed", "best_value", "time_seconds", "phase_switch"])

    done = load_done()
    total = len(RUNNERS) * len(PROBLEMS) * len(SEEDS)
    idx = 0

    for prob_name, n_dims, budget in PROBLEMS:
        for method, runner in RUNNERS.items():
            for seed in SEEDS:
                idx += 1
                key = (method, prob_name, str(seed))
                tag = f"[{idx}/{total}] {method:<18} | {prob_name:<12} | seed={seed}"

                if key in done:
                    print(f"{tag} ... SKIP")
                    continue

                if method == "HEBO" and budget > HEBO_MAX_BUDGET:
                    print(f"{tag} ... SKIP (too_slow)")
                    csv_append(method, prob_name, seed, float("nan"), 0.0, "too_slow")
                    continue

                print(f"{tag} ... ", end="", flush=True)
                try:
                    if method in ("TandemEngine_v2", "TandemEngine_v1"):
                        best_val, elapsed, ps = runner(n_dims, budget, seed)
                        csv_append(method, prob_name, seed, best_val, elapsed, ps)
                        print(f"best={best_val:.4f}  ps={ps}  ({elapsed:.1f}s)")
                    else:
                        result = runner(n_dims, budget, seed)
                        best_val, elapsed = result[0], result[1]
                        csv_append(method, prob_name, seed, best_val, elapsed)
                        print(f"best={best_val:.4f}  ({elapsed:.1f}s)")
                except Exception:
                    tb = traceback.format_exc().replace("\n", " | ")
                    csv_append(method, prob_name, seed, float("nan"), 0.0, "error")
                    print(f"ERROR: {tb[:120]}")

    # ── summary ──────────────────────────────────────────────────────────────
    from collections import defaultdict
    g: dict[tuple, list[float]] = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            try:
                v = float(r["best_value"])
                if not np.isnan(v):
                    g[(r["method"], r["problem"])].append(v)
            except (ValueError, KeyError):
                pass

    methods = list(RUNNERS.keys())
    probs   = [p[0] for p in PROBLEMS]
    print("\n=== SUMMARY (median) ===")
    header = f"{'Method':<20}" + "".join(f"{p:>12}" for p in probs)
    print(header)
    print("-" * len(header))
    for m in methods:
        row = f"{m:<20}"
        for prob in probs:
            vals = g.get((m, prob), [])
            row += f"{np.median(vals):>12.3f}" if vals else f"{'N/A':>12}"
        print(row)

    print("\nDone. Results saved to:", CSV_PATH.resolve())


if __name__ == "__main__":
    main()
