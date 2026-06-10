#!/usr/bin/env python3
"""
benchmark_multi_tr_v2.py -- TuRBO-M 有効性検証 v2
問題ごとにn_trs={1,2,3}を比較:
  - Ackley  50D (unimodal-ish)   : n_trs=1 が有利と予想
  - Rastrigin 20D (真の多峰性)   : n_trs>1 が有利と予想
  - Levy    20D (中間的)         : どちらが有利か不明
"""
import math, statistics, time
from trust_bo import Float, TRustBOEngine

BOUNDS   = (-5.0, 5.0)
SEEDS    = [42, 7, 13, 1, 99]
BATCH    = 10
BASE_CONFIG = {
    "epochs": 300, "ensemble_size": 5,
    "n_cem_samples": 512, "n_cem_iters": 15,
    "acquisition": "ei",
    "tau_succ": 3, "tau_fail": 5,
    "l_max": 1.0, "l_min": 0.0078125,
}


def ackley(x: dict) -> float:
    v = list(x.values()); d = len(v)
    a = -20.0 * math.exp(-0.2 * math.sqrt(sum(xi**2 for xi in v) / d))
    b = -math.exp(sum(math.cos(2*math.pi*xi) for xi in v) / d)
    return a + b + 20.0 + math.e


def rastrigin(x: dict) -> float:
    v = list(x.values()); A = 10.0
    return A * len(v) + sum(xi**2 - A * math.cos(2*math.pi*xi) for xi in v)


def levy(x: dict) -> float:
    v = list(x.values())
    w = [1 + (xi - 1) / 4 for xi in v]
    term1 = math.sin(math.pi * w[0]) ** 2
    term2 = sum((w[i]-1)**2 * (1 + 10*math.sin(math.pi*w[i]+1)**2)
                for i in range(len(w)-1))
    term3 = (w[-1]-1)**2 * (1 + math.sin(2*math.pi*w[-1])**2)
    return term1 + term2 + term3


def run_trm(fn, n_dims, budget, n_trs, seed):
    space = [Float(f"x{i}", BOUNDS[0], BOUNDS[1]) for i in range(n_dims)]
    cfg = {**BASE_CONFIG, "n_trs": n_trs}
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed, config=cfg)
    for _ in range(budget // BATCH):
        cands = engine.ask(batch_size=BATCH)
        engine.tell(cands, [{"value": fn(c), "feasible": True} for c in cands])
    return engine.best()["objective_values"][0]


def run_random(fn, n_dims, budget, seed):
    space = [Float(f"x{i}", BOUNDS[0], BOUNDS[1]) for i in range(n_dims)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed,
                       config={"n_init": budget + 1})
    cands = engine.ask(batch_size=budget)
    engine.tell(cands, [{"value": fn(c), "feasible": True} for c in cands])
    return engine.best()["objective_values"][0]


PROBLEMS = [
    ("Ackley-50D",    ackley,    50, 400),
    ("Rastrigin-20D", rastrigin, 20, 400),
    ("Levy-20D",      levy,      20, 400),
]
N_TRS_LIST = [1, 2, 3]


def run_problem(name, fn, n_dims, budget):
    print(f"\n{'─'*56}")
    print(f"  {name}  (dims={n_dims}, budget={budget})")
    print(f"{'─'*56}")
    results = {}

    # Random
    rnd = [run_random(fn, n_dims, budget, s) for s in SEEDS]
    results["random"] = rnd
    print(f"  Random        median={statistics.median(rnd):.4f}")

    for n_trs in N_TRS_LIST:
        t0 = time.perf_counter()
        vals = [run_trm(fn, n_dims, budget, n_trs, s) for s in SEEDS]
        results[n_trs] = vals
        med = statistics.median(vals)
        rnd_med = statistics.median(rnd)
        med1 = statistics.median(results[1])
        pct_r = (rnd_med - med) / (abs(rnd_med)+1e-9) * 100
        pct_1 = (med1 - med) / (abs(med1)+1e-9) * 100 if n_trs != 1 else 0.0
        tag = f"vs n=1: {pct_1:+.1f}%" if n_trs != 1 else "(base)"
        print(f"  n_trs={n_trs}       median={med:.4f}  vs Rnd:{pct_r:+.1f}%  {tag}  "
              f"({time.perf_counter()-t0:.0f}s)")
    return results


def main():
    print(f"\n{'='*56}")
    print(f"  TuRBO-M Validation v2  |  seeds={len(SEEDS)}")
    print(f"  per_tr_min=1 (最小保証スロット)")
    print(f"{'='*56}")

    all_results = {}
    for problem in PROBLEMS:
        all_results[problem[0]] = run_problem(*problem)

    # Final verdict
    print(f"\n\n{'='*56}")
    print(f"  VERDICT")
    print(f"{'='*56}")
    for name, fn, n_dims, budget in PROBLEMS:
        r = all_results[name]
        med1 = statistics.median(r[1])
        print(f"\n  {name}:")
        for n_trs in [2, 3]:
            med_k = statistics.median(r[n_trs])
            pct = (med1 - med_k) / (abs(med1)+1e-9) * 100
            verdict = "✓ Multi-TR 有効" if pct > 2.0 else ("△ 同程度" if pct > -2.0 else "✗ 劣化")
            print(f"    n_trs={n_trs}: {pct:+.1f}%  {verdict}")
    print()


if __name__ == "__main__":
    main()
