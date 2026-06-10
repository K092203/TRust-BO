"""benchmark_native.py — native Phase 2 (Rust) vs TandemEngineV2 (sklearn) vs TRust-BO。
Ackley 10D/50D, 3 seeds。出力: native_results.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "python"))
warnings.filterwarnings("ignore")
from trust_bo import Float, TRustBOEngine, TandemEngineV2

BATCH = 4
PROBLEMS = [("Ackley_10D", 10, 200), ("Ackley_50D", 50, 500)]
SEEDS = [0, 1, 2]


def ackley(x):
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


def run(engine, d, budget):
    ev = 0
    while ev < budget:
        cands = engine.ask(batch_size=min(BATCH, budget - ev))
        engine.tell(cands, [{"value": ackley(np.array([c[f"x{i}"] for i in range(d)])),
                             "feasible": True} for c in cands])
        ev += len(cands)
    return engine.best()["objective_values"][0]


def make(method, d, budget, seed):
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(d)]
    n_init = max(10, min(2 * (d + 1), 50))
    if method == "Native_P2":
        return TRustBOEngine(space=space, direction="minimize", seed=seed,
                             config={"n_init": n_init, "enable_phase2": True})
    if method == "Tandem_v2":
        return TandemEngineV2(space=space, direction="minimize", seed=seed,
                              budget=budget, phase1_ratio=0.8,
                              config={"n_init": n_init})
    return TRustBOEngine(space=space, direction="minimize", seed=seed,
                         config={"n_init": n_init})


out = open("native_results.csv", "w", buffering=1)
out.write("method,problem,seed,best_value,time_seconds\n")
total = len(PROBLEMS) * 3 * len(SEEDS)
i = 0
for pname, d, budget in PROBLEMS:
    for method in ["Native_P2", "Tandem_v2", "TRust-BO"]:
        for seed in SEEDS:
            i += 1
            t0 = time.time()
            best = run(make(method, d, budget, seed), d, budget)
            dt = time.time() - t0
            out.write(f"{method},{pname},{seed},{best:.6f},{dt:.2f}\n")
            print(f"[{i}/{total}] {method:10s} {pname:11s} seed={seed} "
                  f"best={best:.4f} ({dt:.0f}s)", flush=True)
out.close()
print("done")
