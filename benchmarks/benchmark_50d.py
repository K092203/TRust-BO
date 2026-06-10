#!/usr/bin/env python3
"""
benchmark_50d.py - 50D Ackley benchmark: TRust-BO vs Random vs GP

Compares optimization performance on 50D Ackley (optimum = 0.0) with budget = 1000.
GP is expected to time out due to O(n^3) complexity at this dimensionality.

Usage:
    python benchmark_50d.py
"""
import math
import signal
import statistics
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Optional

import numpy as np


# ── configuration ────────────────────────────────────────────────────────────
N_DIMS           = 50
BUDGET           = 1000
BATCH_TRM        = 10      # TRM proposals per round
SEEDS            = [42, 7, 13]
BOUNDS           = (-5.0, 5.0)
GP_STEP_TIMEOUT  = 60      # seconds: per-fit SIGALRM timeout
GP_TOTAL_TIMEOUT = 600     # seconds: 10-min hard cap per seed


# ── objective ─────────────────────────────────────────────────────────────────
def ackley(x) -> float:
    """50D Ackley.  Accepts dict or array-like.  Optimum = 0.0 at origin."""
    vals = list(x.values()) if isinstance(x, dict) else list(x)
    d = len(vals)
    term_a = -20.0 * math.exp(-0.2 * math.sqrt(sum(v ** 2 for v in vals) / d))
    term_b = -math.exp(sum(math.cos(2.0 * math.pi * v) for v in vals) / d)
    return term_a + term_b + 20.0 + math.e


# ── SIGALRM-based timeout (Linux only) ────────────────────────────────────────
class _StepTimeout(Exception):
    pass

@contextmanager
def _time_limit(seconds: int):
    def _handler(signum, frame):
        raise _StepTimeout()
    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# ── sampler: TRust-BO ───────────────────────────────────────────────────────
def run_trm(seed: int) -> tuple[float, float, list[float]]:
    """Returns (best_value, elapsed_sec, best_so_far_at_each_batch)."""
    from trust_bo import Float, TRustBOEngine

    space  = [Float(f"x{i}", BOUNDS[0], BOUNDS[1]) for i in range(N_DIMS)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed)

    t0 = time.perf_counter()
    evaluated = 0
    current_best = float("inf")
    batch_curve: list[float] = []

    while evaluated < BUDGET:
        batch = min(BATCH_TRM, BUDGET - evaluated)
        cands = engine.ask(batch_size=batch)
        vals  = [ackley(c) for c in cands]
        engine.tell(cands, [{"value": v, "feasible": True} for v in vals])
        evaluated  += batch
        current_best = min(current_best, min(vals))
        batch_curve.append(current_best)

    elapsed   = time.perf_counter() - t0
    best_val  = engine.best()["objective_values"][0]
    tr        = engine.tr_state()
    tr_info   = f"L={tr['side_length']:.4f}" if tr else "N/A"
    print(f"    best={best_val:.4f}  time={elapsed:.1f}s  TR({tr_info})")
    return best_val, elapsed, batch_curve


# ── sampler: Random ───────────────────────────────────────────────────────────
def run_random(seed: int) -> tuple[float, float, list[float]]:
    """Uniform random sampling — true baseline."""
    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()

    best = float("inf")
    curve: list[float] = []
    for _ in range(BUDGET):
        x    = rng.uniform(BOUNDS[0], BOUNDS[1], N_DIMS)
        val  = ackley(x)
        best = min(best, val)
        curve.append(best)

    elapsed = time.perf_counter() - t0
    print(f"    best={best:.4f}  time={elapsed:.3f}s")
    return best, elapsed, curve


# ── sampler: Gaussian Process ─────────────────────────────────────────────────
def run_gp(seed: int) -> tuple[Optional[float], float, int, list[float]]:
    """Sequential GP-EI with hard timeouts.

    Returns (best_value, elapsed_sec, n_evals_completed, best_so_far_curve).
    Times out per step (SIGALRM) and per seed (wall-clock cap).
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        from scipy.stats import norm as sp_norm
    except ImportError as exc:
        print(f"    [GP] dependency missing — {exc}")
        return None, 0.0, 0, []

    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()

    # Cold start: same n_init as TRM for 50D (= 50)
    n_init = 50
    X = rng.uniform(BOUNDS[0], BOUNDS[1], (n_init, N_DIMS))
    y = np.array([ackley(X[i]) for i in range(n_init)])
    best = float(y.min())
    curve: list[float] = list(y)   # per-eval curve for cold start

    # Isotropic Matérn 5/2 — ARD (length_scale per dim) is too slow for 50D
    kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-4, normalize_y=True,
        n_restarts_optimizer=0,   # skip hyperparameter re-optimization for speed
    )

    n_evals       = n_init
    stop_reason   = "budget_exhausted"

    while n_evals < BUDGET:
        # ── total wall-clock cap ────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        if elapsed > GP_TOTAL_TIMEOUT:
            stop_reason = f"total_timeout ({elapsed:.0f}s > {GP_TOTAL_TIMEOUT}s)"
            break

        # ── per-step SIGALRM timeout ────────────────────────────────────
        try:
            with _time_limit(GP_STEP_TIMEOUT):
                gp.fit(X, y)
                X_cand        = rng.uniform(BOUNDS[0], BOUNDS[1], (256, N_DIMS))
                mu, sigma     = gp.predict(X_cand, return_std=True)
                sigma         = np.maximum(sigma, 1e-8)
                z             = (best - mu) / sigma
                ei            = (best - mu) * sp_norm.cdf(z) + sigma * sp_norm.pdf(z)
                x_next        = X_cand[np.argmax(ei)]
        except _StepTimeout:
            elapsed     = time.perf_counter() - t0
            stop_reason = f"step_timeout (eval {n_evals}, {elapsed:.0f}s elapsed)"
            break

        y_next = float(ackley(x_next))
        X      = np.vstack([X, x_next])
        y      = np.append(y, y_next)
        best   = min(best, y_next)
        curve.append(best)
        n_evals += 1

        if n_evals % 50 == 0:
            print(f"    [GP] eval {n_evals:4d}  best={best:.4f}"
                  f"  elapsed={time.perf_counter()-t0:.1f}s")

    elapsed = time.perf_counter() - t0
    status  = (f"completed {n_evals}/{BUDGET} evals" if n_evals == BUDGET
               else f"stopped at eval {n_evals}: {stop_reason}")
    print(f"    [GP] {status}  best={best:.4f}  time={elapsed:.1f}s")
    return best, elapsed, n_evals, curve


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print()
    print("═" * 64)
    print(f"  50D Ackley Benchmark")
    print(f"  Dims={N_DIMS}  Budget={BUDGET}  Seeds={SEEDS}  Optimum=0.0")
    print(f"  GP timeouts: step={GP_STEP_TIMEOUT}s / total={GP_TOTAL_TIMEOUT}s")
    print("═" * 64)

    records: dict[str, dict] = {s: {} for s in SEEDS}

    for seed in SEEDS:
        print(f"\n── Seed {seed} {'─'*50}")

        print(f"  [TRust-BO] running {BUDGET} evals (batch={BATCH_TRM})...")
        trm_best, trm_time, _ = run_trm(seed)

        print(f"  [Random]     running {BUDGET} evals...")
        rnd_best, rnd_time, _ = run_random(seed)

        print(f"  [GP]         running (step_timeout={GP_STEP_TIMEOUT}s, total={GP_TOTAL_TIMEOUT}s)...")
        gp_best, gp_time, gp_evals, _ = run_gp(seed)

        records[seed] = {
            "trm":  {"best": trm_best, "time": trm_time, "evals": BUDGET},
            "rnd":  {"best": rnd_best, "time": rnd_time, "evals": BUDGET},
            "gp":   {"best": gp_best,  "time": gp_time,  "evals": gp_evals},
        }

    # ── summary ──────────────────────────────────────────────────────────────
    print()
    print("═" * 64)
    print("  RESULTS SUMMARY")
    print("═" * 64)

    col_w = 22
    print(f"{'':10}  {'TRust-BO':>{col_w}}  {'Random':>{col_w}}  {'GP (partial)':>{col_w}}")
    print("─" * 80)

    trm_bests, rnd_bests, gp_bests = [], [], []
    trm_times, rnd_times, gp_times = [], [], []

    for seed in SEEDS:
        r   = records[seed]
        tb  = r["trm"]["best"]
        rb  = r["rnd"]["best"]
        gb  = r["gp"]["best"]
        ge  = r["gp"]["evals"]

        trm_bests.append(tb); trm_times.append(r["trm"]["time"])
        rnd_bests.append(rb); rnd_times.append(r["rnd"]["time"])
        if gb is not None:
            gp_bests.append(gb)
        gp_times.append(r["gp"]["time"])

        gp_str = f"{gb:.4f} ({ge} ev)" if gb is not None else "timeout/skip"
        print(f"  seed={seed:<4}  {tb:>{col_w}.4f}  {rb:>{col_w}.4f}  {gp_str:>{col_w}}")

    print("─" * 80)

    trm_med = statistics.median(trm_bests)
    rnd_med = statistics.median(rnd_bests)
    gp_med  = statistics.median(gp_bests) if gp_bests else None

    trm_t   = statistics.mean(trm_times)
    rnd_t   = statistics.mean(rnd_times)
    gp_t    = statistics.mean(gp_times)

    def pct(a, b):
        return (b - a) / (abs(b) + 1e-9) * 100

    gp_med_str = f"{gp_med:.4f} ({sum(r['gp']['evals'] for r in records.values())//len(SEEDS)} ev avg)" \
                 if gp_med is not None else "N/A"
    print(f"  {'Median':10}  {trm_med:>{col_w}.4f}  {rnd_med:>{col_w}.4f}  {gp_med_str:>{col_w}}")
    print(f"  {'Wall-time':10}  {trm_t:>{col_w-1}.1f}s  {rnd_t:>{col_w-1}.3f}s  {gp_t:>{col_w-1}.1f}s")

    print()
    print(f"  TRM vs Random : {pct(trm_med, rnd_med):+.1f}%  (positive = TRM wins for minimization)")
    if gp_med is not None:
        gp_evals_mean = statistics.mean([r["gp"]["evals"] for r in records.values()])
        print(f"  GP  vs Random : {pct(gp_med, rnd_med):+.1f}%  "
              f"(completed ~{gp_evals_mean:.0f}/{BUDGET} evals)")

    print()
    print("  Notes:")
    print(f"    - TRM n_init = 50 (cold-start), then {BUDGET-50} warm evals with TR+surrogate")
    print(f"    - Random = pure uniform sampling, no model")
    print( "    - GP O(n³) complexity makes it impractical at budget=1000 / 50D")
    print("═" * 64)


if __name__ == "__main__":
    main()
