"""GP ベースライン比較テスト。

scikit-learn GaussianProcessRegressor (Matern 5/2 + EI) vs TRM Engine を
いくつかのベンチマーク関数で比較する。

目的: MLP アンサンブルサロゲートが GP と比べて competitive かどうかを検証する。

注: GP は O(n^3) なので大次元・大 budget では TRM の利点が大きい。
このテストは小~中次元 (5D, 10D) の小 budget でどちらが良いかを確認する。
"""
import math
import statistics

import numpy as np
import pytest
from scipy.stats import norm as scipy_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from trust_bo import Float, TRustBOEngine

pytestmark = pytest.mark.eval


# ------------------------------------------------------------------ helpers

def sphere(p: dict) -> float:
    return sum(v ** 2 for v in p.values())


def ackley(p: dict) -> float:
    x = list(p.values())
    d = len(x)
    a = -20 * math.exp(-0.2 * math.sqrt(sum(xi**2 for xi in x) / d))
    b = -math.exp(sum(math.cos(2 * math.pi * xi) for xi in x) / d)
    return a + b + 20 + math.e


def _halton(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Halton 準乱数列 (冷起動用)。"""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    rng = np.random.default_rng(seed)
    bases = primes[:d]
    pts = np.zeros((n, d))
    for j, b in enumerate(bases):
        i, f = 0, 1.0
        for ii in range(n):
            f /= b
            r = rng.random() * 0.01  # tiny jitter for diversity
            pts[ii, j] = (i * f + r) % 1.0
            i += 1
    # Properly generate Halton
    pts = np.zeros((n, d))
    for j, b in enumerate(bases):
        x = np.zeros(n)
        for ii in range(n):
            k, f, v = ii + 1, 1.0, 0.0
            while k > 0:
                f /= b
                v += f * (k % b)
                k //= b
            x[ii] = v
        pts[:, j] = x
    return pts  # [0,1]^d


def _ei_acquisition(gp: GaussianProcessRegressor, X_cand: np.ndarray, y_best: float) -> np.ndarray:
    """GP の予測から Expected Improvement を計算 (最小化方向 → −best で最大化)。"""
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    # 最小化: EI = E[max(f_best - f, 0)], f_best = -y_best (negated for maximization)
    imp = y_best - mu  # y_best は最小値 (最小化)
    z = imp / sigma
    ei = imp * scipy_norm.cdf(z) + sigma * scipy_norm.pdf(z)
    return np.maximum(ei, 0.0)


def run_gp(fn, n_dims, bounds_low, bounds_high, budget, seed):
    """GP-EI による最適化ループ (sequential)。"""
    rng = np.random.default_rng(seed)
    n_init = max(10, min(2 * (n_dims + 1), 50))

    # Cold start: Halton + bounds スケール
    X_init = _halton(n_init, n_dims, seed)
    scale = np.array([bounds_high - bounds_low] * n_dims)
    X_init = X_init * scale + bounds_low
    y_init = np.array([fn({f"x{i}": X_init[j, i] for i in range(n_dims)}) for j in range(n_init)])

    X_obs = X_init.copy()
    y_obs = y_init.copy()

    kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=2
    )

    for _ in range(budget - n_init):
        gp.fit(X_obs, y_obs)
        # ランダム候補から EI 最大点を選ぶ (最大化: negate for minimization)
        n_cand = 1024
        X_cand = rng.random((n_cand, n_dims)) * scale + bounds_low
        ei_vals = _ei_acquisition(gp, X_cand, y_obs.min())
        best_cand = X_cand[np.argmax(ei_vals)]
        y_new = fn({f"x{i}": best_cand[i] for i in range(n_dims)})
        X_obs = np.vstack([X_obs, best_cand])
        y_obs = np.append(y_obs, y_new)

    return float(y_obs.min())


def run_trm(fn, n_dims, bounds_low, bounds_high, budget, seed):
    """TRM Engine による最適化ループ (batch_size=1 で sequential に近似)。"""
    space = [Float(f"x{i}", bounds_low, bounds_high) for i in range(n_dims)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed)
    for _ in range(budget):
        cands = engine.ask(batch_size=1)
        engine.tell(cands, [fn(c) for c in cands])
    best = engine.best()
    return best["objective_values"][0] if best else float("inf")


# ------------------------------------------------------------------ tests

SEEDS = list(range(5))
BUDGET = 80


def _compare(fn_name, fn, n_dims, bounds_low, bounds_high):
    trm_vals, gp_vals = [], []
    for s in SEEDS:
        trm_v = run_trm(fn, n_dims, bounds_low, bounds_high, BUDGET, seed=s)
        gp_v = run_gp(fn, n_dims, bounds_low, bounds_high, BUDGET, seed=s)
        trm_vals.append(trm_v)
        gp_vals.append(gp_v)

    med_trm = statistics.median(trm_vals)
    med_gp = statistics.median(gp_vals)

    print(f"\n{fn_name} {n_dims}D  budget={BUDGET}  seeds={SEEDS}")
    print(f"  TRM  : median={med_trm:.4f}  vals={[round(v,4) for v in trm_vals]}")
    print(f"  GP   : median={med_gp:.4f}  vals={[round(v,4) for v in gp_vals]}")
    ratio = (med_gp - med_trm) / (abs(med_gp) + 1e-9) * 100
    print(f"  TRM vs GP: {ratio:+.1f}% (positive = TRM wins)")

    return med_trm, med_gp


def test_sphere_5d_vs_gp():
    """5D Sphere: TRM と GP の比較。GP は小次元で強いはずなので互角が目標。"""
    med_trm, med_gp = _compare("Sphere", sphere, 5, -5.0, 5.0)
    # TRM が GP の 3 倍以上悪くなければ合格 (競合的であることの確認)
    assert med_trm <= med_gp * 3 + 1.0, \
        f"TRM({med_trm:.4f}) much worse than GP({med_gp:.4f}) on sphere 5D"


def test_ackley_10d_vs_gp():
    """10D Ackley: 高次元では TRM (TR + CEM) の利点が出やすい。"""
    med_trm, med_gp = _compare("Ackley", ackley, 10, -5.0, 5.0)
    # TRM が GP の 1.5 倍以上悪くなければ合格
    assert med_trm <= med_gp * 1.5 + 1.0, \
        f"TRM({med_trm:.4f}) much worse than GP({med_gp:.4f}) on ackley 10D"
