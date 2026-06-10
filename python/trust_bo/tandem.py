"""TandemEngine: TRust-BO (Phase 1) + GP refine (Phase 2).

Phase 1: TRustBOEngine で budget × phase1_ratio を探索。
Phase 2: TR境界内で GP (Matern 5/2) + EI で精密化。

Classes:
    TandemEngine   - v1 (bug-fixed, basic GP EI)
    TandemEngineV2 - v2 (enhanced: WhiteKernel, random-then-refine EI)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from .engine import TRustBOEngine
from .space import ParamDef


# ── shared utility ──────────────────────────────────────────────────────────

def _ei(X: np.ndarray, gp, best_f: float, xi: float = 0.01) -> np.ndarray:
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - best_f - xi) / sigma
    return (mu - best_f - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def _build_tr_bounds(
    space_defs: list[ParamDef],
    bo: TRustBOEngine,
) -> list[tuple[float, float]]:
    """Return raw-space bounds clipped to the current trust region (or full space)."""
    tr = bo.tr_state()
    if tr is not None and tr.get("center") is not None:
        center = tr["center"]
        sl = tr["side_length"]
        raw = []
        for i, p in enumerate(space_defs):
            lo_n = max(0.0, center[i] - sl / 2)
            hi_n = min(1.0, center[i] + sl / 2)
            raw.append((p.low + lo_n * (p.high - p.low),
                         p.low + hi_n * (p.high - p.low)))
        return raw
    return [(p.low, p.high) for p in space_defs]


# ── TandemEngine (v1, bug-fixed) ────────────────────────────────────────────

class TandemEngine:
    """Phase 1: TRust-BO 探索  →  Phase 2: GP 精密化 (v1)"""

    def __init__(
        self,
        space: list[ParamDef],
        direction: str = "minimize",
        seed: int = 42,
        budget: int = 100,
        phase1_ratio: float = 0.8,
        config: dict | None = None,
    ) -> None:
        import warnings
        warnings.warn(
            "TandemEngine/TandemEngineV2 (sklearn-based) are deprecated and will be "
            "removed in v0.2. Use TRustBOEngine(config={'enable_phase2': True}) for "
            "the native Rust Tandem Residual-GP.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._space_defs = space
        self._direction = direction
        self._seed = seed
        self._budget = budget
        self._phase1_ratio = phase1_ratio
        self._config = config or {}

        # Subtract n_init from phase1_budget so that init-phase short batches
        # (TRustBOEngine returns fewer candidates during warm-up) don't prevent
        # the phase transition from triggering.
        n_init_adj = self._config.get("n_init", 0)
        self._phase1_budget = max(1, int(budget * phase1_ratio) - n_init_adj)

        self._bo = TRustBOEngine(
            space=space, direction=direction, seed=seed, config=self._config
        )
        self._phase = 1
        self._eval_count = 0
        self._tell_count = 0

        self._gp = None
        self._gp_bounds: list[tuple[float, float]] = []
        self._X_all: list[list[float]] = []
        self._Y_all: list[float] = []
        self._param_names: list[str] = [p.name for p in space]
        self._rng = np.random.default_rng(seed)

    # ── public API ──────────────────────────────────────────────────────────

    def ask(self, batch_size: int = 1) -> list[dict]:
        if self._phase == 1:
            return self._bo.ask(batch_size=batch_size)
        return self._gp_ask(batch_size)

    def tell(self, candidates: list[dict], results: list) -> None:
        self._bo.tell(candidates, results)
        self._eval_count += len(candidates)
        self._tell_count += 1

        for c, r in zip(candidates, results):
            if isinstance(r, dict) and r.get("feasible", True) and r.get("value") is not None:
                x = [c[n] for n in self._param_names]
                self._X_all.append(x)
                self._Y_all.append(float(r["value"]))

        if self._phase == 1 and self._should_switch():
            self._switch_to_phase2()

    def best(self) -> dict | None:
        return self._bo.best()

    @property
    def phase(self) -> int:
        return self._phase

    # ── internal ────────────────────────────────────────────────────────────

    def _should_switch(self) -> bool:
        """True when Phase-1 budget is exhausted (two independent signals)."""
        if self._eval_count >= self._phase1_budget:
            return True
        # Fallback: adaptive tell-count check using inferred batch size.
        # Fires at phase1_ratio of expected total tells regardless of short batches.
        if self._tell_count > 1 and self._eval_count > 0:
            est_batch = self._eval_count / self._tell_count
            expected_tells = self._budget / max(1.0, est_batch)
            phase1_tells = int(expected_tells * self._phase1_ratio)
            if self._tell_count >= phase1_tells:
                return len(self._X_all) >= max(10, len(self._space_defs) + 1)
        return False

    def _switch_to_phase2(self) -> None:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel

        self._phase = 2
        self._gp_bounds = _build_tr_bounds(self._space_defs, self._bo)

        if not self._X_all:
            return

        X = np.array(self._X_all)
        Y_fit = -np.array(self._Y_all) if self._direction == "minimize" else np.array(self._Y_all)

        kernel = ConstantKernel(1.0) * Matern(nu=2.5)
        self._gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=3, random_state=self._seed,
        )
        self._gp.fit(X, Y_fit)

    def _gp_ask(self, batch_size: int) -> list[dict]:
        if self._gp is None or not self._X_all:
            return self._bo.ask(batch_size=batch_size)

        Y = np.array(self._Y_all)
        best_f = float(np.max(-Y) if self._direction == "minimize" else np.max(Y))
        bounds = self._gp_bounds
        n_dims = len(bounds)
        best_x, best_ei = None, -np.inf

        for _ in range(min(20 * n_dims, 100)):
            x0 = np.array([self._rng.uniform(lo, hi) for lo, hi in bounds])
            res = minimize(
                lambda x: -_ei(x.reshape(1, -1), self._gp, best_f).item(),
                x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 100},
            )
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x = res.x

        if best_x is None:
            best_x = np.array([self._rng.uniform(lo, hi) for lo, hi in bounds])

        candidates = []
        for _ in range(batch_size):
            noise = self._rng.normal(0, 1e-4, size=n_dims)
            x = np.clip(best_x + noise, [b[0] for b in bounds], [b[1] for b in bounds])
            candidates.append({n: float(x[i]) for i, n in enumerate(self._param_names)})
        return candidates


# ── TandemEngineV2 (enhanced Phase 2) ───────────────────────────────────────

class TandemEngineV2(TandemEngine):
    """Phase 2 強化版: WhiteKernel / random-then-refine EI / TR内データのみでGP fit."""

    def _switch_to_phase2(self) -> None:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

        self._phase = 2
        self._gp_bounds = _build_tr_bounds(self._space_defs, self._bo)

        if not self._X_all:
            return

        bounds_arr = np.array(self._gp_bounds)  # (n_dims, 2)
        X_all = np.array(self._X_all)
        Y_all = np.array(self._Y_all)

        # Filter to TR bounds only to avoid GP overfitting on far-away points.
        inside = np.all(
            (X_all >= bounds_arr[:, 0]) & (X_all <= bounds_arr[:, 1]),
            axis=1,
        )
        X = X_all[inside] if inside.sum() >= max(10, len(self._space_defs) + 1) else X_all
        Y = Y_all[inside] if inside.sum() >= max(10, len(self._space_defs) + 1) else Y_all

        Y_fit = -Y if self._direction == "minimize" else Y.copy()

        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        self._gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=10, random_state=self._seed,
        )
        self._gp.fit(X, Y_fit)

    def _gp_ask(self, batch_size: int) -> list[dict]:
        if self._gp is None or not self._X_all:
            return self._bo.ask(batch_size=batch_size)

        Y = np.array(self._Y_all)
        best_f = float(np.max(-Y) if self._direction == "minimize" else np.max(Y))
        bounds = self._gp_bounds
        bounds_arr = np.array(bounds)
        n_dims = len(bounds)

        # Step 1: coarse random sampling (1000 points)
        X_rand = self._rng.uniform(
            bounds_arr[:, 0], bounds_arr[:, 1], size=(1000, n_dims)
        )
        ei_vals = _ei(X_rand, self._gp, best_f)

        # Step 2: refine top-10 candidates with L-BFGS-B
        top_idx = np.argsort(ei_vals)[-10:]
        best_x, best_ei = None, -np.inf

        for idx in top_idx:
            res = minimize(
                lambda x: -_ei(x.reshape(1, -1), self._gp, best_f).item(),
                X_rand[idx],
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200},
            )
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x = res.x

        if best_x is None:
            best_x = X_rand[top_idx[-1]]

        candidates = []
        for _ in range(batch_size):
            noise = self._rng.normal(0, 1e-4, size=n_dims)
            x = np.clip(best_x + noise, bounds_arr[:, 0], bounds_arr[:, 1])
            candidates.append({n: float(x[i]) for i, n in enumerate(self._param_names)})
        return candidates
