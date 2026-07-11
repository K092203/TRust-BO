"""Two-stage low-/high-fidelity cascade for :class:`TRustBOEngine`.

The Rust optimiser itself remains unchanged.  This module only orchestrates
two independent Python ``TRustBOEngine`` instances around an inexpensive
low-fidelity objective followed by a high-fidelity refinement.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .engine import TRustBOEngine
from .space import Float, ParamDef, SearchSpaceManager


Objective = Callable[[dict[str, Any]], dict[str, Any] | float]


class CascadeMFEngine:
    """低忠実度(LF)で広域探索 → 上位候補をHFで評価し、縮小空間でHF最適化を継続。

    Rustコアは不変。TRustBOEngine を2段カスケードするPython層ラッパー。

    ``space`` is deliberately limited to continuous ``Float`` parameters:
    the handoff and diversity criterion are defined in the continuous [0, 1]
    encoded space, so integer and categorical parameters are not silently
    approximated.
    """

    def __init__(
        self,
        space: list[ParamDef],
        direction: str = "minimize",
        seed: int = 42,
        config: dict | None = None,
        lf_budget: int = 200,
        top_k: int = 8,
        shrink: float = 0.5,
        batch_size: int = 4,
        lf_config: dict | None = None,
        hf_config: dict | None = None,
        outside_frac: float = 0.0,
    ) -> None:
        if direction not in ("minimize", "maximize"):
            raise ValueError("direction must be 'minimize' or 'maximize'")
        if any(not isinstance(param, Float) for param in space):
            raise NotImplementedError("CascadeMFEngine supports Float spaces only")
        if any(param.step is not None for param in space):
            # 縮小境界が step 格子と整合しない off-grid 値を生むため明示拒否
            raise NotImplementedError("CascadeMFEngine does not support Float(step=...)")
        if not space:
            raise ValueError("space must contain at least one Float parameter")
        if lf_budget <= 0 or top_k <= 0 or batch_size <= 0:
            raise ValueError("lf_budget, top_k, and batch_size must be positive")
        if lf_budget < top_k:
            raise ValueError("lf_budget must be at least top_k")
        if not 0.0 < shrink <= 1.0:
            raise ValueError("shrink must be in (0, 1]")
        if not 0.0 <= outside_frac < 1.0:
            raise ValueError("outside_frac must be in [0, 1)")

        self.space = list(space)
        self.direction = direction
        self.seed = seed
        base = dict(config or {})
        # LF/HF で設定を分けたい場合 (例: early Phase2 を HF のみに) は個別指定。
        self.lf_config = {**base, **(lf_config or {})}
        self.hf_config = {**base, **(hf_config or {})}
        self.lf_budget = lf_budget
        self.top_k = top_k
        self.shrink = shrink
        self.batch_size = batch_size
        # LF 誤誘導 (縮小 box が真の最適域を切る) への防御: HF 予算のこの割合を
        # 元の全域空間での探索に確保する。0.0 で無効 (従来挙動)。
        self.outside_frac = outside_frac
        self._space_manager = SearchSpaceManager(self.space)

    def run(self, lf_fn: Objective, hf_fn: Objective, hf_budget: int) -> dict:
        """Run the LF global search, then the HF refinement cascade.

        ``hf_budget`` must cover the requested ``top_k`` HF seed evaluations;
        requiring this explicitly avoids a partially seeded Stage B run.
        """
        if hf_budget < self.top_k:
            raise ValueError("hf_budget must be at least top_k")

        lf_engine = TRustBOEngine(
            space=self.space,
            direction=self.direction,
            seed=self.seed,
            config=self.lf_config,
        )
        lf_evals = self._evaluate_budget(lf_engine, lf_fn, self.lf_budget)

        selected = self._select_diverse_feasible(lf_engine)
        if not selected:
            raise RuntimeError("LF stage produced no feasible candidates for HF refinement")

        reduced_space, reduced_bounds = self._reduced_space(selected)
        hf_config = {**self.hf_config, "n_init": len(selected)}
        hf_engine = TRustBOEngine(
            space=reduced_space,
            direction=self.direction,
            seed=self.seed + 1,
            config=hf_config,
        )

        # Replay the selected LF locations only through the actual HF oracle.
        hf_results = [hf_fn(candidate) for candidate in selected]
        hf_engine.tell(selected, hf_results)
        hf_evals = len(selected)

        # LF 誤誘導への防御: HF 予算の一部を元の全域空間での探索に充てる (任意)。
        outside_evals = 0
        outside_best: dict | None = None
        n_outside = int(self.outside_frac * hf_budget)
        if n_outside > 0:
            outside_engine = TRustBOEngine(
                space=self.space,
                direction=self.direction,
                seed=self.seed + 2,
                config=self.hf_config,
            )
            outside_evals = self._evaluate_budget(outside_engine, hf_fn, n_outside)
            outside_best = outside_engine.best()
            hf_evals += outside_evals

        hf_evals += self._evaluate_budget(hf_engine, hf_fn, hf_budget - hf_evals)

        best = hf_engine.best()
        if outside_best is not None and (
            best is None or self._better(outside_best, best)
        ):
            best = outside_best

        n_hf_feasible = len(
            [t for t in hf_engine.history() if t.status == "complete"]
        )
        return {
            "best": best,
            "hf_evals": hf_evals,
            "lf_evals": lf_evals,
            "reduced_bounds": reduced_bounds,
            "hf_feasible": n_hf_feasible,
            "outside_evals": outside_evals,
        }

    def _better(self, a: dict, b: dict) -> bool:
        va = float(a["objective_values"][0])
        vb = float(b["objective_values"][0])
        return va > vb if self.direction == "maximize" else va < vb

    def _evaluate_budget(
        self, engine: TRustBOEngine, objective: Objective, budget: int
    ) -> int:
        evaluated = 0
        while evaluated < budget:
            candidates = engine.ask(batch_size=min(self.batch_size, budget - evaluated))
            if not candidates:
                raise RuntimeError("TRustBOEngine.ask() returned no candidates")
            engine.tell(candidates, [objective(candidate) for candidate in candidates])
            evaluated += len(candidates)
        return evaluated

    def _select_diverse_feasible(self, engine: TRustBOEngine) -> list[dict[str, Any]]:
        feasible = [trial for trial in engine.history() if trial.status == "complete"]
        feasible.sort(
            key=lambda trial: float(trial.objective_values[0]),
            reverse=self.direction == "maximize",
        )

        selected: list[dict[str, Any]] = []
        encoded_selected: list[list[float]] = []
        skipped: list[dict[str, Any]] = []
        for trial in feasible:
            encoded = self._space_manager.encode(trial.parameters)
            if all(
                max(abs(x - y) for x, y in zip(encoded, prior)) > 0.05
                for prior in encoded_selected
            ):
                selected.append(dict(trial.parameters))
                encoded_selected.append(encoded)
                if len(selected) == self.top_k:
                    break
            else:
                skipped.append(dict(trial.parameters))
        # 多様性フィルタで top_k に届かない場合は、値順位の高い未選択点で埋める
        # (HF n_init との整合を優先し、多様性は best effort とする)。
        for params in skipped:
            if len(selected) >= self.top_k:
                break
            selected.append(params)
        return selected

    def _reduced_space(
        self, candidates: list[dict[str, Any]]
    ) -> tuple[list[Float], list[tuple[float, float]]]:
        reduced: list[Float] = []
        bounds: list[tuple[float, float]] = []
        for param in self.space:
            values = [float(candidate[param.name]) for candidate in candidates]
            lo, hi = min(values), max(values)
            span = hi - lo
            # A singleton bounding box has no 10%-of-box margin.  Use 10% of
            # the original span so Float encoding remains well defined.
            margin = 0.1 * (span if span > 0.0 else param.high - param.low)
            reduced_lo = max(param.low, lo - margin)
            reduced_hi = min(param.high, hi + margin)
            if reduced_hi <= reduced_lo:
                reduced_lo, reduced_hi = param.low, param.high
            bounds.append((float(reduced_lo), float(reduced_hi)))
            reduced.append(
                Float(
                    name=param.name,
                    low=float(reduced_lo),
                    high=float(reduced_hi),
                    step=param.step,
                    log=param.log,
                )
            )
        return reduced, bounds
