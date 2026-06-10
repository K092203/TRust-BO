from __future__ import annotations

import hashlib
import json
import struct
import time
import zipfile
from pathlib import Path
from typing import Any

from ._lib import Engine as _RustEngine
from .history import HistoryStore, Trial
from .space import ParamDef, SearchSpaceManager

_DEFAULT_CONFIG: dict = {
    "ensemble_size": 5,
    "epochs": 500,
    "learning_rate": 5e-4,
    "n_cem_samples": 512,
    "n_cem_iters": 25,
    "elite_fraction": 0.1,
    "beta": 2.0,
    "acquisition": "ei",
    # Trust Region  (l_init は __init__ で次元数から適応計算)
    "tau_succ": 3,
    "tau_fail": 5,
    "l_max": 1.0,
    "l_min": 0.0078125,  # 0.5^7
    # TuRBO-M: 並列 TR 数 (1 = 従来の単一 TR、後方互換)
    "n_trs": 1,
}


class TRustBOEngine:
    def __init__(
        self,
        space: list[ParamDef],
        direction: str | list[str] = "minimize",
        seed: int = 42,
        *,
        config: dict | None = None,
    ) -> None:
        self._space = SearchSpaceManager(space)
        self._direction = direction
        self._seed = seed
        self._history = HistoryStore()
        self._tr_states: list = []
        self._ask_count: int = 0
        self._config: dict = {**_DEFAULT_CONFIG, **(config or {})}
        self._rust = _RustEngine()
        self._model_states: list = []       # hex-encoded surrogate weights for warm start
        self._feas_model_states: list = []  # hex-encoded feasibility surrogate weights
        self._phase: str = "global"         # Tandem Phase 2 の sticky 状態
        self._stagnation_count: int = 0     # EI 停滞カウンタ (Rust と往復)

        n = self._space.n_dims
        self._config.setdefault("n_init", max(10, min(2 * (n + 1), 50)))
        # adaptive l_init: TR 内に初期訓練点が ~5 点入るよう次元数で決定
        # l^d = 5/n_init → l = (5/n_init)^(1/d)
        n_init_val = self._config["n_init"]
        adaptive_l = min(0.8, max(0.3, (5.0 / n_init_val) ** (1.0 / max(n, 1))))
        self._config.setdefault("l_init", adaptive_l)

    # --- public API ---

    def ask(self, batch_size: int = 1) -> list[dict]:
        # infeasible trials も渡して feasibility surrogate を訓練させる
        evaluated = [t for t in self._history.all_trials()
                     if t.status in ("complete", "infeasible")]
        params = [self._space.encode(t.parameters) for t in evaluated]
        # [CHANGED]: infeasible 試行の目的値を feasible 試行の最悪値で埋める。
        # 0.0 固定だと feasible 値のスケールに依存した意図しないバイアスが生じる。
        # （Rust 側では infeasible 値は主サロゲート訓練に使われないが、意味的に正しい値を渡す）
        _feasible_vals = [
            self._to_maximize(t.objective_values)
            for t in evaluated if t.status == "complete"
        ]
        _worst_feasible = min(_feasible_vals) if _feasible_vals else 0.0
        values = [
            self._to_maximize(t.objective_values) if t.status == "complete" else _worst_feasible
            for t in evaluated
        ]
        feasibility = [t.status == "complete" for t in evaluated]

        config = {**self._config, "batch_size": batch_size, "n_dims": self._space.n_dims,
                  "model_states": self._model_states,
                  "feas_model_states": self._feas_model_states,
                  "phase": self._phase,
                  "stagnation_count": self._stagnation_count}
        seed = _derive_seed(self._seed, self._ask_count)
        self._ask_count += 1

        result: dict = json.loads(self._rust.propose(
            params,
            values,
            feasibility,
            [],
            json.dumps(self._tr_states),
            json.dumps(config),
            seed,
        ))

        self._tr_states = result["tr_states"]
        self._model_states = result.get("model_states", [])
        self._feas_model_states = result.get("feas_model_states", [])
        self._phase = result.get("phase", "global")
        self._stagnation_count = result.get("stagnation_count", 0)
        return [self._space.decode(c) for c in result["candidates"]]

    def tell(self, candidates: list[dict], results: list[Any]) -> None:
        for candidate, res in zip(candidates, results):
            self._history.add(self._make_trial(candidate, res))

    def best(self) -> dict | None:
        feasible = self._history.feasible_trials()
        if not feasible:
            return None
        best = max(feasible, key=lambda t: self._to_maximize(t.objective_values))
        return {
            "trial_id": best.trial_id,
            "parameters": best.parameters,
            "objective_values": best.objective_values,
        }

    def history(self) -> list[Trial]:
        return self._history.all_trials()

    def tr_state(self) -> dict | None:
        """現在の Trust Region 状態を返す。warm path 未開始なら None。"""
        if not self._tr_states:
            return None
        s = self._tr_states[0]
        return {
            "side_length": s["side_length"],
            "success_count": s["success_count"],
            "failure_count": s["failure_count"],
            "center": s["center"],
            "best_value": s["best_value"],
            "active": s["active"],
        }

    def best_so_far_curve(self) -> list[float]:
        """評価順に best-so-far 値のリストを返す (最小化方向)。"""
        best = float("inf")
        curve = []
        for t in self._history.all_trials():
            if t.status == "complete" and t.objective_values:
                v = t.objective_values[0]
                if self._direction == "minimize":
                    best = min(best, v)
                else:
                    best = min(best, -v)
            curve.append(best)
        return curve

    def save(self, path: str | Path) -> None:
        meta = {
            "version": "0.1.0",
            "direction": self._direction,
            "seed": self._seed,
            "ask_count": self._ask_count,
            "config": self._config,
            "tr_states": self._tr_states,
            "model_states": self._model_states,
            "feas_model_states": self._feas_model_states,
            "phase": self._phase,
            "stagnation_count": self._stagnation_count,
            "space": self._space.to_list(),
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("study.jsonl", self._history.to_jsonl())
            zf.writestr("meta.json", json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> TRustBOEngine:
        with zipfile.ZipFile(path, "r") as zf:
            meta: dict = json.loads(zf.read("meta.json"))
            jsonl: str = zf.read("study.jsonl").decode("utf-8")

        engine = cls.__new__(cls)
        engine._space = SearchSpaceManager.from_list(meta["space"])
        engine._direction = meta["direction"]
        engine._seed = meta["seed"]
        engine._ask_count = meta["ask_count"]
        engine._config = meta["config"]
        engine._tr_states = meta["tr_states"]
        engine._model_states = meta.get("model_states", [])
        engine._feas_model_states = meta.get("feas_model_states", [])
        engine._phase = meta.get("phase", "global")
        engine._stagnation_count = meta.get("stagnation_count", 0)
        engine._history = HistoryStore.from_jsonl(jsonl) if jsonl.strip() else HistoryStore()
        engine._rust = _RustEngine()
        return engine

    # --- internal helpers ---

    def _to_maximize(self, objective_values: list[float] | None) -> float:
        if objective_values is None:
            return float("-inf")
        v = objective_values[0] if isinstance(objective_values, list) else float(objective_values)
        return -v if self._direction == "minimize" else v

    def _make_trial(self, candidate: dict, result: Any) -> Trial:
        tid = self._history.next_id()
        now = time.time()

        if isinstance(result, (int, float)):
            return Trial(tid, candidate, [float(result)], status="complete",
                         started_at=now, completed_at=now)

        if result is None:
            return Trial(tid, candidate, None, status="failed",
                         started_at=now, completed_at=now)

        if isinstance(result, dict):
            v = result.get("value")
            feasible = result.get("feasible", True)
            reason = result.get("failure_reason")
            if v is None:
                status = "failed"
                values = None
            elif not feasible:
                status = "infeasible"
                values = [float(v)]
            else:
                status = "complete"
                values = [float(v)]
            return Trial(tid, candidate, values, status=status,
                         failure_reason=reason, started_at=now, completed_at=now)

        return Trial(tid, candidate, [float(result)], status="complete",
                     started_at=now, completed_at=now)


def _derive_seed(master: int, count: int) -> int:
    data = struct.pack("<QQ", master & 0xFFFFFFFFFFFFFFFF, count & 0xFFFFFFFFFFFFFFFF)
    return struct.unpack("<Q", hashlib.sha256(data).digest()[:8])[0]
