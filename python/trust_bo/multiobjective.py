"""
multiobjective.py — Phase K-1: Chebyshev スカラー化 + Pareto フロント追跡

Rust 変更なし、Python 層のみで実装。

設計方針（"ask-time re-scalarization"）:
  - ask() のたびに現在の重みベクトル w を決め、
    過去のすべての多目的評価値を w で再スカラー化して
    フレッシュな TRustBOEngine に replay してから propose する。
  - tell() は多目的履歴に追記するだけ（スカラー値は保持しない）。
  - これにより GP は常に現在の一貫したスカラー目的関数を見る。

使用例:
    from trust_bo import MultiObjectiveEngine, Float

    space = [Float(f"x{i}", 0.0, 1.0) for i in range(30)]
    engine = MultiObjectiveEngine(
        space=space,
        directions=["minimize", "minimize"],
        seed=42,
    )
    for _ in range(n_iters):
        cands = engine.ask(batch_size=4)
        results = [{"values": [f1(c), f2(c)], "feasible": True} for c in cands]
        engine.tell(cands, results)

    front = engine.pareto_front()
    hv    = engine.hypervolume(ref=[1.1, 1.1])
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from typing import Any

import numpy as np

from ._lib import Engine as _RustEngine
from .engine import TRustBOEngine
from .space import ParamDef, SearchSpaceManager


# ── ユーティリティ ────────────────────────────────────────────────────────────

def _gen_weights(n_obj: int, n_weights: int) -> list[np.ndarray]:
    """確率単体上の重みベクトルを n_weights 個生成する。"""
    if n_obj == 1:
        return [np.array([1.0])] * n_weights
    if n_obj == 2:
        eps = 1e-3
        ts = np.linspace(eps, 1.0 - eps, n_weights)
        return [np.array([t, 1.0 - t]) for t in ts]
    rng = np.random.default_rng(0)
    weights = rng.dirichlet(np.ones(n_obj), size=n_weights)
    weights = np.clip(weights, 1e-3, None)
    weights /= weights.sum(axis=1, keepdims=True)
    return list(weights)


def _derive_seed(master: int, count: int) -> int:
    """ask() 呼び出しごとに決定論的に異なるシードを導出する。"""
    data = struct.pack("<QQ", master & 0xFFFFFFFFFFFFFFFF, count & 0xFFFFFFFFFFFFFFFF)
    return struct.unpack("<Q", hashlib.sha256(data).digest()[:8])[0]


def _pareto_mask(costs: np.ndarray) -> np.ndarray:
    """非支配解のブールマスクを返す（最小化）。costs: (n, m)."""
    n = len(costs)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        diff = costs[i] - costs
        dominators = np.where(
            np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
        )[0]
        if len(dominators) > 0:
            dominated[i] = True
    return ~dominated


def hypervolume_2d(front_f: np.ndarray, ref: np.ndarray) -> float:
    """2D 超体積指標（最小化）。スイープライン法 O(n log n)。

    front_f : (n, 2) 配列
    ref     : (2,) 参照点（全点より大きいこと）
    """
    valid_mask = (front_f[:, 0] < ref[0]) & (front_f[:, 1] < ref[1])
    valid = front_f[valid_mask]
    if len(valid) == 0:
        return 0.0
    pts = valid[_pareto_mask(valid)]
    idx = np.argsort(pts[:, 0])
    pts = pts[idx]
    hv = 0.0
    for i, (x, y) in enumerate(pts):
        x_right = float(pts[i + 1, 0]) if i + 1 < len(pts) else float(ref[0])
        hv += (x_right - float(x)) * (float(ref[1]) - float(y))
    return hv


# ── 多目的エンジン ────────────────────────────────────────────────────────────

# EHVI バックエンド (Rust propose_mo) のデフォルト設定
_EHVI_DEFAULT_CONFIG: dict = {
    "ensemble_size": 5,
    "epochs": 500,
    "learning_rate": 5e-4,
    "n_cem_samples": 512,
    "n_cem_iters": 25,
    "elite_fraction": 0.1,
    "ref_margin": 0.1,
    "sigma_init": 0.2,
    "n_cem_starts": 5,
}


class MultiObjectiveEngine:
    """
    多目的 Bayesian 最適化。2 つのバックエンドを選択できる。

    - ``method="chebyshev"`` (Phase K-1、デフォルト):
      ask() のたびに現在の重みで過去評価を再スカラー化し、フレッシュな
      TRustBOEngine で次候補を生成する（ask-time re-scalarization）。
    - ``method="ehvi"`` (Phase K-2、2 目的のみ):
      目的ごとに独立サロゲートを学習し、Rust コアの 2D Expected
      Hypervolume Improvement を CEM で最大化して次候補を生成する。

    Parameters
    ----------
    space      : パラメータ空間定義
    directions : 各目的の方向 ("minimize" | "maximize")
    seed       : 乱数シード
    method     : "chebyshev" | "ehvi"
    n_weights  : (chebyshev) 重みベクトルの数（デフォルト: max(20, 5*n_obj)）
    config     : バックエンドに渡す追加設定
    """

    def __init__(
        self,
        space: list[ParamDef],
        directions: list[str],
        seed: int = 42,
        method: str = "chebyshev",
        n_weights: int | None = None,
        config: dict | None = None,
    ) -> None:
        self._space = space
        self._directions = list(directions)
        self._n_obj = len(directions)
        self._signs = np.array(
            [-1.0 if d == "maximize" else 1.0 for d in directions]
        )
        if method not in ("chebyshev", "ehvi"):
            raise ValueError(f"unknown method '{method}': expected 'chebyshev' or 'ehvi'")
        if method == "ehvi" and self._n_obj != 2:
            raise ValueError(f"method='ehvi' は 2 目的のみ対応 (n_obj={self._n_obj})")
        self._method = method
        n_w = n_weights or max(20, 5 * self._n_obj)
        self._weights = _gen_weights(self._n_obj, n_w)
        self._seed = seed
        self._config = dict(config or {})

        self._mo_history: list[dict] = []
        self._ask_count: int = 0   # ask() 呼び出し回数（重み選択に使用）

        # EHVI バックエンド用の状態
        self._space_mgr = SearchSpaceManager(space)
        self._rust = _RustEngine()
        n_dims = self._space_mgr.n_dims
        self._ehvi_config = {**_EHVI_DEFAULT_CONFIG, **self._config}
        self._ehvi_config.setdefault("n_init", max(10, min(2 * (n_dims + 1), 50)))
        self._mo_model_states: list[list[str]] = []  # 目的ごとの warm-start weights

    # --- public API ---

    def ask(self, batch_size: int = 1) -> list[dict]:
        """次の候補バッチを返す（method に応じてバックエンドを切替）。"""
        if self._method == "ehvi":
            return self._ask_ehvi(batch_size)
        return self._ask_chebyshev(batch_size)

    def _ask_ehvi(self, batch_size: int) -> list[dict]:
        """Rust propose_mo (2D EHVI) で次候補を生成する。"""
        sm = self._space_mgr
        params_enc = [sm.encode(e["params"]) for e in self._mo_history]
        obj_vals = [
            (self._signs * np.asarray(e["values"], dtype=float)).tolist()
            if (e["feasible"] and e["values"] is not None)
            else [0.0] * self._n_obj
            for e in self._mo_history
        ]
        feas = [bool(e["feasible"] and e["values"] is not None) for e in self._mo_history]

        config = {
            **self._ehvi_config,
            "batch_size": batch_size,
            "n_dims": sm.n_dims,
            "n_obj": 2,
            "model_states": self._mo_model_states,
        }
        seed = _derive_seed(self._seed, self._ask_count)
        self._ask_count += 1

        out = json.loads(self._rust.propose_mo(
            params_enc,
            obj_vals,
            feas,
            [],  # ref_point 空 → Rust が観測 nadir から動的決定
            json.dumps(config),
            seed,
        ))
        self._mo_model_states = out.get("model_states", []) or []
        return [sm.decode(c) for c in out["candidates"]]

    def _ask_chebyshev(self, batch_size: int = 1) -> list[dict]:
        """現在の重みで過去評価を再スカラー化し、次候補を返す。"""
        w = self._weights[self._ask_count % len(self._weights)]
        self._ask_count += 1

        # フレッシュなエンジンを構築
        engine = TRustBOEngine(
            space=self._space,
            direction="minimize",
            seed=self._seed + self._ask_count,
            config=self._config,
        )

        feasible = [e for e in self._mo_history if e["feasible"] and e["values"] is not None]
        infeasible = [e for e in self._mo_history if not e["feasible"]]

        if feasible:
            costs = np.array([
                self._signs * np.asarray(e["values"], dtype=float)
                for e in feasible
            ])
            ideal = costs.min(axis=0)
            nadir = costs.max(axis=0)
            denom = np.maximum(nadir - ideal, 1e-8)

            scalar_results = []
            for f in costs:
                scalar_results.append({
                    "value": float(np.max(w * (f - ideal) / denom)),
                    "feasible": True,
                })
            engine.tell([e["params"] for e in feasible], scalar_results)

        if infeasible:
            engine.tell(
                [e["params"] for e in infeasible],
                [{"value": 0.0, "feasible": False}] * len(infeasible),
            )

        return engine.ask(batch_size=batch_size)

    def tell(self, candidates: list[dict], results: list[Any]) -> None:
        """多目的評価結果を履歴に追記する（スカラー化は ask() 時に実施）。

        results の各要素（どちらでも可）:
            {"values": [f1, f2, ...], "feasible": bool}  ← 推奨
            {"value": scalar, "feasible": bool}           ← 後方互換（単目的）
        """
        now = time.time()
        for cand, res in zip(candidates, results):
            vals, feasible = self._parse_result(res)
            self._mo_history.append({
                "params": cand,
                "values": vals,
                "feasible": feasible,
                "timestamp": now,
            })

    def pareto_front(self) -> list[dict]:
        """現在の非支配解リストを返す。

        Returns
        -------
        list of {"params": dict, "values": list[float]}
        """
        feasible = [
            e for e in self._mo_history
            if e["feasible"] and e["values"] is not None
        ]
        if not feasible:
            return []
        costs = np.array([
            (self._signs * np.asarray(e["values"], dtype=float)).tolist()
            for e in feasible
        ])
        mask = _pareto_mask(costs)
        return [
            {"params": e["params"], "values": e["values"]}
            for e, keep in zip(feasible, mask) if keep
        ]

    def hypervolume(self, ref: list[float] | np.ndarray) -> float:
        """Pareto フロントの超体積指標を計算する（2 目的のみ）。

        ref は最小化空間での参照点。
        directions=["maximize","minimize"] の場合は
        ref=[-r1, r2] のように maximize 目的を符号反転して渡す。
        """
        if self._n_obj != 2:
            raise NotImplementedError("hypervolume() は現在 2 目的のみ対応")
        front = self.pareto_front()
        if not front:
            return 0.0
        costs = np.array([
            (self._signs * np.asarray(e["values"], dtype=float)).tolist()
            for e in front
        ])
        return hypervolume_2d(costs, np.asarray(ref, dtype=float))

    def best(self) -> dict | None:
        """後方互換: 最良スカラー値の点（最後の ask weight 基準）。"""
        feasible = [e for e in self._mo_history if e["feasible"] and e["values"] is not None]
        if not feasible:
            return None
        costs = np.array([self._signs * np.asarray(e["values"]) for e in feasible])
        ideal = costs.min(axis=0)
        nadir = costs.max(axis=0)
        denom = np.maximum(nadir - ideal, 1e-8)
        w = self._weights[(self._ask_count - 1) % len(self._weights)]
        scalars = [float(np.max(w * (f - ideal) / denom)) for f in costs]
        best_idx = int(np.argmin(scalars))
        e = feasible[best_idx]
        return {"params": e["params"], "objective_values": e["values"]}

    def history(self) -> list[dict]:
        return list(self._mo_history)

    def n_evaluated(self) -> int:
        return len(self._mo_history)

    # --- internal ---

    def _parse_result(self, res: Any) -> tuple[list[float] | None, bool]:
        if isinstance(res, dict):
            vals = res.get("values")
            if vals is None and "value" in res:
                vals = [res["value"]]
            feasible = res.get("feasible", True)
            return ([float(v) for v in vals] if vals is not None else None), feasible
        if isinstance(res, (int, float)):
            return [float(res)], True
        return None, False
