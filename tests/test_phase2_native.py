"""tests/test_phase2_native.py — Rust ネイティブ Tandem Phase 2 のテスト。

遷移ロジックは TR state を直接注入して決定的にテストする
(Ackley 多数評価による確率的遷移には依存しない)。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from trust_bo import Float, TRustBOEngine


def _ackley(x: np.ndarray) -> float:
    n = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        + 20 + np.e
    )


def _space(d):
    return [Float(f"x{i}", -5.0, 5.0) for i in range(d)]


def _run(engine, d, n_evals, batch=4, fn=_ackley):
    ev = 0
    while ev < n_evals:
        cands = engine.ask(batch_size=min(batch, n_evals - ev))
        engine.tell(
            cands,
            [{"value": fn(np.array([c[f"x{i}"] for i in range(d)])), "feasible": True}
             for c in cands],
        )
        ev += len(cands)


def _make_warm_engine(d=5, n_evals=40, enable=True, extra_config=None):
    """warm path 到達済み (tr_states あり) のエンジンを作る。"""
    config = {"n_init": 6, "epochs": 50}
    if enable:
        config["enable_phase2"] = True
    config.update(extra_config or {})
    eng = TRustBOEngine(space=_space(d), direction="minimize", seed=0, config=config)
    _run(eng, d, n_evals)
    assert eng._tr_states, "warm path 未到達"
    return eng


# ── 遷移ロジック (TR state 直接注入で deterministic) ─────────────────────────

def test_local_transition_on_lmin():
    """TR を l_min 未満に注入 → 次 ask で local 遷移。"""
    eng = _make_warm_engine(n_evals=40)  # 40 >= 3*n_init=18
    eng._tr_states[0]["side_length"] = 0.001  # < l_min=0.0078125
    cands = eng.ask(batch_size=4)
    assert eng._phase == "local"
    assert len(cands) == 4


def test_min_evals_guard():
    """feasible 評価数 < 3*n_init では local に遷移しない。"""
    eng = _make_warm_engine(d=5, n_evals=16, extra_config={"n_init": 6})  # 16 < 18
    eng._tr_states[0]["side_length"] = 0.001
    eng.ask(batch_size=4)
    assert eng._phase == "global"


def test_sticky_local_and_frozen_tr():
    """local 遷移後: TR state 凍結 + phase 維持 (sticky)。"""
    eng = _make_warm_engine(n_evals=40)
    eng._tr_states[0]["side_length"] = 0.001
    cands = eng.ask(batch_size=4)
    assert eng._phase == "local"
    frozen = [dict(s) for s in eng._tr_states]
    eng.tell(cands, [{"value": 1.0, "feasible": True} for _ in cands])
    eng.ask(batch_size=4)
    assert eng._phase == "local", "sticky でない"
    assert eng._tr_states == frozen, "local 中に TR が変化した"


def test_default_off_restarts_tr():
    """enable_phase2=False (デフォルト): 同じ注入でも従来どおり TR 再起動。"""
    eng = _make_warm_engine(n_evals=40, enable=False)
    eng._tr_states[0]["side_length"] = 0.001
    eng.ask(batch_size=4)
    assert eng._phase == "global"
    # 再起動で side_length が l_init に戻る
    assert eng._tr_states[0]["side_length"] > 0.01


# ── 50D 回帰 (n_local = max(50, n_dims+2) の検証) ────────────────────────────

def test_50d_local_branch_reaches_gp():
    """50D・60 点で local 分岐が GP fit に到達する (52 点必要)。"""
    d = 50
    eng = TRustBOEngine(
        space=_space(d), direction="minimize", seed=0,
        config={"n_init": 12, "epochs": 30, "enable_phase2": True,
                "phase2_min_evals": 55},
    )
    _run(eng, d, 60)
    eng._tr_states[0]["side_length"] = 0.001
    cands = eng.ask(batch_size=4)
    assert eng._phase == "local", "50D で local 遷移失敗 (旧 50 点上限バグ)"
    assert len(cands) == 4


# ── GP 頑健性 / batch 保証 ───────────────────────────────────────────────────

def test_degenerate_residuals_fallback():
    """全点同一値 → GP fit 失敗でも panic せず batch_size 件返る。"""
    d = 5
    eng = TRustBOEngine(
        space=_space(d), direction="minimize", seed=0,
        config={"n_init": 6, "epochs": 30, "enable_phase2": True},
    )
    _run(eng, d, 40, fn=lambda x: 1.0)  # 定数関数
    eng._tr_states[0]["side_length"] = 0.001
    cands = eng.ask(batch_size=4)
    assert len(cands) == 4
    # 定数関数では残差≈0 だが MLP 残差が厳密ゼロとは限らないため
    # phase は local/global どちらも許容。契約は「batch_size 件・無 panic」。


def test_duplicate_points_no_panic():
    """重複点だらけの履歴でも panic せず候補が返る。"""
    d = 3
    eng = TRustBOEngine(
        space=_space(d), direction="minimize", seed=0,
        config={"n_init": 6, "epochs": 30, "enable_phase2": True},
    )
    same = {f"x{i}": 1.0 for i in range(d)}
    # ask は呼ぶが tell は全て同一点で上書き
    for _ in range(10):
        cands = eng.ask(batch_size=4)
        eng.tell([same] * len(cands), [{"value": 2.0, "feasible": True}] * len(cands))
    eng._tr_states and eng._tr_states[0].update(side_length=0.001)
    cands = eng.ask(batch_size=4)
    assert len(cands) == 4


@pytest.mark.parametrize("b", [1, 3, 7])
def test_batch_size_guarantee(b):
    """local phase で常に batch_size 件返る (backfill 検証)。"""
    eng = _make_warm_engine(n_evals=40)
    eng._tr_states[0]["side_length"] = 0.001
    cands = eng.ask(batch_size=b)
    assert len(cands) == b
    for c in cands:
        for i in range(5):
            assert -5.0 <= c[f"x{i}"] <= 5.0


# ── Python 側 ────────────────────────────────────────────────────────────────

def test_no_sklearn_required():
    """sklearn が import 不可でも native Phase 2 は動く。"""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "sklearn" or k.startswith("sklearn.")}
    sys.modules["sklearn"] = None  # import を強制失敗させる
    try:
        eng = TRustBOEngine(
            space=_space(3), direction="minimize", seed=1,
            config={"n_init": 6, "epochs": 30, "enable_phase2": True},
        )
        _run(eng, 3, 30)
        assert eng.best() is not None
    finally:
        del sys.modules["sklearn"]
        sys.modules.update(saved)


def test_quality_sanity_vs_random():
    """5D Ackley budget=120: enable_phase2 >= Random。"""
    d, budget = 5, 120
    eng = TRustBOEngine(
        space=_space(d), direction="minimize", seed=0,
        config={"n_init": 10, "enable_phase2": True},
    )
    _run(eng, d, budget)
    bo_best = eng.best()["objective_values"][0]

    rng = np.random.default_rng(0)
    rand_best = min(
        _ackley(rng.uniform(-5, 5, d)) for _ in range(budget)
    )
    assert bo_best <= rand_best, f"BO {bo_best:.3f} > Random {rand_best:.3f}"
