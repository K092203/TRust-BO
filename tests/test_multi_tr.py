"""Multi-TR (n_trs > 1) smoke tests.

These tests verify that the n_trs > 1 code path does not crash and produces
valid output. Performance validation is not the goal here.

Design note: multi-TR was implemented and benchmarked but is deprioritized for
the CFD use case, where single-TR outperforms on small budgets and is more
stable. The code is kept for future use on multi-modal problems with large budgets.
"""
import pytest
from trust_bo import Float, TRustBOEngine

_SPACE_5D = [Float(f"x{i}", -5.0, 5.0) for i in range(5)]

_MULTI_TR_CONFIG = {
    "n_trs": 2,
    "n_init": 10,
    "epochs": 50,
    "ensemble_size": 3,
    "n_cem_samples": 64,
    "n_cem_iters": 5,
}


def _sphere(c: dict) -> float:
    return sum(v ** 2 for v in c.values())


def test_multi_tr_ask_returns_correct_batch():
    engine = TRustBOEngine(space=_SPACE_5D, direction="minimize", seed=0,
                           config=_MULTI_TR_CONFIG)
    cands = engine.ask(batch_size=4)
    assert len(cands) == 4
    for c in cands:
        assert set(c.keys()) == {f"x{i}" for i in range(5)}


def test_multi_tr_completes_warm_path():
    """n_trs=2 で cold start を超えて warm path に到達してもクラッシュしない。"""
    engine = TRustBOEngine(space=_SPACE_5D, direction="minimize", seed=1,
                           config=_MULTI_TR_CONFIG)
    for _ in range(15):  # n_init=10 を超えて warm path に入る
        cands = engine.ask(batch_size=4)
        engine.tell(cands, [{"value": _sphere(c), "feasible": True} for c in cands])
    assert engine.best() is not None


def test_multi_tr_beats_random():
    """n_trs=2 が Random より良い結果を出す（5D Sphere、budget=200）。"""
    import numpy as np

    engine = TRustBOEngine(space=_SPACE_5D, direction="minimize", seed=42,
                           config={**_MULTI_TR_CONFIG, "epochs": 300,
                                   "n_cem_samples": 256, "n_cem_iters": 15})
    for _ in range(50):  # budget=200
        cands = engine.ask(batch_size=4)
        engine.tell(cands, [{"value": _sphere(c), "feasible": True} for c in cands])

    best = engine.best()["objective_values"][0]

    rng = np.random.default_rng(42)
    random_best = min(_sphere({f"x{i}": v for i, v in enumerate(x)})
                      for x in rng.uniform(-5, 5, (200, 5)))

    assert best < random_best, f"multi-TR ({best:.3f}) should beat random ({random_best:.3f})"


def test_multi_tr_reproducible():
    """同一 seed で同一結果。"""
    def _run(seed):
        engine = TRustBOEngine(space=_SPACE_5D, direction="minimize", seed=seed,
                               config=_MULTI_TR_CONFIG)
        for _ in range(5):
            cands = engine.ask(batch_size=4)
            engine.tell(cands, [{"value": _sphere(c), "feasible": True} for c in cands])
        return engine.best()["objective_values"][0]

    assert _run(7) == _run(7)
