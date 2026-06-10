"""Phase 0 exit criteria: ビルドできる・propose() を呼べる"""
import pytest
from trust_bo import Categorical, Float, Int, TRustBOEngine


def test_instantiate():
    engine = TRustBOEngine(space=[Float("x", -5.0, 5.0)], direction="minimize", seed=42)
    assert engine is not None


def test_ask_shape():
    engine = TRustBOEngine(
        space=[Float("x0", -5.0, 5.0), Float("x1", -3.0, 3.0)],
        direction="minimize",
        seed=42,
    )
    candidates = engine.ask(batch_size=4)
    assert len(candidates) == 4
    for c in candidates:
        assert set(c.keys()) == {"x0", "x1"}


def test_ask_values_in_range():
    engine = TRustBOEngine(
        space=[Float("x", -5.0, 5.0), Int("n", 0, 10), Categorical("m", ["A", "B", "C"])],
        direction="minimize",
        seed=1,
    )
    for c in engine.ask(batch_size=20):
        assert -5.0 <= c["x"] <= 5.0
        assert 0 <= c["n"] <= 10
        assert c["m"] in ["A", "B", "C"]


def test_reproducibility():
    space = [Float("x", -5.0, 5.0), Float("y", -3.0, 3.0)]
    c1 = TRustBOEngine(space=space, direction="minimize", seed=42).ask(batch_size=5)
    c2 = TRustBOEngine(space=space, direction="minimize", seed=42).ask(batch_size=5)
    assert c1 == c2


def test_different_seeds_differ():
    space = [Float("x", -5.0, 5.0)]
    c1 = TRustBOEngine(space=space, direction="minimize", seed=42).ask(batch_size=8)
    c2 = TRustBOEngine(space=space, direction="minimize", seed=99).ask(batch_size=8)
    assert c1 != c2


def test_tell_does_not_crash():
    engine = TRustBOEngine(space=[Float("x", -5.0, 5.0)], direction="minimize", seed=42)
    candidates = engine.ask(batch_size=3)
    results = [{"value": c["x"] ** 2, "feasible": True} for c in candidates]
    engine.tell(candidates, results)
    assert len(engine.history()) == 3


def test_best():
    engine = TRustBOEngine(space=[Float("x", -5.0, 5.0)], direction="minimize", seed=42)
    candidates = engine.ask(batch_size=5)
    engine.tell(candidates, [{"value": c["x"] ** 2, "feasible": True} for c in candidates])
    best = engine.best()
    assert best is not None
    assert "parameters" in best and "objective_values" in best
    # minimize なので最小値の trial が返るはず
    min_val = min(c["x"] ** 2 for c in candidates)
    assert abs(best["objective_values"][0] - min_val) < 1e-6


def test_tell_failed_result():
    engine = TRustBOEngine(space=[Float("x", -5.0, 5.0)], direction="minimize", seed=42)
    candidates = engine.ask(batch_size=2)
    engine.tell(candidates, [None, {"value": 1.0, "feasible": True}])
    trials = engine.history()
    assert trials[0].status == "failed"
    assert trials[1].status == "complete"


def test_save_load_round_trip(tmp_path):
    space = [Float("x", -5.0, 5.0), Float("y", -3.0, 3.0)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=42)
    candidates = engine.ask(batch_size=3)
    engine.tell(candidates, [{"value": c["x"] ** 2 + c["y"] ** 2, "feasible": True} for c in candidates])

    path = tmp_path / "study.trm"
    engine.save(path)

    engine2 = TRustBOEngine.load(path)
    assert len(engine2.history()) == 3
    assert engine2._seed == 42
    assert engine2._ask_count == 1

    # ロード後の ask() も再現性を持つ
    c1 = engine.ask(batch_size=2)
    c2 = engine2.ask(batch_size=2)
    assert c1 == c2


def test_int_param_type():
    engine = TRustBOEngine(space=[Int("n", 0, 10)], direction="minimize", seed=42)
    for c in engine.ask(batch_size=10):
        assert isinstance(c["n"], int)
        assert 0 <= c["n"] <= 10


def test_categorical_choices():
    engine = TRustBOEngine(
        space=[Categorical("mode", ["fast", "balanced", "accurate"])],
        direction="minimize",
        seed=42,
    )
    for c in engine.ask(batch_size=12):
        assert c["mode"] in ["fast", "balanced", "accurate"]


def test_maximize_direction():
    engine = TRustBOEngine(space=[Float("x", 0.0, 1.0)], direction="maximize", seed=42)
    candidates = engine.ask(batch_size=5)
    engine.tell(candidates, [c["x"] for c in candidates])
    best = engine.best()
    expected = max(c["x"] for c in candidates)
    assert abs(best["objective_values"][0] - expected) < 1e-6


def test_ask_count_increments():
    engine = TRustBOEngine(space=[Float("x", 0.0, 1.0)], direction="minimize", seed=42)
    assert engine._ask_count == 0
    engine.ask(batch_size=1)
    assert engine._ask_count == 1
    engine.ask(batch_size=1)
    assert engine._ask_count == 2


def test_sequential_asks_differ():
    """同じエンジンで連続 ask() すると異なる候補が返る"""
    engine = TRustBOEngine(space=[Float("x", 0.0, 1.0)], direction="minimize", seed=42)
    c1 = engine.ask(batch_size=4)
    c2 = engine.ask(batch_size=4)
    assert c1 != c2
