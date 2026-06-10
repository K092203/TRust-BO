"""tests/test_tandem.py — TandemEngine / TandemEngineV2 の単体テスト"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from trust_bo import TandemEngine, TandemEngineV2, Float


# ── helpers ──────────────────────────────────────────────────────────────────

def _ackley(x: np.ndarray) -> float:
    n = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        + 20 + np.e
    )


def _run_engine(engine, n_dims: int, budget: int, batch: int = 4):
    """Run engine for budget evaluations; return (best_value, phase_switch_ev)."""
    ev = 0
    phase_switch = -1
    while ev < budget:
        b = min(batch, budget - ev)
        cands = engine.ask(batch_size=b)
        engine.tell(
            cands,
            [{"value": _ackley(np.array([c[f"x{i}"] for i in range(n_dims)])),
              "feasible": True}
             for c in cands],
        )
        ev += len(cands)
        if hasattr(engine, "phase") and engine.phase == 2 and phase_switch < 0:
            phase_switch = ev
    best = engine.best()
    best_val = best["objective_values"][0] if best else float("inf")
    return best_val, phase_switch


# ── Phase 2 発動テスト ─────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_phase2_triggers_all_seeds(seed):
    """Phase 2 が全 seed で発動することを確認（修正済みロジック）。"""
    n_dims, budget = 5, 40  # small budget to run fast alongside benchmark
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    engine = TandemEngine(
        space=space, direction="minimize", seed=seed,
        budget=budget, phase1_ratio=0.8,
        config={"n_init": 6},
    )
    _, phase_switch = _run_engine(engine, n_dims, budget)
    assert engine.phase == 2, f"seed={seed}: Phase 2 not reached"
    assert phase_switch > 0, f"seed={seed}: phase_switch={phase_switch}"
    assert phase_switch <= budget, f"seed={seed}: phase_switch={phase_switch} > budget"


def test_phase2_gp_is_fitted():
    """Phase 2 移行後に GP が fit 済みであることを確認。"""
    n_dims, budget = 5, 40
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    engine = TandemEngine(
        space=space, direction="minimize", seed=0,
        budget=budget, phase1_ratio=0.8,
        config={"n_init": 6},
    )
    _run_engine(engine, n_dims, budget)
    assert engine._gp is not None, "GP should be fitted after phase 2"


def test_phase2_improves_or_matches():
    """TandemEngine が TRust-BO と同等以上の結果を出す。"""
    from trust_bo import TRustBOEngine
    n_dims, budget = 5, 40
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]

    te = TandemEngine(space=space, direction="minimize", seed=0,
                      budget=budget, phase1_ratio=0.8, config={"n_init": 6})
    bo = TRustBOEngine(space=space, direction="minimize", seed=0,
                       config={"n_init": 6})

    te_val, _ = _run_engine(te, n_dims, budget)
    bo_val, _ = _run_engine(bo, n_dims, budget)
    assert te_val <= bo_val + 1.0, f"TandemEngine {te_val:.3f} much worse than TRust-BO {bo_val:.3f}"


# ── TandemEngineV2 テスト ─────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_v2_phase2_triggers(seed):
    """TandemEngineV2 でも全 seed で Phase 2 が発動する。"""
    n_dims, budget = 5, 40
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    engine = TandemEngineV2(
        space=space, direction="minimize", seed=seed,
        budget=budget, phase1_ratio=0.8,
        config={"n_init": 6},
    )
    _, phase_switch = _run_engine(engine, n_dims, budget)
    assert engine.phase == 2, f"V2 seed={seed}: Phase 2 not reached"
    assert phase_switch > 0, f"V2 seed={seed}: phase_switch={phase_switch}"


def test_v2_gp_uses_white_kernel():
    """V2 の GP が WhiteKernel を含むカーネルを使っていることを確認。"""
    n_dims, budget = 5, 40
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    engine = TandemEngineV2(
        space=space, direction="minimize", seed=0,
        budget=budget, phase1_ratio=0.8, config={"n_init": 6},
    )
    _run_engine(engine, n_dims, budget)
    assert engine._gp is not None
    kernel_str = str(engine._gp.kernel_)
    assert "WhiteKernel" in kernel_str or "white" in kernel_str.lower(), \
        f"WhiteKernel not found in {kernel_str}"


def test_v2_ask_returns_correct_count():
    """V2 の _gp_ask が batch_size 分の候補を返す。"""
    n_dims, budget = 5, 40
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    engine = TandemEngineV2(
        space=space, direction="minimize", seed=0,
        budget=budget, phase1_ratio=0.6, config={"n_init": 6},
    )
    _run_engine(engine, n_dims, budget)
    cands = engine.ask(batch_size=3)
    assert len(cands) == 3
    for c in cands:
        for p in space:
            assert p.low <= c[p.name] <= p.high
