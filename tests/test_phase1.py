"""Phase 1 exit criteria: Sphere/Ackley が解ける・再現性・Optuna adapter"""
import math
import pytest
from trust_bo import TRustBOEngine, Float


def sphere(x: list[float]) -> float:
    return sum(xi ** 2 for xi in x)


def ackley(x: list[float]) -> float:
    d = len(x)
    a = -20 * math.exp(-0.2 * math.sqrt(sum(xi**2 for xi in x) / d))
    b = -math.exp(sum(math.cos(2 * math.pi * xi) for xi in x) / d)
    return a + b + 20 + math.e


def run_ask_tell(fn, n_dims, n_rounds, batch_size, seed):
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed)
    for _ in range(n_rounds):
        candidates = engine.ask(batch_size=batch_size)
        results = [{"value": fn([c[f"x{i}"] for i in range(n_dims)]), "feasible": True}
                   for c in candidates]
        engine.tell(candidates, results)
    return engine.best()["objective_values"][0]


def test_sphere_5d():
    # Phase 1 は LHS のみ。閾値は Phase 2 (surrogate 導入後) に引き締める
    val = run_ask_tell(sphere, n_dims=5, n_rounds=10, batch_size=10, seed=42)
    assert val < 15.0, f"sphere 5D: {val:.4f}"


def test_ackley_5d():
    val = run_ask_tell(ackley, n_dims=5, n_rounds=10, batch_size=10, seed=42)
    assert val < 10.0, f"ackley 5D: {val:.4f}"


def test_reproducibility_across_sessions():
    """同一 seed で 2 セッション分の ask/tell が完全一致する"""
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(3)]

    def run(seed):
        e = TRustBOEngine(space=space, direction="minimize", seed=seed)
        history = []
        for _ in range(3):
            cands = e.ask(batch_size=4)
            e.tell(cands, [{"value": sum(c[k]**2 for k in c), "feasible": True} for c in cands])
            history.append(cands)
        return history

    assert run(42) == run(42)
    assert run(42) != run(99)


def test_optuna_adapter():
    optuna = pytest.importorskip("optuna")
    from trust_bo.integrations.optuna import TrustBoOptunaSampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(trial):
        x = [trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(3)]
        return sphere(x)

    sampler = TrustBoOptunaSampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(obj, n_trials=30)
    assert study.best_value < 5.0
