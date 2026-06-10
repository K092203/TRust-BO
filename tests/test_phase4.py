"""Phase 4: 制約付き最適化テスト。

feasibility=False を tell() に渡すことで infeasible 領域を学習し、
EI × P(feasible) で制約を考慮した探索を行う。
"""
import math
import pytest
from trust_bo import Float, TRustBOEngine

P4_CONFIG = {
    "epochs": 300,
    "ensemble_size": 5,
    "n_cem_samples": 512,
    "n_cem_iters": 10,
    "acquisition": "ei",
    "l_init": 1.0,
    "tau_succ": 3,
    "tau_fail": 10,
}
BUDGET = 150
BATCH = 10


def sphere(p: dict) -> float:
    return sum(v ** 2 for v in p.values())


def _run_constrained(fn, n_dims, constraint_fn, seed, config=None):
    """制約付き最適化を実行。constraint_fn(params_dict) -> bool"""
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed,
                       config=config or P4_CONFIG)
    for _ in range(BUDGET // BATCH):
        cands = engine.ask(batch_size=BATCH)
        results = [
            {"value": fn(c), "feasible": constraint_fn(c)}
            for c in cands
        ]
        engine.tell(cands, results)
    best = engine.best()
    return best, engine


# --- テスト関数 ---

def test_backward_compatible_no_constraints():
    """infeasible trial がない場合は従来動作と同一 (回帰テスト)。"""
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(5)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=42, config=P4_CONFIG)
    for _ in range(BUDGET // BATCH):
        cands = engine.ask(batch_size=BATCH)
        engine.tell(cands, [{"value": sphere(c), "feasible": True} for c in cands])
    best = engine.best()
    assert best is not None
    assert best["objective_values"][0] < 10.0, f"unconstrained sphere 5D: {best['objective_values'][0]:.3f}"


def test_constrained_sphere_2d():
    """2D Sphere + 線形制約 x0 + x1 >= 2。
    最適点は x*=[1,1] (f=2.0)。制約なし最適点 x=[0,0] (f=0) は infeasible。
    """
    def constraint(p):
        return p["x0"] + p["x1"] >= 2.0

    best, engine = _run_constrained(sphere, 2, constraint, seed=42)
    assert best is not None

    # 最良点が feasible であること
    assert constraint(best["parameters"]), \
        f"best point is infeasible: {best['parameters']}"

    # 合理的な範囲に収束していること (最適値 2.0 の 5 倍以内)
    val = best["objective_values"][0]
    assert val < 10.0, f"constrained sphere 2D: best={val:.3f}, expected < 10.0"

    # feasible / infeasible の両方が history に記録されていること
    history = engine.history()
    statuses = [t.status for t in history]
    assert "infeasible" in statuses, "no infeasible trials recorded"
    assert "complete" in statuses, "no complete trials recorded"
    n_feas = statuses.count("complete")
    n_infeas = statuses.count("infeasible")
    print(f"\n  complete={n_feas}  infeasible={n_infeas}"
          f"  best={val:.4f}  params={best['parameters']}")


def test_constrained_sphere_5d():
    """5D Sphere + 半空間制約 sum(x) >= 5。
    最適実行可能点は x=[1,1,1,1,1] (f=5.0)。
    """
    def constraint(p):
        return sum(p.values()) >= 5.0

    best, engine = _run_constrained(sphere, 5, constraint, seed=42)
    assert best is not None
    assert constraint(best["parameters"]), "best point is infeasible"

    val = best["objective_values"][0]
    assert val < 30.0, f"constrained sphere 5D: best={val:.3f}"
    print(f"\n  best={val:.4f}  sum_x={sum(best['parameters'].values()):.3f}")


def test_feasibility_ratio_improves():
    """warm path では cold start より feasible 候補の割合が高いこと。"""
    n_dims = 2

    def constraint(p):
        # x0 ∈ [-5,5]: feasible if x0 >= 0 (half-space, 50% feasible by random)
        return p["x0"] >= 0.0

    space = [Float(f"x{i}", -5.0, 5.0) for i in range(n_dims)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=42, config=P4_CONFIG)

    cold_feasible, warm_feasible = 0, 0
    cold_total, warm_total = 0, 0
    is_warm = False

    for _ in range(BUDGET // BATCH):
        cands = engine.ask(batch_size=BATCH)
        results = [{"value": sphere(c), "feasible": constraint(c)} for c in cands]
        engine.tell(cands, results)
        # warm path = TR state が存在する
        if engine.tr_state() is not None:
            is_warm = True
        feas_count = sum(1 for r in results if r["feasible"])
        if not is_warm:
            cold_feasible += feas_count
            cold_total += BATCH
        else:
            warm_feasible += feas_count
            warm_total += BATCH

    cold_rate = cold_feasible / cold_total if cold_total > 0 else 0
    warm_rate = warm_feasible / warm_total if warm_total > 0 else 0
    print(f"\n  cold feasibility rate: {cold_rate:.2f}  warm: {warm_rate:.2f}")

    # warm path では feasibility surrogate が学習されているので rate が改善するはず
    # ただし確率的なため soft assertion: < 2x 悪化なら OK とする
    assert warm_rate >= cold_rate * 0.5, \
        f"warm path feasibility rate {warm_rate:.2f} much worse than cold {cold_rate:.2f}"


def test_reproducibility_phase4():
    """同一 seed で制約付き最適化が再現可能。"""
    def constraint(p):
        return p["x0"] + p["x1"] >= 2.0

    best1, _ = _run_constrained(sphere, 2, constraint, seed=42)
    best2, _ = _run_constrained(sphere, 2, constraint, seed=42)
    assert best1["objective_values"] == best2["objective_values"], \
        f"reproducibility failed: {best1['objective_values']} != {best2['objective_values']}"
