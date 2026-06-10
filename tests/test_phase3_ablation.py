"""Phase 3.5 アブレーション: TR dynamics の貢献を検証する。

TR dynamic: l_init=0.5 (half space), l_max=1.0, tau_succ=3, tau_fail=5
            → expand/shrink が実際に発動する設定
TR frozen : l_init=1.0 (full space), tau_succ/fail=9999
            → TR なし相当 (全空間でグローバル EI + CEM)

注: l_init=1.0=l_max (現行デフォルト) だと expand 不可 + tau_fail=10 では
200 eval / 15 round では shrink 発動せず → TR on=off になる。
この ablation は TR dynamics が実際に機能することを確認し、貢献度を比較する。

pytest -m eval で実行
"""
import math
import statistics

import pytest
from trust_bo import Float, TRustBOEngine

pytestmark = pytest.mark.eval

SEEDS = list(range(5))
BUDGET = 200
BATCH = 10
N_DIMS = 10

BASE_CFG = {
    "epochs": 200,
    "ensemble_size": 5,
    "n_cem_samples": 512,
    "n_cem_iters": 10,
    "acquisition": "ei",
}

# TR あり: l_init=0.5 → expand/shrink 発動する設定
TR_DYN_CONFIG = {**BASE_CFG, "l_init": 0.5, "l_max": 1.0, "l_min": 0.0078125,
                 "tau_succ": 3, "tau_fail": 5}
# TR なし相当: 常に全空間を探索 (l_init=1.0=l_max, dynamics frozen)
TR_FRZ_CONFIG = {**BASE_CFG, "l_init": 1.0, "l_max": 1.0, "l_min": 0.0078125,
                 "tau_succ": 9999, "tau_fail": 9999}


def ackley(p: dict) -> float:
    x = list(p.values())
    d = len(x)
    a = -20 * math.exp(-0.2 * math.sqrt(sum(xi**2 for xi in x) / d))
    b = -math.exp(sum(math.cos(2 * math.pi * xi) for xi in x) / d)
    return a + b + 20 + math.e


def rosenbrock(p: dict) -> float:
    x = list(p.values())
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def run(fn, seed, config):
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(N_DIMS)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed, config=config)
    for _ in range(BUDGET // BATCH):
        cands = engine.ask(batch_size=BATCH)
        engine.tell(cands, [{"value": fn(c), "feasible": True} for c in cands])
    tr = engine.tr_state()
    best = engine.best()["objective_values"][0]
    return best, tr


def run_random(fn, seed):
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(N_DIMS)]
    engine = TRustBOEngine(space=space, direction="minimize", seed=seed,
                       config={"n_init": BUDGET + 1})
    cands = engine.ask(batch_size=BUDGET)
    engine.tell(cands, [{"value": fn(c), "feasible": True} for c in cands])
    return engine.best()["objective_values"][0]


def _ablate(fn, fn_name):
    dyn_vals, frz_vals, rnd_vals = [], [], []
    dyn_trs, frz_trs = [], []

    for s in SEEDS:
        v_dyn, tr_dyn = run(fn, s, TR_DYN_CONFIG)
        v_frz, tr_frz = run(fn, s, TR_FRZ_CONFIG)
        v_rnd = run_random(fn, s)
        dyn_vals.append(v_dyn)
        frz_vals.append(v_frz)
        rnd_vals.append(v_rnd)
        if tr_dyn:
            dyn_trs.append(tr_dyn["side_length"])
        if tr_frz:
            frz_trs.append(tr_frz["side_length"])

    med_dyn = statistics.median(dyn_vals)
    med_frz = statistics.median(frz_vals)
    med_rnd = statistics.median(rnd_vals)

    print(f"\n{fn_name} {N_DIMS}D  budget={BUDGET}  seeds={SEEDS}")
    print(f"  TR dynamic: median={med_dyn:.3f}  vals={[round(v,3) for v in dyn_vals]}")
    print(f"             final L: {[round(l,4) for l in dyn_trs]}")
    print(f"  TR frozen : median={med_frz:.3f}  vals={[round(v,3) for v in frz_vals]}")
    print(f"             final L: {[round(l,4) for l in frz_trs]}")
    print(f"  Random    : median={med_rnd:.3f}  vals={[round(v,3) for v in rnd_vals]}")
    print(f"  TR dyn vs Random : {(med_rnd - med_dyn)/(abs(med_rnd)+1e-9)*100:.1f}% improvement")
    print(f"  TR frz vs Random : {(med_rnd - med_frz)/(abs(med_rnd)+1e-9)*100:.1f}% improvement")

    return med_dyn, med_frz, med_rnd, dyn_trs


def test_tr_dynamics_fire():
    """TR dynamics が実際に発動する (l が 0.5 から変化する) ことを確認。"""
    _, tr = run(ackley, SEEDS[0], TR_DYN_CONFIG)
    assert tr is not None
    assert tr["side_length"] != 0.5, \
        f"TR side_length stuck at 0.5 — dynamics not firing (succ={tr['success_count']} fail={tr['failure_count']})"
    print(f"\n  final TR: L={tr['side_length']:.4f}  succ={tr['success_count']}  fail={tr['failure_count']}")


def test_ackley_ablation():
    """TR dynamic と TR frozen の両方が random を上回ることを確認。"""
    med_dyn, med_frz, med_rnd, _ = _ablate(ackley, "Ackley")
    assert med_dyn < med_rnd, f"TR dynamic {med_dyn:.3f} >= random {med_rnd:.3f}"
    assert med_frz < med_rnd, f"TR frozen {med_frz:.3f} >= random {med_rnd:.3f}"


def test_rosenbrock_ablation():
    """Rosenbrock でも確認。"""
    med_dyn, med_frz, med_rnd, _ = _ablate(rosenbrock, "Rosenbrock")
    assert med_dyn < med_rnd, f"TR dynamic {med_dyn:.3f} >= random {med_rnd:.3f}"
    assert med_frz < med_rnd, f"TR frozen {med_frz:.3f} >= random {med_rnd:.3f}"
