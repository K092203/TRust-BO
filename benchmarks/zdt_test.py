"""
zdt_test.py — Phase K-1-4/K-1-5 検証スクリプト

テスト方針: "model-guided が random init を超えることを確認"
  比較対象 = 最初の n_init ランダム評価（baseline）
  基準    = HV_final / HV_init >= 2.0

テスト 1 (K-1-4): ZDT1 5D
  budget=100、model-guided による HV が n_init ランダム初期化の 2× 以上

テスト 2 (K-1-5): NeuralFoil 16D CST Cl/Cd 同時最適化
  SMOKE=1 のときはスキップ

使い方:
    cd /home/k0903/trm-engine
    SMOKE=1 .venv/bin/python benchmarks/zdt_test.py   # ~1分
    .venv/bin/python benchmarks/zdt_test.py            # ~10分
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "python"))

from trust_bo import Float, MultiObjectiveEngine, hypervolume_2d
from trust_bo.multiobjective import _pareto_mask

SMOKE   = bool(os.environ.get("SMOKE"))
BUDGET  = 40  if SMOKE else 100
SEEDS   = [0, 1] if SMOKE else [0, 1, 2]
BATCH   = 4
N_INIT  = 10

ZDT1_N   = 5
REF_ZDT1 = np.array([1.1, 2.0])
REF_CLCD = np.array([0.0, 0.05])  # 最小化空間: (-Cl, Cd) の参照点

# 初期 HV が 0 のときに HV_final との比較が不定になるのを防ぐ最小基準
HV_MIN_ABS = 0.05


# ── ユーティリティ ────────────────────────────────────────────────────────────

def hv_from_history(history: list[dict], ref: np.ndarray, n: int | None = None) -> float:
    """history の先頭 n 件（None=全件）の feasible な評価から HV を計算する。"""
    entries = history[:n] if n is not None else history
    feasible = [e for e in entries if e["feasible"] and e["values"] is not None]
    if not feasible:
        return 0.0
    costs = np.array([e["values"] for e in feasible])
    mask = _pareto_mask(costs)
    return hypervolume_2d(costs[mask], ref)


# ── ZDT1 ──────────────────────────────────────────────────────────────────────

def zdt1(x: dict) -> tuple[float, float]:
    """ZDT1 5D（最小化）。真の Pareto: g=1 → f2 = 1 - sqrt(f1)"""
    xs = np.array([x[f"x{i}"] for i in range(ZDT1_N)])
    f1 = float(xs[0])
    g  = 1.0 + 9.0 * float(np.sum(xs[1:])) / (ZDT1_N - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return f1, f2


def run_mo_zdt1(seed: int) -> tuple[float, float]:
    """(HV_init, HV_final) を返す。"""
    space = [Float(f"x{i}", 0.0, 1.0) for i in range(ZDT1_N)]
    engine = MultiObjectiveEngine(
        space=space, directions=["minimize", "minimize"],
        seed=seed, config={"n_init": N_INIT, "batch_size": BATCH},
    )
    evaluated = 0
    while evaluated < BUDGET:
        b = min(BATCH, BUDGET - evaluated)
        cands = engine.ask(batch_size=b)
        engine.tell(cands, [{"values": list(zdt1(c)), "feasible": True} for c in cands])
        evaluated += b
    h = engine.history()
    hv_init  = hv_from_history(h, REF_ZDT1, n=N_INIT)
    hv_final = hv_from_history(h, REF_ZDT1)
    return hv_init, hv_final


# ── NeuralFoil Cl/Cd ──────────────────────────────────────────────────────────

N_UPPER  = 8
N_LOWER  = 8
ALPHA_DEG = 4.0
RE        = 3e6
UPPER_LB  = np.full(N_UPPER, 0.04)
UPPER_UB  = np.full(N_UPPER, 0.45)
LOWER_LB  = np.full(N_LOWER, -0.40)
LOWER_UB  = np.full(N_LOWER,  0.15)


def evaluate_cst_mo(candidate: dict) -> tuple[float, float, bool]:
    import aerosandbox as asb
    import neuralfoil as nf
    upper = np.array([candidate[f"u{i}"] for i in range(N_UPPER)])
    lower = np.array([candidate[f"l{i}"] for i in range(N_LOWER)])
    try:
        af = asb.KulfanAirfoil(
            upper_weights=upper, lower_weights=lower,
            leading_edge_weight=0.0, TE_thickness=0.0,
        )
        r    = nf.get_aero_from_airfoil(airfoil=af, alpha=ALPHA_DEG, Re=RE)
        cl   = float(r["CL"].item())
        cd   = float(r["CD"].item())
        conf = float(r["analysis_confidence"].item())
        if conf < 0.5 or cd <= 0 or cl <= 0:
            return 0.0, 0.0, False
        return cl, cd, True
    except Exception:
        return 0.0, 0.0, False


def run_mo_clcd(seed: int) -> tuple[float, float]:
    """(HV_init, HV_final) を返す。"""
    space  = [Float(f"u{i}", float(UPPER_LB[i]), float(UPPER_UB[i])) for i in range(N_UPPER)]
    space += [Float(f"l{i}", float(LOWER_LB[i]), float(LOWER_UB[i])) for i in range(N_LOWER)]
    engine = MultiObjectiveEngine(
        space=space, directions=["maximize", "minimize"],
        seed=seed, config={"n_init": N_INIT, "batch_size": BATCH},
    )
    evaluated = 0
    while evaluated < BUDGET:
        b = min(BATCH, BUDGET - evaluated)
        cands = engine.ask(batch_size=b)
        results = []
        for c in cands:
            cl, cd, feas = evaluate_cst_mo(c)
            results.append({"values": [cl, cd], "feasible": feas})
        engine.tell(cands, results)
        evaluated += b

    h = engine.history()
    # HV 計算: 最小化空間に変換 (signs=[-1,1])
    signs = np.array([-1.0, 1.0])
    def clcd_hv(entries, n=None):
        es = entries[:n] if n is not None else entries
        feas = [e for e in es if e["feasible"] and e["values"] is not None]
        if not feas:
            return 0.0
        costs = np.array([signs * np.asarray(e["values"]) for e in feas])
        mask = _pareto_mask(costs)
        return hypervolume_2d(costs[mask], REF_CLCD)

    return clcd_hv(h, n=N_INIT), clcd_hv(h)


# ── テストスイート ────────────────────────────────────────────────────────────

def check_improvement(hv_init: float, hv_final: float,
                      min_ratio: float = 2.0) -> tuple[bool, float, str]:
    """(ok, ratio_or_abs, display_str)"""
    if hv_init > 0:
        ratio = hv_final / hv_init
        ok    = ratio >= min_ratio
        return ok, ratio, f"ratio={ratio:.2f} (>={min_ratio})"
    else:
        ok = hv_final >= HV_MIN_ABS
        return ok, hv_final, f"HV_final={hv_final:.4f} (init=0, need>={HV_MIN_ABS})"


def test_zdt1():
    print(f"\n[K-1-4] ZDT1 {ZDT1_N}D  budget={BUDGET}  ref={REF_ZDT1.tolist()}  seeds={SEEDS}")
    print(f"  基準: HV_final / HV_init >= 2.0 (HV_init=0 のときは HV_final >= {HV_MIN_ABS})")
    all_ok = True
    for seed in SEEDS:
        hv_init, hv_final = run_mo_zdt1(seed)
        ok, metric, detail = check_improvement(hv_init, hv_final)
        flag = "OK" if ok else "FAIL"
        print(f"  seed={seed}  HV_init={hv_init:.4f}  HV_final={hv_final:.4f}  {detail}  [{flag}]")
        if not ok:
            all_ok = False
    print(f"  -> {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_clcd():
    if SMOKE:
        print("\n[K-1-5] NeuralFoil Cl/Cd テスト → SMOKE=1 のためスキップ")
        return True
    # 16D CST は TRust-BO 苦手ゾーン（<50D）なので基準を 1.2× に緩める
    min_ratio_clcd = 1.2
    print(f"\n[K-1-5] NeuralFoil Cl/Cd 2目的最適化  budget={BUDGET}  seeds={SEEDS}")
    print(f"  基準: HV_final / HV_init >= {min_ratio_clcd} (16D は TRust-BO 苦手ゾーン)")
    all_ok = True
    for seed in SEEDS:
        t0               = time.perf_counter()
        hv_init, hv_final = run_mo_clcd(seed)
        elapsed          = time.perf_counter() - t0
        ok, metric, detail = check_improvement(hv_init, hv_final, min_ratio=min_ratio_clcd)
        flag = "OK" if ok else "FAIL"
        print(f"  seed={seed}  HV_init={hv_init:.5f}  HV_final={hv_final:.5f}"
              f"  {detail}  ({elapsed:.1f}s)  [{flag}]")
        if not ok:
            all_ok = False
    print(f"  -> {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ── エントリポイント ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase K-1 検証: ZDT1 5D + NeuralFoil Cl/Cd 2目的最適化")
    print(f"  budget={BUDGET}  batch={BATCH}  seeds={SEEDS}  SMOKE={SMOKE}")
    print("=" * 60)

    results = {
        "K-1-4_ZDT1":    test_zdt1(),
        "K-1-5_NF_ClCd": test_clcd(),
    }

    print("\n" + "=" * 60)
    print("  テスト結果サマリ")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name:<25} {'PASS' if ok else 'FAIL'}")
    overall = all(results.values())
    print(f"\n  総合: {'ALL PASSED' if overall else 'SOME FAILED'}")
    print("=" * 60)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
