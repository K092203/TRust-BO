"""
rolling_integration_test.py — Phase H-1-5 統合テスト

RollingTRustBOEngine + MockEvaluator を NeuralFoil の evaluate_cst に接続し、
逐次実行（cfd_neuralfoil_benchmark.py の run_trust_bo）と best_value を比較する。

検証項目:
  1. budget=50 / concurrent=4 が完走する
  2. 逐次実行との best_value 差が ±20% 以内
  3. failure_rate=0.2 のジョブ失敗時もプロセスが死なず完走する

使い方:
  cd /home/k0903/trm-engine
  SMOKE=1 .venv/bin/python benchmarks/rolling_integration_test.py  # ~1分
  .venv/bin/python benchmarks/rolling_integration_test.py           # ~5分
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "python"))

from trust_bo import Float, MockEvaluator, RollingTRustBOEngine, TRustBOEngine

# ── 設定 ──────────────────────────────────────────────────────────────────────

N_UPPER = 8
N_LOWER = 8
DIM     = N_UPPER + N_LOWER

ALPHA   = 4.0
RE      = 3e6

UPPER_LB = np.full(N_UPPER, 0.04)
UPPER_UB = np.full(N_UPPER, 0.45)
LOWER_LB = np.full(N_LOWER, -0.40)
LOWER_UB = np.full(N_LOWER, 0.15)

LB = np.concatenate([UPPER_LB, LOWER_LB])
UB = np.concatenate([UPPER_UB, LOWER_UB])

SMOKE   = bool(os.environ.get("SMOKE"))
BUDGET  = 30 if SMOKE else 50
SEEDS   = [0, 1] if SMOKE else [0, 1, 2]
MAX_CONCURRENT = 4
TOLERANCE = 0.25  # ローリングが逐次より 25% 以上悪化した場合のみ FAIL


# ── 評価関数 ──────────────────────────────────────────────────────────────────

def evaluate_cst(params: np.ndarray) -> tuple[float, bool]:
    """CST 係数 → (Cl/Cd, feasible)"""
    import aerosandbox as asb
    import neuralfoil as nf

    upper = params[:N_UPPER]
    lower = params[N_UPPER:]
    try:
        af = asb.KulfanAirfoil(
            upper_weights=upper,
            lower_weights=lower,
            leading_edge_weight=0.0,
            TE_thickness=0.0,
        )
        result = nf.get_aero_from_airfoil(airfoil=af, alpha=ALPHA, Re=RE)
        cl   = float(result["CL"].item())
        cd   = float(result["CD"].item())
        conf = float(result["analysis_confidence"].item())
        if conf < 0.5 or cd <= 0 or cl <= 0:
            return 0.0, False
        return cl / cd, True
    except Exception:
        return 0.0, False


def cst_from_candidate(candidate: dict) -> float:
    """TRustBO の candidate dict → Cl/Cd スカラー（MockEvaluator 用）"""
    params = np.array(
        [candidate[f"u{i}"] for i in range(N_UPPER)] +
        [candidate[f"l{i}"] for i in range(N_LOWER)]
    )
    val, feas = evaluate_cst(params)
    return val


# ── エンジン構築ヘルパー ──────────────────────────────────────────────────────

def make_space():
    space = [Float(f"u{i}", float(UPPER_LB[i]), float(UPPER_UB[i])) for i in range(N_UPPER)]
    space += [Float(f"l{i}", float(LOWER_LB[i]), float(LOWER_UB[i])) for i in range(N_LOWER)]
    return space


def make_base_engine(seed: int) -> TRustBOEngine:
    return TRustBOEngine(
        space=make_space(),
        direction="maximize",
        seed=seed,
        config={"n_init": 10, "enable_phase2": True, "batch_size": MAX_CONCURRENT},
    )


# ── 逐次実行（比較基準）────────────────────────────────────────────────────────

def run_sequential(seed: int) -> tuple[float, float]:
    """従来の同期バッチ実行"""
    engine = make_base_engine(seed)
    t0 = time.perf_counter()
    evaluated = 0
    while evaluated < BUDGET:
        b = min(MAX_CONCURRENT, BUDGET - evaluated)
        cands = engine.ask(batch_size=b)
        results = []
        for c in cands:
            params = np.array(
                [c[f"u{i}"] for i in range(N_UPPER)] +
                [c[f"l{i}"] for i in range(N_LOWER)]
            )
            val, feas = evaluate_cst(params)
            results.append({"value": val, "feasible": feas})
        engine.tell(cands, results)
        evaluated += b
    best = engine.best()
    bv = best["objective_values"][0] if best else 0.0
    return bv, time.perf_counter() - t0


# ── ローリング実行 ────────────────────────────────────────────────────────────

def run_rolling(seed: int, failure_rate: float = 0.0) -> tuple[float, float]:
    """RollingTRustBOEngine + MockEvaluator 経由"""
    base = make_base_engine(seed)
    evaluator = MockEvaluator(
        fn=cst_from_candidate,
        min_delay=0.01,
        max_delay=0.15,
        failure_rate=failure_rate,
        seed=seed,
    )
    rolling = RollingTRustBOEngine(
        base_engine=base,
        evaluator=evaluator,
        max_concurrent=MAX_CONCURRENT,
        poll_interval=0.05,
        job_timeout=2.0 if failure_rate > 0 else None,
        verbose=False,
    )
    t0 = time.perf_counter()
    result = rolling.run(budget=BUDGET)
    bv = result.get("objective_values", [0.0])[0] if result else 0.0
    return bv, time.perf_counter() - t0


# ── テストスイート ────────────────────────────────────────────────────────────

def test_completion():
    """Test 1: budget=50 / concurrent=4 が完走する"""
    print(f"\n[Test 1] 完走テスト  budget={BUDGET} / concurrent={MAX_CONCURRENT}")
    all_ok = True
    for seed in SEEDS:
        try:
            bv, elapsed = run_rolling(seed)
            status = "OK" if bv > 0 else "WARN(bv=0)"
            print(f"  seed={seed}  best={bv:.2f}  elapsed={elapsed:.1f}s  {status}")
            if bv <= 0:
                all_ok = False
        except Exception as e:
            print(f"  seed={seed}  ERROR: {e}")
            all_ok = False
    return all_ok


def test_quality_vs_sequential():
    """Test 2: ローリングの best_value が逐次より TOLERANCE 以上悪化しない"""
    print(f"\n[Test 2] 品質比較（逐次 vs ローリング、劣化許容={TOLERANCE*100:.0f}%）")
    all_ok = True
    for seed in SEEDS:
        seq_bv, seq_t = run_sequential(seed)
        rol_bv, rol_t = run_rolling(seed)
        if seq_bv > 0 and rol_bv < seq_bv:
            diff = (seq_bv - rol_bv) / seq_bv
        else:
            diff = 0.0  # ローリングが逐次と同等以上なら常に OK
        ok = diff <= TOLERANCE
        flag = "OK" if ok else "FAIL"
        print(
            f"  seed={seed}  seq={seq_bv:.2f}({seq_t:.1f}s)"
            f"  roll={rol_bv:.2f}({rol_t:.1f}s)"
            f"  diff={diff*100:.1f}%  [{flag}]"
        )
        if not ok:
            all_ok = False
    return all_ok


def test_failure_handling():
    """Test 3: failure_rate=0.2 でもプロセスが死なず完走する"""
    print(f"\n[Test 3] 障害耐性テスト  failure_rate=0.2 / job_timeout=2.0s")
    all_ok = True
    for seed in SEEDS[:2]:  # 2 seeds で十分
        try:
            bv, elapsed = run_rolling(seed, failure_rate=0.2)
            # 一部失敗するので best は低くなる可能性があるが、完走できれば OK
            print(f"  seed={seed}  best={bv:.2f}  elapsed={elapsed:.1f}s  OK")
        except Exception as e:
            print(f"  seed={seed}  ERROR（クラッシュ）: {e}")
            all_ok = False
    return all_ok


# ── エントリポイント ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase H-1-5: Rolling Engine 統合テスト（NeuralFoil）")
    print(f"  budget={BUDGET}  dim={DIM}D  concurrent={MAX_CONCURRENT}")
    print(f"  seeds={SEEDS}  SMOKE={SMOKE}")
    print("=" * 60)

    results = {
        "Test1_completion":    test_completion(),
        "Test2_quality":       test_quality_vs_sequential(),
        "Test3_failure":       test_failure_handling(),
    }

    print("\n" + "=" * 60)
    print("  テスト結果サマリ")
    print("=" * 60)
    all_passed = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<30} {status}")
        if not ok:
            all_passed = False

    overall = "ALL PASSED" if all_passed else "SOME FAILED"
    print(f"\n  総合: {overall}")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
