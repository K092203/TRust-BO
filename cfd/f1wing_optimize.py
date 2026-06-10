"""f1wing_optimize.py — F1ウイング2D断面最適化

変数: main_camber, main_thickness, flap_angle, flap_gap
目的: |Cl|/Cd 最大化（ダウンフォース効率）
流体: 60m/s, 迎角-3°, 地面効果あり
"""
from __future__ import annotations
import csv, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np
from trust_bo import TRustBOEngine, Float
from mock_cfd import mock_cfd

CSV_PATH = Path("f1wing_results.csv")
ERROR_LOG = Path("openfoam_errors.txt")


def f1wing_cfd(main_camber: float, main_thickness: float,
               flap_angle: float, flap_gap: float) -> dict | None:
    """
    F1ウイング（メイン翼 + フラップ）の空力特性を推定するモック。

    地面効果: ground clearance=0.05m → 誘導抵抗を約40%増加
    フラップ効果:
      - flap_angle 大 → Cl 増加、Cd も増加
      - flap_gap 小 → 干渉抵抗増加
    """
    # メイン翼型（inverted: ダウンフォース方向）
    base = mock_cfd(main_camber, 0.4, main_thickness,
                    aoa_deg=-3.0,   # inverted, 負の迎角
                    noise_seed=hash((main_camber, main_thickness, flap_angle, flap_gap)) & 0xFFFF)
    if base is None:
        return None

    # フラップ効果（簡易モデル）
    flap_rad = math.radians(flap_angle)
    cl_flap = 0.8 * flap_rad                         # フラップ追加揚力（実際は負方向）
    cd_flap = 0.015 * (flap_angle / 30.0)**2         # フラップ抵抗

    # ギャップ干渉抵抗
    if flap_gap < 0.015:
        cd_flap += 0.02 * (0.015 - flap_gap) / 0.015

    # 地面効果（鏡像法近似: 誘導抵抗40%増）
    cd_ground = 0.40 * base["Cd"]

    cl_total = -(abs(base["Cl"]) + cl_flap)   # ダウンフォース（負）
    cd_total = base["Cd"] + cd_flap + cd_ground
    cd_total = max(cd_total, 0.01)

    time.sleep(0.05)

    return {
        "Cl": round(cl_total, 4),
        "Cd": round(cd_total, 6),
        "abs_Cl_Cd": round(abs(cl_total) / cd_total, 2),
    }


def run(budget: int = 20):
    space = [
        Float("main_camber",    0.02, 0.09),
        Float("main_thickness", 0.08, 0.20),
        Float("flap_angle",     15.0, 45.0),
        Float("flap_gap",       0.01, 0.04),
    ]
    engine = TRustBOEngine(
        space=space,
        direction="maximize",
        seed=42,
        config={"n_init": 6, "batch_size": 1},
    )

    with open(CSV_PATH, "w", newline="", buffering=1) as f:
        csv.writer(f).writerow(
            ["trial", "main_camber", "main_thickness", "flap_angle",
             "flap_gap", "Cl", "Cd", "abs_Cl_Cd", "feasible", "time_s"])

    print(f"F1ウイング2D最適化: budget={budget}, 4変数, mock_cfd=True")
    print(f"{'Trial':>5} {'m_cam':>6} {'m_thk':>6} {'f_ang':>6} "
          f"{'f_gap':>6} {'|Cl|/Cd':>8}")
    print("-" * 55)

    best_val = -float("inf")
    best_params = None
    evaluated = 0

    while evaluated < budget:
        cands = engine.ask(batch_size=1)
        c = cands[0]
        t0 = time.perf_counter()

        try:
            result = f1wing_cfd(
                c["main_camber"], c["main_thickness"],
                c["flap_angle"], c["flap_gap"])
            elapsed = time.perf_counter() - t0

            if result is None:
                engine.tell(cands, [{"value": 0.0, "feasible": False}])
                with open(CSV_PATH, "a", newline="", buffering=1) as f:
                    csv.writer(f).writerow(
                        [evaluated+1, c["main_camber"], c["main_thickness"],
                         c["flap_angle"], c["flap_gap"], "", "", "", False,
                         f"{elapsed:.2f}"])
            else:
                v = result["abs_Cl_Cd"]
                engine.tell(cands, [{"value": v, "feasible": True}])
                with open(CSV_PATH, "a", newline="", buffering=1) as f:
                    csv.writer(f).writerow(
                        [evaluated+1, c["main_camber"], c["main_thickness"],
                         c["flap_angle"], c["flap_gap"],
                         result["Cl"], result["Cd"], v, True,
                         f"{elapsed:.2f}"])
                if v > best_val:
                    best_val = v
                    best_params = dict(c)
                print(f"{evaluated+1:5d} {c['main_camber']:6.4f} "
                      f"{c['main_thickness']:6.4f} {c['flap_angle']:6.1f} "
                      f"{c['flap_gap']:6.4f} {v:8.2f}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            with open(ERROR_LOG, "a") as ef:
                ef.write(f"f1wing trial={evaluated+1} error={e}\n")
            engine.tell(cands, [{"value": 0.0, "feasible": False}])

        evaluated += 1

    print("-" * 55)
    print(f"\n最良 |Cl|/Cd = {best_val:.2f}")
    print(f"最良パラメータ: {best_params}")
    return {"best_abs_Cl_Cd": best_val, "best_params": best_params}


if __name__ == "__main__":
    result = run(budget=20)
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
