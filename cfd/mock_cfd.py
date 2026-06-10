"""mock_cfd.py — OpenFOAM接続前の開発・テスト用モックCFD関数

薄翼理論 + 境界層近似 + ランダムノイズで Cl/Cd を推定する。
実際のOpenFOAM接続時はrun_openfoam_case()を差し替えるだけでよい。
"""
from __future__ import annotations
import math, time
import numpy as np


def _thin_airfoil_cl(m: float, aoa_deg: float) -> float:
    """薄翼理論による揚力係数推定 Cl = 2π(α + 2m)"""
    aoa_rad = math.radians(aoa_deg)
    return 2 * math.pi * (aoa_rad + 2 * m)


def _blasius_cd(t: float, re: float = 2e6) -> float:
    """平板境界層 (Blasius) + 翼厚抵抗の近似
    Cd_friction ≈ 1.328/√Re (層流), 翼厚補正 +4t
    """
    cd_friction = 1.328 / math.sqrt(re)
    cd_thickness = 4 * t * cd_friction
    return cd_friction + cd_thickness


def _induced_drag(cl: float, ar: float = 8.0) -> float:
    """誘導抵抗 Cd_i = Cl^2 / (π e AR), e=0.85"""
    return cl**2 / (math.pi * 0.85 * ar)


def mock_cfd(m: float, p: float, t: float, aoa_deg: float = 5.0,
             noise_seed: int | None = None) -> dict | None:
    """
    NACA翼型パラメータ → Cl, Cd, Cl/Cd を返す（モック）

    Args:
        m: 最大キャンバー (0.0-0.09)
        p: キャンバー位置 (0.2-0.8)
        t: 最大翼厚 (0.08-0.35)
        aoa_deg: 迎角 [deg]
        noise_seed: 再現性用シード (None=乱数)

    Returns:
        {"Cl": float, "Cd": float, "Cl_Cd": float}
        失敗時: None
    """
    # 物理的に不合理なパラメータは失敗扱い
    if t < 0.06 or t > 0.40:
        return None
    if m < 0.0 or m > 0.12:
        return None

    rng = np.random.default_rng(noise_seed)

    cl_theory = _thin_airfoil_cl(m, aoa_deg)
    cd_theory = _blasius_cd(t) + _induced_drag(cl_theory)

    # 失速モデル: aoa > 12度 または t < 0.08 で急激な Cl 低下・Cd 増大
    stall_factor = 1.0
    if aoa_deg > 12.0:
        excess = aoa_deg - 12.0
        stall_factor = math.exp(-0.3 * excess)
        cd_theory *= (1.0 + 0.5 * excess)
    if t < 0.08:
        stall_factor *= 0.8
        cd_theory *= 1.5

    cl = cl_theory * stall_factor
    cd = max(cd_theory, 0.005)  # 下限

    # CFDノイズ（実測値の変動を模倣）
    noise_scale = 0.02
    cl += rng.normal(0, noise_scale * abs(cl))
    cd += rng.normal(0, noise_scale * cd)
    cd = max(cd, 0.001)

    # ダウンウォッシュ補正（キャンバー位置効果）
    p_effect = 1.0 + 0.05 * (p - 0.4)
    cl *= p_effect

    # 実行時間をシミュレート（実OpenFOAMは1-5分）
    time.sleep(0.05)

    return {"Cl": round(cl, 4), "Cd": round(cd, 6), "Cl_Cd": round(cl / cd, 2)}


def run_openfoam_case(m: float, p: float, t: float, aoa_deg: float = 5.0,
                      case_dir: str = "/tmp/of_case",
                      timeout: int = 1800) -> dict | None:
    """
    OpenFOAM実行インターフェース（現在はモック）

    実OpenFOAM接続時はこの関数を以下のように差し替える:
        1. generate_airfoil.py で翼型座標生成
        2. blockMesh でC型メッシュ生成
        3. simpleFoam 実行（mpirun -np 12）
        4. postProcess で forceCoeffs 抽出
        5. {"Cl": float, "Cd": float, "Cl_Cd": float} を返す

    実OpenFOAM実行例:
        import subprocess
        subprocess.run(["blockMesh", "-case", case_dir], check=True, timeout=timeout)
        subprocess.run(["mpirun", "-np", "12", "simpleFoam", "-parallel",
                        "-case", case_dir], check=True, timeout=timeout)
    """
    seed = hash((m, p, t, aoa_deg)) & 0xFFFFFFFF
    return mock_cfd(m, p, t, aoa_deg, noise_seed=seed)


if __name__ == "__main__":
    result = run_openfoam_case(0.04, 0.4, 0.12, aoa_deg=5.0)
    print(f"NACA4412 @ 5°: Cl={result['Cl']}, Cd={result['Cd']}, "
          f"Cl/Cd={result['Cl_Cd']}")

    # 薄翼理論との比較
    cl_theory = _thin_airfoil_cl(0.04, 5.0)
    print(f"薄翼理論 Cl={cl_theory:.4f}")
