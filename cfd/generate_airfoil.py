"""generate_airfoil.py — NACA 4桁翼型座標生成"""
from __future__ import annotations
import numpy as np


def generate_naca_coords(m: float, p: float, t: float,
                         n_points: int = 100) -> dict:
    """
    m: 最大キャンバー (0.0-0.09)
    p: キャンバー最大位置 (0.2-0.8)
    t: 最大翼厚 (0.08-0.35)
    戻り値: {"upper": [(x,y),...], "lower": [(x,y),...], "chord": 1.0}
    """
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2  # cosine spacing [0,1]

    # 翼厚分布
    yt = 5 * t * (0.2969 * np.sqrt(x)
                  - 0.1260 * x
                  - 0.3516 * x**2
                  + 0.2843 * x**3
                  - 0.1015 * x**4)

    # キャンバーライン
    yc = np.where(
        x < p,
        m / p**2 * (2 * p * x - x**2),
        m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2)
    ) if m > 0 else np.zeros_like(x)

    dyc_dx = np.where(
        x < p,
        2 * m / p**2 * (p - x),
        2 * m / (1 - p)**2 * (p - x)
    ) if m > 0 else np.zeros_like(x)

    theta = np.arctan(dyc_dx)

    xu = x  - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x  + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return {
        "upper": list(zip(xu.tolist(), yu.tolist())),
        "lower": list(zip(xl.tolist(), yl.tolist())),
        "chord": 1.0,
        "naca_label": f"NACA{int(m*100):1d}{int(p*10):1d}{int(t*100):02d}",
    }


def write_obj_points(coords: dict, path: str) -> None:
    """翼型座標を点群ファイルに書き出す（OpenFOAM blockMesh用）"""
    upper = coords["upper"]
    lower = coords["lower"]
    with open(path, "w") as f:
        f.write("// NACA airfoil points\n")
        f.write(f"// {coords['naca_label']}\n")
        for x, y in upper:
            f.write(f"({x:.6f} {y:.6f} 0)\n")
        for x, y in reversed(lower[1:-1]):
            f.write(f"({x:.6f} {y:.6f} 0)\n")


if __name__ == "__main__":
    coords = generate_naca_coords(0.04, 0.4, 0.12)
    print(f"Generated {coords['naca_label']}: "
          f"{len(coords['upper'])} upper + {len(coords['lower'])} lower points")
    print(f"Max thickness at x≈0.3: y={max(y for _,y in coords['upper']):.4f}")
