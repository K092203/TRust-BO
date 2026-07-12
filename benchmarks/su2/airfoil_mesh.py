"""
airfoil_mesh.py — Phase H-2-2: 純 Python 構造 O-mesh 生成器

CST(Kulfan)翼型 → 境界層クラスタリング付き構造 O-mesh → SU2 ネイティブ形式。
gmsh 不要(libGLU 依存を回避)、座標分布・y+ を完全制御でき再現性も高い。

O-mesh トポロジ:
  - 内側リング(j=0): 翼型表面。外側リング(j=nj-1): 遠方境界の円。
  - 各表面点から中心(0.5, 0)を通る放射線に沿って外側へ押し出し、
    壁近傍は幾何級数で密にする(y+~1 を狙う)。
  - i 方向(周方向)は周期的に閉じる。

使い方:
    from airfoil_mesh import cst_coords, generate_omesh, write_su2
    xu, yu, xl, yl = cst_coords(w_upper, w_lower, n_half=100)
    mesh = generate_omesh(w_upper, w_lower, ...)
    write_su2(mesh, "airfoil.su2")

    python airfoil_mesh.py            # NACA0012 検証メッシュを出力 + 統計
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ── CST(Kulfan)翼型 ───────────────────────────────────────────────────────────

def _bernstein(n: int, x: np.ndarray) -> np.ndarray:
    """次数 n の Bernstein 基底 (n+1, len(x))。"""
    from math import comb
    out = np.empty((n + 1, x.size))
    for i in range(n + 1):
        out[i] = comb(n, i) * x**i * (1.0 - x) ** (n - i)
    return out


def cst_surface(weights: np.ndarray, x: np.ndarray,
                n1: float = 0.5, n2: float = 1.0,
                te_thickness: float = 0.0) -> np.ndarray:
    """CST 単面: y = C(x)·S(x) + x·(te/2)。weights は Bernstein 係数。"""
    weights = np.asarray(weights, dtype=float)
    n = len(weights) - 1
    cls = x**n1 * (1.0 - x) ** n2
    shape = (_bernstein(n, x) * weights[:, None]).sum(axis=0)
    return cls * shape + x * (te_thickness / 2.0)


def cst_coords(w_upper: np.ndarray, w_lower: np.ndarray, n_half: int = 100,
               te_thickness: float = 0.0):
    """CST 上下面の座標を cosine 分布(前縁・後縁で密)で返す。"""
    beta = np.linspace(0.0, math.pi, n_half)
    x = 0.5 * (1.0 - np.cos(beta))          # 0..1、両端クラスタ
    yu = cst_surface(w_upper, x, te_thickness=te_thickness)
    yl = cst_surface(w_lower, x, te_thickness=te_thickness)
    return x, yu, x, yl


def cst_geometry_metrics(w_upper: np.ndarray, w_lower: np.ndarray,
                         n_points: int = 401) -> dict:
    """Return deterministic thickness metrics for a CST airfoil."""
    if n_points < 3:
        raise ValueError("n_points must be at least 3")
    x, yu, _, yl = cst_coords(
        w_upper, w_lower, n_half=n_points
    )
    thickness = yu - yl
    dx = x[1:] - x[:-1]
    area = float(np.sum(0.5 * (thickness[:-1] + thickness[1:]) * dx))
    return {
        "max_thickness": float(np.max(thickness)),
        "area": area,
        "min_internal_thickness": float(np.min(thickness[1:-1])),
    }


def validate_cst_geometry(
    w_upper: np.ndarray,
    w_lower: np.ndarray,
    *,
    min_max_thickness: float = 0.06,
    min_area: float = 0.05,
    crossing_tol: float = 1e-10,
    n_points: int = 401,
) -> tuple[bool, dict, str | None]:
    """Validate basic physical CST geometry before mesh generation or CFD."""
    wu = np.asarray(w_upper, dtype=float)
    wl = np.asarray(w_lower, dtype=float)
    if wu.ndim != 1 or wl.ndim != 1 or wu.size == 0 or wu.size != wl.size:
        return False, {}, "invalid_weights"
    if not (np.all(np.isfinite(wu)) and np.all(np.isfinite(wl))):
        return False, {}, "nonfinite_geometry"
    metrics = cst_geometry_metrics(wu, wl, n_points)
    if not all(np.isfinite(value) for value in metrics.values()):
        return False, metrics, "nonfinite_geometry"
    if metrics["min_internal_thickness"] <= crossing_tol:
        return False, metrics, "surface_crossing"
    if metrics["max_thickness"] < min_max_thickness:
        return False, metrics, "max_thickness"
    if metrics["area"] < min_area:
        return False, metrics, "section_area"
    return True, metrics, None


def naca00xx_weights() -> tuple[np.ndarray, np.ndarray]:
    """NACA0012 を近似する 8 項 CST 重み(検証用)。

    NACA0012 表面を CST(N1=0.5,N2=1.0,8 係数)に最小二乗フィットして得た重み。
    """
    # NACA0012 厚み式から最小二乗で求めた上面重み(下面は対称で符号反転)
    wu = np.array([0.1718, 0.1528, 0.1606, 0.1206, 0.2010, 0.0931, 0.1719, 0.1063])
    wl = -wu
    return wu, wl


def naca00xx_coords(thickness: float = 0.12, n_half: int = 100):
    """解析式による NACA00xx(検証の基準)。sharp TE。"""
    beta = np.linspace(0.0, math.pi, n_half)
    x = 0.5 * (1.0 - np.cos(beta))
    t = thickness
    # sharp TE 係数 (-0.1036) 版
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1036 * x**4)
    return x, yt, x, -yt


# ── O-mesh 生成 ────────────────────────────────────────────────────────────────

@dataclass
class Mesh2D:
    """構造 O-mesh。points: (npoin,2)、quads: (nelem,4) インデックス、
    airfoil_edges / farfield_edges: (n,2) の線分インデックス。"""
    points: np.ndarray
    quads: np.ndarray
    airfoil_edges: np.ndarray
    farfield_edges: np.ndarray
    ni: int  # 周方向点数
    nj: int  # 半径方向点数


def _airfoil_loop(xu, yu, xl, yl) -> np.ndarray:
    """上下面から閉ループ(周方向、重複なし)を構築する。

    順序: TE →(上面)→ LE →(下面)→ TE 手前。
    LE・TE は一度だけ含める。戻り値 (ni, 2)。
    """
    upper = np.column_stack([xu, yu])     # LE→TE
    lower = np.column_stack([xl, yl])     # LE→TE
    # TE..LE(上面を反転) + LE と TE を除いた下面内部
    loop = np.vstack([upper[::-1], lower[1:-1]])
    return loop


def _radial_spacing(nj: int, first_height: float, total_length: float,
                    growth: float = 1.2) -> np.ndarray:
    """壁(0)→遠方(1)の正規化半径分布。幾何級数クラスタリング。

    first_height/total_length を最初のセル比とし、growth 比で成長。
    nj-1 セルで [0,1] を覆うよう正規化する。
    """
    h0 = first_height / total_length
    # 幾何級数のセル幅
    widths = h0 * growth ** np.arange(nj - 1)
    s = np.concatenate([[0.0], np.cumsum(widths)])
    s /= s[-1]  # 正規化して [0,1]
    return s


def generate_omesh(w_upper=None, w_lower=None, *,
                   coords=None,
                   n_half: int = 100, nj: int = 80,
                   far_radius: float = 50.0,
                   first_cell: float = 1.0e-5,
                   growth: float = 1.2,
                   te_thickness: float = 0.0,
                   center: tuple[float, float] | None = None) -> Mesh2D:
    """CST 重み(または coords=(xu,yu,xl,yl))から O-mesh を生成する。

    center=None のとき翼型の重心を射線中心に用いる(キャンバ付き・厚翼でも
    星形になりやすく、メッシュ生成の成功率が上がる)。
    """
    if coords is not None:
        xu, yu, xl, yl = coords
    else:
        xu, yu, xl, yl = cst_coords(w_upper, w_lower, n_half=n_half,
                                    te_thickness=te_thickness)
    loop = _airfoil_loop(xu, yu, xl, yl)       # (ni, 2)
    ni = len(loop)
    if center is None:
        cx, cy = float(loop[:, 0].mean()), float(loop[:, 1].mean())
    else:
        cx, cy = center

    # 放射方向の押し出し: 各表面点から中心(0.5,0)を通る射線に沿って遠方の円へ。
    # 翼型が中心に対して星形である限り射線は交差せず、任意の CST 形状に対して
    # 頑健・一様なメッシュを与える(壁直交版と RANS 結果が一致したため放射版を採用)。
    rel = loop - np.array([cx, cy])
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    outer = np.column_stack([cx + far_radius * np.cos(ang),
                             cy + far_radius * np.sin(ang)])
    s = _radial_spacing(nj, first_cell, far_radius, growth)  # (nj,) 0..1
    pts = np.empty((ni, nj, 2))
    for j in range(nj):
        pts[:, j, :] = loop + s[j] * (outer - loop)
    points = pts.reshape(ni * nj, 2)

    def pid(i, j):
        return (i % ni) * nj + j

    # quad セル: (i,j),(i+1,j),(i+1,j+1),(i,j+1)、周方向は周期的
    quads = np.empty((ni * (nj - 1), 4), dtype=np.int64)
    k = 0
    for i in range(ni):
        for j in range(nj - 1):
            quads[k] = [pid(i, j), pid(i + 1, j), pid(i + 1, j + 1), pid(i, j + 1)]
            k += 1

    # 内側(翼型)・外側(遠方)境界の線分
    airfoil_edges = np.array([[pid(i, 0), pid(i + 1, 0)] for i in range(ni)],
                             dtype=np.int64)
    farfield_edges = np.array([[pid(i, nj - 1), pid(i + 1, nj - 1)] for i in range(ni)],
                              dtype=np.int64)

    mesh = Mesh2D(points, quads, airfoil_edges, farfield_edges, ni, nj)
    _ensure_ccw(mesh)
    return mesh


def _ensure_ccw(mesh: Mesh2D) -> None:
    """全 quad を反時計回り(正の符号付き面積)に揃える。"""
    p = mesh.points
    q = mesh.quads
    # 最初のセルの符号付き面積で全体の向きを判定(構造格子なので一様)
    area = _signed_area(p[q[0]])
    if area < 0:
        mesh.quads = q[:, ::-1].copy()


def _signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


# ── SU2 形式書き出し ──────────────────────────────────────────────────────────

def write_su2(mesh: Mesh2D, path: str,
              airfoil_tag: str = "airfoil",
              farfield_tag: str = "farfield") -> None:
    """SU2 ネイティブ(.su2)2D メッシュを書き出す。quad=VTK 9, line=VTK 3。"""
    p = mesh.points
    q = mesh.quads
    lines = []
    lines.append("NDIME= 2")
    lines.append(f"NELEM= {len(q)}")
    for e in q:
        lines.append(f"9 {e[0]} {e[1]} {e[2]} {e[3]}")
    lines.append(f"NPOIN= {len(p)}")
    for xy in p:
        lines.append(f"{xy[0]:.16e} {xy[1]:.16e}")
    lines.append("NMARK= 2")
    lines.append(f"MARKER_TAG= {airfoil_tag}")
    lines.append(f"MARKER_ELEMS= {len(mesh.airfoil_edges)}")
    for e in mesh.airfoil_edges:
        lines.append(f"3 {e[0]} {e[1]}")
    lines.append(f"MARKER_TAG= {farfield_tag}")
    lines.append(f"MARKER_ELEMS= {len(mesh.farfield_edges)}")
    for e in mesh.farfield_edges:
        lines.append(f"3 {e[0]} {e[1]}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── メッシュ品質チェック ──────────────────────────────────────────────────────

def mesh_quality(mesh: Mesh2D) -> dict:
    """最小セル面積・負面積数・最小法線方向セル高さを返す。"""
    p = mesh.points
    q = mesh.quads
    areas = np.array([_signed_area(p[e]) for e in q])
    # 壁第一層の高さ(j=0 と j=1 の半径差)
    ni, nj = mesh.ni, mesh.nj
    p0 = p[0 * nj + 0]
    p1 = p[0 * nj + 1]
    first_h = float(np.linalg.norm(p1 - p0))
    return {
        "n_cells": len(q),
        "n_points": len(p),
        "min_area": float(areas.min()),
        "n_negative_area": int((areas <= 0).sum()),
        "first_cell_height": first_h,
        "ni": ni, "nj": nj,
    }


# ── エントリポイント(検証)─────────────────────────────────────────────────────

def main():
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/k0903/su2/cases/naca0012_rans/mesh.su2"
    # NACA0012(解析式)で検証メッシュを生成
    coords = naca00xx_coords(thickness=0.12, n_half=120)
    mesh = generate_omesh(coords=coords, nj=90, far_radius=50.0,
                          first_cell=1.0e-5, growth=1.18)
    qual = mesh_quality(mesh)
    print("=== O-mesh 品質 (NACA0012) ===")
    for k, v in qual.items():
        print(f"  {k:20s} = {v}")
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_su2(mesh, out)
    print(f"  written: {out}")
    if qual["n_negative_area"] > 0:
        print("  !! 負面積セルあり — メッシュ無効")
        return 1
    print("  OK: 全セル正面積")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
