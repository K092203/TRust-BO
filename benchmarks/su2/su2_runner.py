"""
su2_runner.py — Phase H-2-3: SU2 RANS 設定テンプレート + Python 実行ラッパー

CST 翼型重み → O-mesh 生成 → SU2 RANS 実行 → Cl/Cd パース、を 1 関数に集約。
評価条件: Ma=0.3, Re=3e6, SA 乱流モデル(roadmap H-2 準拠)。

使い方:
    from su2_runner import run_cst, SU2Settings
    cl, cd, feasible, info = run_cst(w_upper, w_lower, aoa=2.0)
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

from airfoil_mesh import generate_omesh, mesh_quality, validate_cst_geometry, write_su2

# 環境変数 SU2_RUN / SU2_WORK で上書き可能 (未設定時は従来のパスのまま)
SU2_RUN_DEFAULT = os.environ.get("SU2_RUN", "/home/k0903/su2/install/bin")
SU2_WORK_DEFAULT = os.environ.get("SU2_WORK", "/home/k0903/su2/work")

# ── RANS 設定テンプレート ────────────────────────────────────────────────────
# {aoa} {mach} {re} {iter} {mesh} {history} はラッパーが埋める。
RANS_CFG_TEMPLATE = """\
SOLVER= RANS
KIND_TURB_MODEL= SA
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

% --- 自由流条件 ---
MACH_NUMBER= {mach}
AOA= {aoa}
SIDESLIP_ANGLE= 0.0
INIT_OPTION= REYNOLDS
FREESTREAM_TEMPERATURE= 288.15
REYNOLDS_NUMBER= {re}
REYNOLDS_LENGTH= 1.0

% --- 参照値 ---
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= 1.0
REF_AREA= 1.0
REF_DIMENSIONALIZATION= FREESTREAM_VEL_EQ_ONE

% --- 境界条件 ---
MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )

% --- 数値法 ---
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.1, 2.0, 1.0, 100.0 )
ITER= {iter}

% --- 線形ソルバ ---
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10

% --- 流れの離散化 ---
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
TIME_DISCRE_FLOW= EULER_IMPLICIT

% --- 乱流の離散化 ---
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
SLOPE_LIMITER_TURB= VENKATAKRISHNAN
TIME_DISCRE_TURB= EULER_IMPLICIT
CFL_REDUCTION_TURB= 1.0

% --- 収束: 抗力係数の Cauchy 収束で停止(RANS 残差は -7.5 で停滞しがちなため
%     最適化目的である力係数の安定をもって収束とみなす)。max_iter で打ち切り。
CONV_FIELD= ( DRAG )
CONV_RESIDUAL_MINVAL= -12
CONV_STARTITER= 100
CONV_CAUCHY_ELEMS= 200
CONV_CAUCHY_EPS= 5E-6

% --- 入出力 ---
MESH_FILENAME= {mesh}
MESH_FORMAT= SU2
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, RMS_NU_TILDE, LIFT, DRAG )
SCREEN_WRT_FREQ_INNER= 100
HISTORY_OUTPUT= ( INNER_ITER, RMS_RES, AERO_COEFF )
CONV_FILENAME= {history}
OUTPUT_FILES= ( RESTART, CSV )
OUTPUT_WRT_FREQ= 1000
WRT_PERFORMANCE= NO
"""


@dataclass
class SU2Settings:
    mach: float = 0.3
    reynolds: float = 3.0e6
    aoa: float = 2.0
    max_iter: int = 3000
    # メッシュ
    n_half: int = 120
    nj: int = 90
    far_radius: float = 50.0
    first_cell: float = 1.0e-5
    growth: float = 1.18
    te_thickness: float = 0.0
    min_max_thickness: float = 0.06
    min_section_area: float = 0.05
    # 非物理 CD の下限。Re=3e6 の乱流翼型の物理下限 (~4e-3) に対し余裕側の 1e-3。
    # 粗メッシュ・未収束で CD が異常に小さく出る「薄翼 CD≈0」型アーティファクト対策
    # (旧 cd<=1e-6 では Cl/Cd=561 級のケースがすり抜けた — BENCHMARK.md §20.3)。
    min_cd: float = 1.0e-3
    # 実行
    su2_run: str = SU2_RUN_DEFAULT
    n_threads: int = 4
    timeout_s: float = 1800.0
    keep_workdir: bool = False
    workroot: str = field(default_factory=lambda: SU2_WORK_DEFAULT)


def _parse_clcd(history_csv: str) -> tuple[float | None, float | None, int]:
    """history.csv の最終行から CL, CD と反復数を返す。"""
    try:
        rows = list(csv.reader(open(history_csv)))
    except OSError:
        return None, None, 0
    if len(rows) < 2:
        return None, None, 0
    hdr = [h.strip().strip('"') for h in rows[0]]
    last = dict(zip(hdr, [c.strip() for c in rows[-1]]))
    cl = cd = None
    for k, v in last.items():
        ku = k.upper()
        if ku == "CL":
            cl = float(v)
        elif ku == "CD":
            cd = float(v)
    return cl, cd, len(rows) - 1


def run_cst(w_upper, w_lower, aoa: float | None = None,
            settings: SU2Settings | None = None) -> tuple[float, float, bool, dict]:
    """CST 重みから SU2 RANS を実行し (cl, cd, feasible, info) を返す。

    feasible=False は メッシュ不正 / SU2 失敗 / 発散 / 非物理値 を意味する。
    """
    s = settings or SU2Settings()
    if aoa is not None:
        s.aoa = aoa

    t0 = time.perf_counter()
    valid_geometry, geometry, geometry_error = validate_cst_geometry(
        np.asarray(w_upper), np.asarray(w_lower),
        min_max_thickness=s.min_max_thickness,
        min_area=s.min_section_area,
    )
    info: dict = {"workdir": None, "aoa": s.aoa, "geometry": geometry}
    if not valid_geometry:
        info["error"] = f"geometry_{geometry_error}"
        info["elapsed_s"] = time.perf_counter() - t0
        return 0.0, 0.0, False, info

    os.makedirs(s.workroot, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="su2_", dir=s.workroot)
    info["workdir"] = workdir
    try:
        # 1) メッシュ生成
        try:
            mesh = generate_omesh(
                np.asarray(w_upper), np.asarray(w_lower),
                n_half=s.n_half, nj=s.nj, far_radius=s.far_radius,
                first_cell=s.first_cell, growth=s.growth,
                te_thickness=s.te_thickness,
            )
        except Exception as e:  # noqa: BLE001
            info["error"] = f"mesh_gen: {e}"
            return 0.0, 0.0, False, info
        qual = mesh_quality(mesh)
        info["mesh"] = qual
        if qual["n_negative_area"] > 0 or qual["min_area"] <= 0:
            info["error"] = "negative_area"
            return 0.0, 0.0, False, info

        mesh_path = os.path.join(workdir, "mesh.su2")
        write_su2(mesh, mesh_path)

        # 2) 設定書き出し
        history_base = os.path.join(workdir, "history")
        cfg = RANS_CFG_TEMPLATE.format(
            mach=s.mach, aoa=s.aoa, re=s.reynolds, iter=s.max_iter,
            mesh=mesh_path, history=history_base,
        )
        cfg_path = os.path.join(workdir, "case.cfg")
        with open(cfg_path, "w") as f:
            f.write(cfg)

        # 3) SU2 実行
        env = dict(os.environ)
        env["PATH"] = s.su2_run + ":" + env.get("PATH", "")
        env["SU2_RUN"] = s.su2_run
        env["OMP_NUM_THREADS"] = str(s.n_threads)
        log_path = os.path.join(workdir, "run.log")
        with open(log_path, "w") as logf:
            try:
                rc = subprocess.run(
                    [os.path.join(s.su2_run, "SU2_CFD"), cfg_path],
                    cwd=workdir, env=env, stdout=logf, stderr=subprocess.STDOUT,
                    timeout=s.timeout_s,
                ).returncode
            except subprocess.TimeoutExpired:
                info["error"] = "timeout"
                return 0.0, 0.0, False, info
        info["returncode"] = rc

        # 4) Cl/Cd パース
        cl, cd, iters = _parse_clcd(history_base + ".csv")
        info["iters"] = iters
        if cl is None or cd is None:
            info["error"] = "no_clcd"
            return 0.0, 0.0, False, info
        info["cl"] = cl
        info["cd"] = cd
        # 物理性チェック: cd>min_cd、有限、揚抗比が異常でない
        if not (np.isfinite(cl) and np.isfinite(cd)) or cd <= s.min_cd:
            info["error"] = "nonphysical"
            return cl, cd, False, info
        return cl, cd, True, info
    finally:
        info["elapsed_s"] = time.perf_counter() - t0
        # keep_workdir 指定時のみ残す。infeasible/error も含め通常は削除し
        # 本番ベンチ(数千評価)でのディスク逼迫を防ぐ(原因は info["error"] に残る)。
        if not s.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    # NACA0012 相当(CST 近似重み)で動作確認
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from airfoil_mesh import naca00xx_weights

    wu, wl = naca00xx_weights()
    st = SU2Settings(aoa=2.0, max_iter=int(os.environ.get("ITER", "3000")),
                     n_threads=int(os.environ.get("NTHREAD", "4")),
                     keep_workdir=True)
    print(f"running NACA0012-CST RANS: Ma={st.mach} Re={st.reynolds} AoA={st.aoa} ...")
    cl, cd, feas, info = run_cst(wu, wl, settings=st)
    print(f"  feasible={feas}  CL={cl:.5f}  CD={cd:.6f}  "
          f"CL/CD={cl/cd if cd else float('nan'):.2f}")
    print(f"  iters={info.get('iters')}  elapsed={info.get('elapsed_s',0):.1f}s  "
          f"err={info.get('error')}")
    print(f"  mesh={info.get('mesh')}")
    print(f"  workdir={info.get('workdir')}")
