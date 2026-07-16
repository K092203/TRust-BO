"""Structured half-wing mesh generator for the Formula Student 3-D case.

The mesh is a spanwise extrusion of the O-grid topology used by
``airfoil_mesh.py``.  Coordinates are (streamwise x, spanwise y, vertical z).
The outer O-grid ring is rectangular, so its lower side is an exact ground
plane.  At the tip, the first radial bands form a planar endplate; the rest of
the tip plane is farfield.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .airfoil_mesh import _airfoil_loop, cst_surface
except ImportError:  # direct execution from benchmarks/su2
    from airfoil_mesh import _airfoil_loop, cst_surface


SPAN_CONTROL_ETA = np.linspace(0.0, 1.0, 6)
GROUND_Z = 0.0
WING_HEIGHT = 0.03
ENDPLATE_BOUNDS = (-0.10, 0.50, 0.002, 0.25)


SU2_CONFIG_TEMPLATE = """SOLVER= INC_RANS
KIND_TURB_MODEL= SA
SA_OPTIONS= NONE
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
INC_DENSITY_MODEL= CONSTANT
INC_ENERGY_EQUATION= NO
INC_DENSITY_INIT= {density}
INC_VELOCITY_INIT= ( {vx}, {vy}, {vz} )
INC_NONDIM= DIMENSIONAL
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= {mu}
FREESTREAM_NU_FACTOR= 3.0
REF_ORIGIN_MOMENT_X= {moment_x}
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= {ref_length}
REF_AREA= {half_ref_area}
MARKER_HEATFLUX= ( wing, 0.0, endplate, 0.0, ground, 0.0 )
MARKER_FAR= ( farfield )
MARKER_SYM= ( symmetry )
MARKER_PLOTTING= ( wing, endplate, ground )
MARKER_MONITORING= ( wing, endplate )
GRID_MOVEMENT= NONE
SURFACE_MOVEMENT= MOVING_WALL
MARKER_MOVING= ( ground )
SURFACE_TRANSLATION_RATE= ( {vx}, {vy}, {vz} )
SURFACE_ROTATION_RATE= ( 0.0, 0.0, 0.0 )
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 1.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.1, 1.5, 0.5, 50.0 )
ITER= {iter}
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 15
CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
TIME_DISCRE_FLOW= EULER_IMPLICIT
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
TIME_DISCRE_TURB= EULER_IMPLICIT
CFL_REDUCTION_TURB= 0.5
CONV_FIELD= ( LIFT, DRAG )
CONV_RESIDUAL_MINVAL= -10
CONV_STARTITER= 300
CONV_CAUCHY_ELEMS= {cauchy_elems}
CONV_CAUCHY_EPS= 1E-5
MESH_FILENAME= {mesh}
MESH_FORMAT= SU2
SCREEN_OUTPUT= ( INNER_ITER, RMS_PRESSURE, RMS_VELOCITY-X, RMS_NU_TILDE, LIFT, DRAG )
SCREEN_WRT_FREQ_INNER= 100
HISTORY_OUTPUT= ( INNER_ITER, RMS_RES, AERO_COEFF )
CONV_FILENAME= {history}
OUTPUT_FILES= ( RESTART, CSV )
OUTPUT_WRT_FREQ= 1000
WRT_PERFORMANCE= NO
"""


@dataclass(frozen=True)
class WingParameters:
    camber: np.ndarray       # (6, 5)
    thickness: np.ndarray    # (6, 5)
    half_span: float
    root_chord: float
    taper_ratio: float
    le_sweep_offset: float
    aoa_deg: float


@dataclass
class Mesh3D:
    points: np.ndarray
    hexes: np.ndarray
    markers: dict[str, np.ndarray]
    ni: int
    nj: int
    nk: int
    endplate_j: int


def parse_wing_parameters(values: np.ndarray) -> WingParameters:
    """Parse the fixed 65-D ordering described in the benchmark specification."""
    x = np.asarray(values, dtype=float)
    if x.shape != (65,):
        raise ValueError(f"expected a finite 65-D vector, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("nonfinite_parameters")
    sections = x[:60].reshape(6, 10)
    return WingParameters(
        camber=sections[:, :5].copy(), thickness=sections[:, 5:].copy(),
        half_span=float(x[60]), root_chord=float(x[61]),
        taper_ratio=float(x[62]), le_sweep_offset=float(x[63]),
        aoa_deg=float(x[64]),
    )


def _clamped_bspline_controls(controls: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Evaluate a clamped cubic B-spline whose six rows are control points."""
    c = np.asarray(controls, dtype=float)
    u = np.asarray(eta, dtype=float)
    if c.shape[0] != 6 or np.any((u < 0.0) | (u > 1.0)):
        raise ValueError("invalid B-spline controls or eta")
    degree = 3
    # Open uniform/clamped knot vector: the six station values are B-spline
    # control ordinates (not interpolation samples).  SPAN_CONTROL_ETA defines
    # their uniformly spaced control-polygon locations.
    knots = np.array([0.0, 0.0, 0.0, 0.0, 1 / 3, 2 / 3, 1.0, 1.0, 1.0, 1.0])
    out = np.empty(u.shape + c.shape[1:])
    for index, value in np.ndenumerate(u):
        if value == 1.0:
            span = 5
        else:
            span = int(np.searchsorted(knots, value, side="right") - 1)
            span = min(max(span, degree), 5)
        d = [c[span - degree + j].copy() for j in range(degree + 1)]
        for r in range(1, degree + 1):
            for j in range(degree, r - 1, -1):
                i = span - degree + j
                den = knots[i + degree - r + 1] - knots[i]
                alpha = 0.0 if den == 0.0 else (value - knots[i]) / den
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
        out[index] = d[degree]
    return out


def interpolate_span_coefficients(params: WingParameters, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cubic B-spline camber and thickness coefficients at ``eta``."""
    return (_clamped_bspline_controls(params.camber, eta),
            _clamped_bspline_controls(params.thickness, eta))


def _section_surfaces(camber: np.ndarray, thickness: np.ndarray,
                      n_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = np.linspace(0.0, math.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))
    yc = cst_surface(camber, x)
    yt = cst_surface(thickness, x)
    return x, yc + 0.5 * yt, yc - 0.5 * yt


def validate_wing_geometry(values: np.ndarray, *, n_eta: int = 41,
                           n_points: int = 401,
                           max_surface_rate: float = 0.20,
                           max_thickness_rate: float = 0.15) -> tuple[bool, dict, str | None]:
    """Dense section and span-continuity validation, before any mesh/CFD work."""
    try:
        p = parse_wing_parameters(values)
    except ValueError as exc:
        return False, {}, str(exc)
    if np.any(p.thickness <= 0.0):
        return False, {}, "nonpositive_thickness_coefficient"
    if p.half_span <= 0 or p.root_chord <= 0 or p.taper_ratio <= 0:
        return False, {}, "invalid_planform"
    eta = np.linspace(0.0, 1.0, n_eta)
    camber, thick = interpolate_span_coefficients(p, eta)
    max_t, areas, max_c = [], [], []
    surfaces, thicknesses = [], []
    x_ref = None
    for ca, th in zip(camber, thick):
        x, yu, yl = _section_surfaces(ca, th, n_points)
        thickness = yu - yl
        area = float(np.sum(0.5 * (thickness[:-1] + thickness[1:]) * np.diff(x)))
        metrics = (float(thickness.max()), area,
                   float(np.max(np.abs(0.5 * (yu + yl)))))
        if float(thickness[1:-1].min()) <= 1e-10:
            return False, {"eta": eta, "section": metrics}, "surface_crossing"
        if not 0.06 <= metrics[0] <= 0.20:
            return False, {"eta": eta, "section": metrics}, "thickness_ratio"
        if area < 0.05:
            return False, {"eta": eta, "section": metrics}, "section_area_ratio"
        if metrics[2] > 0.08:
            return False, {"eta": eta, "section": metrics}, "camber_ratio"
        max_t.append(metrics[0]); areas.append(area); max_c.append(metrics[2])
        surfaces.append(np.concatenate([yu, yl]))
        thicknesses.append(thickness)
        x_ref = x
    surfaces = np.asarray(surfaces)
    thicknesses = np.asarray(thicknesses)
    d_eta = np.diff(eta)
    surface_rate = np.max(np.abs(np.diff(surfaces, axis=0)), axis=1) / d_eta
    thickness_rate = np.max(np.abs(np.diff(thicknesses, axis=0)), axis=1) / d_eta
    if float(surface_rate.max()) > max_surface_rate:
        return False, {"max_surface_rate": float(surface_rate.max())}, "span_surface_rate"
    if float(thickness_rate.max()) > max_thickness_rate:
        return False, {"max_thickness_rate": float(thickness_rate.max())}, "span_thickness_rate"
    # Physical dense sections must remain above ground.  This catches a large
    # chord/AoA combination before allocating the 3-D mesh.
    physical = [_physical_section(p, float(e), ca, th, min(n_points, 401))
                for e, ca, th in zip(eta, camber, thick)]
    min_z = float(min(section[:, 1].min() for section in physical))
    if min_z <= GROUND_Z:
        return False, {"min_wing_z": min_z}, "wing_ground_intersection"
    return True, {
        "eta": eta, "x": x_ref, "min_thickness_ratio": float(np.min(max_t)),
        "max_thickness_ratio": float(np.max(max_t)), "min_area_ratio": float(np.min(areas)),
        "max_camber_ratio": float(np.max(max_c)),
        "max_surface_rate": float(surface_rate.max()),
        "max_thickness_rate": float(thickness_rate.max()),
        "min_wing_z": min_z,
    }, None


def _ray_rectangle(center: np.ndarray, points: np.ndarray,
                   bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Intersect rays center->points with an axis-aligned x/z rectangle."""
    xmin, xmax, zmin, zmax = bounds
    direction = points - center
    candidates = np.full((len(points), 4), np.inf)
    pos_x, neg_x = direction[:, 0] > 0, direction[:, 0] < 0
    pos_z, neg_z = direction[:, 1] > 0, direction[:, 1] < 0
    candidates[pos_x, 0] = (xmax - center[0]) / direction[pos_x, 0]
    candidates[neg_x, 1] = (xmin - center[0]) / direction[neg_x, 0]
    candidates[pos_z, 2] = (zmax - center[1]) / direction[pos_z, 1]
    candidates[neg_z, 3] = (zmin - center[1]) / direction[neg_z, 1]
    scale = np.min(np.where(candidates > 0, candidates, np.inf), axis=1)
    return center + direction * scale[:, None]


def _physical_section(p: WingParameters, eta: float, camber: np.ndarray,
                      thickness: np.ndarray, n_half: int) -> np.ndarray:
    x, yu, yl = _section_surfaces(camber, thickness, n_half)
    loop = _airfoil_loop(x, yu, x, yl)
    chord = p.root_chord * (1.0 - eta + p.taper_ratio * eta)
    le = p.le_sweep_offset * eta
    local = np.column_stack([le + chord * loop[:, 0], WING_HEIGHT + chord * loop[:, 1]])
    pivot = np.array([le + 0.25 * chord, WING_HEIGHT])
    angle = math.radians(p.aoa_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    return pivot + (local - pivot) @ rotation.T


def _wall_spacing(n_cells: int, first_fraction: float) -> np.ndarray:
    """Geometric spacing on [0,1] with an exact first-cell fraction."""
    if n_cells < 1 or not 0.0 < first_fraction < 1.0:
        raise ValueError("invalid wall spacing")
    if n_cells == 1:
        return np.array([0.0, 1.0])
    if first_fraction * n_cells >= 1.0:
        widths = np.full(n_cells, 1.0 / n_cells)
    else:
        lo, hi = 1.0, 2.0
        while first_fraction * (hi**n_cells - 1.0) / (hi - 1.0) < 1.0:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            total = first_fraction * (mid**n_cells - 1.0) / (mid - 1.0)
            if total < 1.0:
                lo = mid
            else:
                hi = mid
        ratio = 0.5 * (lo + hi)
        widths = first_fraction * ratio ** np.arange(n_cells)
        widths[-1] += 1.0 - float(widths.sum())
    return np.concatenate([[0.0], np.cumsum(widths)])


def generate_wing_mesh(values: np.ndarray, *, n_half: int = 41, nj: int = 36,
                       nk: int = 17, first_cell: float = 2.0e-5,
                       growth: float = 1.22, endplate_j: int = 20,
                       domain_bounds: tuple[float, float, float, float] = (-0.8, 1.2, 0.0, 0.8),
                       endplate_bounds: tuple[float, float, float, float] = ENDPLATE_BOUNDS) -> Mesh3D:
    """Generate the structured hex mesh after all dense geometry checks pass."""
    valid, _, reason = validate_wing_geometry(values)
    if not valid:
        raise ValueError(f"infeasible:{reason}")
    if nk < 2 or nj < 4 or not 1 <= endplate_j < nj - 1:
        raise ValueError("invalid mesh dimensions")
    p = parse_wing_parameters(values)
    eta = np.linspace(0.0, 1.0, nk)
    cambers, thicknesses = interpolate_span_coefficients(p, eta)
    sections = [_physical_section(p, e, ca, th, n_half)
                for e, ca, th in zip(eta, cambers, thicknesses)]
    ni = len(sections[0])
    tip = sections[-1]
    ep_bounds = endplate_bounds
    ex0, ex1, ez0, ez1 = ep_bounds
    all_sections = np.concatenate(sections)
    if (all_sections[:, 0].min() <= ex0 or all_sections[:, 0].max() >= ex1 or
            all_sections[:, 1].min() <= ez0 or all_sections[:, 1].max() >= ez1):
        raise ValueError("infeasible:wing_outside_fixed_endplate")
    dx0, dx1, dz0, dz1 = domain_bounds
    if (all_sections[:, 0].min() <= dx0 or all_sections[:, 0].max() >= dx1 or
            all_sections[:, 1].min() <= dz0 or all_sections[:, 1].max() >= dz1):
        raise ValueError("infeasible:wing_outside_domain_or_ground_intersection")
    s_outer = np.linspace(0.0, 1.0, nj - endplate_j)
    grid = np.empty((nk, ni, nj, 3))
    _ = growth  # compatibility hint; ratio is solved from first_cell and length
    for k, (e, inner) in enumerate(zip(eta, sections)):
        center = inner.mean(axis=0)
        guide = _ray_rectangle(center, inner, ep_bounds)
        outer = _ray_rectangle(center, inner, domain_bounds)
        rings = np.empty((ni, nj, 2))
        guide_distance = np.linalg.norm(guide - inner, axis=1)
        for i in range(ni):
            # ``growth`` is retained for API compatibility; the actual ratio
            # is solved so first_cell and the fixed guide boundary are exact.
            spacing = _wall_spacing(endplate_j, first_cell / guide_distance[i])
            rings[i, :endplate_j + 1] = (inner[i] + spacing[:, None] *
                                         (guide[i] - inner[i]))
        for jj in range(1, len(s_outer)):
            rings[:, endplate_j + jj] = guide + s_outer[jj] * (outer - guide)
        grid[k, :, :, 0] = rings[:, :, 0]
        grid[k, :, :, 1] = e * p.half_span
        grid[k, :, :, 2] = rings[:, :, 1]
    points = grid.reshape(-1, 3)

    def pid(k: int, i: int, j: int) -> int:
        return (k * ni + i % ni) * nj + j

    hexes = []
    for k in range(nk - 1):
        for i in range(ni):
            for j in range(nj - 1):
                hexes.append([pid(k, i, j), pid(k, i + 1, j), pid(k, i + 1, j + 1), pid(k, i, j + 1),
                              pid(k + 1, i, j), pid(k + 1, i + 1, j), pid(k + 1, i + 1, j + 1), pid(k + 1, i, j + 1)])
    hexes = np.asarray(hexes, dtype=np.int64)
    if _hex_signed_volumes(points, hexes[:1])[0] < 0:
        hexes = hexes[:, [0, 3, 2, 1, 4, 7, 6, 5]]

    wing = [[pid(k, i, 0), pid(k, i + 1, 0), pid(k + 1, i + 1, 0), pid(k + 1, i, 0)]
            for k in range(nk - 1) for i in range(ni)]
    symmetry = [[pid(0, i, j), pid(0, i, j + 1), pid(0, i + 1, j + 1), pid(0, i + 1, j)]
                for i in range(ni) for j in range(nj - 1)]
    endplate = [[pid(nk - 1, i, j), pid(nk - 1, i + 1, j),
                 pid(nk - 1, i + 1, j + 1), pid(nk - 1, i, j + 1)]
                for i in range(ni) for j in range(endplate_j)]
    tip_far = [[pid(nk - 1, i, j), pid(nk - 1, i + 1, j),
                pid(nk - 1, i + 1, j + 1), pid(nk - 1, i, j + 1)]
               for i in range(ni) for j in range(endplate_j, nj - 1)]
    ground, outer_far = [], []
    tol = 1e-12
    for k in range(nk - 1):
        for i in range(ni):
            face = [pid(k, i, nj - 1), pid(k + 1, i, nj - 1),
                    pid(k + 1, i + 1, nj - 1), pid(k, i + 1, nj - 1)]
            (ground if np.all(np.abs(points[face, 2] - GROUND_Z) < tol) else outer_far).append(face)
    markers = {
        "wing": np.asarray(wing, dtype=np.int64),
        "endplate": np.asarray(endplate, dtype=np.int64),
        "ground": np.asarray(ground, dtype=np.int64).reshape(-1, 4),
        "symmetry": np.asarray(symmetry, dtype=np.int64),
        "farfield": np.asarray(tip_far + outer_far, dtype=np.int64),
    }
    mesh = Mesh3D(points, hexes, markers, ni, nj, nk, endplate_j)
    ok, report, why = validate_mesh_topology(mesh)
    if not ok:
        raise ValueError(f"infeasible:{why}:{report}")
    return mesh


def _tetra_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def _hex_signed_volumes(points: np.ndarray, hexes: np.ndarray) -> np.ndarray:
    """Signed volume using six tetrahedra sharing body diagonal 0--6."""
    h = points[hexes]
    tets = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
    return sum((_tetra_volume(h[:, a], h[:, b], h[:, c], h[:, d])
                for a, b, c, d in tets), start=np.zeros(len(h)))


def _hex_tetra_volumes(points: np.ndarray, hexes: np.ndarray) -> np.ndarray:
    h = points[hexes]
    tets = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
    return np.column_stack([_tetra_volume(h[:, a], h[:, b], h[:, c], h[:, d])
                            for a, b, c, d in tets])


def validate_mesh_topology(mesh: Mesh3D) -> tuple[bool, dict, str | None]:
    """Check positive hex volume, marker partition, watertightness and intersections."""
    volumes = _hex_signed_volumes(mesh.points, mesh.hexes)
    tetra_volumes = _hex_tetra_volumes(mesh.points, mesh.hexes)
    report = {"n_points": len(mesh.points), "n_cells": len(mesh.hexes),
              "min_signed_volume": float(volumes.min()),
              "n_nonpositive": int(np.count_nonzero(volumes <= 0.0))}
    report["min_subtetra_volume"] = float(tetra_volumes.min())
    if (not np.all(np.isfinite(volumes)) or np.any(volumes <= 1e-18) or
            np.any(tetra_volumes <= 1e-20)):
        return False, report, "nonpositive_hex_volume"
    faces: dict[tuple[int, ...], int] = {}
    patterns = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
                (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
    for cell in mesh.hexes:
        for pattern in patterns:
            key = tuple(sorted(int(cell[i]) for i in pattern))
            faces[key] = faces.get(key, 0) + 1
    boundary = {key for key, count in faces.items() if count == 1}
    if any(count > 2 for count in faces.values()):
        return False, report, "nonmanifold_volume_face"
    marked = [tuple(sorted(map(int, face))) for values in mesh.markers.values() for face in values]
    report["n_boundary_faces"] = len(boundary)
    report["n_marked_faces"] = len(marked)
    if len(marked) != len(set(marked)) or set(marked) != boundary:
        return False, report, "non_watertight_marker_partition"
    ep_nodes = np.unique(mesh.markers["endplate"])
    if len(ep_nodes) and not np.allclose(mesh.points[ep_nodes, 1], mesh.points[ep_nodes[0], 1]):
        return False, report, "nonplanar_endplate"
    wing_faces = {tuple(sorted(map(int, f))) for f in mesh.markers["wing"]}
    ep_faces = {tuple(sorted(map(int, f))) for f in mesh.markers["endplate"]}
    if wing_faces & ep_faces:
        return False, report, "wing_endplate_face_intersection"
    tip_wing_nodes = {int(n) for face in mesh.markers["wing"] for n in face
                      if np.isclose(mesh.points[n, 1], mesh.points[:, 1].max())}
    ep_inner_nodes = {int(n) for face in mesh.markers["endplate"] for n in face
                      if n % mesh.nj == 0}
    if tip_wing_nodes != ep_inner_nodes:
        return False, report, "endplate_not_shared_with_wing_tip"
    return True, report, None


def mesh_quality(mesh: Mesh3D) -> dict:
    volumes = _hex_signed_volumes(mesh.points, mesh.hexes)
    return {"n_cells": len(mesh.hexes), "n_points": len(mesh.points),
            "min_signed_volume": float(volumes.min()),
            "max_signed_volume": float(volumes.max()),
            "n_nonpositive_volume": int(np.count_nonzero(volumes <= 0)),
            **{f"n_{tag}_faces": len(faces) for tag, faces in mesh.markers.items()}}


def write_su2(mesh: Mesh3D, path: str | Path) -> None:
    """Write SU2 native 3-D format (VTK 12 hexes, VTK 9 boundary quads)."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as out:
        out.write("NDIME= 3\n")
        out.write(f"NELEM= {len(mesh.hexes)}\n")
        for cell in mesh.hexes:
            out.write("12 " + " ".join(map(str, cell)) + "\n")
        out.write(f"NPOIN= {len(mesh.points)}\n")
        for point in mesh.points:
            out.write(" ".join(f"{v:.16e}" for v in point) + "\n")
        out.write("NMARK= 5\n")
        for tag in ("wing", "endplate", "ground", "symmetry", "farfield"):
            faces = mesh.markers[tag]
            out.write(f"MARKER_TAG= {tag}\nMARKER_ELEMS= {len(faces)}\n")
            for face in faces:
                out.write("9 " + " ".join(map(str, face)) + "\n")


def baseline_parameters() -> np.ndarray:
    """A smooth, symmetric 12%-class baseline for smoke tests."""
    values = np.empty(65)
    camber = np.array([0.015, 0.020, 0.020, 0.015, 0.008])
    thickness = np.array([0.24, 0.22, 0.20, 0.17, 0.12])
    for station in range(6):
        values[station * 10:station * 10 + 5] = camber
        values[station * 10 + 5:station * 10 + 10] = thickness
    values[60:] = [0.65, 0.28, 0.82, 0.025, 4.0]
    return values


def render_su2_config(mesh: str, history: str, values: np.ndarray, *, iterations: int = 5,
                      density: float = 1.225, mu: float = 1.81e-5,
                      velocity: tuple[float, float, float] = (11.0, 0.0, 0.0),
                      cauchy_elems: int = 300) -> str:
    p = parse_wing_parameters(values)
    return SU2_CONFIG_TEMPLATE.format(
        density=density, vx=velocity[0], vy=velocity[1], vz=velocity[2], mu=mu,
        moment_x=0.25 * p.root_chord, ref_length=p.root_chord,
        half_ref_area=0.5 * p.half_span * p.root_chord * (1.0 + p.taper_ratio),
        iter=iterations, cauchy_elems=cauchy_elems, mesh=mesh, history=history,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="wing3d.su2")
    args = parser.parse_args()
    params = baseline_parameters()
    generated = generate_wing_mesh(params)
    write_su2(generated, args.path)
    print(mesh_quality(generated))
