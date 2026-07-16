import sys
from pathlib import Path

import numpy as np
import pytest


SU2_DIR = Path(__file__).parents[1] / "benchmarks" / "su2"
sys.path.insert(0, str(SU2_DIR))

from wing3d_mesh import (  # noqa: E402
    baseline_parameters,
    generate_wing_mesh,
    parse_wing_parameters,
    validate_mesh_topology,
    validate_wing_geometry,
)


def test_baseline_geometry_and_mesh_are_valid():
    values = baseline_parameters()
    valid, metrics, reason = validate_wing_geometry(values)
    assert valid, (reason, metrics)
    mesh = generate_wing_mesh(values, n_half=21, nj=12, nk=5, endplate_j=4)
    valid, metrics, reason = validate_mesh_topology(mesh)
    assert valid, (reason, metrics)
    assert set(mesh.markers) == {"wing", "endplate", "ground", "symmetry", "farfield"}
    assert all(len(faces) > 0 for faces in mesh.markers.values())
    grid = mesh.points.reshape(mesh.nk, mesh.ni, mesh.nj, 3)
    first_heights = np.linalg.norm(grid[:, :, 1] - grid[:, :, 0], axis=2)
    assert np.allclose(first_heights, 2.0e-5, rtol=1e-10, atol=1e-14)


def test_large_root_small_tip_still_has_external_radial_cells():
    values = baseline_parameters()
    values[60:65] = [0.65, 0.35, 0.65, 0.0, 0.0]
    mesh = generate_wing_mesh(values, n_half=21, nj=12, nk=5, endplate_j=4)
    valid, metrics, reason = validate_mesh_topology(mesh)
    assert valid, (reason, metrics)


def test_nonpositive_thickness_is_infeasible_before_meshing():
    values = baseline_parameters()
    values[5] = -0.01
    valid, _, reason = validate_wing_geometry(values)
    assert not valid
    assert reason == "nonpositive_thickness_coefficient"
    with pytest.raises(ValueError, match="infeasible:nonpositive_thickness"):
        generate_wing_mesh(values)


def test_parameter_order_is_station_major():
    values = baseline_parameters()
    parsed = parse_wing_parameters(values)
    assert parsed.camber.shape == (6, 5)
    assert parsed.thickness.shape == (6, 5)
    assert np.all(parsed.thickness > 0)
