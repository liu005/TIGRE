"""Check GUI geometry against geometry definitions in TIGRE's shipped demos."""
import ast
from pathlib import Path

import numpy as np
import pytest
import tigre
from scipy.spatial.transform import Rotation

from tigre.utilities.detector_geometry import surface_from_geometry
from tigre.utilities.detector_geometry_builder import build_modular_surface


DEMOS = Path(__file__).resolve().parents[1] / "demos"


def _demo_geometry(filename, stop_target):
    # Execute only the demo's geometry assignments, before data/GPU/plot calls.
    tree = ast.parse((DEMOS / filename).read_text())
    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == stop_target for t in node.targets):
                break
            assignments.append(node)
        elif isinstance(node, ast.Expr):
            break
    namespace = {"np": np, "tigre": tigre}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def _gui(geo, rotation, angle, cor=0):
    x, y, z = np.rad2deg(rotation)
    return build_modular_surface(
        "Piecewise linear", 1, float(geo.DSO), 0,
        tuple(int(n) for n in geo.nDetector), tuple(geo.dDetector),
        float(geo.DSD - geo.DSO), 0, 1, 0,
        [float(geo.offDetector[1])], [0], [z], [float(geo.offDetector[0])],
        orbit_start_deg=float(np.rad2deg(angle)),
        panel_x_angles_deg=[x], panel_y_angles_deg=[y], cor_mm=cor)


def test_detector_rotation_demo_all_100_poses():
    demo = _demo_geometry("d17_DetectorRotation.py", "head")
    geo, angles = demo["geo"], demo["angles"]
    expected = surface_from_geometry(geo, angles)
    # TIGRE check_geo quantizes angles to float32; allow 0.0005 mm at 1 m DSO.
    assert geo.rotDetector.shape == (100, 3)
    for view, angle in enumerate(angles):
        actual = _gui(geo, geo.rotDetector[view], angle)
        np.testing.assert_allclose(actual.pixel_centers_mm[0],
                                   expected.pixel_centers_mm[view], rtol=0, atol=5e-4)
        np.testing.assert_allclose(actual.source_positions_mm[0],
                                   expected.source_positions_mm[view], rtol=0, atol=5e-4)
        # Independent Euler-angle reference using SciPy, including panel edges.
        local = np.zeros((*geo.nDetector, 3))
        local[..., 1] = (np.arange(geo.nDetector[1]) - (geo.nDetector[1]-1)/2) * geo.dDetector[1]
        local[..., 2] = ((np.arange(geo.nDetector[0]) - (geo.nDetector[0]-1)/2) * geo.dDetector[0])[:, None]
        detector = Rotation.from_euler("xyz", geo.rotDetector[view]).apply(local.reshape(-1, 3))
        detector[:, 0] -= geo.DSD - geo.DSO
        reference = Rotation.from_euler("z", angle).apply(detector).reshape(local.shape)
        np.testing.assert_allclose(actual.pixel_centers_mm[0], reference, rtol=0, atol=5e-4)


@pytest.mark.parametrize("cor", [0.0, -7.5, 7.5])
def test_create_geometry_demo_with_cor_sign_checks(cor):
    demo = _demo_geometry("d01_CreateGeometry.py", "unused")
    geo = demo["geo"]
    # The demo defines COR=0; nonzero cases explicitly extend that baseline.
    assert geo.COR == 0
    geo.COR = cor
    angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    expected = surface_from_geometry(geo, angles)
    for view, angle in enumerate(angles):
        actual = _gui(geo, geo.rotDetector, angle, cor)
        np.testing.assert_allclose(actual.pixel_centers_mm[0],
                                   expected.pixel_centers_mm[view], rtol=0, atol=5e-4)
        np.testing.assert_allclose(actual.source_positions_mm[0],
                                   expected.source_positions_mm[view], rtol=0, atol=5e-4)
        np.testing.assert_allclose(actual.source_positions_mm[0],
            [geo.DSO*np.cos(angle)-cor*np.sin(angle),
             geo.DSO*np.sin(angle)+cor*np.cos(angle), 0], rtol=0, atol=5e-4)


@pytest.mark.parametrize("layout", ["Cylindrical", "Piecewise linear"])
@pytest.mark.parametrize("motion", ["Rotating source + detector",
    "Fixed detector, moving source", "Fixed detector, arc source"])
def test_panel_tilts_preserve_centres_spacing_masks_and_export(layout, motion, tmp_path):
    parameters = dict(layout=layout, n_views=5, source_radius_mm=1000,
        source_z_mm=0, module_shape=(3, 4), pixel_spacing_mm=(2, 3),
        detector_radius_mm=536, column_gap=2, n_modules=3,
        angular_span_deg=40, transverse_centers_mm=[-20, 0, 20],
        normal_offsets_mm=[0, 5, 0], panel_angles_deg=[20, 0, -20],
        z_offsets_mm=[0, 0, 0], source_motion=motion,
        source_start_mm=(1000, -100, 0), source_end_mm=(1000, 100, 0),
        orbit_start_deg=-30, scan_span_deg=60)
    baseline = build_modular_surface(**parameters)
    tilted = build_modular_surface(**parameters,
        panel_x_angles_deg=[13, 0, -9], panel_y_angles_deg=[-17, 0, 21])
    np.testing.assert_array_equal(tilted.valid_mask, baseline.valid_mask)
    np.testing.assert_array_equal(tilted.source_positions_mm, baseline.source_positions_mm)
    for panel in range(3):
        grid = tilted.pixel_centers_mm[:, :, panel*6:panel*6+4]
        original = baseline.pixel_centers_mm[:, :, panel*6:panel*6+4]
        np.testing.assert_allclose(grid.mean(axis=(1, 2)), original.mean(axis=(1, 2)), atol=1e-4)
        np.testing.assert_allclose(np.linalg.norm(np.diff(grid, axis=1), axis=-1), 2, atol=1e-4)
        np.testing.assert_allclose(np.linalg.norm(np.diff(grid, axis=2), axis=-1), 3, atol=1e-4)
        if panel == 1:
            np.testing.assert_array_equal(grid, original)
        else:
            assert not np.allclose(grid, original)
    if motion != "Rotating source + detector":
        for view in range(1, 5):
            np.testing.assert_array_equal(tilted.pixel_centers_mm[view], tilted.pixel_centers_mm[0])
    path = tmp_path / "tilted_surface.npz"
    np.savez_compressed(path, source_positions_mm=tilted.source_positions_mm,
        pixel_centers_mm=tilted.pixel_centers_mm, valid_mask=tilted.valid_mask)
    with np.load(path, allow_pickle=False) as saved:
        for name in ("source_positions_mm", "pixel_centers_mm", "valid_mask"):
            np.testing.assert_array_equal(saved[name], getattr(tilted, name))
