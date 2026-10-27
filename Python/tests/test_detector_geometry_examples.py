"""Check GUI geometry against geometry definitions in TIGRE's shipped demos."""
import ast
from pathlib import Path

import numpy as np
import pytest
import tigre
from scipy.spatial.transform import Rotation

from tigre.utilities.detector_geometry import surface_from_geometry
from tigre.utilities.detector_geometry_gui import build_modular_surface


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


@pytest.mark.parametrize("preset", ["TIGRE: circular cone-beam CT",
    "3D detector tilt: x, y, z", "TIGRE: global COR offset"])
def test_tigre_gui_preset_matches_package_geometry(preset):
    from tigre.utilities.detector_geometry_gui import GUI_EXAMPLES
    from unittest.mock import Mock
    from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder
    geo = tigre.geometry_default(high_resolution=False)
    if preset == "3D detector tilt: x, y, z":
        geo.rotDetector = np.deg2rad([20.0, 15.0, 10.0])
    elif preset == "TIGRE: global COR offset":
        geo.COR = 150.0
    layout, stationary, fields, _ = GUI_EXAMPLES[preset]
    builder = object.__new__(DetectorGeometryBuilder)
    builder._fields = {name: Mock(get=Mock(return_value=value)) for name, value in fields.items()}
    builder._layout = Mock(get=Mock(return_value=layout))
    builder._detector_stationary = stationary
    actual = build_modular_surface(**builder._read_parameters())
    expected = surface_from_geometry(geo, np.linspace(0, 2*np.pi, 50, endpoint=False))
    np.testing.assert_allclose(actual.pixel_centers_mm, expected.pixel_centers_mm, rtol=0, atol=5e-4)
    np.testing.assert_allclose(actual.source_positions_mm, expected.source_positions_mm, rtol=0, atol=5e-4)


@pytest.mark.parametrize("cor", [0, 7, -7])
def test_preview_marks_origin_and_animates_cor(cor):
    from matplotlib.figure import Figure
    from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder
    from types import SimpleNamespace
    builder = object.__new__(DetectorGeometryBuilder)
    builder._axes = Figure().add_subplot(111, projection="3d")
    builder.surface = SimpleNamespace(source_positions_mm=np.zeros((4, 3)))
    parameters = dict(cor_mm=cor, source_motion="Rotating source + detector",
                      orbit_start_deg=0, scan_span_deg=360)
    builder._add_coordinate_markers(parameters)
    assert [arrow.get_label() for arrow in builder._origin_arrows] == [
        "World +x", "World +y", "World +z"]
    from matplotlib.colors import to_rgba
    for index, (arrow, color) in enumerate(zip(builder._origin_arrows, ("red", "green", "blue"))):
        np.testing.assert_allclose(arrow.get_colors()[0], to_rgba(color, alpha=0.3))
        assert arrow.get_alpha() == 0.3
        shaft = np.asarray(arrow._segments3d[0])
        np.testing.assert_allclose(shaft[1], [0, 0, 0])
        np.testing.assert_allclose(shaft[0], np.eye(3)[index] * 0.12)
    assert [text.get_text() for text in builder._axes.texts[:3]] == ["x", "y", "z"]
    assert all(text.get_alpha() == 0.3 for text in builder._axes.texts[:3])
    if cor == 0:
        assert builder._cor_artist is None
        assert builder._cor_line is None
    else:
        for view, expected in enumerate(([0, cor, 0], [-cor, 0, 0],
                                         [0, -cor, 0], [cor, 0, 0])):
            builder._update_cor_marker(parameters, view)
            np.testing.assert_allclose(builder._cor_center, expected, atol=1e-12)
            assert builder._cor_label.get_text() == "COR"
            np.testing.assert_allclose(builder._cor_label.get_position_3d(),
                np.asarray(expected) + builder._cor_label_offset, atol=1e-12)
            assert builder._cor_sphere_vertices.shape == (288, 4, 3)
            line = np.asarray(builder._cor_line.get_data_3d())
            np.testing.assert_allclose(line[:, 0], [0, 0, 0])
            np.testing.assert_allclose(line[:, 1], expected, atol=1e-12)



def test_gct_preset_is_last_and_builds_53_panels():
    from unittest.mock import Mock
    from tigre.utilities.detector_geometry_gui import (
        GUI_EXAMPLES, DetectorGeometryBuilder, validate_gui_size)
    assert next(reversed(GUI_EXAMPLES)) == "GCT: 53-panel circular scan"
    layout, stationary, fields, _ = GUI_EXAMPLES["GCT: 53-panel circular scan"]
    builder = object.__new__(DetectorGeometryBuilder)
    builder._fields = {name: Mock(get=Mock(return_value=value)) for name, value in fields.items()}
    builder._layout = Mock(get=Mock(return_value=layout))
    builder._detector_stationary = stationary
    parameters = builder._read_parameters()
    assert parameters["n_views"] == 5701
    for key in ("transverse_centers_mm", "normal_offsets_mm", "panel_angles_deg", "z_offsets_mm"):
        assert len(parameters[key]) == 53
    validate_gui_size(5701, 1, 64, 53, 0)
    # Exercise the same layout without allocating the full 240 MiB acquisition.
    parameters["n_views"] = 3
    surface = build_modular_surface(**parameters)
    assert surface.pixel_centers_mm.shape == (3, 1, 3392, 3)
    assert surface.valid_mask.all()
    angles = np.deg2rad([156, 270, 384])
    expected = 6100 * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(3)))
    np.testing.assert_allclose(surface.source_positions_mm, expected, atol=0.001)



def test_list_apply_requires_preview_for_panel_y_tilt():
    import tkinter as tk
    from unittest.mock import patch
    from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    try:
        with patch.object(DetectorGeometryBuilder, "update_preview"):
            builder = DetectorGeometryBuilder(root)
        builder._layout.set("Piecewise linear")
        for key, value in {"n_views": "4", "rows": "3", "columns": "4",
            "transverse_centers_mm": "-20, 0, 20", "normal_offsets_mm": "0, 0, 0",
            "panel_angles_deg": "0, 0, 0", "z_offsets_mm": "0, 0, 0"}.items():
            builder._fields[key].set(value)
        builder._update_module_count_field()
        builder.update_preview()
        before = builder.surface.pixel_centers_mm.copy()
        before_panels = np.asarray(builder._panel_vertices(builder._parameters, 0))
        builder._show_panel_list("panel_y_angles_deg", "Panel y angles")
        window = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
        window.withdraw()

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        widgets = list(descendants(window))
        editor = next(w for w in widgets if isinstance(w, tk.Text))
        editor.delete("1.0", "end")
        editor.insert("1.0", "30\n0\n0")
        apply_button = next(w for w in widgets if w.winfo_class() == "TButton" and w.cget("text") == "Apply")
        apply_button.invoke()
        assert builder._fields["panel_y_angles_deg"].get() == "30, 0, 0"
        np.testing.assert_array_equal(builder.surface.pixel_centers_mm, before)
        assert builder._parameters["panel_y_angles_deg"].tolist() == [0]
        builder.update_preview()
        assert builder._parameters["panel_y_angles_deg"].tolist() == [30, 0, 0]
        assert not np.allclose(builder.surface.pixel_centers_mm[:, :, :4], before[:, :, :4])
        np.testing.assert_array_equal(builder.surface.pixel_centers_mm[:, :, 6:], before[:, :, 6:])
        after_panels = np.asarray(builder._panel_vertices(builder._parameters, 0))
        assert not np.allclose(after_panels[0], before_panels[0])
        np.testing.assert_allclose(after_panels[1:], before_panels[1:])
        builder._render_frame(2)
        assert builder._frame_index == 2
        assert not np.allclose(builder.surface.pixel_centers_mm[2, :, :4], before[2, :, :4])
    finally:
        root.destroy()
