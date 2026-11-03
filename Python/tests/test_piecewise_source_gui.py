"""Joined source paths preserve the fixed detector, archives and native rays."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tigre.utilities.detector_geometry_gui import (
    DetectorGeometryBuilder, _piecewise_source_positions, build_modular_surface)
from tigre.utilities.detector_geometry import linear_source_surface
from tigre.utilities.geometry_io import load_geometry_file, save_geometry_yaml


def settings(layout="Piecewise linear"):
    return dict(layout=layout, n_views=6, source_radius_mm=1000, source_z_mm=0,
        module_shape=(4, 5), pixel_spacing_mm=(2, 2), detector_radius_mm=500,
        column_gap=1, n_modules=2, angular_span_deg=20,
        transverse_centers_mm=[-10, 10], normal_offsets_mm=[0, 2],
        panel_angles_deg=[3, -3], z_offsets_mm=[0, 1],
        panel_x_angles_deg=[2, -2], panel_y_angles_deg=[1, -1],
        source_motion="Fixed detector, moving source",
        source_waypoints_mm=[[1000, -60, 0], [1000, 0, 0], [900, 40, 20]],
        source_segment_views=[3, 4])


@pytest.mark.parametrize("layout", ["Cylindrical", "Piecewise linear"])
def test_join_matches_two_linear_scans_on_the_same_detector(layout):
    parameters = settings(layout)
    surface = build_modular_surface(**parameters)
    points = parameters["source_waypoints_mm"]
    scans = [linear_source_surface(surface.pixel_centers_mm[0], start, end, count,
             valid_mask=surface.valid_mask[0])
             for start, end, count in zip(points[:-1], points[1:], [3, 4])]
    expected = np.concatenate([scans[0].source_positions_mm, scans[1].source_positions_mm[1:]])
    np.testing.assert_array_equal(surface.source_positions_mm, expected)
    assert np.count_nonzero(np.all(surface.source_positions_mm == points[1], axis=1)) == 1
    np.testing.assert_array_equal(surface.pixel_centers_mm,
        np.broadcast_to(surface.pixel_centers_mm[0], surface.pixel_centers_mm.shape))
    np.testing.assert_array_equal(surface.valid_mask,
        np.broadcast_to(surface.valid_mask[0], surface.valid_mask.shape))
    assert len(surface._module_slices) == 2
    assert not surface.valid_mask[:, :, 5].any()


@pytest.mark.parametrize("points,counts", [
    ([[1, 0, 0], [2, 0, 0]], [3]),
    ([[1, 0, 0], [2, np.nan, 0], [3, 0, 0]], [3, 3]),
    ([[1, 0], [2, 0], [3, 0]], [3, 3]),
    ([[1, 0, 0], [2, 0, 0], [3, 0, 0]], [3]),
    ([[1, 0, 0], [2, 0, 0], [3, 0, 0]], [1, 3]),
    ([[1, 0, 0], [2, 0, 0], [3, 0, 0]], [3.5, 3]),
    ([[1, 0, 0], [2, 0, 0], [3, 0, 0]], [True, True]),
    ([[1, 0, 0], [2, 0, 0], [3, 0, 0]], [6000, 2]),
])
def test_invalid_segments_are_rejected_before_building(points, counts):
    with pytest.raises(ValueError):
        _piecewise_source_positions(points, counts)


@pytest.mark.parametrize("changes", [dict(n_views=7), dict(fixed_hardware=True),
    dict(source_motion="Rotating source + detector"),
    dict(acquisition={}), dict(source_segment_views=None)])
def test_incompatible_modes_and_sample_counts_are_rejected(changes):
    with pytest.raises(ValueError):
        build_modular_surface(**(settings() | changes))


def test_three_segments_retain_endpoints_and_nonuniform_sampling():
    result = _piecewise_source_positions([[10, 0, 0], [10, 4, 0], [8, 4, 0], [8, 4, 6]], [3, 2, 4])
    np.testing.assert_allclose(result, [[10, 0, 0], [10, 2, 0], [10, 4, 0],
        [8, 4, 0], [8, 4, 2], [8, 4, 4], [8, 4, 6]])


def test_recipe_exact_and_export_preserve_piecewise_path(tmp_path):
    from tigre.utilities.geometry_recon_export import reconstruction_script_text
    parameters = settings()
    expected = build_modular_surface(**parameters)
    geo = SimpleNamespace(nVoxel=np.array([8, 8, 8]), sVoxel=np.array([40., 40., 40.]), offOrigin=np.zeros(3))
    recipe, exact = tmp_path / "recipe.yaml", tmp_path / "exact.yaml"
    save_geometry_yaml(recipe, settings=parameters, geometry=geo)
    save_geometry_yaml(exact, surface=expected, geometry=geo)
    for path in (recipe, exact):
        actual = load_geometry_file(path).surface
        np.testing.assert_array_equal(actual.source_positions_mm, expected.source_positions_mm)
        np.testing.assert_array_equal(actual.pixel_centers_mm, expected.pixel_centers_mm)
        np.testing.assert_array_equal(actual.valid_mask, expected.valid_mask)
    script = tmp_path / "reconstruct.py"
    namespace = dict(__file__=str(script), __name__="test_export")
    exec(compile(reconstruction_script_text(recipe, script), str(script), "exec"), namespace)
    _, actual = namespace["prepare_geometry"](recipe)
    np.testing.assert_array_equal(actual.source_positions_mm, expected.source_positions_mm)


def test_piecewise_recipe_matlab_bridge_round_trip(tmp_path):
    from tigre.utilities.geometry_matlab_bridge import to_mat, from_mat
    recipe, mat, restored = (tmp_path / name for name in ("recipe.yaml", "recipe.mat", "restored.yaml"))
    parameters = settings()
    save_geometry_yaml(recipe, settings=parameters)
    to_mat(recipe, mat)
    from_mat(mat, restored)
    expected, actual = load_geometry_file(recipe), load_geometry_file(restored)
    np.testing.assert_array_equal(actual.settings["source_waypoints_mm"], parameters["source_waypoints_mm"])
    np.testing.assert_array_equal(actual.settings["source_segment_views"], parameters["source_segment_views"])
    np.testing.assert_array_equal(actual.surface.source_positions_mm, expected.surface.source_positions_mm)
    np.testing.assert_array_equal(actual.surface.pixel_centers_mm, expected.surface.pixel_centers_mm)
    np.testing.assert_array_equal(actual.surface.valid_mask, expected.surface.valid_mask)


def test_gui_confirm_save_load_inspect_and_switch_modes(tmp_path):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    try:
        with patch.object(DetectorGeometryBuilder, "update_preview"):
            app = DetectorGeometryBuilder(root)
        app._source_trajectory.set("Piecewise linear")
        app._on_source_trajectory_changed()
        assert app._detector_stationary is True
        assert str(app._entries["source_start_mm"].cget("state")) == "disabled"
        assert str(app._entries["source_waypoints_mm"].cget("state")) == "normal"
        assert str(app._entries["n_views"].cget("state")) == "disabled"
        app._fields["source_waypoints_mm"].set("[[1000, -60, 0], [1000, 0, 0], [900, 40, 20]]")
        app._fields["source_segment_views"].set("3, 4")
        app._fields["n_views"].set("ignored while derived")
        app._fields["rows"].set("4")
        app._fields["columns"].set("5")
        with patch("tkinter.messagebox.showerror", side_effect=AssertionError):
            app._preview_button.invoke()
        assert app._confirmed_once
        assert app._fields["n_views"].get() == "6"
        expected = app.surface.source_positions_mm.copy()
        app._render_frame(5)
        recipe, exact = tmp_path / "gui.yaml", tmp_path / "inspection.yaml"
        save_geometry_yaml(recipe, settings=app._read_parameters())
        save_geometry_yaml(exact, surface=app.surface)
        app.load_geometry_path(exact)
        assert str(app._entries["source_waypoints_mm"].cget("state")) == "disabled"
        app.return_to_builder()
        assert app._source_trajectory.get() == "Piecewise linear"
        app.load_geometry_path(recipe)
        assert app._source_trajectory.get() == "Piecewise linear"
        np.testing.assert_array_equal(app.surface.source_positions_mm, expected)
        app._source_trajectory.set("Linear")
        app._on_source_trajectory_changed()
        assert str(app._entries["source_waypoints_mm"].cget("state")) == "disabled"
        assert app._read_parameters()["source_waypoints_mm"] is None
        app._source_trajectory.set("Piecewise linear")
        app._on_source_trajectory_changed()
        app._detector_motion.set("Moving with Source")
        app._on_detector_motion_changed()
        assert app._source_trajectory.get() == "Cylindrical"
        app._example.set("Tomosynthesis: piecewise linear source")
        app.load_example()
        with patch("tkinter.messagebox.showerror", side_effect=AssertionError):
            app._preview_button.invoke()
        assert app._source_trajectory.get() == "Piecewise linear"
        assert len(app.surface.source_positions_mm) == 41
        np.testing.assert_array_equal(app.surface.source_positions_mm[[0, 20, 40]],
            [[1000, -300, 0], [1000, 0, 0], [800, 300, 100]])
    finally:
        root.destroy()


def test_native_piecewise_projection_and_reconstruction():
    pytest.importorskip("_AxSurface")
    from tigre.utilities.Ax_surface import Ax_surface
    from tigre.algorithms.surface_algorithms import cgls_surface
    from test_surface_algorithms import volume_geo, phantom
    parameters = settings() | dict(module_shape=(48, 64), pixel_spacing_mm=(1, 1),
        transverse_centers_mm=[0], normal_offsets_mm=[0], panel_angles_deg=[0],
        z_offsets_mm=[0], panel_x_angles_deg=[0], panel_y_angles_deg=[0], column_gap=0,
        source_waypoints_mm=[[300, -60, 0], [300, 0, 0], [280, 60, 20]],
        source_segment_views=[21, 21], n_views=41, detector_radius_mm=200)
    geo, truth = volume_geo(), phantom()
    surface = build_modular_surface(**parameters)
    data = Ax_surface(truth, geo, surface)
    assert data.shape == (41, 48, 64) and np.isfinite(data).all()
    # Project two independent linear scans onto the same fixed detector,
    # then verify that joining their data preserves acquisition order.
    points = parameters["source_waypoints_mm"]
    scans = [linear_source_surface(surface.pixel_centers_mm[0], start, end, 21,
             valid_mask=surface.valid_mask[0]) for start, end in zip(points[:-1], points[1:])]
    expected = np.concatenate([Ax_surface(truth, geo, scans[0]), Ax_surface(truth, geo, scans[1])[1:]])
    np.testing.assert_allclose(data, expected, rtol=1e-6, atol=1e-5)
    reconstruction = cgls_surface(data, geo, surface, niter=40)
    assert np.isfinite(reconstruction).all()
    residual = np.linalg.norm(Ax_surface(reconstruction, geo, surface) - data) / np.linalg.norm(data)
    assert residual < .05, residual
