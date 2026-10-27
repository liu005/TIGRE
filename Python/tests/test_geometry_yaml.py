"""YAML round trips, exact companion surfaces, animation and GUI loading."""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import yaml
from pathlib import Path
from unittest.mock import patch
from tigre.utilities.geometry_io import save_geometry_yaml, load_geometry_file
from tigre.utilities.detector_geometry_gui import build_modular_surface, DetectorGeometryBuilder
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry
from tigre.utilities.visualization.animate_geometry import animate_geometry


def settings(layout="Piecewise linear", motion="Rotating source + detector"):
    return dict(layout=layout, n_views=3, source_radius_mm=1000, source_z_mm=5,
        module_shape=(3, 4), pixel_spacing_mm=(2, 3), detector_radius_mm=536,
        column_gap=1, n_modules=3, angular_span_deg=40,
        transverse_centers_mm=[-20, 0, 20], normal_offsets_mm=[0, 5, 0],
        panel_angles_deg=[10, 0, -10], z_offsets_mm=[0, 3, 0],
        panel_x_angles_deg=[5, 0, -5], panel_y_angles_deg=[0, 10, -10],
        source_motion=motion, source_start_mm=[1000, -100, 0],
        source_end_mm=[1000, 100, 20], orbit_start_deg=-30, scan_span_deg=60,
        cor_mm=7 if motion == "Rotating source + detector" else 0)


@pytest.mark.parametrize("layout", ["Piecewise linear", "Cylindrical"])
@pytest.mark.parametrize("motion", ["Rotating source + detector",
    "Fixed detector, moving source", "Fixed detector, arc source"])
def test_editable_yaml_round_trip(layout, motion, tmp_path):
    parameters = settings(layout, motion)
    expected = build_modular_surface(**parameters)
    path = tmp_path / "geometry.yaml"
    save_geometry_yaml(path, settings=parameters)
    assert "!!python" not in path.read_text()
    assert not list(tmp_path.glob("*.npz"))
    loaded = load_geometry_file(path)
    assert loaded.settings["layout"] == layout
    assert loaded.settings["source_motion"] == motion
    np.testing.assert_array_equal(loaded.surface.pixel_centers_mm, expected.pixel_centers_mm)
    np.testing.assert_array_equal(loaded.surface.source_positions_mm, expected.source_positions_mm)
    np.testing.assert_array_equal(loaded.surface.valid_mask, expected.valid_mask)
    assert loaded.surface._module_slices == expected._module_slices


def test_exact_surface_yaml_companion_and_animation(tmp_path):
    from types import SimpleNamespace
    original = build_modular_surface(**settings())
    mask = original.valid_mask.copy()
    mask[1, 0, 0] = False
    surface = DetectorSurfaceGeometry(original.source_positions_mm,
        original.pixel_centers_mm, mask, module_slices=original._module_slices)
    volume = SimpleNamespace(sVoxel=np.array([20, 30, 40]), offOrigin=np.array([1, 2, 3]))
    path = tmp_path / "geometry.yml"
    save_geometry_yaml(path, surface=surface, geometry=volume)
    data = yaml.safe_load(path.read_text())
    assert data["kind"] == "surface"
    assert not Path(data["surface_file"]).is_absolute()
    loaded = load_geometry_file(path)
    assert loaded.settings is None
    np.testing.assert_array_equal(loaded.surface.pixel_centers_mm, surface.pixel_centers_mm)
    np.testing.assert_array_equal(loaded.surface.valid_mask, surface.valid_mask)
    assert loaded.surface._module_slices == surface._module_slices
    np.testing.assert_array_equal(loaded.geometry.sVoxel, volume.sVoxel)
    ax = animate_geometry(path, pos=1, animate=False)
    ax.figure.canvas.draw()
    assert "view 1/3" in ax.get_title()
    plt.close(ax.figure)


def test_yaml_animation_and_explicit_argument_conflict(tmp_path):
    path = tmp_path / "geometry.yaml"
    save_geometry_yaml(path, settings=settings())
    ax = animate_geometry(str(path), animate=False)
    ax.figure.canvas.draw()
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="omit"):
        animate_geometry(path, angles=[0, 1])


@pytest.mark.parametrize("change", [
    {"version": 99}, {"units": {"length": "cm", "angle": "degree"}},
    {"rotation_convention": "unknown"}, {"kind": "unknown"}, {"typo": 1},
    {"builder": {"unknown": 1}},
])
def test_yaml_rejects_incompatible_or_malformed_settings(change, tmp_path):
    path = tmp_path / "geometry.yaml"
    save_geometry_yaml(path, settings=settings())
    data = yaml.safe_load(path.read_text())
    data.update(change)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_geometry_file(path)


def test_yaml_does_not_construct_python_objects(tmp_path):
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object:builtins.object {}")
    with pytest.raises(ValueError, match="Invalid geometry YAML"):
        load_geometry_file(path)


def test_gui_yaml_restores_editable_fields_and_surface_yaml_is_inspection(tmp_path):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    try:
        with patch.object(DetectorGeometryBuilder, "update_preview"):
            gui = DetectorGeometryBuilder(root)
        path = tmp_path / "editable.yaml"
        parameters = settings(motion="Fixed detector, arc source")
        save_geometry_yaml(path, settings=parameters)
        gui.load_geometry_path(path)
        expected = load_geometry_file(path)
        saved = tmp_path / "saved_by_gui.yaml"
        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(saved)):
            gui.save_geometry_settings()
        np.testing.assert_array_equal(load_geometry_file(saved).surface.pixel_centers_mm,
                                      expected.surface.pixel_centers_mm)
        assert not gui._inspection_mode
        assert str(gui._entries["panel_y_angles_deg"].cget("state")) == "normal"
        assert str(gui._entries["source_start_mm"].cget("state")) == "disabled"
        assert gui._detector_stationary == "arc"
        np.testing.assert_array_equal(gui.surface.pixel_centers_mm, expected.surface.pixel_centers_mm)
        for name in ("panel_x_angles_deg", "panel_y_angles_deg", "z_offsets_mm"):
            np.testing.assert_allclose(gui._read_parameters()[name], parameters[name])
        exact = tmp_path / "exact.yaml"
        save_geometry_yaml(exact, surface=gui.surface)
        gui.load_geometry_path(exact)
        assert gui._inspection_mode
        inspected = tmp_path / "saved_inspection.yaml"
        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(inspected)):
            gui.save_geometry_settings()
        assert load_geometry_file(inspected).settings is None
        np.testing.assert_array_equal(load_geometry_file(inspected).surface.pixel_centers_mm,
                                      expected.surface.pixel_centers_mm)
        assert str(gui._entries["panel_y_angles_deg"].cget("state")) == "disabled"
        gui.return_to_builder()
        np.testing.assert_array_equal(gui.surface.pixel_centers_mm, expected.surface.pixel_centers_mm)
    finally:
        root.destroy()



def test_shipped_three_panel_yaml():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "demos" / "geometry_examples" / "three_panel.yaml"
    loaded = load_geometry_file(path)
    assert loaded.settings["panel_x_angles_deg"] == [5, 0, -5]
    assert loaded.settings["panel_y_angles_deg"] == [0, 10, 0]
    assert loaded.surface.pixel_centers_mm.shape == (48, 16, 100, 3)
    assert len(loaded.surface._module_slices) == 3
