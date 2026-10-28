"""Numeric loading and Tk inspection regression tests (no GPU required)."""
import numpy as np
import pytest
from unittest.mock import patch
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry, surface_from_geometry
from tigre.utilities.detector_geometry_gui import (
    DetectorGeometryBuilder, load_geometry_archive, surface_archive_arrays)


def sample_surface():
    p = np.zeros((3, 3, 6, 3))
    p[..., 0] = -100
    p[..., 1] = np.arange(6)
    p[..., 2] = np.arange(3)[None, :, None]
    p[1] += [0, 5, 3]
    mask = np.ones(p.shape[:3], dtype=bool)
    mask[1, 0, 0] = False
    return DetectorSurfaceGeometry([[100, 0, 0], [0, 100, 20], [-100, 0, 0]], p, mask,
        module_slices=[(slice(0, 3), slice(0, 3)), (slice(0, 3), slice(3, 6))])


def test_surface_archive_preserves_views_masks_modules_and_volume(tmp_path):
    from types import SimpleNamespace
    surface = sample_surface()
    volume = SimpleNamespace(sVoxel=np.array([20, 30, 40]), offOrigin=np.array([1, 2, 3]),
                             nVoxel=np.array([10, 15, 8]))
    path = tmp_path / "surface.npz"
    np.savez_compressed(path, **surface_archive_arrays(surface, volume))
    geo, loaded = load_geometry_archive(path)
    for name in ("source_positions_mm", "pixel_centers_mm", "valid_mask"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(surface, name))
    assert loaded._module_slices == surface._module_slices
    np.testing.assert_array_equal(geo.sVoxel, volume.sVoxel)
    np.testing.assert_array_equal(geo.offOrigin, volume.offOrigin)
    np.testing.assert_array_equal(geo.nVoxel, volume.nVoxel)
    np.testing.assert_allclose(geo.dVoxel, [2, 2, 5])


@pytest.mark.parametrize("counts", [[0, 2, 3], [1.5, 2, 3], [1, 2], [1, np.inf, 3]])
def test_volume_rejects_invalid_voxel_counts(counts):
    from tigre.utilities.geometry_io import _volume
    with pytest.raises(ValueError):
        _volume(dict(sVoxel=[20, 30, 40], nVoxel=counts))


def test_standard_archive_converts_per_view_tilt_and_cor(tmp_path):
    import tigre
    geo = tigre.geometry_default(high_resolution=False)
    geo.nDetector = np.array([3, 4])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.rotDetector = np.deg2rad([[0, 0, 0], [10, 15, 20], [-10, -15, -20]])
    geo.COR = np.array([0, 10, -10])
    angles = np.array([0, 0.5, 1.2])
    arrays = {name: getattr(geo, name) for name in (
        "DSO", "DSD", "nDetector", "dDetector", "rotDetector", "COR")}
    path = tmp_path / "standard.npz"
    np.savez(path, angles=angles, **arrays)
    _, actual = load_geometry_archive(path)
    expected = surface_from_geometry(geo, angles)
    np.testing.assert_allclose(actual.pixel_centers_mm, expected.pixel_centers_mm)
    np.testing.assert_allclose(actual.source_positions_mm, expected.source_positions_mm)


def test_old_surface_archive_and_bad_module_metadata(tmp_path):
    surface = sample_surface()
    arrays = surface_archive_arrays(surface)
    arrays.pop("module_slices")
    path = tmp_path / "old.npz"
    np.savez(path, **arrays)
    geo, loaded = load_geometry_archive(path)
    assert geo is None
    assert loaded._module_slices is None
    np.savez(path, **arrays, module_slices=np.array([[0, 99, 0, 6]]))
    with pytest.raises(ValueError):
        load_geometry_archive(path)


def test_inspection_controls_playback_and_return_to_builder():
    import tkinter as tk
    from tigre.utilities.visualization.surface_renderer import SurfaceRenderer
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    try:
        with patch.object(DetectorGeometryBuilder, "update_preview"):
            gui = DetectorGeometryBuilder(root)
        gui._fields["n_views"].set("3")
        gui._fields["rows"].set("3")
        gui._fields["columns"].set("4")
        gui._volume_fields["offOrigin"].set("-100, 0, 0")
        gui._volume_motion.set(True)
        gui._volume_end.set("100, 0, 0")
        gui.update_preview()
        np.testing.assert_allclose(gui._builder_geometry.offOrigin,
                                   [[-100, 0, 0], [0, 0, 0], [100, 0, 0]])
        assert isinstance(gui._surface_renderer, SurfaceRenderer)
        retained = {key: value.get() for key, value in gui._fields.items()}
        surface = sample_surface()
        gui.inspect_geometry(surface=surface)
        with pytest.raises(ValueError, match="angles"):
            import tigre
            gui.inspect_geometry(tigre.geometry_default(high_resolution=False))
        assert gui.surface is surface
        assert gui._inspection_mode
        for entry in list(gui._entries.values()) + list(gui._list_buttons.values()):
            assert str(entry.cget("state")) == "disabled"
        assert str(gui._layout_box.cget("state")) == "disabled"
        assert str(gui._preview_button.cget("state")) == "disabled"
        gui.update_preview()  # Cannot overwrite inspection data.
        assert gui.surface is surface
        gui._render_frame(1)
        assert gui._frame_index == 1
        assert "17 valid pixels" in gui._axes.get_title()
        gui._canvas.draw()
        gui.return_to_builder()
        assert not gui._inspection_mode
        assert gui.surface is not surface
        assert {key: value.get() for key, value in gui._fields.items()} == retained
        assert str(gui._entries["rows"].cget("state")) == "normal"
        assert str(gui._entries["cor_mm"].cget("state")) == "normal"
        assert str(gui._entries["transverse_centers_mm"].cget("state")) == "disabled"
        assert str(gui._layout_box.cget("state")) == "readonly"
        gui.inspect_geometry(surface=surface)
        gui._example.set("TIGRE: circular cone-beam CT")
        gui.load_example()
        assert not gui._inspection_mode
        assert gui.surface.pixel_centers_mm.shape == (50, 128, 128, 3)
        import tigre
        geo = tigre.geometry_default(high_resolution=False)
        geo.nDetector = np.array([3, 4])
        geo.sDetector = geo.nDetector * geo.dDetector
        geo.rotDetector = np.deg2rad([10, 15, 20])
        geo.COR = 12
        angles = np.array([0.1, 0.5])
        expected = surface_from_geometry(geo, angles)
        gui.inspect_geometry(geo, angles=angles)
        np.testing.assert_allclose(gui.surface.pixel_centers_mm, expected.pixel_centers_mm)
        assert np.asarray(geo.rotDetector).shape == (3,)  # Inspection did not mutate input.
        gui._render_frame(1)
        gui._canvas.draw()
    finally:
        root.destroy()
