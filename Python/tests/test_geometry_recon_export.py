"""Exported examples preserve saved geometry and dispatch FDK/CGLS without a GPU."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tigre.utilities.geometry_io import save_geometry_yaml, load_geometry_file
from tigre.utilities.geometry_recon_export import reconstruction_script_text


def saved_example(tmp_path, moving=False):
    path = tmp_path / "saved geometry.yaml"
    settings = dict(layout="Piecewise linear", n_views=3, source_radius_mm=1000,
        source_z_mm=0, module_shape=(2, 3), pixel_spacing_mm=(1, 1),
        detector_radius_mm=500, column_gap=0, n_modules=1, angular_span_deg=0,
        transverse_centers_mm=[0], normal_offsets_mm=[0], panel_angles_deg=[0], z_offsets_mm=[0])
    offsets = np.array([[0, 1, 2], [10, 1, 2], [20, 1, 2]]) if moving else np.array([0, 1, 2])
    geo = SimpleNamespace(nVoxel=np.array([2, 3, 4]), sVoxel=np.array([20, 30, 40]), offOrigin=offsets)
    save_geometry_yaml(path, settings=settings, geometry=geo)
    script = tmp_path / "reconstruct.py"
    text = reconstruction_script_text(path, script)
    namespace = dict(__file__=str(script), __name__="exported_example")
    exec(compile(text, str(script), "exec"), namespace)
    return path, namespace


def test_exported_script_uses_saved_voxels_and_both_algorithms(tmp_path):
    path, script = saved_example(tmp_path)
    assert script["DEFAULT_GEOMETRY"] == path
    geo, surface = script["prepare_geometry"](path)
    np.testing.assert_array_equal(geo.nVoxel, [2, 3, 4])
    np.testing.assert_allclose(geo.dVoxel, [10, 10, 10])
    projections = tmp_path / "projections.npy"
    np.save(projections, np.ones(surface.valid_mask.shape, np.float32))
    script["fdk_surface"] = Mock(return_value=np.ones((2, 3, 4)))
    script["cgls_surface"] = Mock(return_value=(np.zeros((2, 3, 4)), [3, 2, 1]))
    output = tmp_path / "output.npz"
    script["main"](["--projections", str(projections), "--output", str(output), "--iterations", "3"])
    assert script["fdk_surface"].call_count == 1
    assert script["cgls_surface"].call_args.kwargs["niter"] == 3
    with np.load(output, allow_pickle=False) as result:
        assert result["fdk"].shape == (2, 3, 4)
        assert result["cgls"].shape == (2, 3, 4)


def test_moving_volume_changes_both_source_and_detector_to_object_frame(tmp_path):
    path, script = saved_example(tmp_path, moving=True)
    document = load_geometry_file(path)
    geo, surface = script["prepare_geometry"](path)
    shifts = document.geometry.offOrigin[:, ::-1]
    np.testing.assert_allclose(surface.source_positions_mm, document.surface.source_positions_mm - shifts)
    np.testing.assert_allclose(surface.pixel_centers_mm, document.surface.pixel_centers_mm - shifts[:, None, None])
    np.testing.assert_array_equal(geo.offOrigin, [0, 0, 0])
    np.testing.assert_array_equal(surface.valid_mask, document.surface.valid_mask)


def test_exported_script_rejects_nontransmitting_views_before_reconstruction(tmp_path):
    path, script = saved_example(tmp_path, moving=True)
    document = load_geometry_file(path)
    document.geometry.offOrigin[:, 0] += 100
    save_geometry_yaml(path, settings=document.settings, geometry=document.geometry)
    script["fdk_surface"] = Mock()
    script["cgls_surface"] = Mock()
    with pytest.raises(ValueError, match="entire scan"):
        script["main"](["--projections", str(tmp_path / "not_loaded.npy")])
    script["fdk_surface"].assert_not_called()
    script["cgls_surface"].assert_not_called()


def test_unsupported_fdk_does_not_prevent_cgls(tmp_path):
    path, script = saved_example(tmp_path)
    surface = load_geometry_file(path).surface
    projections = tmp_path / "projections.npy"
    np.save(projections, np.zeros(surface.valid_mask.shape, np.float32))
    script["fdk_surface"] = Mock(side_effect=NotImplementedError("unsupported orbit"))
    script["cgls_surface"] = Mock(return_value=(np.zeros((2, 3, 4)), []))
    output = tmp_path / "result.npz"
    script["main"](["--projections", str(projections), "--output", str(output)])
    with np.load(output, allow_pickle=False) as result:
        assert "cgls" in result and "fdk" not in result
        assert str(result["fdk_skip_reason"]) == "unsupported orbit"
    with pytest.raises(NotImplementedError):
        script["main"](["--projections", str(projections), "--algorithm", "fdk"])


def test_export_button_requires_successful_yaml_save(tmp_path):
    import tkinter as tk
    from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    try:
        with patch.object(DetectorGeometryBuilder, "update_preview"):
            gui = DetectorGeometryBuilder(root)
        assert str(gui._export_recon_button.cget("state")) == "disabled"
        path = tmp_path / "saved.yaml"
        with patch("tkinter.filedialog.asksaveasfilename", return_value=""):
            gui.save_geometry_settings()
        assert gui._last_saved_yaml is None
        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(path)), \
                patch("tigre.utilities.detector_geometry_gui.save_geometry_yaml", side_effect=OSError("failed")), \
                patch("tkinter.messagebox.showerror"):
            gui.save_geometry_settings()
        assert str(gui._export_recon_button.cget("state")) == "disabled"
        gui._fields["n_views"].set("3")
        gui._fields["rows"].set("2")
        gui._fields["columns"].set("3")
        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(path)):
            gui.save_geometry_settings()
        assert gui._last_saved_yaml == path
        assert str(gui._export_recon_button.cget("state")) == "normal"
        target = tmp_path / "saved_recon.py"
        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(target)):
            gui._export_recon_button.invoke()
        compile(target.read_text(encoding="utf-8"), str(target), "exec")
        assert "saved.yaml" in target.read_text(encoding="utf-8")
    finally:
        root.destroy()
