"""Finite transmission rays, off-center objects and moving-volume validation."""
from types import SimpleNamespace

import numpy as np
import pytest

from tigre.utilities.ct_physics import validate_transmission_geometry
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry


def geometry(offset=(0, 0, 0)):
    return SimpleNamespace(sVoxel=np.array([10, 10, 10]), offOrigin=np.asarray(offset))


def surface(source, detector):
    return DetectorSurfaceGeometry(np.atleast_2d(source), np.asarray(detector, dtype=float).reshape(-1, 1, 1, 3))


def test_opposite_sides_transmit():
    assert validate_transmission_geometry(surface([20, 0, 0], [-20, 0, 0]), geometry())["crossing_samples"] == 1


def test_fixed_detector_scan_allows_empty_same_side_view():
    values = surface([[20, 0, 0], [-20, 0, 0]], [[-30, 0, 0], [-30, 0, 0]])
    report = validate_transmission_geometry(values, geometry())
    assert report["crossing_samples"] == 1
    assert report["empty_views"] == 1


def test_valid_corner_transmission_is_not_rejected_by_hemisphere_test():
    # Both endpoints are in the center's +x hemisphere, yet the segment crosses the box.
    assert validate_transmission_geometry(surface([6, 0, 0], [4, 6, 0]), geometry())["crossing_samples"] == 1


@pytest.mark.parametrize("source,detector,match", [
    ([0, 0, 0], [-20, 0, 0], "source is inside"),
    ([20, 0, 0], [0, 0, 0], "detector panel 1 is inside"),
    ([20, 20, 0], [-20, 20, 0], "no measured.*crosses"),
    ([20, 0, 0], [30, 0, 0], "entire scan"),
])
def test_nontransmission_geometry_is_rejected(source, detector, match):
    with pytest.raises(ValueError, match=match):
        validate_transmission_geometry(surface(source, detector), geometry())


def test_moving_volume_allows_helical_entry_and_exit_empty_views():
    rays = surface([[20, 0, 0]] * 2, [[-20, 0, 0]] * 2)
    report = validate_transmission_geometry(rays, geometry([[0, 0, 0], [20, 0, 0]]))
    assert report["crossing_samples"] == 1 and report["empty_views"] == 1
    with pytest.raises(ValueError, match="entire scan"):
        validate_transmission_geometry(rays, geometry([[20, 0, 0], [40, 0, 0]]))


def test_masked_same_side_pixel_is_ignored():
    pixels = np.array([[[[-20, 0, 0], [30, 0, 0]]]], dtype=float)
    rays = DetectorSurfaceGeometry([[20, 0, 0]], pixels, np.array([[[True, False]]]))
    assert validate_transmission_geometry(rays, geometry())["crossing_samples"] == 1


def test_empty_same_side_panel_does_not_block_transmitting_panel():
    pixels = np.array([[[[-20, 0, 0], [30, 0, 0]]]], dtype=float)
    rays = DetectorSurfaceGeometry([[20, 0, 0]], pixels,
        module_slices=[(slice(0, 1), slice(0, 1)), (slice(0, 1), slice(1, 2))])
    assert validate_transmission_geometry(rays, geometry())["crossing_samples"] == 1


def test_invalid_hardware_position_still_blocks_partially_transmitting_scan():
    rays = surface([[20, 0, 0]] * 2, [[-20, 0, 0]] * 2)
    with pytest.raises(ValueError, match="source is inside.*view 2"):
        validate_transmission_geometry(rays, geometry([[0, 0, 0], [0, 0, 20]]))


def test_gui_allows_empty_views_but_rejects_completely_empty_scan():
    import tkinter as tk
    from unittest.mock import patch
    from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    try:
        with patch.object(DetectorGeometryBuilder, "update_preview"):
            builder = DetectorGeometryBuilder(root)
        for name, value in (("n_views", "4"), ("rows", "2"), ("columns", "3"),
                            ("n_modules", "1"), ("angular_span_deg", "1")):
            builder._fields[name].set(value)
        builder.confirm_parameters()
        previous = builder.surface
        assert previous is not None
        assert "Physical CT: PASS" in builder._validation_text.get("1.0", "end")
        assert "CGLS geometry: supported" in builder._validation_text.get("1.0", "end")
        assert "FDK geometry:" in builder._validation_text.get("1.0", "end")
        builder._detector_motion.set("Fixed Position")
        builder._on_detector_motion_changed()
        builder.confirm_parameters()
        previous = builder.surface
        assert builder._physical_report["empty_views"] > 0
        assert "Empty views:" in builder._validation_text.get("1.0", "end")
        builder._volume_fields["offOrigin"].set("1000, 0, 0")
        with patch("tkinter.messagebox.showerror") as error, patch("tkinter.messagebox.askyesnocancel", return_value=False):
            builder.confirm_parameters()
        assert error.call_count == 1
        assert "Physical CT geometry" in error.call_args.args[1]
        assert "Volume movement is OFF" in error.call_args.args[1]
        assert "End" in error.call_args.args[1] and "ignored" in error.call_args.args[1]
        assert builder.surface is previous
        assert "Confirm: FAILED" in builder._validation_text.get("1.0", "end")
        assert "previous scene retained" in builder._status.get()
    finally:
        root.destroy()
