"""The diamond preset forms four physical sides without rotating its detector."""

from unittest.mock import patch

import numpy as np
import pytest

from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder, build_modular_surface
from tigre.utilities.ct_physics import validate_transmission_geometry


def test_diamond_preset_gui_geometry_and_transmission(tmp_path):
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    try:
        with patch.object(DetectorGeometryBuilder, "update_preview"):
            app = DetectorGeometryBuilder(root)
            app._example.set("Diamond: L-shaped source + detector")
            app.load_example()
        assert app._source_trajectory.get() == "Piecewise linear"
        assert app._detector_stationary is True
        parameters = app._read_parameters()
        surface = build_modular_surface(**parameters)
        assert surface.source_beam is not None
        np.testing.assert_allclose(surface.source_beam.opening_angles_deg, np.tile([95, 20], (61, 1)))
        np.testing.assert_allclose(surface.source_beam.axis_directions[[0, 30]],
            np.tile(np.array([-1, -1, 0]) / np.sqrt(2), (2, 1)))
        np.testing.assert_allclose(surface.source_beam.axis_directions[31], np.array([1, -1, 0]) / np.sqrt(2))
        assert np.count_nonzero(surface.valid_mask) < np.count_nonzero(surface.detector_valid_mask)
        source = surface.source_positions_mm
        np.testing.assert_allclose(source[[0, 30, 60]],
                                   [[600, 0, 0], [0, 600, 0], [-600, 0, 0]])
        np.testing.assert_allclose(np.abs(source[:, 0]) + source[:, 1], 600, atol=1e-4)
        assert np.count_nonzero(np.all(source == [0, 600, 0], axis=1)) == 1
        first, second = np.diff(source[[0, 30, 60]], axis=0)
        assert np.dot(first, second) == pytest.approx(0)
        np.testing.assert_array_equal(surface.pixel_centers_mm,
            np.broadcast_to(surface.pixel_centers_mm[0], surface.pixel_centers_mm.shape))
        assert len(surface._module_slices) == 2
        edge_sets = []
        for rows, columns in surface._module_slices:
            panel = surface.pixel_centers_mm[0, rows, columns]
            xy = panel[..., :2]
            np.testing.assert_allclose(np.abs(xy[..., 0]) - xy[..., 1], 600, atol=1e-4)
            assert np.all(xy[..., 1] < 0)
            # Extend the first/last measured pixel centres by half a pixel
            # to check real panel edges rather than the packed array layout.
            midline = panel.mean(axis=0, dtype=np.float64)
            step = midline[1] - midline[0]
            edges = np.array([midline[0] - step / 2, midline[-1] + step / 2])
            edge_sets.append(edges)
        expected = [np.array([[0, -600, 0], [-600, 0, 0]]),
                    np.array([[600, 0, 0], [0, -600, 0]])]
        for actual, target in zip(edge_sets, expected):
            np.testing.assert_allclose(actual, target, atol=1e-3)
        geometry = app._read_volume()
        report = validate_transmission_geometry(surface, geometry)
        assert report["views"] == 61
        assert report["empty_views"] == 0
        assert report["crossing_samples"] > 0
        from tigre.utilities.geometry_io import save_geometry_yaml
        path = tmp_path / "diamond.yaml"
        save_geometry_yaml(path, settings=parameters, geometry=geometry)
        with patch("tkinter.messagebox.showerror", side_effect=AssertionError):
            app.load_geometry_path(path)
        assert app._fields["beam_model"].get() == "On"
        assert app._fields["beam_axis_mode"].get() == "Segment normal"
        np.testing.assert_array_equal(app.surface.valid_mask, surface.valid_mask)
        assert app._surface_renderer.update(31)
        app._fields["beam_aperture_angles_deg"].set("0, 0")
        no_aperture = build_modular_surface(**app._read_parameters())
        assert no_aperture.source_beam.aperture_angles_deg is None
        assert no_aperture.valid_mask.all()
        app._fields["beam_roll_deg"].set("90")
        rolled = build_modular_surface(**app._read_parameters())
        np.testing.assert_allclose(np.abs(rolled.source_beam.wide_axis_directions[:, 2]), 1)
        app._fields["beam_model"].set("Off")
        with patch("tkinter.messagebox.showerror", side_effect=AssertionError):
            app._on_beam_model_changed()
        assert str(app._entries["beam_opening_angles_deg"].cget("state")) == "disabled"
        uncollimated = build_modular_surface(**app._read_parameters())
        assert uncollimated.source_beam is None and uncollimated.valid_mask.all()
        assert app.surface.source_beam is None and app.surface.valid_mask.all()
        assert not any(line.get_label() in ("Beam flux contour", "Hard aperture") for line in app._axes.lines)
        app._fields["beam_aperture_angles_deg"].set("140, 40")
        app._fields["beam_roll_deg"].set("0")
        app._fields["beam_model"].set("On")
        with patch("tkinter.messagebox.showerror", side_effect=AssertionError):
            app._on_beam_model_changed()
        np.testing.assert_array_equal(app.surface.valid_mask, surface.valid_mask)
        assert {"Beam flux contour", "Hard aperture"} <= {line.get_label() for line in app._axes.lines}
    finally:
        root.destroy()
