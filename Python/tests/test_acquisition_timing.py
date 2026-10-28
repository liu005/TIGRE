"""Acquisition cadence, multi-turn orbits and synchronized volume motion."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tigre.utilities.detector_geometry_gui import (
    DetectorGeometryBuilder, acquisition_timing, build_modular_surface,
)
from tigre.utilities.geometry_io import save_geometry_yaml, load_geometry_file, surface_archive_arrays, load_geometry_archive


def timing(**changes):
    return dict(readout_period_ms=100, firing_rate_hz=4, rotation_period_s=1,
                table_speed_mm_s=-10, playback_rate=2, **changes)


def parameters():
    return dict(layout="Piecewise linear", n_views=9, source_radius_mm=1000,
                source_z_mm=0, module_shape=(2, 3), pixel_spacing_mm=(1, 1),
                detector_radius_mm=500, column_gap=0, n_modules=1,
                angular_span_deg=0, transverse_centers_mm=[0], normal_offsets_mm=[0],
                panel_angles_deg=[0], z_offsets_mm=[0], acquisition=timing())


def volume_builder():
    builder = object.__new__(DetectorGeometryBuilder)
    builder._timing_enabled = Mock(get=Mock(return_value=True))
    builder._timing_fields = {name: Mock(get=Mock(return_value=str(value)))
                              for name, value in timing().items()}
    builder._fields = {"n_views": Mock(get=Mock(return_value="9"))}
    builder._volume_fields = {name: Mock(get=Mock(return_value=value)) for name, value in
        (("nVoxel", "10, 20, 30"), ("sVoxel", "20, 40, 60"), ("offOrigin", "100, 2, 3"))}
    return builder


def test_multi_turn_source_and_volume_share_pulse_times():
    values, times = acquisition_timing(9, timing())
    np.testing.assert_allclose(times, np.arange(9) / 4)
    surface = build_modular_surface(**parameters())
    np.testing.assert_allclose(surface.source_positions_mm[[0, 4, 8]], [[1000, 0, 0]] * 3, atol=1e-10)
    np.testing.assert_allclose(surface.source_positions_mm[1], [0, 1000, 0], atol=1e-10)
    volume = volume_builder()._read_volume()
    np.testing.assert_allclose(volume.offOrigin[:, 0], 100 - 10 * times)
    np.testing.assert_allclose(volume.offOrigin[:, 1:], [[2, 3]] * 9)
    np.testing.assert_allclose(volume.dVoxel, [2, 2, 2])


@pytest.mark.parametrize("field,value", [
    ("readout_period_ms", 0), ("readout_period_ms", 300),
    ("firing_rate_hz", -1), ("rotation_period_s", 0),
    ("table_speed_mm_s", np.nan), ("playback_rate", 0),
])
def test_invalid_timing_is_rejected(field, value):
    values = timing()
    values[field] = value
    with pytest.raises(ValueError):
        acquisition_timing(9, values)


def test_single_view_and_detector_rate_boundary():
    values = timing()
    values["readout_period_ms"] = 250
    np.testing.assert_array_equal(acquisition_timing(1, values)[1], [0])
    builder = volume_builder()
    builder._fields["n_views"].get.return_value = "1"
    np.testing.assert_allclose(builder._read_volume().offOrigin, [[100, 2, 3]])


def test_timed_endpoints_determine_speed_and_all_three_coordinates():
    builder = volume_builder()
    builder._volume_motion = Mock(get=Mock(return_value=True))
    builder._volume_end = Mock(get=Mock(return_value="140, -2, 9"))
    geo = builder._read_volume()
    np.testing.assert_allclose(geo.offOrigin[0], [100, 2, 3])
    np.testing.assert_allclose(geo.offOrigin[-1], [140, -2, 9])
    np.testing.assert_allclose(geo.offOrigin[4], [120, 0, 6])
    assert geo.acquisition["table_speed_mm_s"] == 20
    assert builder._read_acquisition()["table_speed_mm_s"] == 20


def test_single_view_rejects_distinct_movement_endpoints():
    builder = volume_builder()
    builder._fields["n_views"].get.return_value = "1"
    builder._volume_motion = Mock(get=Mock(return_value=True))
    builder._volume_end = Mock(get=Mock(return_value="140, 2, 3"))
    with pytest.raises(ValueError, match="two views"):
        builder._read_volume()


def test_timing_yaml_round_trip(tmp_path):
    path = tmp_path / "timing.yaml"
    geo = volume_builder()._read_volume()
    save_geometry_yaml(path, settings=parameters(), geometry=geo)
    loaded = load_geometry_file(path)
    assert loaded.settings["acquisition"] == timing()
    np.testing.assert_allclose(loaded.geometry.offOrigin, geo.offOrigin)
    np.testing.assert_allclose(loaded.surface.source_positions_mm,
                               build_modular_surface(**parameters()).source_positions_mm)
    np.testing.assert_allclose(loaded.geometry.view_times_s, np.arange(9) / 4)
    assert loaded.geometry.motion_mode == "speed"


def test_timing_npz_round_trip(tmp_path):
    surface = build_modular_surface(**parameters())
    geo = volume_builder()._read_volume()
    path = tmp_path / "timing.npz"
    np.savez_compressed(path, **surface_archive_arrays(surface, geo))
    loaded, _ = load_geometry_archive(path)
    assert loaded.acquisition == timing()
    np.testing.assert_allclose(loaded.view_times_s, np.arange(9) / 4)
    np.testing.assert_allclose(loaded.offOrigin, geo.offOrigin)
    assert loaded.motion_mode == "speed"
    payload = surface_archive_arrays(surface, geo)
    payload["view_times_s"] = np.zeros(9)
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="timestamps"):
        load_geometry_archive(path)


def test_timing_exact_surface_yaml_preserves_playback(tmp_path):
    path = tmp_path / "exact.yaml"
    save_geometry_yaml(path, surface=build_modular_surface(**parameters()),
                       geometry=volume_builder()._read_volume())
    loaded = load_geometry_file(path)
    assert loaded.geometry.acquisition == timing()
    np.testing.assert_allclose(loaded.geometry.view_times_s, np.arange(9) / 4)


def test_gui_timing_controls_derive_orbit_and_offsets(monkeypatch, tmp_path):
    import tkinter as tk
    from tkinter import messagebox
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    def fail_dialog(*args, **kwargs):
        pytest.fail(str(args))
    monkeypatch.setattr(messagebox, "showerror", fail_dialog)
    try:
        from unittest.mock import patch
        with patch.object(DetectorGeometryBuilder, "update_preview"):
            builder = DetectorGeometryBuilder(root)
        for name, value in (("n_views", "9"), ("rows", "2"), ("columns", "3")):
            builder._fields[name].set(value)
        for name, value in timing().items():
            builder._timing_fields[name].set(str(value))
        builder._timing_enabled.set(True)
        builder._update_field_states()
        builder.update_preview()
        assert builder._parameters["scan_span_deg"] == 810
        assert str(builder._entries["scan_span_deg"].cget("state")) == "disabled"
        np.testing.assert_allclose(builder._builder_geometry.offOrigin[-1], [-20, 0, 0])
        builder._render_frame(8)
        assert "2 s" in builder._frame_slider.cget("label")
        builder._volume_motion.set(True)
        builder._volume_fields["offOrigin"].set("10, 2, 3")
        builder._volume_end.set("30, 2, 3")
        builder._update_field_states()
        assert str(builder._entries["volume_end"].cget("state")) == "normal"
        assert str(builder._entries["table_speed_mm_s"].cget("state")) == "disabled"
        builder.update_preview()
        np.testing.assert_allclose(builder._builder_geometry.offOrigin[[0, -1]], [[10, 2, 3], [30, 2, 3]])
        assert builder._parameters["acquisition"]["table_speed_mm_s"] == 10
        builder._render_frame(8)
        expected = builder.surface.source_positions_mm[8] - builder._builder_geometry.offOrigin[8, ::-1]
        np.testing.assert_allclose(builder._motion_positions[8], expected)
        np.testing.assert_allclose(np.mean(builder._surface_renderer.volume._vec[2]), 30)
        builder._timing_enabled.set(False)
        builder._update_field_states()
        assert str(builder._entries["scan_span_deg"].cget("state")) == "normal"
        builder._timing_enabled.set(True)
        builder._volume_motion.set(False)
        builder._volume_fields["offOrigin"].set("0, 0, 0")
        builder._timing_fields["table_speed_mm_s"].set("10")
        path = tmp_path / "speed_mode.yaml"
        save_geometry_yaml(path, settings=builder._read_parameters(), geometry=builder._read_volume())
        builder.load_geometry_path(path)
        assert not builder._volume_motion.get()
        builder._fields["n_views"].set("17")
        np.testing.assert_allclose(builder._read_volume().offOrigin[-1], [40, 0, 0])
    finally:
        root.destroy()


def test_playback_uses_elapsed_time_and_skips_frames(monkeypatch):
    import tigre.utilities.detector_geometry_gui as module
    builder = object.__new__(DetectorGeometryBuilder)
    builder.surface = SimpleNamespace(source_positions_mm=np.zeros((9, 3)))
    builder._parameters = {"acquisition": timing()}
    builder._animation_playing = True
    builder._animation_started = 10
    builder._animation_start_frame = 1
    builder._frame_slider = Mock()
    builder.root = Mock()
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.5)
    builder._schedule_next_frame()
    builder._frame_slider.set.assert_called_once_with(5)
    assert builder.root.after.call_args.args[0] == 100
