"""Tab overviews replace stale field help; keyboard navigation explains parameters."""
from unittest.mock import patch

import pytest

from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder, TAB_HELP


def test_help_follows_tabs_hover_and_keyboard_focus():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    try:
        with patch.object(DetectorGeometryBuilder, "update_preview"):
            builder = DetectorGeometryBuilder(root)
        for tab in builder._settings_tabs.tabs():
            builder._set_help("COR (mm)")
            builder._settings_tabs.select(tab)
            root.update()
            name = builder._settings_tabs.tab(tab, "text")
            assert builder._help_text.get("1.0", "end").strip() == TAB_HELP[name]
            assert builder._help_frame.cget("text") == f"Parameter help: {name}"
        builder._entries["volume_start"].event_generate("<FocusIn>")
        root.update()
        assert "Initial volume center" in builder._help_text.get("1.0", "end")
        builder._settings_tabs.select(1)
        root.update()
        builder._detector_motion_button.event_generate("<Enter>")
        root.update()
        assert "Choose Moving with Source" in builder._help_text.get("1.0", "end")
        builder._inspection_mode = True
        builder._on_help_tab_changed()
        assert "Inspection mode" in builder._help_text.get("1.0", "end")
    finally:
        root.destroy()


def test_play_requires_first_successful_confirm():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    try:
        with patch.object(DetectorGeometryBuilder, "update_preview"):
            builder = DetectorGeometryBuilder(root)
        builder._fields["n_views"].set("3")
        builder._fields["rows"].set("2")
        builder._fields["columns"].set("3")
        builder.update_preview()
        assert builder.surface is not None
        assert str(builder._play_button.cget("state")) == "disabled"
        builder._fields["rows"].set("0")
        with patch("tkinter.messagebox.showerror"):
            builder._preview_button.invoke()
        assert not builder._confirmed_once
        assert str(builder._play_button.cget("state")) == "disabled"
        builder._fields["rows"].set("2")
        builder._preview_button.invoke()
        assert builder._confirmed_once
        assert str(builder._play_button.cget("state")) == "normal"
        confirmed_surface = builder.surface
        builder._fields["rows"].set("0")
        with patch("tkinter.messagebox.showerror"):
            builder._preview_button.invoke()
        assert builder.surface is confirmed_surface
    finally:
        root.destroy()


def test_source_trajectory_selection_and_detector_motion_stay_in_sync():
    import tkinter as tk
    import numpy as np
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    try:
        with patch.object(DetectorGeometryBuilder, "update_preview"):
            builder = DetectorGeometryBuilder(root)
        for name, value in (("n_views", "3"), ("rows", "2"), ("columns", "3")):
            builder._fields[name].set(value)
        builder._source_trajectory.set("Linear")
        builder._source_trajectory_box.event_generate("<<ComboboxSelected>>")
        assert builder._detector_stationary is True
        assert str(builder._entries["source_start_mm"].cget("state")) == "normal"
        assert str(builder._entries["source_radius_mm"].cget("state")) == "disabled"
        builder._preview_button.invoke()
        np.testing.assert_allclose(builder.surface.source_positions_mm[[0, -1]], [[1000, 0, 0], [1000, 200, 0]])
        builder._source_trajectory.set("Cylindrical")
        builder._on_source_trajectory_changed()
        assert builder._detector_stationary == "arc"
        assert str(builder._entries["source_start_mm"].cget("state")) == "disabled"
        assert str(builder._entries["source_radius_mm"].cget("state")) == "normal"
        builder.toggle_detector_motion()
        assert builder._source_trajectory.get() == "Cylindrical"
        assert builder._detector_stationary is False
        builder.toggle_detector_motion()
        assert builder._source_trajectory.get() == "Linear"
        previous = builder.surface
        builder._detector_motion.set("Moving with Source")
        builder._detector_motion_box.event_generate("<<ComboboxSelected>>")
        assert builder._detector_stationary is False
        assert builder._source_trajectory.get() == "Cylindrical"
        builder._detector_motion.set("Fixed Position")
        builder._on_detector_motion_changed()
        assert builder._detector_stationary == "arc"
        assert builder.surface is previous
        assert str(builder._detector_motion_box.cget("state")) == "readonly"
        builder._inspection_mode = True
        builder._update_field_states()
        assert str(builder._source_trajectory_box.cget("state")) == "disabled"
        assert str(builder._detector_motion_box.cget("state")) == "disabled"
    finally:
        root.destroy()
