"""Linked voxel counts, extent and spacing; no desktop window required."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder


def builder():
    import tkinter as tk
    interpreter = tk.Tcl()
    gui = object.__new__(DetectorGeometryBuilder)
    gui._test_interpreter = interpreter
    gui._volume_fields = {name: tk.StringVar(master=interpreter, value=value) for name, value in
        (("nVoxel", "10, 20, 30"), ("sVoxel", "20, 40, 60"), ("offOrigin", "0, 0, 0"))}
    gui._voxel_size = tk.StringVar(master=interpreter, value="2, 2, 2")
    gui._voxel_spacing = tk.StringVar(master=interpreter)
    gui._voxel_fields_syncing = False
    gui._volume_motion = Mock(get=Mock(return_value=False))
    gui._volume_end = Mock()
    for name, variable in (("nVoxel", gui._volume_fields["nVoxel"]),
                           ("sVoxel", gui._volume_fields["sVoxel"]), ("dVoxel", gui._voxel_size)):
        variable.trace_add("write", lambda *_args, field=name: gui._sync_voxel_fields(field))
    return gui


def test_editing_each_volume_dimension_updates_the_related_field():
    gui = builder()
    gui._volume_fields["nVoxel"].set("20, 30, 40")
    np.testing.assert_allclose(gui._parse_numbers(gui._volume_fields["sVoxel"].get(), "size"), [40, 60, 80])
    gui._voxel_size.set("1, 0.5, 3")
    np.testing.assert_allclose(gui._parse_numbers(gui._volume_fields["sVoxel"].get(), "size"), [20, 15, 120])
    gui._volume_fields["sVoxel"].set("80, 90, 160")
    np.testing.assert_allclose(gui._parse_numbers(gui._voxel_size.get(), "spacing"), [4, 3, 4])
    np.testing.assert_array_equal(gui._read_volume().nVoxel, [20, 30, 40])
    np.testing.assert_allclose(gui._read_volume().dVoxel, [4, 3, 4])


def test_partial_edit_is_not_overwritten_and_invalid_spacing_cannot_confirm():
    gui = builder()
    gui._voxel_size.set("1,")
    assert gui._voxel_size.get() == "1,"
    assert gui._volume_fields["sVoxel"].get() == "20, 40, 60"
    with pytest.raises(ValueError):
        gui._read_volume()


def test_loading_volume_preserves_counts_and_extent_before_deriving_spacing():
    gui = builder()
    gui._set_volume_controls(SimpleNamespace(nVoxel=np.array([3, 5, 7]),
        sVoxel=np.array([1., 2., 3.]), offOrigin=np.array([4., 5., 6.])))
    np.testing.assert_allclose(gui._read_volume().sVoxel, [1, 2, 3])
    np.testing.assert_allclose(gui._read_volume().dVoxel, [1/3, 2/5, 3/7])
