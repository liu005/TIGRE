"""Differing inactive endpoints require an explicit movement/stationary choice."""
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder


def builder():
    gui = object.__new__(DetectorGeometryBuilder)
    gui._volume_motion = Mock(get=Mock(return_value=False))
    gui._timing_enabled = Mock(get=Mock(return_value=False))
    gui._movement_endpoints = lambda: (np.array([-200, 0, 0]), np.array([200, 0, 0]))
    gui._volume_fields = {"offOrigin": Mock(get=Mock(return_value="-200, 0, 0"))}
    gui._volume_end = Mock(get=Mock(return_value="200, 0, 0"))
    gui._update_field_states = Mock()
    gui._status = Mock()
    gui.root = Mock()
    return gui


@pytest.mark.parametrize("answer", [True, False, None])
def test_differing_inactive_endpoints_prompt_for_intent(answer):
    gui = builder()
    with patch("tkinter.messagebox.askyesnocancel", return_value=answer) as prompt:
        assert gui._resolve_endpoint_motion() is (answer is not None)
    prompt.assert_called_once()
    if answer:
        gui._volume_motion.set.assert_called_once_with(True)
        gui._update_field_states.assert_called_once()
    else:
        gui._volume_motion.set.assert_not_called()
    if answer is False:
        with patch("tkinter.messagebox.askyesnocancel") as prompt:
            assert gui._resolve_endpoint_motion()
            prompt.assert_not_called()  # Same explicit stationary choice is retained.


def test_cancel_does_not_run_confirm_or_change_the_scene():
    gui = builder()
    gui.update_preview = Mock()
    with patch("tkinter.messagebox.askyesnocancel", return_value=None):
        gui.confirm_parameters()
    gui.update_preview.assert_not_called()


@pytest.mark.parametrize("mode", ["timed", "moving", "equal"])
def test_unambiguous_settings_do_not_prompt(mode):
    gui = builder()
    if mode == "timed":
        gui._timing_enabled.get.return_value = True
    elif mode == "moving":
        gui._volume_motion.get.return_value = True
    else:
        gui._movement_endpoints = lambda: (np.zeros(3), np.zeros(3))
    with patch("tkinter.messagebox.askyesnocancel") as prompt:
        assert gui._resolve_endpoint_motion()
        prompt.assert_not_called()
