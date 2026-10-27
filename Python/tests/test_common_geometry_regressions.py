from types import SimpleNamespace

import numpy as np
import pytest

from tigre.utilities.common_geometry import (
    ArbitrarySourceDetMoveGeo, find_intersection_planes, staticDetLinearSourceGeo,
)


def geometry():
    return SimpleNamespace(DSO=100.0, DSD=200.0,
                           offDetector=np.zeros(2), rotDetector=np.zeros(3))


def test_oblique_linear_source_distance():
    result = staticDetLinearSourceGeo(geometry(), np.array([50.0]), s_rot=30)
    np.testing.assert_allclose(result.DSO, [np.sqrt(17500.0)])


def test_arbitrary_detector_preserves_offset_components():
    result = ArbitrarySourceDetMoveGeo(
        geometry(), [[100., 0., 0.]], [[-100., 3., 4.]])
    np.testing.assert_allclose(result.offDetector, [[4., 3.]])


def test_arbitrary_detector_rejects_same_side():
    with pytest.raises(RuntimeError, match="opposite sides"):
        ArbitrarySourceDetMoveGeo(geometry(), [[100., 0., 0.]], [[50., 0., 0.]])


def test_plane_intersection_accepts_numpy_line_origins():
    intersection, _ = find_intersection_planes(
        np.array([[2., 0., 0.]]), np.array([[1., 0., 0.]]),
        np.array([[0., 3., 4.]]))
    np.testing.assert_allclose(intersection, [[2., 3., 4.]])
