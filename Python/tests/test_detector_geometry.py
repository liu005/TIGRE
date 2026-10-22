import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tigre.utilities.detector_geometry import (
    DetectorModule,
    DetectorSurfaceGeometry,
)


def _pose(translation, rotation=None):
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
    transform[:3, 3] = translation
    return transform


def test_explicit_pixel_centers_support_per_view_curved_coordinates_and_mask():
    source = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    centers = np.zeros((2, 2, 3, 3))
    centers[0, :, :, 0] = -500.0
    centers[1, :, :, 0] = -450.0
    centers[:, :, :, 1] = np.array([[-2.0, 0.0, 3.0], [-2.0, 0.0, 3.0]])
    centers[:, :, :, 2] = np.array([[-1.0], [1.0]])
    mask = np.array([[True, False, True], [True, True, True]])

    geometry = DetectorSurfaceGeometry(source, centers, mask)

    assert geometry.source_positions_mm.dtype == np.float32
    assert geometry.pixel_centers_mm.shape == (2, 2, 3, 3)
    assert geometry.valid_mask.shape == (2, 2, 3)
    assert not geometry.valid_mask[:, 0, 1].any()
    assert np.allclose(geometry.pixel_centers_mm, centers)


def test_modules_pack_gaps_nonuniform_centers_and_per_view_poses():
    source = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    left_pixels = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.5, 1.0]],
    ])
    right_pixels = np.array([
        [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 2.5, 1.0]],
    ])
    left_pose = np.stack([
        _pose([-500.0, 0.0, 0.0]),
        _pose([-500.0, 10.0, 0.0], [0.0, 0.0, 0.1]),
    ])
    right_pose = _pose([-500.0, 0.0, 0.0])
    modules = [
        DetectorModule(left_pixels, left_pose, row_offset=0, column_offset=0),
        DetectorModule(right_pixels, right_pose, row_offset=0, column_offset=3),
    ]

    geometry = DetectorSurfaceGeometry.from_modules(source, modules)
    module_data = [
        np.full((2, 2, 2), 4.0, dtype=np.float32),
        np.full((2, 2, 2), 9.0, dtype=np.float32),
    ]
    packed = geometry.pack_module_data(module_data)

    assert geometry.pixel_centers_mm.shape == (2, 2, 5, 3)
    expected_mask = np.array([[True, True, False, True, True]] * 2)
    assert np.array_equal(
        geometry.valid_mask, np.broadcast_to(expected_mask, (2, 2, 5)))
    assert np.all(packed[:, :, 0:2] == 4.0)
    assert np.all(packed[:, :, 2] == 0.0)
    assert np.all(packed[:, :, 3:5] == 9.0)
    unpacked = geometry.unpack_module_data(packed)
    assert np.array_equal(unpacked[0], module_data[0])
    assert np.array_equal(unpacked[1], module_data[1])
    assert not np.allclose(
        geometry.pixel_centers_mm[0, 0, 0],
        geometry.pixel_centers_mm[1, 0, 0])
    assert np.allclose(
        geometry.pixel_centers_mm[:, 0, 1, 1],
        [1.0, 10.0 + np.cos(0.1)])


def test_static_module_pose_broadcasts_to_source_views():
    source = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    module = DetectorModule(
        [[[0.0, 0.0, 0.0]]],
        _pose([-20.0, 0.0, 0.0]),
    )

    geometry = DetectorSurfaceGeometry.from_modules(source, [module])

    assert np.array_equal(
        geometry.pixel_centers_mm[:, 0, 0],
        [[-20.0, 0.0, 0.0], [-20.0, 0.0, 0.0]])


@pytest.mark.parametrize(
    "source, centers, mask, error",
    [
        ([1.0, 2.0, 3.0], np.zeros((1, 1, 3)), None, "source_positions_mm"),
        ([[1.0, 2.0, np.inf]], np.zeros((1, 1, 1, 3)), None, "finite"),
        ([[1.0, 2.0, 3.0]], np.zeros((1, 1, 1, 3)), [[True, False]], "valid_mask"),
    ],
)
def test_explicit_geometry_rejects_invalid_shapes_and_values(source, centers, mask, error):
    with pytest.raises(ValueError, match=error):
        DetectorSurfaceGeometry(source, centers, mask)


def test_modules_reject_overlap_and_invalid_rigid_transforms():
    pixel = np.zeros((1, 1, 3))
    identity = np.eye(4)
    with pytest.raises(ValueError, match="overlap"):
        DetectorSurfaceGeometry.from_modules(
            [[10.0, 0.0, 0.0]],
            [DetectorModule(pixel, identity), DetectorModule(pixel, identity)],
        )

    shear = np.eye(4)
    shear[0, 1] = 0.2
    with pytest.raises(ValueError, match="rigid rotations"):
        DetectorModule(pixel, shear)


def test_module_data_shape_must_match_geometry():
    module = DetectorModule(np.zeros((2, 3, 3)), np.eye(4))
    geometry = DetectorSurfaceGeometry.from_modules([[10.0, 0.0, 0.0]], [module])
    with pytest.raises(ValueError, match="module data must have shape"):
        geometry.pack_module_data([np.zeros((1, 2, 2))])
