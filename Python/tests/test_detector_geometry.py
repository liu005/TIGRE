from unittest.mock import Mock
import queue

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tigre.utilities.detector_geometry import (
    DetectorModule,
    DetectorSurfaceGeometry,
    cylindrical_modular_detector_surface,
    piecewise_linear_modular_detector_surface,
)
from tigre.utilities.detector_geometry_builder import MAX_RECIPE_VIEWS, MAX_MODULE_DIMENSION, MAX_MODULES, MAX_TOTAL_PIXELS, build_modular_surface, check_tigre_compatibility, validate_surface_size


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


def test_piecewise_linear_modular_surface_builds_rotated_panels_and_gaps():
    source = np.array([[100.0, 0.0, 5.0], [0.0, 100.0, 5.0]])
    geometry = piecewise_linear_modular_detector_surface(
        source,
        module_shape=(2, 2),
        pixel_spacing_mm=(2.0, 3.0),
        detector_radius_mm=500.0,
        module_centers_mm=[[0.0, -10.0, 1.0], [0.0, 10.0, -1.0]],
        panel_angles_rad=[np.pi / 4.0, -np.pi / 4.0],
        column_gap=1,
        detector_center_z_mm=2.0,
    )

    assert geometry.pixel_centers_mm.shape == (2, 2, 5, 3)
    assert np.array_equal(
        geometry.valid_mask[0, 0], [True, True, False, True, True])
    assert np.allclose(geometry.pixel_centers_mm[0, 0, :2], [
        [-500.0 + 1.5 / np.sqrt(2), -10.0 - 1.5 / np.sqrt(2), 2.0],
        [-500.0 - 1.5 / np.sqrt(2), -10.0 + 1.5 / np.sqrt(2), 2.0],
    ])
    assert np.allclose(geometry.pixel_centers_mm[1, 0, 0], [
        10.0 + 1.5 / np.sqrt(2),
        -500.0 + 1.5 / np.sqrt(2),
        2.0,
    ])


def test_cylindrical_modular_surface_places_tangent_panels_on_arc():
    source = np.array([[100.0, 0.0, 0.0]])
    geometry = cylindrical_modular_detector_surface(
        source,
        module_shape=(1, 2),
        pixel_spacing_mm=4.0,
        detector_radius_mm=100.0,
        n_modules=3,
        angular_span_rad=np.pi / 2.0,
    )

    assert geometry.pixel_centers_mm.shape == (1, 1, 6, 3)
    assert np.array_equal(
        geometry.valid_mask[0, 0], [True, True, True, True, True, True])
    centers = geometry.pixel_centers_mm[0, 0].reshape(3, 2, 3).mean(axis=1)
    expected_angles = np.array([np.pi / 4.0, 0.0, -np.pi / 4.0])
    expected_centers = np.column_stack((
        -100.0 * np.cos(expected_angles),
        -100.0 * np.sin(expected_angles),
        np.zeros(3),
    ))
    assert np.allclose(centers, expected_centers)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"module_shape": (0, 2)}, "module_shape"),
        ({"pixel_spacing_mm": (1.0, 0.0)}, "pixel_spacing_mm"),
        ({"module_centers_mm": [[0.0, 0.0]]}, "module_centers_mm"),
        ({"panel_angles_rad": [np.nan]}, "panel_angles_rad"),
        ({"column_gap": -1}, "column_gap"),
    ],
)
def test_piecewise_modular_surface_rejects_invalid_layout(kwargs, error):
    arguments = {
        "source_positions_mm": [[100.0, 0.0, 0.0]],
        "module_shape": (1, 2),
        "pixel_spacing_mm": 1.0,
        "detector_radius_mm": 100.0,
        "module_centers_mm": [[0.0, 0.0, 0.0]],
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=error):
        piecewise_linear_modular_detector_surface(**arguments)


def test_gui_builder_creates_cylindrical_and_piecewise_layouts():
    common = {
        "n_views": 4,
        "source_radius_mm": 1000.0,
        "source_z_mm": 0.0,
        "orbit_start_deg": 0.0,
        "scan_span_deg": 360.0,
        "module_shape": (2, 3),
        "pixel_spacing_mm": (1.0, 2.0),
        "detector_radius_mm": 600.0,
        "column_gap": 1,
        "n_modules": 3,
        "angular_span_deg": 90.0,
        "transverse_centers_mm": [-10.0, 0.0, 10.0],
        "normal_offsets_mm": [0.0, 10.0, 0.0],
        "panel_angles_deg": [45.0, 0.0, -45.0],
        "z_offsets_mm": [0.0, 0.0, 0.0],
    }
    cylinder = build_modular_surface("Cylindrical", **common)
    u_shape = build_modular_surface("Piecewise linear", **common)

    assert cylinder.pixel_centers_mm.shape == (4, 2, 11, 3)
    assert u_shape.pixel_centers_mm.shape == (4, 2, 11, 3)
    assert np.array_equal(cylinder.valid_mask[0, 0], [
        True, True, True, False, True, True, True, False, True, True, True])
    assert np.allclose(
        u_shape.source_positions_mm[:, 2], np.zeros(4))





def test_rotating_source_supports_partial_scan_angles_and_full_scan_defaults():
    parameters = {
        "layout": "Cylindrical",
        "n_views": 3,
        "source_radius_mm": 6100.0,
        "source_z_mm": 0.0,
        "module_shape": (1, 1),
        "pixel_spacing_mm": 1.0,
        "detector_radius_mm": 4500.0,
        "column_gap": 0,
        "n_modules": 1,
        "angular_span_deg": 90.0,
        "transverse_centers_mm": [0.0],
        "normal_offsets_mm": [0.0],
        "panel_angles_deg": [0.0],
        "z_offsets_mm": [0.0],
    }
    partial = build_modular_surface(
        **parameters, orbit_start_deg=156.0, scan_span_deg=228.0)
    assert np.allclose(
        partial.source_positions_mm,
        6100.0 * np.column_stack((
            np.cos(np.deg2rad([156.0, 270.0, 384.0])),
            np.sin(np.deg2rad([156.0, 270.0, 384.0])),
            np.zeros(3),
        )),
        atol=1e-3,
    )

    full = build_modular_surface(**parameters)
    assert np.allclose(
        full.source_positions_mm,
        6100.0 * np.column_stack((
            np.cos(np.deg2rad([0.0, 120.0, 240.0])),
            np.sin(np.deg2rad([0.0, 120.0, 240.0])),
            np.zeros(3),
        )),
        atol=1e-3,
    )


@pytest.mark.parametrize("scan_span", [0.0, 360.1, np.nan])
def test_rotating_source_rejects_invalid_scan_span(scan_span):
    with pytest.raises(ValueError, match="Scan span"):
        build_modular_surface(
            "Cylindrical", 3, 1000.0, 0.0, (1, 1), 1.0, 600.0, 0, 1,
            90.0, [0.0], [0.0], [0.0], [0.0], scan_span_deg=scan_span)


def test_gui_size_guard_accepts_full_gct_demo_geometry_dimensions():
    gct_views = 5701
    gct_dense_columns = 53 * 64
    validate_surface_size(gct_views, 1, 64, 53, 0)
    assert gct_views * gct_dense_columns == 19_337_792
    assert gct_views <= MAX_RECIPE_VIEWS
    assert gct_views * gct_dense_columns <= MAX_TOTAL_PIXELS


def test_compatibility_check_accepts_cylindrical_orbit_for_fdk():
    geometry = build_modular_surface(
        "Cylindrical",
        n_views=20,
        source_radius_mm=1000.0,
        source_z_mm=0.0,
        module_shape=(3, 4),
        pixel_spacing_mm=(1.0, 1.0),
        detector_radius_mm=600.0,
        column_gap=1,
        n_modules=5,
        angular_span_deg=100.0,
        transverse_centers_mm=[-10.0, 0.0, 10.0],
        normal_offsets_mm=[0.0, 0.0, 0.0],
        panel_angles_deg=[30.0, 0.0, -30.0],
        z_offsets_mm=[0.0, 0.0, 0.0],
    )

    compatibility = check_tigre_compatibility(geometry)

    assert compatibility["iterative_supported"]
    assert compatibility["fdk_supported"], compatibility["fdk_message"]




def test_gui_builder_rejects_unequal_piecewise_panel_lists():
    with pytest.raises(ValueError, match="equal lengths"):
        build_modular_surface(
            "Piecewise linear",
            n_views=1,
            source_radius_mm=1000.0,
            source_z_mm=0.0,
            module_shape=(1, 1),
            pixel_spacing_mm=1.0,
            detector_radius_mm=600.0,
            column_gap=0,
            n_modules=3,
            angular_span_deg=90.0,
            transverse_centers_mm=[-10.0, 10.0],
            normal_offsets_mm=[0.0],
            panel_angles_deg=[45.0, -45.0],
            z_offsets_mm=[0.0, 0.0],
        )

def test_gui_size_guard_rejects_excessive_views_modules_and_pixel_counts():
    with pytest.raises(ValueError, match="views"):
        validate_surface_size(MAX_RECIPE_VIEWS + 1, 1, 1, 1, 0)
    with pytest.raises(ValueError, match="dimension"):
        build_modular_surface(
            "Cylindrical", 1, 1000.0, 0.0, (MAX_MODULE_DIMENSION + 1, 1),
            1.0, 600.0, 0, 1, 90.0, [0.0], [0.0], [0.0], [0.0])
    with pytest.raises(ValueError, match="modules"):
        build_modular_surface(
            "Cylindrical", 1, 1000.0, 0.0, (1, 1), 1.0, 600.0, 0,
            MAX_MODULES + 1, 90.0, [0.0], [0.0], [0.0], [0.0])
    with pytest.raises(ValueError, match="detector samples"):
        validate_surface_size(MAX_RECIPE_VIEWS, 3334, 1, 1, 0)














@pytest.mark.parametrize("shape", [(3, 4), (1, 1)])
def test_gui_full_panel_rotation_and_cor_match_tigre_geometry(shape):
    import tigre
    from tigre.utilities.detector_geometry import surface_from_geometry

    geo = tigre.geometry_default(high_resolution=False)
    geo.nDetector = np.array(shape)
    geo.dDetector = np.array([2.0, 3.0])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.rotDetector = np.deg2rad([13.0, -17.0, 23.0])
    geo.COR = 7.0
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    expected = surface_from_geometry(geo, angles)
    actual = build_modular_surface(
        "Piecewise linear", 4, 1000.0, 0.0, shape, (2.0, 3.0),
        536.0, 0, 1, 0.0, [0], [0], [23], [0],
        panel_x_angles_deg=[13], panel_y_angles_deg=[-17], cor_mm=7)
    np.testing.assert_allclose(actual.pixel_centers_mm, expected.pixel_centers_mm, atol=1e-4)
    np.testing.assert_allclose(actual.source_positions_mm, expected.source_positions_mm, atol=1e-4)


def test_gui_rejects_invalid_panel_tilts_and_cor():
    args = ("Piecewise linear", 4, 1000, 0, (2, 3), (1, 1),
            536, 0, 1, 0, [0], [0], [0], [0])
    for kwargs in ({"panel_x_angles_deg": [1, 2]},
                   {"panel_y_angles_deg": [np.nan]}, {"cor_mm": np.inf},
                   {"cor_mm": 1, "source_motion": "Fixed detector, moving source"}):
        with pytest.raises(ValueError):
            build_modular_surface(*args, **kwargs)
