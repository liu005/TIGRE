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
from tigre.utilities.detector_geometry_gui import (
    DETECTOR_FIELD_HELP,
    DetectorGeometryBuilder,
    MAX_GUI_VIEWS,
    MAX_MODULE_DIMENSION,
    MAX_MODULES,
    MAX_TOTAL_PIXELS,
    build_modular_surface,
    check_tigre_compatibility,
    validate_gui_size,
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


def test_gui_builder_supports_fixed_detector_and_linearly_moving_source():
    geometry = build_modular_surface(
        "Piecewise linear",
        n_views=5,
        source_radius_mm=1000.0,
        source_z_mm=10.0,
        module_shape=(2, 2),
        pixel_spacing_mm=(1.0, 2.0),
        detector_radius_mm=600.0,
        column_gap=1,
        n_modules=2,
        angular_span_deg=90.0,
        transverse_centers_mm=[-10.0, 10.0],
        normal_offsets_mm=[0.0, 0.0],
        panel_angles_deg=[45.0, -45.0],
        z_offsets_mm=[0.0, 0.0],
        source_motion="Fixed detector, moving source",
        source_start_mm=[700.0, -250.0, 5.0],
        source_end_mm=[800.0, 100.0, 30.0],
    )

    assert np.allclose(
        geometry.source_positions_mm[[0, -1]],
        [[700.0, -250.0, 5.0], [800.0, 100.0, 30.0]],
    )
    assert np.allclose(
        geometry.source_positions_mm[1:-1],
        [[725.0, -162.5, 11.25], [750.0, -75.0, 17.5], [775.0, 12.5, 23.75]],
    )
    assert np.array_equal(
        geometry.pixel_centers_mm,
        np.broadcast_to(geometry.pixel_centers_mm[0], geometry.pixel_centers_mm.shape),
    )
    assert np.array_equal(
        geometry.valid_mask,
        np.broadcast_to(geometry.valid_mask[0], geometry.valid_mask.shape),
    )

    builder = object.__new__(DetectorGeometryBuilder)
    builder.surface = geometry
    builder._panel_count = 2
    parameters = {
        "source_motion": "Fixed detector, moving source",
        "layout": "Piecewise linear",
        "module_shape": (2, 2),
        "pixel_spacing_mm": (1.0, 2.0),
        "column_gap": 1,
        "panel_angles_deg": [45.0, -45.0],
    }
    assert np.allclose(
        builder._panel_vertices(parameters, 0),
        builder._panel_vertices(parameters, 4),
    )

def test_fixed_detector_accepts_arbitrary_source_start_and_ignores_orbit_fields():
    geometry = build_modular_surface(
        "Piecewise linear",
        n_views=3,
        source_radius_mm=-1.0,
        source_z_mm=np.nan,
        module_shape=(1, 2),
        pixel_spacing_mm=1.0,
        detector_radius_mm=600.0,
        column_gap=0,
        n_modules=1,
        angular_span_deg=90.0,
        transverse_centers_mm=[0.0],
        normal_offsets_mm=[0.0],
        panel_angles_deg=[0.0],
        z_offsets_mm=[0.0],
        source_motion="Fixed detector, moving source",
        source_start_mm=[0.0, 1000.0, 5.0],
        source_end_mm=[0.0, 0.0, 25.0],
    )

    assert np.allclose(
        geometry.source_positions_mm,
        [[0.0, 1000.0, 5.0], [0.0, 500.0, 15.0], [0.0, 0.0, 25.0]],
    )
    assert np.array_equal(
        geometry.pixel_centers_mm,
        np.broadcast_to(geometry.pixel_centers_mm[0], geometry.pixel_centers_mm.shape),
    )

    builder = object.__new__(DetectorGeometryBuilder)
    builder.surface = geometry
    builder._panel_count = 1
    parameters = {
        "source_motion": "Fixed detector, moving source",
        "layout": "Piecewise linear",
        "module_shape": (1, 2),
        "pixel_spacing_mm": (1.0, 1.0),
        "column_gap": 0,
        "panel_angles_deg": [0.0],
    }
    vertices = builder._panel_vertices(parameters, 0)
    assert np.allclose(
        vertices[0][1] - vertices[0][0], [-2.0, 0.0, 0.0], atol=1e-4)
    assert np.allclose(
        vertices, builder._panel_vertices(parameters, len(geometry.source_positions_mm) - 1))
    compatibility = check_tigre_compatibility(geometry)
    assert compatibility["iterative_supported"]
    assert not compatibility["fdk_supported"]
    assert "circular source orbit" in compatibility["fdk_message"]


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
    validate_gui_size(gct_views, 1, 64, 53, 0)
    assert gct_views * gct_dense_columns == 19_337_792
    assert gct_views <= MAX_GUI_VIEWS
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


def test_gui_panel_vertices_rotate_with_source_in_rotating_mode():
    builder = object.__new__(DetectorGeometryBuilder)
    builder.surface = build_modular_surface(
        "Piecewise linear",
        n_views=4,
        source_radius_mm=1000.0,
        source_z_mm=0.0,
        module_shape=(1, 2),
        pixel_spacing_mm=1.0,
        detector_radius_mm=600.0,
        column_gap=0,
        n_modules=1,
        angular_span_deg=90.0,
        transverse_centers_mm=[0.0],
        normal_offsets_mm=[0.0],
        panel_angles_deg=[0.0],
        z_offsets_mm=[0.0],
    )
    builder._panel_count = 1
    parameters = {
        "source_motion": "Rotating source + detector",
        "layout": "Piecewise linear",
        "module_shape": (1, 2),
        "pixel_spacing_mm": (1.0, 1.0),
        "column_gap": 0,
        "panel_angles_deg": [0.0],
    }

    assert not np.allclose(
        builder._panel_vertices(parameters, 0),
        builder._panel_vertices(parameters, 1),
    )


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
        validate_gui_size(MAX_GUI_VIEWS + 1, 1, 1, 1, 0)
    with pytest.raises(ValueError, match="dimension"):
        build_modular_surface(
            "Cylindrical", 1, 1000.0, 0.0, (MAX_MODULE_DIMENSION + 1, 1),
            1.0, 600.0, 0, 1, 90.0, [0.0], [0.0], [0.0], [0.0])
    with pytest.raises(ValueError, match="modules"):
        build_modular_surface(
            "Cylindrical", 1, 1000.0, 0.0, (1, 1), 1.0, 600.0, 0,
            MAX_MODULES + 1, 90.0, [0.0], [0.0], [0.0], [0.0])
    with pytest.raises(ValueError, match="detector samples"):
        validate_gui_size(MAX_GUI_VIEWS, 3334, 1, 1, 0)


def test_gui_parameter_reader_ignores_fields_unused_by_selected_modes():
    builder = object.__new__(DetectorGeometryBuilder)
    values = {
        "n_views": "5",
        "orbit_start_deg": "not used",
        "scan_span_deg": "not used",
        "source_radius_mm": "not used",
        "source_z_mm": "not used",
        "source_start_mm": "700, -250, 5",
        "source_end_mm": "800, 100, 30",
        "rows": "2",
        "columns": "2",
        "row_pitch_mm": "1",
        "column_pitch_mm": "2",
        "detector_radius_mm": "600",
        "column_gap": "1",
        "n_modules": "not used",
        "angular_span_deg": "not used",
        "transverse_centers_mm": "-10, 10",
        "normal_offsets_mm": "0, 0",
        "panel_angles_deg": "45, -45",
        "z_offsets_mm": "0, 0",
    }
    builder._fields = {key: Mock(get=Mock(return_value=value)) for key, value in values.items()}
    builder._layout = Mock(get=Mock(return_value="Piecewise linear"))
    builder._detector_stationary = True

    parameters = builder._read_parameters()
    geometry = build_modular_surface(**parameters)

    assert np.allclose(geometry.source_positions_mm[[0, -1]],
                       [[700, -250, 5], [800, 100, 30]])
    assert geometry.pixel_centers_mm.shape == (5, 2, 5, 3)

    values["source_start_mm"] = "not used"
    values["source_end_mm"] = "not used"
    builder._detector_stationary = False
    builder._layout = Mock(get=Mock(return_value="Cylindrical"))
    values["source_radius_mm"] = "1000"
    values["source_z_mm"] = "0"
    values["orbit_start_deg"] = "156"
    values["scan_span_deg"] = "228"
    values["n_modules"] = "3"
    values["angular_span_deg"] = "90"
    builder._fields = {
        key: Mock(get=Mock(return_value=value)) for key, value in values.items()}
    parameters = builder._read_parameters()
    assert parameters["orbit_start_deg"] == 156.0
    assert parameters["scan_span_deg"] == 228.0
    assert build_modular_surface(**parameters).pixel_centers_mm.shape == (5, 2, 8, 3)


def test_gui_worker_surfaces_unexpected_compatibility_exceptions(monkeypatch):
    import tigre.utilities.detector_geometry_gui as gui

    def fail_check(_):
        raise RuntimeError("unexpected checker failure")

    monkeypatch.setattr(gui, "check_tigre_compatibility", fail_check)
    builder = object.__new__(DetectorGeometryBuilder)
    builder._compatibility_queue = queue.Queue()

    builder._run_compatibility_check(object())

    result = builder._compatibility_queue.get_nowait()
    assert result == {"error": "RuntimeError: unexpected checker failure"}


def test_failed_preview_clears_old_scene_and_disables_animation():
    builder = object.__new__(DetectorGeometryBuilder)
    builder.surface = None
    builder._panel_collection = object()
    builder._source_artist = object()
    builder._axes = Mock()
    builder._canvas = Mock()
    builder._play_button = Mock()
    builder._frame_slider = Mock()
    builder._animation_playing = False
    builder.tk = type("TkConstants", (), {"DISABLED": "disabled"})
    builder._stop_animation = Mock()

    builder._clear_preview()

    builder._axes.clear.assert_called_once_with()
    builder._canvas.draw_idle.assert_called_once_with()
    assert builder._panel_collection is None
    assert builder._source_artist is None
    builder._play_button.configure.assert_called_once_with(state="disabled")
    builder._frame_slider.configure.assert_called_once_with(state="disabled", to=0)
    builder._frame_slider.set.assert_called_once_with(0)

    builder.toggle_animation()
    builder._schedule_next_frame()


def test_gui_has_help_text_for_every_input_label():
    assert set(DETECTOR_FIELD_HELP) == {
        "Source trajectory",
        "Voxel counts",
        "Voxel size (z, y, x mm)",
        "Full size (mm)",
        "Center offset (mm)",
        "Start position (z, y, x mm)",
        "End position (z, y, x mm)",
        "Move between start and end positions",
        "Use timing to derive orbit and z motion",
        "Detector readout period (ms)",
        "Source firing rate (Hz)",
        "Rotation period (s/turn)",
        "Volume z speed (mm/s)",
        "Playback speed (x real time)",
        "Views / source pulses",
        "COR (mm)",
        "Panel roll / x (degrees)",
        "Panel pitch / y (degrees)",
        "Detector motion",
        "Layout",
        "Views",
        "Orbit start (degrees)",
        "Scan span (degrees)",
        "Source radius (mm)",
        "Source height (mm)",
        "Source start (x, y, z mm)",
        "Source end (x, y, z mm)",
        "Module rows",
        "Module columns",
        "Row pitch (mm)",
        "Column pitch (mm)",
        "Detector radius (mm)",
        "Gap columns",
        "Modules (cylindrical)",
        "Panels (from lists)",
        "Arc span (degrees)",
        "Panel transverse offsets (mm)",
        "Panel normal offsets (mm)",
        "Panel yaw / z (degrees)",
        "Panel z offsets (mm)",
    }
    assert all(description.strip() for description in DETECTOR_FIELD_HELP.values())


def test_layout_selection_updates_module_count_label_and_entry_state():
    builder = object.__new__(DetectorGeometryBuilder)
    builder._layout = Mock()
    builder._update_field_states = Mock()
    builder._fields = {"n_modules": Mock(), "transverse_centers_mm": Mock(get=Mock(return_value="-1, 0, 1"))}
    builder._panel_count_display = Mock()
    builder._module_count_label = Mock()
    builder._module_count_entry = Mock()
    builder.tk = type("TkConstants", (), {"NORMAL": "normal", "DISABLED": "disabled"})

    builder._layout.get.return_value = "Piecewise linear"
    builder._update_module_count_field()
    builder._module_count_label.configure.assert_called_with(text="Panels (from lists)")
    builder._module_count_entry.configure.assert_any_call(state="disabled")
    builder._panel_count_display.set.assert_called_with("3")
    builder._fields["n_modules"].set.assert_not_called()

    builder._layout.get.return_value = "Cylindrical"
    builder._update_module_count_field()
    builder._module_count_label.configure.assert_called_with(text="Modules (cylindrical)")
    builder._module_count_entry.configure.assert_called_with(state="normal")


def test_detector_motion_button_changes_controls_without_applying_them():
    builder = object.__new__(DetectorGeometryBuilder)
    builder._detector_stationary = False
    builder._update_field_states = Mock()
    builder._detector_motion_button = Mock()
    builder._help_text = Mock()
    builder.tk = type(
        "TkConstants", (),
        {"NORMAL": "normal", "DISABLED": "disabled", "END": "end"})
    builder.surface = object()
    previews = []
    builder.update_preview = lambda: previews.append(True)

    builder.toggle_detector_motion()

    assert builder._detector_stationary is True
    builder._detector_motion_button.configure.assert_called_once_with(text="Stationary")
    builder._help_text.insert.assert_any_call("end", "\n\nSelected: Stationary")
    assert previews == []

    builder.toggle_detector_motion()

    assert builder._detector_stationary == "arc"
    builder.toggle_detector_motion()
    assert builder._detector_stationary is False
    assert builder._detector_motion_button.configure.call_args_list[-1] == (
        (), {"text": "Moving with source"})
    assert previews == []


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



def test_gui_preview_full_tilt_and_cor_matches_pixel_axes():
    parameters = dict(layout="Piecewise linear", n_views=4,
        source_radius_mm=1000, source_z_mm=0, module_shape=(3, 4),
        pixel_spacing_mm=(2, 3), detector_radius_mm=536, column_gap=0,
        n_modules=1, angular_span_deg=0, transverse_centers_mm=[0],
        normal_offsets_mm=[0], panel_angles_deg=[23], z_offsets_mm=[0],
        panel_x_angles_deg=[13], panel_y_angles_deg=[-17], cor_mm=70,
        source_motion="Rotating source + detector", orbit_start_deg=15,
        scan_span_deg=360)
    builder = object.__new__(DetectorGeometryBuilder)
    builder.surface = build_modular_surface(**parameters)
    builder._panel_count = 1
    for view in range(4):
        vertices = np.asarray(builder._panel_vertices(parameters, view))[0]
        pixels = builder.surface.pixel_centers_mm[view]
        np.testing.assert_allclose(vertices.mean(axis=0), pixels.mean(axis=(0, 1)), atol=1e-4)
        np.testing.assert_allclose(vertices[1] - vertices[0],
                                   4 * (pixels[0, 1] - pixels[0, 0]), atol=2e-4)
        np.testing.assert_allclose(vertices[3] - vertices[0],
                                   3 * (pixels[1, 0] - pixels[0, 0]), atol=2e-4)


@pytest.mark.parametrize("layout", ["Cylindrical", "Piecewise linear"])
@pytest.mark.parametrize("motion", [False, True, "arc"])
def test_gui_disabled_fields_and_list_buttons(layout, motion):
    builder = object.__new__(DetectorGeometryBuilder)
    builder.tk = type("TkConstants", (), {"NORMAL": "normal", "DISABLED": "disabled"})
    builder._layout = Mock(get=Mock(return_value=layout))
    builder._detector_stationary = motion
    names = ["cor_mm", "n_modules", "angular_span_deg", "transverse_centers_mm",
        "normal_offsets_mm", "panel_angles_deg", "z_offsets_mm", "source_start_mm",
        "source_end_mm", "source_radius_mm", "source_z_mm", "orbit_start_deg",
        "scan_span_deg", "panel_x_angles_deg", "panel_y_angles_deg"]
    builder._entries = {name: Mock() for name in names}
    builder._list_buttons = {name: Mock() for name in names if name.startswith("panel_") or name.endswith("offsets_mm")}
    builder._update_field_states()
    enabled = {"cor_mm": True, "n_modules": layout == "Cylindrical",
        "angular_span_deg": layout == "Cylindrical", "panel_x_angles_deg": True,
        "panel_y_angles_deg": True}
    for name in names:
        if name not in enabled:
            enabled[name] = (motion is True if name in ("source_start_mm", "source_end_mm")
                else motion is not True if name in ("source_radius_mm", "source_z_mm", "orbit_start_deg", "scan_span_deg")
                else layout == "Piecewise linear")
        builder._entries[name].configure.assert_called_with(state="normal" if enabled[name] else "disabled")
        if name in builder._list_buttons:
            builder._list_buttons[name].configure.assert_called_with(state="normal" if enabled[name] else "disabled")


def test_gui_panel_list_expands_shared_tilt_and_validates_entries():
    from tigre.utilities.detector_geometry_gui import _panel_list_values
    np.testing.assert_array_equal(_panel_list_values("5", 3, True), [5, 5, 5])
    np.testing.assert_array_equal(_panel_list_values("5, 0, -5", 3, False), [5, 0, -5])
    for text, shared in (("5", False), ("1, 2", True), ("nan, 0, 1", False)):
        with pytest.raises(ValueError):
            _panel_list_values(text, 3, shared)
