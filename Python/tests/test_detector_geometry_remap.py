import numpy as np
import pytest

from tigre.utilities.detector_geometry import (
    DetectorSurfaceGeometry,
    piecewise_linear_modular_detector_surface,
)
from tigre.utilities.detector_geometry_gui import _load_gui_projections
from tigre.utilities.detector_geometry_remap import remap_projections_for_fdk


def _piecewise_surface():
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    source = np.column_stack((
        1000.0 * np.cos(angles),
        1000.0 * np.sin(angles),
        np.zeros(angles.size),
    ))
    return piecewise_linear_modular_detector_surface(
        source,
        module_shape=(1, 8),
        pixel_spacing_mm=(1.0, 8.0),
        detector_radius_mm=500.0,
        module_centers_mm=[
            [0.0, -32.0, 0.0],
            [60.0, 32.0, 0.0],
        ],
        panel_angles_rad=[0.0, 0.0],
        column_gap=2,
    )


def test_gui_remap_archive_retains_volume_metadata(tmp_path):
    import queue
    from types import SimpleNamespace
    from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder
    from tigre.utilities.geometry_io import load_geometry_archive
    surface = _piecewise_surface()
    projections = tmp_path / "projections.npy"
    np.save(projections, np.zeros(surface.valid_mask.shape, np.float32))
    output = tmp_path / "remapped.npz"
    volume = SimpleNamespace(nVoxel=np.array([2, 3, 4]), sVoxel=np.array([20, 30, 40]),
                             offOrigin=np.array([1, 2, 3]))
    builder = object.__new__(DetectorGeometryBuilder)
    builder._remap_queue = queue.Queue()
    builder._run_projection_remap(projections, output, surface, volume)
    result = builder._remap_queue.get_nowait()
    assert "error" not in result, result
    geo, _ = load_geometry_archive(output)
    np.testing.assert_array_equal(geo.nVoxel, volume.nVoxel)
    np.testing.assert_array_equal(geo.offOrigin, volume.offOrigin)


def test_remap_projection_data_to_fdk_surface_without_filling_panel_gap():
    surface = _piecewise_surface()
    projections = np.broadcast_to(
        np.arange(surface.pixel_centers_mm.shape[2], dtype=np.float32)[None, None, :],
        surface.valid_mask.shape,
    ).copy()

    remapped, target, report = remap_projections_for_fdk(projections, surface)

    assert remapped.shape[0:2] == surface.valid_mask.shape[0:2]
    assert target.pixel_centers_mm.shape[:3] == remapped.shape
    assert target.valid_mask.shape == remapped.shape
    assert target.valid_mask.all(axis=(0, 1)).sum() < remapped.shape[-1]
    assert np.array_equal(
        target.valid_mask,
        np.broadcast_to(target.valid_mask[0], target.valid_mask.shape),
    )
    assert "interpolated" in report["warning"]
    from tigre.algorithms.fdk_surface import _analyse

    _analyse(target)


def test_remapping_rejects_non_circular_source_path():
    surface = _piecewise_surface()
    source = surface.source_positions_mm.copy()
    source[3, 0] += 4.0
    irregular = DetectorSurfaceGeometry(
        source, surface.pixel_centers_mm, surface.valid_mask)
    with pytest.raises(ValueError, match="circular source orbit"):
        remap_projections_for_fdk(
            np.zeros(irregular.valid_mask.shape, dtype=np.float32), irregular)


def test_remap_report_identifies_insufficient_short_scan_coverage():
    angles = np.linspace(0.0, np.pi, 8)
    source = np.column_stack((
        1000.0 * np.cos(angles), 1000.0 * np.sin(angles), np.zeros(8)))
    surface = piecewise_linear_modular_detector_surface(
        source, (1, 8), 30.0, 500.0,
        [[0.0, 0.0, 0.0]], [0.0], 0)

    _, _, report = remap_projections_for_fdk(
        np.zeros(surface.valid_mask.shape, dtype=np.float32), surface)

    assert report["short_scan_supported"] is False
    assert report["short_scan_deficiency_deg"] > 2.0


def test_remapping_rejects_invalid_data_shape_or_partial_column_mask():
    surface = _piecewise_surface()
    with pytest.raises(ValueError, match="projections must have shape"):
        remap_projections_for_fdk(np.zeros((2, 2, 2)), surface)

    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    source = np.column_stack((
        1000.0 * np.cos(angles), 1000.0 * np.sin(angles), np.zeros(8)))
    multirow = piecewise_linear_modular_detector_surface(
        source, (2, 8), 8.0, 500.0,
        [[0.0, 0.0, 0.0]], [0.0], 0)
    partial_mask = multirow.valid_mask.copy()
    partial_mask[:, 0, 0] = False
    partial = DetectorSurfaceGeometry(
        multirow.source_positions_mm, multirow.pixel_centers_mm, partial_mask)
    with pytest.raises(ValueError, match="whole-column"):
        remap_projections_for_fdk(
            np.zeros(partial.valid_mask.shape, dtype=np.float32), partial)


def test_remap_supports_two_dimensional_detector_coordinates():
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    source = np.column_stack((
        1000.0 * np.cos(angles), 1000.0 * np.sin(angles), np.zeros(8)))
    surface = piecewise_linear_modular_detector_surface(
        source, (2, 8), (5.0, 5.0), 500.0,
        [[0.0, 0.0, 0.0]], [np.deg2rad(20.0)], 0)
    src = surface.source_positions_mm[:, None, None, :]
    inward = -src[..., :2] / np.linalg.norm(src[..., :2], axis=-1, keepdims=True)
    delta = surface.pixel_centers_mm[..., :2] - src[..., :2]
    gamma = np.arctan2(
        inward[..., 0] * delta[..., 1] - inward[..., 1] * delta[..., 0],
        inward[..., 0] * delta[..., 0] + inward[..., 1] * delta[..., 1],
    )
    rho = np.linalg.norm(delta, axis=-1)
    eta = np.arctan2(
        surface.pixel_centers_mm[..., 2] - src[..., 2], rho)
    projections = (gamma + 2.0 * eta).astype(np.float32)

    remapped, target, _ = remap_projections_for_fdk(projections, surface)

    assert remapped.shape[1] == 2
    assert target.valid_mask.any()
    assert np.array_equal(
        target.valid_mask,
        np.broadcast_to(target.valid_mask[0], target.valid_mask.shape),
    )
    target_source = target.source_positions_mm[:, None, None, :].astype(np.float64)
    target_pixels = target.pixel_centers_mm.astype(np.float64)
    inward = -target_source[..., :2] / np.linalg.norm(
        target_source[..., :2], axis=-1, keepdims=True)
    delta = target_pixels[..., :2] - target_source[..., :2]
    target_gamma = np.arctan2(
        inward[..., 0] * delta[..., 1] - inward[..., 1] * delta[..., 0],
        inward[..., 0] * delta[..., 0] + inward[..., 1] * delta[..., 1],
    )
    target_eta = np.arctan2(
        target_pixels[..., 2] - target_source[..., 2],
        np.linalg.norm(delta, axis=-1),
    )
    assert np.allclose(
        remapped[target.valid_mask],
        (target_gamma + 2.0 * target_eta)[target.valid_mask],
        atol=2e-5,
    )
    from tigre.algorithms.fdk_surface import _analyse

    _analyse(target)


@pytest.mark.parametrize("suffix", [".npy", ".npz", ".raw"])
def test_gui_projection_loader_accepts_supported_array_formats(tmp_path, suffix):
    expected = (3, 1, 4)
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / f"projections{suffix}"
    if suffix == ".npy":
        np.save(path, values)
    elif suffix == ".npz":
        np.savez(path, projections=values)
    else:
        values.tofile(path)

    loaded = _load_gui_projections(path, expected)

    assert loaded.shape == expected
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded[:, 0], values)
