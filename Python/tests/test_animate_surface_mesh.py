"""CPU tests of curved and tiled detector rendering, without GPU fixtures."""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from types import SimpleNamespace
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry
from tigre.utilities.visualization.animate_geometry import animate_geometry, _surface_mesh


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def curved_surface():
    angle = np.linspace(-0.5, 0.5, 7)
    pixels = np.zeros((2, 4, 7, 3))
    pixels[..., 0] = -100 * np.cos(angle)
    pixels[..., 1] = 100 * np.sin(angle)
    pixels[..., 2] = np.arange(4)[None, :, None] * 3
    pixels[1] += [10, -5, 8]
    return DetectorSurfaceGeometry([[200, 0, 0], [100, 160, 30]], pixels)


def volume():
    return SimpleNamespace(sVoxel=np.array([30, 40, 50]), offOrigin=np.zeros(3))


def test_curved_grid_uses_nonplanar_surface_triangles():
    surface = curved_surface()
    faces, lines = _surface_mesh(surface, 0)
    assert len(faces) == 36
    assert not lines
    assert np.ptp(np.asarray(faces)[..., 0]) > 1
    ax = animate_geometry(volume(), surface=surface, animate=False, view=(70, 20))
    ax.figure.canvas.draw()
    assert (ax.elev, ax.azim) == (70, 20)


def test_module_row_and_column_bounds_are_respected_without_gaps():
    surface = curved_surface()
    surface = DetectorSurfaceGeometry(surface.source_positions_mm, surface.pixel_centers_mm,
        module_slices=[(slice(0, 2), slice(0, 7)), (slice(2, 4), slice(0, 7))])
    faces, _ = _surface_mesh(surface, 0)
    assert len(faces) == 24  # No triangles connect the two row-stacked tiles.
    for triangle in faces:
        z = np.asarray(triangle)[:, 2]
        assert z.max() <= 3 or z.min() >= 6


def test_masks_do_not_bridge_holes_even_with_mesh_downsampling():
    surface = curved_surface()
    mask = surface.valid_mask.copy()
    mask[0, :, 3] = False
    mask[1] = False  # A view with no active pixels is still drawable.
    surface = DetectorSurfaceGeometry(surface.source_positions_mm, surface.pixel_centers_mm, mask)
    faces, _ = _surface_mesh(surface, 0)
    assert len(faces) == 24
    explicit = DetectorSurfaceGeometry(surface.source_positions_mm, surface.pixel_centers_mm, mask,
        module_slices=[(slice(0, 4), slice(0, 7))])
    faces_coarse, _ = _surface_mesh(explicit, 0, max_rows=2, max_columns=2)
    assert not faces_coarse
    ani = animate_geometry(volume(), surface=surface)
    ani._fig.canvas.draw()
    ani._func(1)
    ani._fig.canvas.draw()
    assert "0 valid pixels" in ani._fig.axes[0].get_title()
    ani._draw_was_started = True


@pytest.mark.parametrize("shape", [(1, 7), (4, 1), (1, 1)])
def test_single_row_column_and_pixel_surfaces_draw(shape):
    base = curved_surface()
    pixels = base.pixel_centers_mm[:, :shape[0], :shape[1]]
    surface = DetectorSurfaceGeometry(base.source_positions_mm, pixels)
    faces, segments = _surface_mesh(surface, 0)
    assert not faces
    assert len(segments) == max(shape) - 1
    ax = animate_geometry(volume(), surface=surface, animate=False)
    ax.figure.canvas.draw()


def test_animation_updates_per_view_mesh_coordinates():
    surface = curved_surface()
    ani = animate_geometry(volume(), surface=surface)
    artists = ani._func(1)
    assert isinstance(artists[2], Poly3DCollection)
    assert isinstance(artists[3], Line3DCollection)
    expected_faces, _ = _surface_mesh(surface, 1)
    np.testing.assert_allclose(artists[2]._vec[:3].T.reshape(-1, 3, 3), expected_faces)
    np.testing.assert_allclose(artists[4]._offsets3d, surface.pixel_centers_mm[1].reshape(-1, 3).T)
    ani._fig.canvas.draw()
    ani._draw_was_started = True



def test_thin_surface_has_vertical_space_without_stretching_geometry():
    surface = curved_surface()
    original = surface.pixel_centers_mm.copy()
    ax = animate_geometry(volume(), surface=surface, animate=False)
    spans = np.array([np.ptp(ax.get_xlim3d()), np.ptp(ax.get_ylim3d()), np.ptp(ax.get_zlim3d())])
    assert spans[2] >= 0.5 * max(spans[:2]) - 1e-6
    aspect = ax.get_box_aspect()
    np.testing.assert_allclose(aspect / aspect[0], spans / spans[0])
    np.testing.assert_array_equal(surface.pixel_centers_mm, original)
