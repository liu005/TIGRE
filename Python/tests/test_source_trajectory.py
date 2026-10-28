"""Object-frame source trajectories and playback markers, without a Tk window."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pytest

from tigre.utilities.visualization.source_trajectory import SourceTrajectoryRenderer


@pytest.fixture
def axes():
    figure = plt.figure()
    yield figure.add_subplot(111, projection="3d")
    plt.close(figure)


def test_rotating_source_with_z_translation_is_object_frame_helix(axes):
    angles = np.linspace(0, 4 * np.pi, 41)
    source = np.column_stack((100 * np.cos(angles), 100 * np.sin(angles), np.zeros(41)))
    offsets = np.column_stack((np.linspace(-20, 20, 41), np.full(41, 2), np.full(41, 3)))
    renderer = SourceTrajectoryRenderer(axes, SimpleNamespace(source_positions_mm=source),
                                        SimpleNamespace(offOrigin=offsets))
    np.testing.assert_allclose(renderer.positions[:, :2], source[:, :2] - [3, 2])
    np.testing.assert_allclose(renderer.positions[:, 2], -offsets[:, 0])
    renderer.update(17)
    np.testing.assert_allclose(np.asarray(renderer.marker.get_data_3d()).ravel(), renderer.positions[17])
    assert "View 17" in axes.get_title()
    axes.figure.canvas.draw()


@pytest.mark.parametrize("offset", [[5, 6, 7], [[5, 6, 7]]])
def test_fixed_volume_and_xyz_conversion(axes, offset):
    source = np.array([[100, 0, 0], [0, 100, 0]])
    renderer = SourceTrajectoryRenderer(axes, SimpleNamespace(source_positions_mm=source),
                                        SimpleNamespace(offOrigin=offset))
    np.testing.assert_allclose(renderer.positions, source - [7, 6, 5])


def test_no_volume_uses_world_origin_and_handles_single_view(axes):
    source = np.array([[100, 2, 3]])
    renderer = SourceTrajectoryRenderer(axes, SimpleNamespace(source_positions_mm=source))
    np.testing.assert_array_equal(renderer.positions, source)
    renderer.update(0)
    axes.figure.canvas.draw()


@pytest.mark.parametrize("offset", [[[0, 0, 0]] * 3, [0, 0], [0, np.nan, 0]])
def test_invalid_offsets_rejected(axes, offset):
    with pytest.raises(ValueError, match="offOrigin"):
        SourceTrajectoryRenderer(axes, SimpleNamespace(source_positions_mm=np.zeros((2, 3))),
                                 SimpleNamespace(offOrigin=offset))
