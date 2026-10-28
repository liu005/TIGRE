"""MATLAB interchange preserves physical axes and singleton dimensions."""
import numpy as np
import pytest
from scipy.io import loadmat, savemat
from tigre.utilities.geometry_matlab_bridge import to_mat, from_mat
from tigre.utilities.geometry_io import save_geometry_yaml, load_geometry_file
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry
from test_geometry_yaml import settings


def test_builder_mat_yaml_round_trip(tmp_path):
    yaml = tmp_path / 'geometry.yaml'
    mat = tmp_path / 'geometry.mat'
    output = tmp_path / 'output.yaml'
    save_geometry_yaml(yaml, settings=settings())
    to_mat(yaml, mat)
    document = loadmat(mat, simplify_cells=True)
    assert document['surface']['pixel_centers_mm'].shape == (3, 14, 3, 3)
    from_mat(mat, output)
    expected, actual = load_geometry_file(yaml), load_geometry_file(output)
    np.testing.assert_array_equal(actual.surface.pixel_centers_mm, expected.surface.pixel_centers_mm)
    assert actual.settings['panel_y_angles_deg'] == expected.settings['panel_y_angles_deg']


def test_single_panel_recipe_survives_matlab_scalar_squeezing(tmp_path):
    parameters = settings()
    for name in ('transverse_centers_mm', 'normal_offsets_mm', 'z_offsets_mm',
                 'panel_angles_deg', 'panel_x_angles_deg', 'panel_y_angles_deg'):
        parameters[name] = [0]
    yaml, mat, output = (tmp_path / name for name in ('one.yaml', 'one.mat', 'out.yaml'))
    save_geometry_yaml(yaml, settings=parameters)
    to_mat(yaml, mat)
    from_mat(mat, output)
    np.testing.assert_allclose(load_geometry_file(output).surface.pixel_centers_mm,
                               load_geometry_file(yaml).surface.pixel_centers_mm)


def test_moving_volume_and_voxel_counts_survive_matlab_bridge(tmp_path):
    from types import SimpleNamespace
    geo = SimpleNamespace(nVoxel=np.array([2, 4, 8]), sVoxel=np.array([20, 40, 80]),
        offOrigin=np.array([[0, 1, 2], [10, 2, 3], [20, 3, 4]]), motion_mode='endpoints')
    yaml, mat, output = (tmp_path / name for name in ('moving.yaml', 'moving.mat', 'output.yaml'))
    save_geometry_yaml(yaml, settings=settings(), geometry=geo)
    to_mat(yaml, mat)
    raw = loadmat(mat, simplify_cells=True)
    np.testing.assert_array_equal(raw['geo']['nVoxel'], [8, 4, 2])
    np.testing.assert_array_equal(raw['geo']['offOrigin'], geo.offOrigin[:, ::-1].T)
    from_mat(mat, output)
    actual = load_geometry_file(output)
    np.testing.assert_array_equal(actual.geometry.offOrigin, geo.offOrigin)
    np.testing.assert_array_equal(actual.geometry.nVoxel, geo.nVoxel)
    assert actual.geometry.motion_mode == 'endpoints'


@pytest.mark.parametrize('shape', [(1, 4, 1), (1, 1, 1), (3, 4, 2)])
def test_exact_surface_preserves_singletons_and_volume_axes(shape, tmp_path):
    from types import SimpleNamespace
    rows, columns, views = shape
    source = np.tile([100, 0, 0], (views, 1))
    pixels = np.arange(views*rows*columns*3).reshape(views, rows, columns, 3)
    surface = DetectorSurfaceGeometry(source, pixels)
    geo = SimpleNamespace(sVoxel=np.array([10, 20, 30]), offOrigin=np.array([1, 2, 3]))
    yaml, mat, output = (tmp_path / name for name in ('geometry.yaml', 'geometry.mat', 'output.yaml'))
    save_geometry_yaml(yaml, surface=surface, geometry=geo)
    to_mat(yaml, mat)
    raw = loadmat(mat, simplify_cells=True)
    np.testing.assert_array_equal(raw['geo']['sVoxel'], [30, 20, 10])
    from_mat(mat, output)
    actual = load_geometry_file(output)
    np.testing.assert_array_equal(actual.surface.pixel_centers_mm, surface.pixel_centers_mm)
    np.testing.assert_array_equal(actual.geometry.offOrigin, geo.offOrigin)
