"""Dense matrix oracles and real backend checks for the shared operator path."""
import copy
import importlib
from unittest.mock import patch

import numpy as np
import pytest
import tigre
from tigre import algorithms as algs
from tigre.utilities.projection_operator import (
    ProjectionOperator, SurfaceOperator, TigreOperator, prepare_projections,
)


class MatrixOperator:
    """Positive rectangular system with an unobserved ray and voxel."""
    volume_shape = (1, 2, 2)

    def __init__(self, matrix=None, mask=None):
        self.matrix = np.array([
            [1, 2, 0, 0], [0, 1, 3, 0], [2, 0, 1, 0],
            [1, 1, 1, 0], [0, 2, 1, 0], [3, 1, 0, 0],
        ], np.float32) if matrix is None else matrix.copy()
        self.projection_shape = (len(self.matrix) // 2, 1, 2)
        self.valid_mask = np.ones(self.projection_shape, bool) if mask is None else mask.copy()
        if matrix is None:
            self.valid_mask[1, 0, 0] = False
        self.matrix[~self.valid_mask.ravel()] = 0

    def forward(self, volume):
        return (self.matrix @ volume.ravel()).reshape(self.projection_shape)

    def adjoint(self, projections):
        clean = np.where(self.valid_mask, projections, 0)
        return (self.matrix.T @ clean.ravel()).reshape(self.volume_shape)

    def subset(self, indices):
        matrix = self.matrix.reshape(self.projection_shape + (4,))[indices].reshape(-1, 4)
        return MatrixOperator(matrix, self.valid_mask[indices])


def test_cgls_matches_independent_least_squares_and_preserves_inputs():
    op = MatrixOperator()
    assert isinstance(op, ProjectionOperator)
    truth = np.array([1, 2, 3, 0], np.float32).reshape(op.volume_shape)
    b = op.forward(truth)
    b[~op.valid_mask] = np.nan
    init = np.full(op.volume_shape, .25, np.float32)
    saved = init.copy()
    result, history = algs.cgls(b, operator=op, niter=3, init=init, return_residuals=True)
    # The fourth voxel is in the nullspace: it must retain the initial value.
    expected = np.linalg.lstsq(op.matrix, np.nan_to_num(b).ravel(), rcond=None)[0]
    expected[-1] = init.ravel()[-1]
    np.testing.assert_allclose(result.ravel(), expected, atol=2e-5)
    np.testing.assert_array_equal(init, saved)
    assert np.isnan(b[~op.valid_mask]).all()
    assert history[-1] < history[0]
    assert result.dtype == np.float32


@pytest.mark.parametrize('solver', [algs.sirt_operator, algs.os_sart_operator, algs.fast_os_sart_operator])
def test_normalization_and_interleaved_subsets_match_matrix_oracle(solver):
    op = MatrixOperator()
    b = np.arange(1, 7, dtype=np.float32).reshape(op.projection_shape)
    b[~op.valid_mask] = np.inf
    is_sirt = solver is algs.sirt_operator
    options = {} if is_sirt else dict(blocksize=2)
    actual = solver(b, op, niter=3, lmbda=.8, nonneg=True, **options)
    expected = np.zeros(4)
    previous = expected.copy()
    t = 1.
    groups = [np.arange(3)] if is_sirt else [np.array([0, 2]), np.array([1])]
    for _ in range(3):
        if solver is algs.fast_os_sart_operator:
            new_t = (1 + np.sqrt(1 + 4*t*t))/2
            saved = expected.copy()
            expected += (t-1)/new_t * (expected-previous)
            previous, t = saved, new_t
        for indices in groups:
            a = op.matrix.reshape(3, 2, 4)[indices].reshape(-1, 4).astype(float)
            data = np.where(op.valid_mask[indices], b[indices], 0).ravel()
            row, col = a.sum(axis=1), a.sum(axis=0)
            residual = np.divide(data - a@expected, row, out=np.zeros_like(row), where=row>1e-6)
            correction = np.divide(a.T@residual, col, out=np.zeros_like(col), where=col>1e-6)
            expected = np.maximum(expected + .8*correction, 0)
    np.testing.assert_allclose(actual.ravel(), expected, atol=2e-6)


@pytest.mark.parametrize('solver', [algs.cgls_operator, algs.sirt_operator, algs.os_sart_operator])
def test_initialization_zero_iterations_validation_and_exact_solution(solver):
    op = MatrixOperator()
    init = np.ones(op.volume_shape, np.float32)
    b = op.forward(init)
    np.testing.assert_array_equal(solver(b, op, niter=0, init=init), init)
    np.testing.assert_allclose(solver(b, op, niter=2, init=init), init)
    for niter in [-1, True, 1.5]:
        with pytest.raises(ValueError, match='niter'):
            solver(b, op, niter=niter)
    for invalid in ['FDK', np.zeros((2, 2)), np.full(op.volume_shape, np.nan)]:
        with pytest.raises(ValueError):
            solver(b, op, init=invalid)
    b[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='valid samples'):
        solver(b, op)


def test_entry_points_reject_unsupported_legacy_options():
    op = MatrixOperator()
    b = np.zeros(op.projection_shape, np.float32)
    np.testing.assert_array_equal(algs.lsqr(b, operator=op, niter=1), 0)
    with pytest.raises(TypeError):
        algs.sirt(b, operator=op, niter=1, W=np.ones_like(b))
    with pytest.raises(ValueError, match='omit geo'):
        algs.sirt(b, object(), None, 1, operator=op)
    np.testing.assert_array_equal(algs.sirt(b, operator=op, niter=1, noneg=True), 0)


ITERATIVE_NAMES = [
    'sart', 'sirt', 'os_sart', 'fast_os_sart', 'sart_tv', 'os_sart_tv',
    'iterativereconalg', 'cgls', 'lsqr', 'hybrid_lsqr', 'lsmr', 'irn_tv_cgls',
    'hybrid_flsqr_tv', 'ab_gmres', 'ba_gmres', 'fista', 'ista', 'mlem',
    'asd_pocs', 'os_asd_pocs', 'awasd_pocs', 'os_awasd_pocs',
    'pcsd', 'aw_pcsd', 'os_pcsd', 'os_aw_pcsd',
]


def test_interface_matrix_covers_every_exported_iterative_algorithm():
    exported = {name for name, value in vars(algs).items()
                if callable(value) and getattr(value, '__module__', '') ==
                'tigre.algorithms.iterative_recon_alg'}
    assert exported == set(ITERATIVE_NAMES)


@pytest.mark.parametrize('name', ITERATIVE_NAMES)
def test_every_public_iterative_entry_point_supports_operator(name):
    op = MatrixOperator()
    data = op.forward(np.ones(op.volume_shape, np.float32))
    solver = getattr(algs, name)
    if name in ('cgls', 'sirt', 'os_sart', 'fast_os_sart', 'asd_pocs', 'os_asd_pocs', 'awasd_pocs', 'os_awasd_pocs', 'pcsd'):
        options = dict(maxl2err=0., tviter=0) if name in ('asd_pocs', 'os_asd_pocs', 'awasd_pocs', 'os_awasd_pocs', 'pcsd') else {}
        result = solver(data, operator=op, niter=2, **options)
        assert result.shape == op.volume_shape and np.isfinite(result).all()
        assert np.linalg.norm(op.forward(result)-data) < np.linalg.norm(data)
    else:
        options=dict(maxl2err=0.,tviter=0) if 'pcsd' in name else {}
        with patch('tigre.utilities.Ax.Ax', side_effect=AssertionError('legacy projector')):
            result=solver(data,operator=op,niter=0,**options)
        assert result.shape==op.volume_shape


@pytest.mark.parametrize('name', ['fdk', 'fbp'])
def test_analytical_methods_require_their_geometry_specific_api(name):
    op = MatrixOperator()
    with pytest.raises(NotImplementedError, match='geometry-specific API'):
        getattr(algs, name)(np.zeros(op.projection_shape, np.float32), None, None, operator=op)


def geometry():
    geo = tigre.geometry_default(high_resolution=False)
    geo.nVoxel = np.array([3, 4, 5])
    geo.sVoxel = geo.nVoxel.astype(float)
    geo.dVoxel = np.ones(3)
    geo.nDetector = np.array([4, 5])
    geo.sDetector = geo.nDetector * geo.dDetector
    return geo


def test_conventional_nested_subset_selects_every_per_view_field_without_aliases():
    geo = geometry()
    angles = np.arange(12, dtype=float).reshape(4, 3)/10
    geo.DSO = np.arange(4, dtype=float) + 100
    geo.DSD = geo.DSO + 200
    geo.COR = np.arange(4, dtype=float)
    geo.offOrigin = np.arange(12).reshape(4, 3)
    geo.offDetector = np.arange(8).reshape(4, 2)
    geo.rotDetector = np.arange(12).reshape(4, 3)/100
    original = copy.deepcopy(geo)
    op = TigreOperator(geo, angles)
    sub = op.subset([3, 1, 3]).subset([2, 0])
    for name in ('DSO', 'DSD', 'COR', 'offOrigin', 'offDetector', 'rotDetector'):
        np.testing.assert_allclose(getattr(sub.geo, name), getattr(op.geo, name)[[3, 3]])
        np.testing.assert_array_equal(getattr(geo, name), getattr(original, name))
    np.testing.assert_array_equal(sub.angles, angles[[3, 3]])
    np.testing.assert_array_equal(sub.geo.nVoxel, [3, 4, 5])
    sub.geo.offOrigin[:] = 0
    assert op.geo.offOrigin.any()
    for invalid in [[], [True], [1.5], [-1], [4], [[0]]]:
        with pytest.raises(ValueError):
            op.subset(invalid)
    ax = importlib.import_module('tigre.utilities.Ax')
    atb = importlib.import_module('tigre.utilities.Atb')
    with patch.object(ax, 'Ax', return_value=np.zeros(op.projection_shape, np.float32)) as forward:
        op.forward(np.zeros(op.volume_shape))
        assert forward.call_args.args[0].dtype == np.float32
        assert forward.call_args.args[3] == 'Siddon'
    with patch.object(atb, 'Atb', return_value=np.zeros(op.volume_shape, np.float32)) as adjoint:
        op.adjoint(np.zeros(op.projection_shape))
        assert adjoint.call_args.args[3] == 'matched'


def test_surface_adapter_exact_adjoint_and_subset_on_gpu():
    geo = geometry()
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    surface = tigre.surface_from_geometry(geo, angles)
    mask = surface.valid_mask.copy()
    mask[:, 1, 2] = False
    surface = tigre.DetectorSurfaceGeometry(surface.source_positions_mm, surface.pixel_centers_mm, mask)
    op = SurfaceOperator(geo, surface, gpuids=0, n_slabs=2)
    rng = np.random.default_rng(7)
    x = rng.random(op.volume_shape, dtype=np.float32)
    b = rng.random(op.projection_shape, dtype=np.float32)
    b[~mask] = np.nan
    clean = prepare_projections(b, op)
    ax = op.forward(x)
    atb = op.adjoint(b)
    np.testing.assert_allclose(np.vdot(ax, clean), np.vdot(x, atb), rtol=2e-5)
    sub = op.subset([3, 1]).subset([1])
    np.testing.assert_allclose(sub.forward(x), ax[[1]], atol=1e-5)
    assert sub.n_slabs == 2 and sub.gpuids == 0
    np.testing.assert_array_equal(sub.valid_mask, mask[[1]])
    result = algs.cgls(ax, operator=op, niter=2)
    wrapped = algs.cgls_surface(ax, geo, surface, niter=2, gpuids=0)
    np.testing.assert_allclose(result, wrapped, atol=2e-5)


def test_conventional_adapter_matches_native_calls_and_single_view_subset_on_gpu():
    geo = geometry()
    angles = np.array([0., .4, .8])
    geo.DSO = np.array([100., 110., 120.])
    geo.DSD = np.array([200., 220., 240.])
    geo.offOrigin = np.array([[0., 0., 0.], [0., .2, 0.], [0., 0., .3]])
    from tigre.utilities.gpu import GpuIds
    gpuids = GpuIds()
    gpuids.devices = [0]
    op = TigreOperator(geo, angles, gpuids)
    image = np.ones(op.volume_shape, np.float32)
    projections = op.forward(image)
    np.testing.assert_allclose(projections, tigre.Ax(image, geo, angles, gpuids=gpuids))
    np.testing.assert_allclose(op.adjoint(projections),
                               tigre.Atb(projections, geo, angles, 'matched', gpuids=gpuids))
    np.testing.assert_allclose(op.subset([2]).forward(image), projections[[2]])
    reconstructed = algs.sirt(projections, operator=op, niter=2, nonneg=True)
    assert np.linalg.norm(projections - op.forward(reconstructed)) < np.linalg.norm(projections)


@pytest.mark.parametrize('name', ['cgls', 'sirt', 'os_sart', 'fast_os_sart'])
def test_conventional_call_keeps_legacy_path_and_options(name):
    geo = geometry()
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    data = tigre.Ax(np.ones(tuple(geo.nVoxel), np.float32), geo, angles)
    init = np.zeros(tuple(geo.nVoxel), np.float32)
    saved = init.copy()
    # Legacy-only options must still work and must never enter the new path.
    with patch('tigre.algorithms.operator_algorithms.reconstruct',
               side_effect=AssertionError('unexpected operator dispatch')):
        result, metrics = getattr(algs, name)(data, geo, angles, 2,
            init=init, verbose=False, computel2=True, mu_max=2.)
    assert result.shape == init.shape and np.isfinite(result).all()
    assert metrics.shape == (1, 2)
    assert result.max() <= 2.
    np.testing.assert_array_equal(init, saved)
