"""Audit regressions: shortcuts must preserve the full public contract."""

import numpy as np
import pytest
import tigre
from tigre import algorithms as algs
from tigre.utilities.projection_operator import prepare_projections


class IdentityOperator:
    volume_shape = (3, 3, 3)
    projection_shape = (3, 3, 3)
    valid_mask = None

    def forward(self, value):
        return value.copy()

    def adjoint(self, value):
        return value.copy()

    def subset(self, indices):
        raise AssertionError("An exact solution must not need subsets")


@pytest.mark.parametrize(
    "name", ["lsqr", "lsmr", "hybrid_lsqr", "ab_gmres", "ba_gmres", "hybrid_flsqr_tv"]
)
def test_exact_warm_start_is_owned_finite_and_unchanged(name):
    operator = IdentityOperator()
    init = np.arange(1, 28, dtype=np.float32).reshape(operator.volume_shape)
    saved = init.copy()
    result = getattr(algs, name)(init, operator=operator, niter=3, init=init, verbose=False)
    np.testing.assert_array_equal(result, saved)
    np.testing.assert_array_equal(init, saved)
    assert not np.shares_memory(result, init)


@pytest.mark.parametrize("niter,zero_data", [(0, False), (2, True)])
@pytest.mark.parametrize(
    "options,error",
    [
        ({"unknown_option": 1}, TypeError),
        ({"support": np.zeros((2, 2))}, ValueError),
        ({"mu_max": -1}, ValueError),
        ({"hyper": -1}, ValueError),
        ({"noneg": True, "nonneg": False}, TypeError),
    ],
)
def test_shortcuts_validate_options(niter, zero_data, options, error):
    operator = IdentityOperator()
    data = np.full(operator.projection_shape, 0 if zero_data else 1, np.float32)
    with pytest.raises(error):
        algs.fista(data, operator=operator, niter=niter, **options)


@pytest.mark.parametrize("shape", [(0, 3, 3), (3, 3), (3.5, 3, 3), (True, 3, 3)])
def test_protocol_rejects_invalid_shapes(shape):
    operator = IdentityOperator()
    operator.volume_shape = shape
    with pytest.raises((TypeError, ValueError)):
        prepare_projections(np.ones((3, 3, 3), np.float32), operator)


@pytest.mark.parametrize("slabs", [0, -1, True, 1.5])
def test_surface_adapter_rejects_invalid_slab_count(slabs):
    geo = tigre.geometry_default(high_resolution=False)
    geo.nVoxel = np.array([3, 3, 3])
    surface = tigre.surface_from_geometry(geo, np.array([0.0, 0.5]))
    with pytest.raises(ValueError, match="n_slabs"):
        tigre.SurfaceOperator(geo, surface, n_slabs=slabs)


def test_all_exported_names_resolve():
    namespace = {}
    exec("from tigre.algorithms import *", namespace)
    for name in algs.__all__:
        assert name in namespace


@pytest.mark.parametrize("gpuids", [-1, 0.5, True, [], [0, 0.5], [[0]]])
def test_adapters_reject_invalid_devices_without_native_execution(gpuids):
    geo = tigre.geometry_default(high_resolution=False)
    surface = tigre.surface_from_geometry(geo, np.array([0.0, 0.5]))
    for factory in (
        lambda: tigre.SurfaceOperator(geo, surface, gpuids=gpuids),
        lambda: tigre.TigreOperator(geo, np.array([0.0, 0.5]), gpuids=gpuids),
    ):
        with pytest.raises(ValueError, match="gpuids"):
            factory()


@pytest.mark.parametrize(
    "name", ["lsqr", "lsmr", "hybrid_lsqr", "ab_gmres", "ba_gmres", "irn_tv_cgls"]
)
def test_exact_krylov_breakdown_returns_a_finite_solution(name):
    operator = IdentityOperator()
    data = np.zeros(operator.projection_shape, np.float32)
    data[0, 0, 0] = 1
    options = {"lmbda": 0.0} if name in ("lsmr", "hybrid_lsqr", "irn_tv_cgls") else {}
    if name == "irn_tv_cgls":
        options["niter_outer"] = 1
    result = getattr(algs, name)(data, operator=operator, niter=3, verbose=False, **options)
    np.testing.assert_allclose(result, data, atol=1e-7)
