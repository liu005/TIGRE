"""Input validation and slab/offset invariants of the surface routines."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("_AxSurface")
from test_surface_algorithms import arc_surface, phantom, volume_geo, rel_err  # noqa: E402
from tigre.algorithms.fdk_surface import fdk_surface  # noqa: E402
from tigre.algorithms.surface_algorithms import (  # noqa: E402
    cgls_surface, os_sart_surface, sirt_surface)
from tigre.utilities.Ax_surface import Ax_surface, Atb_surface  # noqa: E402
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry  # noqa: E402


def _data():
    geo, surface = volume_geo(), arc_surface()
    return geo, surface, Ax_surface(phantom(), geo, surface)


@pytest.mark.parametrize("bad", [0, -3, 2.5])
def test_os_sart_rejects_bad_blocksize(bad):
    geo, surface, data = _data()
    with pytest.raises(ValueError, match="blocksize"):
        os_sart_surface(data, geo, surface, niter=1, blocksize=bad)


@pytest.mark.parametrize("solver", [cgls_surface, sirt_surface, os_sart_surface, fdk_surface])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_solvers_reject_nonfinite_data(solver, value):
    geo, surface, data = _data()
    data[3, 5, 7] = value
    kwargs = {} if solver is fdk_surface else {"niter": 1}
    with pytest.raises(ValueError, match="NaN or Inf"):
        solver(data, geo, surface, **kwargs)


@pytest.mark.parametrize("cutoff", [0, -0.5, 1.5, np.nan])
def test_fdk_rejects_bad_cutoff(cutoff):
    geo, surface, data = _data()
    with pytest.raises(ValueError, match="cutoff"):
        fdk_surface(data, geo, surface, cutoff=cutoff)


def test_float32_overflow_is_rejected():
    surface = arc_surface()
    with pytest.raises(ValueError, match="float32"):
        DetectorSurfaceGeometry(surface.source_positions_mm.astype(np.float64) * 0 + 1e300,
                                surface.pixel_centers_mm)


def test_slabs_with_anisotropic_voxels_and_offset():
    geo = volume_geo()
    geo.nVoxel = np.array([24, 20, 28])
    geo.dVoxel = np.array([1.5, 1.0, 0.75])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.offOrigin = np.array([2.0, -3.0, 4.0])
    surface = arc_surface()
    rng = np.random.default_rng(0)
    img = rng.random(tuple(geo.nVoxel)).astype(np.float32)
    proj = rng.random(surface.valid_mask.shape).astype(np.float32)

    ref = Ax_surface(img, geo, surface, gpuids=0, n_slabs=1)
    bref = Atb_surface(proj, geo, surface, gpuids=0, n_slabs=1)
    for n in (2, 3, 7):
        assert rel_err(Ax_surface(img, geo, surface, gpuids=0, n_slabs=n), ref) < 1e-5
        assert rel_err(Atb_surface(proj, geo, surface, gpuids=0, n_slabs=n), bref) < 1e-5

    # adjoint identity <Ax, y> = <x, Atb y> holds with slabs
    lhs = float(np.sum(Ax_surface(img, geo, surface, gpuids=0, n_slabs=3).astype(np.float64) * proj))
    rhs = float(np.sum(img.astype(np.float64) * Atb_surface(proj, geo, surface, gpuids=0, n_slabs=3)))
    assert abs(lhs - rhs) / abs(lhs) < 1e-4
