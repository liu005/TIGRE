import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_surface_algorithms import arc_surface, phantom, volume_geo, rel_err  # noqa: E402
from tigre.utilities.Ax_surface import Ax_surface, Atb_surface  # noqa: E402


def _setup():
    return volume_geo(), arc_surface()


@pytest.mark.parametrize("n_slabs", [2, 3, 5])
def test_slabs_match_single_volume(n_slabs):
    geo, surface = _setup()
    img = phantom()
    ref = Ax_surface(img, geo, surface)
    got = Ax_surface(img, geo, surface, n_slabs=n_slabs)
    assert rel_err(got, ref) < 1e-5

    proj = ref.astype(np.float32)
    bref = Atb_surface(proj, geo, surface)
    bgot = Atb_surface(proj, geo, surface, n_slabs=n_slabs)
    assert rel_err(bgot, bref) < 1e-5


def test_gpuids_object_and_default():
    from tigre.utilities.gpu import GpuIds
    geo, surface = _setup()
    img = phantom()
    ref = Ax_surface(img, geo, surface, gpuids=GpuIds()[0])
    assert rel_err(Ax_surface(img, geo, surface), ref) < 1e-5
    assert rel_err(Ax_surface(img, geo, surface, gpuids=GpuIds()), ref) < 1e-5
    assert rel_err(Ax_surface(img, geo, surface, gpuids=[0, 0]), ref) < 1e-5


def test_fdk_and_cgls_slabs():
    from tigre.utilities.gpu import GpuIds
    from tigre.algorithms.fdk_surface import fdk_surface
    from tigre.algorithms.surface_algorithms import cgls_surface
    geo, surface = _setup()
    data = Ax_surface(phantom(), geo, surface)
    one = GpuIds()[0]
    assert rel_err(fdk_surface(data, geo, surface, gpuids=[0, 0, 0]),
                   fdk_surface(data, geo, surface, gpuids=one)) < 1e-4
    assert rel_err(cgls_surface(data, geo, surface, niter=3, gpuids=[0, 0]),
                   cgls_surface(data, geo, surface, niter=3, gpuids=one)) < 1e-3
