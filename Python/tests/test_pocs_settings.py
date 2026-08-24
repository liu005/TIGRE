"""POCS defaults and dispatch, pinned against MATLAB/Algorithms/*.m.

Reviewed line by line on 2026-08-24. The ASD family's control flow is
faithful to ASD_POCS.m: f0 reset at the top of the loop, dp_vec from the data
step and dg_vec from the TV step, dtvg = alpha*dp on the first iteration,
alpha_red gated on `dg > rmax*dp && dd > epsilon`, beta *= beta_red, the
c-cosine, and `iter >= maxiter`.

Several of these constants have been wrong here before - alpha was 0.1
against MATLAB's 0.002, which collapses the reconstruction, and lmbda_red was
inherited as 1 from the SART family so beta never decayed and the
`beta < 0.005` stop could not fire. Hence pinning them.

`maxl2err` is passed throughout only to skip the default epsilon, which
reconstructs an FDK volume just to size itself.
"""
import numpy as np
import pytest

import tigre
from tigre.algorithms.pocs_algorithms import (ASD_POCS, AwASD_POCS,
                                              OS_ASD_POCS, OS_AwASD_POCS,
                                              PCSD, OS_PCSD)


@pytest.fixture(scope="module")
def tiny():
    geo = tigre.geometry(mode="cone", nVoxel=np.array([32, 32, 32]), default=True)
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False, dtype=np.float32)
    geo.check_geo(angles)
    proj = tigre.Ax(np.zeros((32, 32, 32), dtype=np.float32), geo, angles)
    return geo, angles, proj


@pytest.mark.parametrize("cls", [ASD_POCS, AwASD_POCS, OS_ASD_POCS, OS_AwASD_POCS])
def test_asd_family_matches_matlab_defaults(tiny, cls):
    geo, angles, proj = tiny
    a = cls(proj, geo, angles, 2, maxl2err=1.0)
    assert a.alpha == pytest.approx(0.002)      # ASD_POCS.m; 0.1 collapses it
    assert a.alpha_red == pytest.approx(0.95)
    assert a.rmax == pytest.approx(0.95)
    assert a.beta_red == pytest.approx(0.99)    # NOT the SART family's 1.0
    assert a.numiter_tv == 20
    assert a.beta == pytest.approx(a.lmbda)     # one quantity, two names


@pytest.mark.parametrize("cls,expected", [
    (ASD_POCS, "minimizeTV"), (AwASD_POCS, "minimizeAwTV"),
    (OS_ASD_POCS, "minimizeTV"), (OS_AwASD_POCS, "minimizeAwTV"),
    (PCSD, "minimizeTV"), (OS_PCSD, "minimizeTV")])
def test_regularizer_dispatch(tiny, cls, expected):
    geo, angles, proj = tiny
    a = cls(proj, geo, angles, 2, maxl2err=1.0)
    assert a.regularization == expected
    assert callable(getattr(a, a.regularization))


@pytest.mark.parametrize("cls,expected", [
    (ASD_POCS, 1), (AwASD_POCS, 1), (OS_ASD_POCS, 20), (OS_AwASD_POCS, 20)])
def test_blocksize(tiny, cls, expected):
    geo, angles, proj = tiny
    a = cls(proj, geo, angles, 2, maxl2err=1.0)
    assert a.blocksize == expected


def test_lmbda_red_is_not_silently_decoupled_from_beta(tiny):
    """beta and lmbda are ONE quantity in MATLAB: applied in the data step,
    decayed each iteration, and stop-tested. Splitting them into two Python
    attributes once left the decay on a copy nobody read, so lambda_red had no
    effect on the solver at all."""
    geo, angles, proj = tiny
    a = ASD_POCS(proj, geo, angles, 2, maxl2err=1.0, lmbda=1.0, lmbda_red=0.5)
    assert a.beta_red == pytest.approx(0.5)
    assert a.beta == pytest.approx(1.0)
