import numpy as np
import pytest

pytest.importorskip("_AxSurface")

from test_surface_algorithms import NVOX, ROWS, phantom, rel_err, volume_geo
from tigre.algorithms.surface_algorithms import cgls_surface, fast_os_sart_surface
from tigre.utilities.Ax_surface import Ax_surface
from tigre.utilities.detector_geometry import linear_source_surface


def fixed_detector(curved):
    """Static detector behind the object: flat panel or an arc centred on the source line."""
    cols = 64
    u = np.linspace(-90, 90, cols)
    z = np.linspace(-30, 30, ROWS)
    px = np.zeros((ROWS, cols, 3))
    px[..., 1], px[..., 2] = u[None], z[:, None]
    px[..., 0] = -150.0
    if curved:
        px[..., 0] = -150.0 + 0.002 * u[None] ** 2   # mild bend, still single-valued
    return px


@pytest.mark.parametrize("curved", [False, True])
def test_linear_source_limited_angle_reconstruction(curved):
    geo = volume_geo()
    # source travels +-60 mm along y, 250 mm from the volume centre: ~+-13 deg
    surf = linear_source_surface(fixed_detector(curved), (250, -60, 0), (250, 60, 0), 41)
    assert surf.valid_mask.shape == (41, ROWS, 64)
    truth = phantom()
    data = Ax_surface(truth, geo, surf)
    assert data.shape == (41, ROWS, 64) and np.isfinite(data).all()
    cg = cgls_surface(data, geo, surf, niter=40)
    fos = fast_os_sart_surface(data, geo, surf, niter=15, blocksize=8, nonneg=True)
    # limited angle (+-13 deg): an incomplete problem, so only loose accuracy is expected
    # (measured: cgls 0.57, fast os-sart 0.47, both detectors)
    assert rel_err(cg, truth) < 0.65, rel_err(cg, truth)
    assert rel_err(fos, truth) < 0.55, rel_err(fos, truth)
    # data consistency is what iterative methods can guarantee here (measured 0.008 / 0.022)
    for rec in (cg, fos):
        resid = np.linalg.norm(Ax_surface(rec, geo, surf) - data) / np.linalg.norm(data)
        assert resid < 0.05, resid


def test_linear_source_surface_validation():
    with pytest.raises(ValueError):
        linear_source_surface(np.zeros((4, 3)), (0, 0, 0), (1, 0, 0), 3)
    with pytest.raises(ValueError):
        linear_source_surface(np.zeros((2, 2, 3)), (0, 0, 0), (1, 0, 0), 0)
