import numpy as np
import pytest

pytest.importorskip("_AxSurface")

from test_surface_algorithms import (DSO, N_VIEWS, ROWS, NVOX, gantry, phantom, rel_err,
                                     volume_geo)
from tigre.algorithms.surface_algorithms import cgls_surface
from tigre.utilities.Ax_surface import Ax_surface, Atb_surface
from tigre.utilities.detector_geometry import DetectorModule, DetectorSurfaceGeometry


def u_surface():
    """Three flat panels forming a U that opens toward the source (rotating with it)."""
    source, pose = gantry(N_VIEWS)
    z = np.linspace(-30, 30, ROWS)
    cols = 40
    modules = []
    # bottom panel behind the object, two side arms running toward the source;
    # the arms are long, so many of their pixels see nothing of the 32 mm volume
    s = np.linspace(-45, 45, cols)
    bottom = np.zeros((ROWS, cols, 3))
    bottom[..., 0], bottom[..., 1], bottom[..., 2] = -150.0, s[None], z[:, None]
    arm = np.zeros((ROWS, cols, 3))
    for sign in (1, -1):
        arm = np.zeros((ROWS, cols, 3))
        arm[..., 0] = np.linspace(-150, 120, cols)[None]
        arm[..., 1] = sign * 45.0
        arm[..., 2] = z[:, None]
        modules.append(arm)
    modules.append(bottom)
    mods = [DetectorModule(m, pose, column_offset=(cols + 2) * k) for k, m in enumerate(modules)]
    return DetectorSurfaceGeometry.from_modules(source, mods)


def test_mask_outside_volume_drops_pixels_missing_the_volume():
    geo = volume_geo()
    surf = u_surface()
    masked = surf.mask_outside_volume(geo)
    assert masked.valid_mask.sum() < surf.valid_mask.sum()
    data = Ax_surface(np.ones((NVOX,) * 3, np.float32), geo, surf)
    kept = masked.valid_mask
    # every pixel that was dropped really has no path through the volume
    assert np.all(data[surf.valid_mask & ~kept] == 0)
    # and every kept pixel does
    assert np.all(data[kept] > 0)


def test_u_shaped_detector_reconstruction_with_masking():
    geo = volume_geo()
    surf = u_surface().mask_outside_volume(geo)
    truth = phantom()
    data = Ax_surface(truth, geo, surf)
    rec = cgls_surface(data, geo, surf, niter=60)
    assert rel_err(rec, truth) < 0.15, rel_err(rec, truth)


def test_pixel_inside_volume_ends_ray_at_the_pixel():
    geo = volume_geo()
    source = np.array([[DSO, 0.0, 0.0]], np.float32)
    inside = np.zeros((1, 1, 2, 3), np.float32)
    inside[0, 0, 0] = (0.0, 0.0, 0.0)      # volume centre
    inside[0, 0, 1] = (-100.0, 0.0, 0.0)   # beyond the volume
    surf = DetectorSurfaceGeometry(source, inside)
    data = Ax_surface(np.ones((NVOX,) * 3, np.float32), geo, surf)
    # half the 32 mm box for the first pixel, the whole box for the second
    assert data[0, 0, 0] == pytest.approx(NVOX / 2, rel=1e-3)
    assert data[0, 0, 1] == pytest.approx(NVOX, rel=1e-3)


def test_adjoint_with_pixels_inside_volume():
    geo = volume_geo()
    rng = np.random.default_rng(0)
    source = np.array([[DSO, 3.0, 1.0], [-DSO, 2.0, 0.5]], np.float32)
    pix = rng.uniform(-30, 30, (2, 3, 5, 3)).astype(np.float32)
    surf = DetectorSurfaceGeometry(source, pix)
    x = rng.random((NVOX,) * 3).astype(np.float32)
    y = rng.random((2, 3, 5)).astype(np.float32)
    lhs = float(np.sum(Ax_surface(x, geo, surf).astype(np.float64) * y))
    rhs = float(np.sum(x.astype(np.float64) * Atb_surface(y, geo, surf)))
    assert lhs == pytest.approx(rhs, rel=1e-4)
