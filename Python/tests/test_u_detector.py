"""Three-panel U-shaped detector: bottom panel plus two arms folding toward the source.

Source at 100 mm from the rotation axis, a 32 mm object (half-fan about 9.2 deg).
The bottom panel (x = -100, |y| <= 22) sees only +-6.3 deg; the two arms
(|y| = 22, x from -100 to -30) see the outer fan up to about 9.8 deg, so the arms are
needed to cover the whole object. Whole U rotates with the source.
"""
import numpy as np
import pytest

pytest.importorskip("_AxSurface")

from test_surface_algorithms import NVOX, ROWS, gantry, phantom, rel_err, volume_geo
from tigre.algorithms.fdk_surface import fdk_surface
from tigre.algorithms.surface_algorithms import cgls_surface, fast_os_sart_surface
from tigre.utilities.Ax_surface import Ax_surface
from tigre.utilities.detector_geometry import DetectorModule, DetectorSurfaceGeometry

U_DSO = 100.0
U_VIEWS = 90
BOTTOM_X, HALF_WIDTH, ARM_END_X = -100.0, 22.0, -30.0


def u_surface(n_views=U_VIEWS):
    ang = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
    c, s = np.cos(ang), np.sin(ang)
    rot = np.zeros((n_views, 3, 3))
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1
    pose = np.tile(np.eye(4), (n_views, 1, 1))
    pose[:, :3, :3] = rot
    source = rot @ np.array([U_DSO, 0, 0])

    z = np.linspace(-30, 30, ROWS)

    def panel(x, y):
        local = np.zeros((ROWS, len(x), 3))
        local[..., 0], local[..., 1], local[..., 2] = x[None], y[None], z[:, None]
        return local

    n_arm = int(round((ARM_END_X - BOTTOM_X) / 3.0))
    n_bottom = int(round(2 * HALF_WIDTH / 1.2)) + 1
    arm_x = np.linspace(ARM_END_X, BOTTOM_X + 3.0, n_arm)    # stops one pitch short of the corner
    side = np.linspace(-HALF_WIDTH, HALF_WIDTH, n_bottom)
    panels = [panel(arm_x, np.full(n_arm, -HALF_WIDTH)),
              panel(np.full(n_bottom, BOTTOM_X), side),
              panel(arm_x[::-1], np.full(n_arm, HALF_WIDTH))]
    modules, start = [], 0
    for local in panels:
        modules.append(DetectorModule(local, pose, column_offset=start))
        start += local.shape[1] + 2                           # 2-column gap between panels
    return DetectorSurfaceGeometry.from_modules(source, modules)


def test_u_surface_fan_angles_cover_object_only_with_arms():
    surf = u_surface()
    pix = surf.pixel_centers_mm[0, 0].astype(np.float64)
    src = surf.source_positions_mm[0].astype(np.float64)
    h = pix[:, :2] - src[:2]
    gamma = np.degrees(np.abs(np.arctan2(h[:, 1], -h[:, 0])))
    valid = surf.valid_mask[0, 0]
    object_half_fan = np.degrees(np.arcsin(16.0 / U_DSO))
    n_arm = int(round((ARM_END_X - BOTTOM_X) / 3.0))
    n_bottom = int(round(2 * HALF_WIDTH / 1.2)) + 1
    bottom_cols = np.arange(n_arm + 2, n_arm + 2 + n_bottom)
    assert gamma[bottom_cols].max() < object_half_fan < gamma[valid].max()


def test_u_detector_iterative_methods():
    geo = volume_geo()
    surf = u_surface().mask_outside_volume(geo)
    full = u_surface()
    assert surf.valid_mask.sum() < full.valid_mask.sum()     # beam masking removes pixels
    truth = phantom()
    data = Ax_surface(truth, geo, surf)
    cg = cgls_surface(data, geo, surf, niter=60)
    fos = fast_os_sart_surface(data, geo, surf, niter=10, blocksize=15, nonneg=True)
    assert rel_err(cg, truth) < 0.15, rel_err(cg, truth)
    assert rel_err(fos, truth) < 0.3, rel_err(fos, truth)


def test_u_detector_fdk():
    geo = volume_geo()
    surf = u_surface()
    truth = phantom()
    data = Ax_surface(truth, geo, surf)
    rec = fdk_surface(data, geo, surf)
    assert rel_err(rec, truth) < 0.3, rel_err(rec, truth)
    mask = truth > 0
    assert 0.8 < rec[mask].mean() / truth[mask].mean() < 1.15
