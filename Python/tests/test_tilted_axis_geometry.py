"""Validate tilted_axis_geometry against analytic pinhole projection.

STRUCTURE, so failures localize. Test 1 runs at TILT = 0: it compares
tigre.Ax ball centroids under a plain nominal geometry against
`project_points_tilted(..., 0, 0)`. That validates the analytic projector's
conventions (angle sign, pixel indexing, u/v axes) against TIGRE itself, with
the builder not involved. Only when that holds does test 2 mean anything: the
same comparison at real tilts, with the geometry built by `tilted_axis_geo`.
A failure in test 2 alone is a builder bug; a failure in test 1 is a
convention bug in the reference.

THE BAR. The deleted predecessor utility (axis tilt as per-view offsets)
scored 12.48 px at its best setting on this style of test - and that was its
own validation falsifying it. Acceptance here is SUB-PIXEL, mean and max.

Balls are compact Gaussians so centroids are sub-voxel; points are placed at
several radii and heights because the tilt's signature grows with both, and a
single on-axis point would validate almost nothing.
"""
import numpy as np
import pytest

import tigre
from tigre.utilities.tilted_axis_geometry import (
    tilted_axis_geo, project_points_tilted)

NVOX = 256
SVOX = 100.0
DSD, DSO = 1260.0, 296.851


def base_geo():
    geo = tigre.geometry()
    geo.DSD, geo.DSO = DSD, DSO
    geo.nDetector = np.array([512, 512])
    geo.dDetector = np.array([0.8, 0.8])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([NVOX] * 3)
    geo.sVoxel = np.array([SVOX] * 3)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.zeros(3)
    geo.offDetector = np.zeros(2)
    geo.accuracy = 0.5
    geo.COR = 0.0
    geo.rotDetector = np.zeros(3)
    geo.mode = "cone"
    return geo


# Ball centres in the axis-aligned frame (mm): spread in radius AND height.
POINTS = np.array([
    [0.0, 0.0, 0.0],
    [18.0, 6.0, 22.0],
    [-14.0, 12.0, -25.0],
    [6.0, -20.0, 30.0],
    [-9.0, -15.0, -33.0],
])
ANGLES = np.linspace(0, 2 * np.pi, 12, endpoint=False, dtype=np.float32)


def ball_volume(geo):
    n = int(geo.nVoxel[0])
    d = float(geo.dVoxel[0])
    ax = (np.arange(n) + 0.5 - n / 2.0) * d
    vol = np.zeros((n, n, n), dtype=np.float32)
    zz, yy, xx = np.meshgrid(ax, ax, ax, indexing="ij")
    for (px, py, pz) in POINTS:
        # TIGRE volume index order is (z, y, x)
        r2 = (xx - px) ** 2 + (yy - py) ** 2 + (zz - pz) ** 2
        vol += np.exp(-r2 / (2 * 1.2 ** 2)).astype(np.float32)
    return np.ascontiguousarray(vol)


def centroids(proj, expected, win=14):
    """Blob centroid near each expected (u, v), NaN when out of view."""
    nv, nu = proj.shape
    out = np.full((len(expected), 2), np.nan)
    for j, (ue, ve) in enumerate(expected):
        if not (0 <= ue < nu and 0 <= ve < nv):
            continue
        u0, v0 = int(round(ue)), int(round(ve))
        us, vs = max(u0 - win, 0), max(v0 - win, 0)
        w = proj[vs:v0 + win, us:u0 + win].astype(np.float64)
        w = np.clip(w - 0.1 * w.max(), 0, None)
        if w.sum() <= 0:
            continue
        vv, uu = np.mgrid[vs:vs + w.shape[0], us:us + w.shape[1]]
        out[j] = [(w * uu).sum() / w.sum(), (w * vv).sum() / w.sum()]
    return out


def run_compare(geo, angles, tilt):
    gt = ball_volume(base_geo())
    ref = project_points_tilted(POINTS, base_geo(), np.asarray(ANGLES, float),
                                *tilt)
    proj = tigre.Ax(gt, geo, angles, "Siddon")
    errs = []
    for i in range(len(ANGLES)):
        got = centroids(proj[i], ref[i])
        ok = ~np.isnan(got[:, 0])
        errs.append(np.abs(got[ok] - ref[i][ok]))
    e = np.vstack(errs)
    return float(e.mean()), float(e.max())


def test_projector_conventions_at_zero_tilt():
    """The analytic reference must match plain TIGRE before any tilt enters."""
    mean, mx = run_compare(base_geo(), ANGLES, (0.0, 0.0))
    print("tilt=0: mean %.3f px, max %.3f px" % (mean, mx))
    assert mx < 1.0, "reference projector conventions disagree with TIGRE"


@pytest.mark.parametrize("tilt", [(0.017, 0.0), (0.0, -0.012), (0.010, 0.008)])
def test_builder_matches_physical_model(tilt):
    geo, ang = tilted_axis_geo(base_geo(), np.asarray(ANGLES, float), *tilt)
    mean, mx = run_compare(geo, ang.astype(np.float32), tilt)
    print("tilt=%s: mean %.3f px, max %.3f px" % (tilt, mean, mx))
    assert mx < 1.0, ("builder disagrees with the physical model "
                      "(predecessor's best was 12.48 px - the bar is <1)")


def test_zero_tilt_builder_is_identity():
    geo, ang = tilted_axis_geo(base_geo(), np.asarray(ANGLES, float), 0.0, 0.0)
    assert np.allclose(geo.DSO, DSO) and np.allclose(geo.DSD, DSD)
    assert np.allclose(geo.rotDetector, 0.0, atol=1e-12)
    assert np.allclose(geo.offDetector, 0.0, atol=1e-9)
    assert np.allclose(geo.offOrigin, 0.0, atol=1e-9)
    assert np.allclose(ang, ANGLES, atol=1e-9)
