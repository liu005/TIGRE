import numpy as np
import pytest

pytest.importorskip("_AxSurface")

from test_surface_algorithms import (DSD, DSO, ROWS, SURFACES, phantom, rel_err, volume_geo,
                                     modular_surface)
from tigre.algorithms.fdk_surface import fdk_surface, parker_weights
from tigre.utilities.Ax_surface import Ax_surface
from tigre.utilities.detector_geometry import DetectorModule, DetectorSurfaceGeometry


def arc_surface_angles(angles, ccw=True):
    angles = np.asarray(angles)
    n = angles.size
    c, s = np.cos(angles), np.sin(angles)
    rot = np.zeros((n, 3, 3))
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1
    pose = np.tile(np.eye(4), (n, 1, 1))
    pose[:, :3, :3] = rot
    source = rot @ np.array([DSO, 0, 0])
    t = np.linspace(-1, 1, 64)
    gam = 0.1 * (t + 0.25 * t ** 3) / 1.25
    local = np.zeros((ROWS, 64, 3))
    local[..., 0] = DSO - DSD * np.cos(gam)[None]
    local[..., 1] = DSD * np.sin(gam)[None]
    local[..., 2] = np.linspace(-30, 30, ROWS)[:, None]
    return DetectorSurfaceGeometry.from_modules(source, [DetectorModule(local, pose)])


@pytest.mark.parametrize("name", list(SURFACES))
def test_full_scan(name):
    geo = volume_geo()
    surf = SURFACES[name]()
    truth = phantom()
    rec = fdk_surface(Ax_surface(truth, geo, surf), geo, surf)
    # measured: arc 0.16, modular_gaps 0.24 (hard-edged 32^3 phantom, 60 views)
    assert rel_err(rec, truth) < 0.3, rel_err(rec, truth)
    mask = truth > 0
    assert 0.8 < rec[mask].mean() / truth[mask].mean() < 1.1


@pytest.mark.parametrize("sign", [1, -1])
def test_short_scan_parker(sign):
    geo = volume_geo()
    truth = phantom()
    span = np.pi + 2 * 0.1 + 0.15
    angles = sign * np.linspace(0, span, 90, endpoint=True) + 0.7
    surf = arc_surface_angles(angles)
    rec = fdk_surface(Ax_surface(truth, geo, surf), geo, surf)
    assert rel_err(rec, truth) < 0.22, rel_err(rec, truth)
    mask = truth > 0
    assert 0.8 < rec[mask].mean() / truth[mask].mean() < 1.1


def test_parker_conjugate_weights_sum_to_one():
    gm = 0.3
    for g in np.linspace(-0.25, 0.25, 11):
        for b in np.linspace(0, 2 * gm + 2 * np.pi * 0.0 + 0.5, 9):
            b2 = b + np.pi + 2 * g
            if b2 > np.pi + 2 * gm:
                continue
            w1 = parker_weights([b], [g], gm)[0, 0]
            w2 = parker_weights([b2], [-g], gm)[0, 0]
            assert w1 + w2 == pytest.approx(1.0, abs=1e-6)


def test_rejects_non_circular_orbit():
    geo = volume_geo()
    surf = SURFACES["arc"]()
    bad = surf.subset(np.arange(surf.valid_mask.shape[0]))
    bad.source_positions_mm = bad.source_positions_mm.copy()
    bad.source_positions_mm[3, 2] += 5.0
    with pytest.raises(NotImplementedError):
        fdk_surface(np.zeros(bad.valid_mask.shape, np.float32), geo, bad)


def test_proj_shape_checked():
    geo = volume_geo()
    with pytest.raises(ValueError):
        fdk_surface(np.zeros((1, 2, 3), np.float32), geo, SURFACES["arc"]())
