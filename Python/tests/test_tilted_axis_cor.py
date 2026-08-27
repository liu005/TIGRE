"""Tilted axis COMPOSED with a displaced axis (COR), a detector offset and a
detector rotation (rotDetector).

Same structure as test_tilted_axis_geometry.py, so failures localize:

  1. tilt = 0, COR / offDetector / rotDetector NON-zero, PLAIN TIGRE geometry
     vs the analytic projector. This pins how the reference models each
     ingredient against TIGRE's own kernels, with the builder not involved.
  2. tilt = 0 through the BUILDER: must return the input geometry EXACTLY
     (COR, offDetector, rotDetector, DSD, DSO) - not folded, not rearranged.
     That identity is load-bearing: TIGRE's Wang weighting is gated on
     offDetector_u == 0, and a fold of COR into offDetector_u switched it on
     for a centred detector and halved the reconstruction (scan 18,
     2026-08-27).
  3. tilt != 0 with everything present: builder vs analytic model.
  4. a guard that ignoring the offsets fails by many pixels, so 3 cannot pass
     vacuously.

Acceptance is sub-pixel, as in the parent test.
"""
import numpy as np
import pytest

import tigre
from tigre.utilities.tilted_axis_geometry import (
    tilted_axis_geo, project_points_tilted)

from test_tilted_axis_geometry import (base_geo, ball_volume, centroids,
                                       POINTS, ANGLES)

COR_MM = -3.0                              # ~16 px at this magnification
OFF_DET = (2.4, -4.0)                      # (v, u) mm
ROT_DET = (0.003868, -0.017851, -0.001027)  # setup 4's roll/pitch/yaw, rad


def geo_with_offsets(cor=COR_MM, off=OFF_DET, rot=ROT_DET):
    geo = base_geo()
    geo.COR = float(cor)
    geo.offDetector = np.array(off, dtype=np.float64)
    geo.rotDetector = np.array(rot, dtype=np.float64)
    return geo


def compare(geo, angles, tilt, ref_geo):
    gt = ball_volume(base_geo())
    ref = project_points_tilted(POINTS, ref_geo, np.asarray(ANGLES, float), *tilt)
    proj = tigre.Ax(gt, geo, angles, "Siddon")
    errs = []
    for i in range(len(ANGLES)):
        got = centroids(proj[i], ref[i])
        ok = ~np.isnan(got[:, 0])
        errs.append(np.abs(got[ok] - ref[i][ok]))
    e = np.vstack(errs)
    return float(e.mean()), float(e.max())


@pytest.mark.parametrize("kw,label", [
    (dict(rot=(0.0, 0.0, 0.0)), "COR + offDetector"),
    (dict(cor=0.0, off=(0.0, 0.0)), "rotDetector only"),
    (dict(), "COR + offDetector + rotDetector"),
])
def test_reference_models_offsets_like_tigre(kw, label):
    """Plain TIGRE geometry vs the analytic model, no tilt, no builder."""
    g = geo_with_offsets(**kw)
    mean, mx = compare(g, ANGLES, (0.0, 0.0), g)
    print("%s, tilt=0: mean %.3f px, max %.3f px" % (label, mean, mx))
    assert mx < 1.0, "reference disagrees with TIGRE's own handling of " + label


def test_zero_tilt_builder_is_the_identity():
    geo, ang = tilted_axis_geo(geo_with_offsets(), np.asarray(ANGLES, float), 0.0, 0.0)
    assert np.allclose(geo.COR, COR_MM)
    assert np.allclose(geo.offDetector, OFF_DET, atol=1e-9)
    assert np.allclose(geo.rotDetector, ROT_DET, atol=1e-12)
    assert np.allclose(geo.DSO, base_geo().DSO) and np.allclose(geo.DSD, base_geo().DSD)
    assert np.allclose(geo.offOrigin, 0.0, atol=1e-9)
    assert np.allclose(ang, ANGLES, atol=1e-9)


def test_zero_tilt_keeps_a_centred_detector_centred():
    """offDetector_u must stay exactly 0 when it was 0 (the Wang gate)."""
    geo, _ = tilted_axis_geo(geo_with_offsets(off=(2.4, 0.0)), np.asarray(ANGLES, float), 0.0, 0.0)
    assert geo.offDetector[0, 1] == 0.0


@pytest.mark.parametrize("tilt", [(0.017, 0.0), (0.0, -0.012), (0.010, 0.008)])
def test_builder_matches_physical_model_with_offsets(tilt):
    geo, ang = tilted_axis_geo(geo_with_offsets(), np.asarray(ANGLES, float), *tilt)
    mean, mx = compare(geo, ang, tilt, geo_with_offsets())
    print("tilt=%s + COR/off/rot: mean %.3f px, max %.3f px" % (tilt, mean, mx))
    assert mx < 1.0


def test_small_tilt_keeps_offdetector_u_small():
    """A 1 deg lean must not move offDetector_u by more than the geometry
    demands (~COR*tilt^2 scale) - the Wang-gate hazard in numbers."""
    geo, _ = tilted_axis_geo(geo_with_offsets(off=(2.4, 0.0)), np.asarray(ANGLES, float), 0.0, 0.0175)
    assert abs(geo.offDetector[0, 1]) < 0.05
    assert abs(geo.COR[0] - COR_MM) < 0.05


def test_offsets_change_the_answer():
    """Guard against a vacuous pass: with the offsets ignored the same
    comparison must fail by many pixels."""
    geo, ang = tilted_axis_geo(base_geo(), np.asarray(ANGLES, float), 0.010, 0.008)
    mean, mx = compare(geo, ang, (0.010, 0.008), geo_with_offsets())
    assert mx > 5.0
