"""Tilted axis COMPOSED with a displaced axis (COR) and a detector offset.

Same structure as test_tilted_axis_geometry.py, so failures localize:

  1. tilt = 0, COR and offDetector NON-zero, PLAIN TIGRE geometry (geo.COR and
     geo.offDetector set the ordinary way) vs the analytic projector. This
     pins how the reference models COR - a rigid lateral shift of source and
     detector - against TIGRE's own kernels, with the builder not involved.
  2. tilt = 0 through the BUILDER (which folds COR into offDetector and
     returns COR = 0): must match TIGRE's plain-COR projection. This is the
     documented fold-in offDetector_u + (DSD/DSO)*COR, measured not assumed.
  3. tilt != 0 with COR and offDetector: builder vs analytic model.

Acceptance is sub-pixel, as in the parent test.
"""
import numpy as np
import pytest

import tigre
from tigre.utilities.tilted_axis_geometry import (
    tilted_axis_geo, project_points_tilted)

from test_tilted_axis_geometry import (base_geo, ball_volume, centroids,
                                       POINTS, ANGLES)

COR_MM = -3.0            # ~100 detector px at this magnification
OFF_DET = (2.4, -4.0)    # (v, u) mm


def geo_with_offsets(cor=COR_MM, off=OFF_DET):
    geo = base_geo()
    geo.COR = float(cor)
    geo.offDetector = np.array(off, dtype=np.float64)
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


def test_reference_models_cor_and_offdetector_like_tigre():
    """Plain TIGRE COR/offDetector vs the analytic model, no tilt, no builder."""
    mean, mx = compare(geo_with_offsets(), ANGLES, (0.0, 0.0), geo_with_offsets())
    print("plain COR/off, tilt=0: mean %.3f px, max %.3f px" % (mean, mx))
    assert mx < 1.0, "reference disagrees with TIGRE's own COR/offDetector handling"


def test_zero_tilt_builder_reproduces_tigre_cor():
    """The fold of COR into offDetector must reproduce TIGRE's plain-COR rays."""
    geo, ang = tilted_axis_geo(geo_with_offsets(), np.asarray(ANGLES, float), 0.0, 0.0)
    assert np.allclose(geo.COR, 0.0)
    # documented first-order equivalence, offDetector_u + (DSD/DSO)*COR
    expect_u = OFF_DET[1] + geo_with_offsets().DSD / geo_with_offsets().DSO * COR_MM
    assert abs(geo.offDetector[0, 1] - expect_u) < 0.05 * abs(expect_u)
    mean, mx = compare(geo, ang, (0.0, 0.0), geo_with_offsets())
    print("builder COR fold, tilt=0: mean %.3f px, max %.3f px" % (mean, mx))
    assert mx < 1.0


@pytest.mark.parametrize("tilt", [(0.017, 0.0), (0.0, -0.012), (0.010, 0.008)])
def test_builder_matches_physical_model_with_cor(tilt):
    geo, ang = tilted_axis_geo(geo_with_offsets(), np.asarray(ANGLES, float), *tilt)
    mean, mx = compare(geo, ang, tilt, geo_with_offsets())
    print("tilt=%s + COR/off: mean %.3f px, max %.3f px" % (tilt, mean, mx))
    assert mx < 1.0


def test_offsets_change_the_answer():
    """Guard against a vacuous pass: with COR/offDetector ignored the same
    comparison must fail by many pixels (COR_MM alone is ~16 px at this
    magnification; measured 12.6 px max after the detector offset partly
    cancels it for some balls)."""
    geo, ang = tilted_axis_geo(base_geo(), np.asarray(ANGLES, float), 0.010, 0.008)
    mean, mx = compare(geo, ang, (0.010, 0.008), geo_with_offsets())
    assert mx > 5.0
