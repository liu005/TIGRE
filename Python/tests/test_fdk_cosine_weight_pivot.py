"""The chunked FDK filter's cosine pre-weight must place the pixels where the
CUDA kernels place them: the detector rotated by rotDetector about its OWN
CENTRE (computeDeltas: P.x = 0, rollPitchYaw, translate back), so a pixel sits
at (DSD, 0, 0) + R (0, u, v) relative to the source.

An earlier fork version rotated the whole (DSD, u, v) vector - a rotation about
the SOURCE - which tilts the beam and puts a spurious first-order gradient on
the weight (~0.1 % across our 3072 px detector at 1 degree of pitch). These
tests rebuild the weight independently from the kernel's own construction and
compare. Pure NumPy, no GPU.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tigre.utilities.filtering import cosine_weight

NU, NV, DU, DV, DSD, DSO = 96, 64, 0.8, 0.8, 1536.0, 1000.0


def kernel_roll_pitch_yaw(rot, x, y, z):
    """Verbatim Common/CUDA/Siddon_projection.cu::rollPitchYaw."""
    yaw, pitch, roll = rot[0], rot[1], rot[2]
    cR, sR = np.cos(roll), np.sin(roll)
    cP, sP = np.cos(pitch), np.sin(pitch)
    cY, sY = np.cos(yaw), np.sin(yaw)
    return np.array([
        cR * cP * x + (cR * sP * sY - sR * cY) * y + (cR * sP * cY + sR * sY) * z,
        sR * cP * x + (sR * sP * sY + cR * cY) * y + (sR * sP * cY - cR * sY) * z,
        -sP * x + cP * sY * y + cP * cY * z,
    ])


def reference_weight(rot):
    """Straight from computeDeltas: corner P and steps Pu0/Pv0, rotated about
    the detector centre, translated to x = -(DSD - DSO); source at (DSO, 0, 0)."""
    P = np.array([0.0, DU * (-NU / 2 + 0.5), DV * (NV / 2 - 0.5)])
    Pu0 = np.array([0.0, DU, 0.0])
    Pv0 = np.array([0.0, 0.0, -DV])
    P, Pu0, Pv0 = (kernel_roll_pitch_yaw(rot, *q) for q in (P, Pu0, Pv0))
    P = P + np.array([-(DSD - DSO), 0.0, 0.0])
    i = np.arange(NU)[None, :]
    j = np.arange(NV)[:, None]
    pix = P[:, None, None] + Pu0[:, None, None] * i + Pv0[:, None, None] * j
    D = pix - np.array([DSO, 0.0, 0.0])[:, None, None]
    return -D[0] / np.linalg.norm(D, axis=0)


def grids():
    u = np.arange(-NU / 2 + 0.5, NU / 2) * DU
    v = np.arange(-NV / 2 + 0.5, NV / 2) * DV
    uu, vv = np.meshgrid(u, v)          # (V, U): uu = u, vv = v, as in the filter
    return uu, vv


def R_of(rot):
    rot = np.asarray(rot, dtype=float)
    return Rotation.from_euler("ZYX", rot[[2, 1, 0]]).as_matrix()


ROTS = [
    [0.003868, -0.017851, -0.001027],   # setup 4: roll about the beam, 1 deg pitch, yaw
    [0.05, 0.0, 0.0],
    [0.0, 0.05, 0.0],
    [0.0, 0.0, 0.05],
    [0.02, -0.03, 0.04],
]


@pytest.mark.parametrize("rot", ROTS)
def test_matches_the_kernel_pixel_positions(rot):
    uu, vv = grids()
    w = cosine_weight(uu, vv, DSD, R_of(rot))
    assert np.allclose(w, reference_weight(np.asarray(rot)), rtol=0, atol=1e-12)


def test_untilted_is_bit_identical_to_the_classic_expression():
    uu, vv = grids()
    classic = DSD / np.sqrt(DSD ** 2 + uu ** 2 + vv ** 2)
    assert np.array_equal(cosine_weight(uu, vv, DSD), classic)
    assert np.allclose(cosine_weight(uu, vv, DSD, np.eye(3)), classic, atol=1e-13)


def test_tilt_effect_is_second_order_not_a_gradient():
    """On setup 4's rotDetector the source-pivot form had a ~0.1 % gradient;
    the centre pivot moves the weight by < 2e-5 anywhere on this detector."""
    uu, vv = grids()
    w0 = cosine_weight(uu, vv, DSD)
    w1 = cosine_weight(uu, vv, DSD, R_of(ROTS[0]))
    assert np.abs(w1 - w0).max() < 2e-5
    # the OLD construction, for contrast: rotate (DSD, u, v) about the source
    xyz = R_of(ROTS[0]) @ np.vstack((np.full(uu.size, DSD), uu.ravel(), vv.ravel()))
    w_old = (xyz[0] / np.linalg.norm(xyz, axis=0)).reshape(uu.shape)
    assert np.abs(w_old - w0).max() > 3e-4          # the spurious gradient it had
    # (measured: centre pivot 3e-7 here / 8e-6 on a 427 mm detector; source
    #  pivot 4.8e-4 / 5.8e-4 - i.e. ~0.12 % peak to peak across the detector)


def test_roll_about_the_beam_keeps_the_weight_set():
    uu, vv = grids()
    w0 = cosine_weight(uu, vv, DSD)
    w1 = cosine_weight(uu, vv, DSD, R_of([0.3, 0.0, 0.0]))
    assert w1[NV // 2, NU // 2] == pytest.approx(w0[NV // 2, NU // 2], abs=1e-12)
    yy, xx = np.mgrid[:NV, :NU]
    disc = (yy - (NV - 1) / 2) ** 2 + (xx - (NU - 1) / 2) ** 2 <= 25 ** 2
    assert w1[disc].mean() == pytest.approx(w0[disc].mean(), rel=1e-9)
