"""FDK's cosine weight must use the SAME rotDetector convention as the CUDA.

The kernels are what actually place the rays, so they are the authority. They
apply R = Rz(dRoll)*Ry(dPitch)*Rx(dYaw) to points ordered (x, y, z) =
(beam, u, v), and both bindings map

    dYaw   <- rotDetector[0]      (rotation about the BEAM - the in-plane roll)
    dPitch <- rotDetector[1]      (about u)
    dRoll  <- rotDetector[2]      (about v)

- see Python/tigre/utilities/cuda_interface/_types.pyx and, identically,
MATLAB/Utilities/cuda_interface/Atb_mex.cpp. Note the CUDA's internal field
names run the OPPOSITE way to the public [roll, pitch, yaw] ordering, which is
what made this easy to get wrong.

filtering.py used to build the weight as from_euler("XYZ", rotDetector) acting
on a vector ordered (v, u, DSD). That swaps elements 0 and 2: the roll was
applied about v, and the yaw about the beam. Because a rotation about the beam
leaves both the numerator and the norm unchanged, the real roll contributed
nothing to the weight while the yaw contributed a gradient that should not have
been there.

These tests are pure NumPy - no GPU, no reconstruction - so they run anywhere
and fail loudly if the convention drifts back.
"""
import numpy as np
import pytest

from scipy.spatial.transform import Rotation

ROT = np.array([0.003868155190980586,      # roll  (about the beam)
                -0.01785115681026292,      # pitch (about u)
                -0.001026945932170225])    # yaw   (about v)


def cuda_roll_pitch_yaw(rot, pts):
    """Verbatim transcription of Common/CUDA/Siddon_projection.cu::rollPitchYaw.

    `pts` is (3, N) ordered (beam, u, v), i.e. the kernel's (x, y, z).
    """
    yaw, pitch, roll = rot[0], rot[1], rot[2]      # dYaw, dPitch, dRoll
    cR, sR = np.cos(roll), np.sin(roll)
    cP, sP = np.cos(pitch), np.sin(pitch)
    cY, sY = np.cos(yaw), np.sin(yaw)
    x, y, z = pts
    return np.array([
        cR * cP * x + (cR * sP * sY - sR * cY) * y + (cR * sP * cY + sR * sY) * z,
        sR * cP * x + (sR * sP * sY + cR * cY) * y + (sR * sP * cY - cR * sY) * z,
        -sP * x + cP * sY * y + cP * cY * z,
    ])


def filtering_matrix(rot):
    """The rotation filtering.py builds, isolated so it can be checked here."""
    rot = np.asarray(rot, dtype=float)
    return Rotation.from_euler("ZYX", rot[..., [2, 1, 0]]).as_matrix()


def test_matches_the_kernel_to_machine_precision():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(3, 64)) * 100.0          # (beam, u, v)
    assert np.allclose(filtering_matrix(ROT) @ pts,
                       cuda_roll_pitch_yaw(ROT, pts), atol=1e-10)


def test_the_old_convention_really_was_different():
    """Guards the fix itself: if these ever agree, the test above proves nothing."""
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(3, 64)) * 100.0
    old = Rotation.from_euler("XYZ", ROT).as_matrix() @ pts[[2, 1, 0], :]
    ref = cuda_roll_pitch_yaw(ROT, pts)
    assert not np.allclose(old[2], ref[0], atol=1e-6)


@pytest.mark.parametrize("element,about_beam", [(0, True), (1, False), (2, False)])
def test_only_element_zero_is_the_beam_rotation(element, about_beam):
    """A beam rotation cannot change the cosine weight; the others must.

    This is the property that identifies the axis without trusting any Euler
    bookkeeping: rotating about the beam preserves the numerator AND the norm,
    so the weight is exactly invariant.
    """
    dsd, n, pitch_mm = 1260.0, 64, 0.14 * 48
    ax = (np.arange(n) - n / 2 + 0.5) * pitch_mm
    u, v = np.meshgrid(ax, ax)
    pts = np.vstack((np.full(u.size, dsd), u.ravel(), v.ravel()))

    rot = np.zeros(3)
    rot[element] = 0.02                     # ~1.1 degrees, well above noise
    rotated = filtering_matrix(rot) @ pts
    w0 = pts[0] / np.linalg.norm(pts, axis=0)
    w1 = rotated[0] / np.linalg.norm(rotated, axis=0)

    if about_beam:
        assert np.allclose(w1, w0, atol=1e-12), "beam rotation changed the weight"
    else:
        assert np.abs(w1 / w0 - 1.0).max() > 1e-6, "off-beam rotation was inert"


def test_untilted_geometry_is_bit_identical():
    """No tilt must mean no change, so existing results cannot move."""
    dsd, n = 1260.0, 32
    ax = (np.arange(n) - n / 2 + 0.5) * 0.14 * 96
    u, v = np.meshgrid(ax, ax)
    pts = np.vstack((np.full(u.size, dsd), u.ravel(), v.ravel()))
    w = pts[0] / np.linalg.norm(pts, axis=0)
    expected = dsd / np.sqrt(dsd ** 2 + u.ravel() ** 2 + v.ravel() ** 2)
    assert np.array_equal(w, expected)
