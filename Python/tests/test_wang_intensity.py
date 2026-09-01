"""Wang redundancy weighting must preserve absolute reconstruction intensity.

FDK's classic ramp normalization assumes double ray coverage on a full
circle; Wang weights convert the stack to a single-count convention
(redundant pairs sum to 1), so without the w*2 compensation every
Wang-weighted reconstruction came out at exactly half the non-Wang scale
(measured 0.500 on a uniform cylinder, 2026-09-02 - surfaced by a 3 um
detector-offset change reading as Brenner -70% in a calibration figure).

GPU required (tigre.Ax / algs.fdk).
"""
import numpy as np
import pytest
import tigre
import tigre.algorithms as algs

MU = 0.02


def _geo():
    geo = tigre.geometry_default(high_resolution=False)
    geo.nVoxel = np.array([64, 64, 64])
    geo.sVoxel = np.array([64.0, 64.0, 64.0])
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    return geo


def _phantom():
    zz, yy, xx = np.mgrid[:64, :64, :64]
    cyl = ((yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= 20 ** 2)
    return (cyl.astype(np.float32) * MU), ((yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= 14 ** 2)


def _interior_mean(vol, core):
    return float(vol[16:48][core[16:48]].mean())


class TestWangAbsoluteScale:
    def test_subpixel_offset_matches_centred(self):
        """A sub-pixel u-offset flips the Wang gate; the reconstruction's
        absolute scale must NOT change (this was the 0.5x regression)."""
        geo = _geo()
        phantom, core = _phantom()
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        geo.offDetector = np.array([0.0, 0.0])
        proj = tigre.Ax(phantom, geo, angles)

        ref = _interior_mean(algs.fdk(proj, geo, angles), core)

        geo_off = _geo()
        geo_off.offDetector = np.array([0.0, 0.1])   # 0.1 mm << 1 pixel
        wang = _interior_mean(algs.fdk(proj, geo_off, angles), core)

        assert ref == pytest.approx(MU, rel=0.02)
        assert wang == pytest.approx(ref, rel=0.02), (
            f"Wang-weighted mean {wang:.6f} vs centred {ref:.6f} - "
            "the single-count halving is back")

    def test_displaced_detector_recovers_mu(self):
        """The case Wang exists for: a genuinely displaced detector
        (projections truncated on one side) must reconstruct the interior
        at the true attenuation, not half of it."""
        geo = _geo()
        phantom, core = _phantom()
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        # displace by ~20% of the detector width - one side of the object
        # leaves the field of view; Wang's overlap weighting handles it
        geo.offDetector = np.array([0.0, 0.2 * geo.sDetector[1]])
        proj = tigre.Ax(phantom, geo, angles)
        vol = algs.fdk(proj, geo, angles)
        mean = _interior_mean(vol, core)
        assert mean == pytest.approx(MU, rel=0.05), (
            f"displaced-detector interior mean {mean:.6f}, expected ~{MU}")
        # and left/right halves of the interior must stay balanced
        left = float(vol[16:48, :, :32][core[16:48, :, :32]].mean())
        right = float(vol[16:48, :, 32:][core[16:48, :, 32:]].mean())
        assert left == pytest.approx(right, rel=0.05)
