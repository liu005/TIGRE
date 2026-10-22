"""Validate Ax_surface (explicit source/pixel Siddon projector).

1. A flat uniform detector expressed as a surface reproduces tigre.Ax.
2. A curved (arc) detector with a moving source matches the analytic line
   integral through a uniform sphere, independent of TIGRE's detector model.
3. Masked samples are zero; per-view and shared pixel tables agree.
"""
import numpy as np
import pytest

import tigre
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry, surface_from_geometry

pytest.importorskip("_AxSurface")
from tigre.utilities.Ax_surface import Ax_surface, Atb_surface  # noqa: E402


def make_geo(nvox=48, nrows=24, ncols=40, off_det=(0.0, 0.0)):
    geo = tigre.geometry()
    geo.DSD, geo.DSO = 900.0, 600.0
    geo.nDetector = np.array([nrows, ncols])
    geo.dDetector = np.array([1.5, 1.2])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([nvox] * 3)
    geo.sVoxel = np.array([nvox * 1.0] * 3)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.zeros(3)
    geo.offDetector = np.array(off_det)
    geo.accuracy = 0.5
    geo.COR = 0.0
    geo.rotDetector = np.zeros(3)
    geo.mode = "cone"
    return geo


def flat_surface(geo, angles):
    nrows, ncols = (int(v) for v in geo.nDetector)
    du, dv = geo.dDetector[1], geo.dDetector[0]
    u = du * (np.arange(ncols) - (ncols - 1) / 2) + geo.offDetector[1]
    v = dv * (np.arange(nrows) - (nrows - 1) / 2) + geo.offDetector[0]
    local = np.zeros((nrows, ncols, 3))
    local[..., 0] = -(geo.DSD - geo.DSO)
    local[..., 1] = u[None, :]
    local[..., 2] = v[:, None]
    c, s = np.cos(angles), np.sin(angles)
    rot = np.stack([np.stack([c, -s, 0 * c], -1),
                    np.stack([s, c, 0 * c], -1),
                    np.stack([0 * c, 0 * c, 1 + 0 * c], -1)], 1)
    source = np.einsum("vij,j->vi", rot, [geo.DSO, 0, 0])
    pixels = np.einsum("vij,hwj->vhwi", rot, local)
    return DetectorSurfaceGeometry(source, pixels)


def gaussian_phantom(geo, offsets=((8, -5, 6), (-9, 7, -8)), weights=(1.0, 0.5), var=18.0):
    n = int(geo.nVoxel[0])
    ax = np.arange(n) - (n - 1) / 2
    z, y, x = np.meshgrid(ax, ax, ax, indexing="ij")
    img = np.zeros((n, n, n))
    for (cx, cy, cz), w in zip(offsets, weights):
        img += w * np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * var))
    return img.astype(np.float32)


def test_flat_surface_matches_tigre_ax():
    geo = make_geo(off_det=(2.0, -3.0))
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False).astype(np.float32)
    img = gaussian_phantom(geo)
    ref = tigre.Ax(img, geo, angles)
    got = Ax_surface(img, geo, flat_surface(geo, angles))
    assert got.shape == ref.shape
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-4, err


def animate_geo(**overrides):
    """The geometry used in test_animate_geometry.py (every offset/rotation path)."""
    geo = tigre.geometry(mode="cone", default=True)
    geo.nVoxel = np.array([32, 48, 64])
    geo.sVoxel = geo.nVoxel.astype(np.float64)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.array([60, 80])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.offDetector = np.array([10.0, -15.0])
    geo.rotDetector = np.radians([5.0, -2.0, 1.0])
    # Same as test_animate_geometry.py except offOrigin, which is moved off the
    # voxel grid: at (0, 5, -8) the source lies exactly on voxel planes and a
    # ray grazing a plane picks its voxel row by float rounding (TIGRE's own
    # kernel is equally sensitive), so the two implementations legitimately differ.
    geo.offOrigin = np.array([0.21, 5.37, -8.13])
    geo.COR = 1.5
    for key, value in overrides.items():
        setattr(geo, key, value)
    return geo


def assert_equivalent(got, ref):
    """Two float32 Siddon implementations agree except at rare tie rays.

    A ray through a voxel edge/corner can land in a different voxel after
    float32 rounding, in either implementation (checked against an exact
    float64 integral: each side is right for some such rays). So compare the
    outlier fraction and the relative L2 error, not the maximum. A convention
    error (shift, sign, wrong rotation) breaks both by orders of magnitude.
    """
    err = np.abs(got.astype(np.float64) - ref)
    assert (err > 1e-3 * ref.max()).mean() < 1e-3
    assert np.linalg.norm(err) < 2e-3 * np.linalg.norm(ref)


@pytest.mark.parametrize("overrides", [
    {},
    {"rotDetector": np.zeros(3), "COR": 0.0},
    {"rotDetector": np.radians([0.0, 0.0, 4.0])},
    {"rotDetector": np.radians([3.0, 0.0, 0.0])},
    {"rotDetector": np.radians([0.0, 3.0, 0.0])},
    {"offOrigin": np.zeros(3), "offDetector": np.zeros(2)},
], ids=["animate", "no_rot_no_cor", "roll", "yaw", "pitch", "centered"])
def test_surface_from_geometry_matches_tigre_ax(overrides):
    geo = animate_geo(**overrides)
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False).astype(np.float32)
    img = gaussian_phantom_shape(geo)
    ref = tigre.Ax(img, geo, angles)
    got = Ax_surface(img, geo, surface_from_geometry(geo, angles))
    assert ref.max() > 1
    assert_equivalent(got, ref)


def test_surface_from_geometry_per_view_euler_angles():
    geo = animate_geo()
    angles = np.stack([np.linspace(0, 2 * np.pi, 5, endpoint=False),
                       np.full(5, 0.15), np.full(5, -0.1)], 1).astype(np.float32)
    img = gaussian_phantom_shape(geo)
    ref = tigre.Ax(img, geo, angles)
    got = Ax_surface(img, geo, surface_from_geometry(geo, angles))
    assert_equivalent(got, ref)


def gaussian_phantom_shape(geo):
    nz, ny, nx = (int(v) for v in geo.nVoxel)
    z, y, x = np.meshgrid(np.arange(nz) - (nz - 1) / 2, np.arange(ny) - (ny - 1) / 2,
                          np.arange(nx) - (nx - 1) / 2, indexing="ij")
    img = (np.exp(-((x - 9) ** 2 + (y + 6) ** 2 + (z - 4) ** 2) / 30)
           + 0.5 * np.exp(-((x + 12) ** 2 + (y - 8) ** 2 + (z + 5) ** 2) / 50))
    return img.astype(np.float32)


def test_arc_detector_moving_source_matches_analytic_gaussian():
    geo = make_geo(nvox=64)
    sigma = 6.0
    img = gaussian_phantom(geo, offsets=((0, 0, 0),), weights=(1.0,), var=sigma ** 2)
    n_views, rows, cols = 6, 6, 48
    # Source moves on a limited circular arc; detector is an arc centred on it.
    phis = np.linspace(-0.4, 0.4, n_views)
    gam = np.linspace(-0.12, 0.12, cols)
    zs = np.linspace(-8, 8, rows)
    dso, dsd = 600.0, 1100.0
    source = np.stack([dso * np.cos(phis), dso * np.sin(phis), 0 * phis], -1)
    pixels = np.zeros((n_views, rows, cols, 3))
    for i, phi in enumerate(phis):
        ang = phi + np.pi + gam
        px = source[i, 0] + dsd * np.cos(ang)
        py = source[i, 1] + dsd * np.sin(ang)
        pixels[i, :, :, 0] = px[None, :]
        pixels[i, :, :, 1] = py[None, :]
        pixels[i, :, :, 2] = zs[:, None]
    surf = DetectorSurfaceGeometry(source, pixels)
    got = Ax_surface(img, geo, surf)

    d = pixels - source[:, None, None, :]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    s = source[:, None, None, :]
    dist2 = (s ** 2).sum(-1) - ((s * d).sum(-1)) ** 2
    expected = np.sqrt(2 * np.pi) * sigma * np.exp(-dist2 / (2 * sigma ** 2))
    assert expected.max() > 10
    # Siddon uses a piecewise-constant volume, so rays near voxel faces carry a
    # transverse discretization error (~2%); exactness is covered by the flat
    # comparison with tigre.Ax above.
    assert np.abs(got - expected).max() < 0.03 * expected.max()
    assert np.abs(got - expected).mean() < 0.005 * expected.max()


def test_mask_and_per_view_tables():
    geo = make_geo()
    angles = np.linspace(0, np.pi, 4, endpoint=False).astype(np.float32)
    img = gaussian_phantom(geo)
    surf = flat_surface(geo, angles)
    full = Ax_surface(img, geo, surf)

    mask = np.ones(surf.valid_mask.shape[1:], dtype=bool)
    mask[:, 10:13] = False
    gapped = DetectorSurfaceGeometry(
        surf.source_positions_mm, surf.pixel_centers_mm, mask)
    out = Ax_surface(img, geo, gapped)
    assert np.all(out[:, :, 10:13] == 0)
    keep = np.broadcast_to(mask, out.shape)
    np.testing.assert_array_equal(out[keep], full[keep])

    shared = DetectorSurfaceGeometry(
        surf.source_positions_mm[:1], surf.pixel_centers_mm[0])
    one = Ax_surface(img, geo, shared)
    np.testing.assert_array_equal(one, full[:1])


def arc_surface(n_views=6, rows=6, cols=48):
    phis = np.linspace(-0.4, 0.4, n_views)
    gam = np.linspace(-0.12, 0.12, cols) ** 3 * 8 + np.linspace(-0.12, 0.12, cols)
    zs = np.linspace(-8, 8, rows)
    source = np.stack([600 * np.cos(phis), 600 * np.sin(phis), 0 * phis], -1)
    pixels = np.zeros((n_views, rows, cols, 3))
    for i, phi in enumerate(phis):
        a = phi + np.pi + gam
        pixels[i, :, :, 0] = (source[i, 0] + 1100 * np.cos(a))[None]
        pixels[i, :, :, 1] = (source[i, 1] + 1100 * np.sin(a))[None]
        pixels[i, :, :, 2] = zs[:, None]
    return source, pixels


@pytest.mark.parametrize("case", ["flat", "arc_gapped", "shared"])
def test_atb_is_exact_adjoint(case):
    geo = make_geo(nvox=40, off_det=(1.0, 2.0))
    geo.offOrigin = np.array([1.5, -2.0, 3.0])
    rng = np.random.default_rng(3)
    if case == "flat":
        surf = flat_surface(geo, np.linspace(0, 2 * np.pi, 5, endpoint=False))
    elif case == "arc_gapped":
        source, pixels = arc_surface()
        mask = np.ones(pixels.shape[1:3], dtype=bool)
        mask[:, 20:24] = False
        mask[2, :] = False
        surf = DetectorSurfaceGeometry(source, pixels, mask)
    else:
        flat = flat_surface(geo, np.array([0.3]))
        surf = DetectorSurfaceGeometry(flat.source_positions_mm, flat.pixel_centers_mm[0])
    x = rng.random(tuple(geo.nVoxel)).astype(np.float32)
    y = rng.random(surf.valid_mask.shape).astype(np.float32)
    ax = Ax_surface(x, geo, surf)
    aty = Atb_surface(y, geo, surf)
    lhs = float(np.sum(ax.astype(np.float64) * y))
    rhs = float(np.sum(x.astype(np.float64) * aty))
    assert abs(lhs) > 1
    assert abs(lhs - rhs) < 1e-4 * abs(lhs), (lhs, rhs)


def test_atb_ignores_masked_samples():
    geo = make_geo(nvox=32)
    surf = flat_surface(geo, np.linspace(0, np.pi, 3, endpoint=False))
    mask = np.ones(surf.valid_mask.shape[1:], dtype=bool)
    mask[:, :20] = False
    gapped = DetectorSurfaceGeometry(surf.source_positions_mm, surf.pixel_centers_mm, mask)
    y = np.ones(surf.valid_mask.shape, dtype=np.float32)
    y_zeroed = y.copy()
    y_zeroed[:, :, :20] = 0
    np.testing.assert_allclose(
        Atb_surface(y, geo, gapped), Atb_surface(y_zeroed, geo, surf), rtol=1e-5, atol=1e-5)

