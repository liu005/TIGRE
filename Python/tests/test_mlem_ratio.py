"""MLEM's multiplicative update must stay finite on data that a real scan
produces: noisy, slightly negative in air, warm-started, with a support mask.

Root cause fixed here (2026-09-10): `den[den == 0] = inf` guards only an exact
zero; under the multiplicative update the air voxels decay to float32
denormals, a ray seeing only those has den ~1e-40, proj/den ~1e38, the
backprojection sums to inf and res*inf is inf / NaN. A real 1024^3 run came
back [nan, nan]."""
import numpy as np
import pytest

import tigre
import tigre.algorithms as algs
from tigre.algorithms.statistical_algorithms import mlem_ratio, MLEM_DEN_FLOOR, MLEM_RATIO_MAX


def test_ratio_is_finite_for_denormal_den():
    proj = np.full((4, 8, 8), 0.05, dtype=np.float32)
    den = np.full((4, 8, 8), 1e-40, dtype=np.float32)       # the textbook guard misses this
    naive = proj / np.where(den == 0, np.inf, den)
    assert not np.isfinite(naive).all() or naive.max() > 1e37  # what the old expression did
    out = mlem_ratio(proj, den.copy())
    assert np.isfinite(out).all() and out.max() == 0.0        # no information -> empty ray


def test_ratio_unchanged_where_den_is_real_and_negative_data_clipped():
    rng = np.random.default_rng(0)
    proj = rng.uniform(-0.05, 3.0, (3, 16, 16)).astype(np.float32)
    den = rng.uniform(0.5, 3.0, (3, 16, 16)).astype(np.float32)
    out = mlem_ratio(proj, den.copy())
    exp = np.clip(proj / den, 0.0, MLEM_RATIO_MAX)
    np.testing.assert_allclose(out, exp, rtol=1e-6)
    assert out.min() == 0.0 and (proj < 0).any()


def test_ratio_floor_is_negligible_for_real_rays():
    # one millionth of an attenuation length: any voxel of a real object seen
    # by one sample is orders of magnitude above it
    assert MLEM_DEN_FLOOR <= 1e-5


def test_ratio_is_in_place_and_allocates_no_full_buffer():
    proj = np.ones((2, 4, 4), dtype=np.float32)
    den = np.full((2, 4, 4), 2.0, dtype=np.float32)
    out = mlem_ratio(proj, den)
    assert out is den and np.allclose(den, 0.5)


@pytest.mark.gpu
def test_mlem_finite_on_noisy_negative_data_with_support_and_collapsed_warm_start():
    geo = tigre.geometry_default(high_resolution=False)
    angles = np.linspace(0, 2 * np.pi, 60, dtype=np.float32)
    from tigre.utilities import sample_loader
    head = (sample_loader.load_head_phantom(geo.nVoxel) * 0.2).astype(np.float32)
    rng = np.random.default_rng(1)
    proj = (tigre.Ax(head, geo, angles) + rng.normal(0, 0.05, (60,) + tuple(geo.nDetector))).astype(np.float32)
    n = geo.nVoxel
    _, yy, xx = np.ogrid[:n[0], :n[1], :n[2]]
    support = np.broadcast_to(np.hypot(yy - (n[1] - 1) / 2, xx - (n[2] - 1) / 2) <= min(n[1], n[2]) / 2 * 0.98, tuple(n))
    init = head.copy()
    init[head == 0] = np.float32(1e-44)                       # air already collapsed to denormals
    vol = algs.mlem(proj, geo, angles, 8, init=init, support=support)
    assert np.isfinite(vol).all()
    assert vol[~support].max() == 0.0
    assert np.sqrt(np.mean((vol - head) ** 2)) < 0.01


def test_ratio_is_capped_for_near_empty_rays():
    proj = np.full((2, 4, 4), 0.05, dtype=np.float32)          # positive noise on an empty ray
    den = np.full((2, 4, 4), 1e-5, dtype=np.float32)           # above the floor, but nearly empty
    out = mlem_ratio(proj, den.copy())
    assert out.max() == MLEM_RATIO_MAX and 1.0 < MLEM_RATIO_MAX <= 100.0


@pytest.mark.gpu
def test_mlem_does_not_explode_on_empty_rays_with_positive_noise():
    """Air inside the support + noisy log data: without the cap the air voxels
    grow by the ratio every iteration (real run: max 1.3e15)."""
    geo = tigre.geometry_default(high_resolution=False)
    angles = np.linspace(0, 2 * np.pi, 60, dtype=np.float32)
    from tigre.utilities import sample_loader
    head = (sample_loader.load_head_phantom(geo.nVoxel) * 0.2).astype(np.float32)
    rng = np.random.default_rng(2)
    proj = tigre.Ax(head, geo, angles)
    proj = (proj + np.abs(rng.normal(0, 0.05, proj.shape))).astype(np.float32)   # one-sided noise: worst case
    vol = algs.mlem(proj, geo, angles, 30, init=np.maximum(head, 1e-6))
    assert np.isfinite(vol).all()
    assert vol.max() < 5 * head.max()
