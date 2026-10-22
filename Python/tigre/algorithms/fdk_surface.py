"""Approximate FDK reconstruction for a circular orbit with a curved/modular detector.

The detector is described by a DetectorSurfaceGeometry (per-pixel positions). The
algorithm is the equiangular fan-beam FDK (Kak & Slaney, 3.5) with a Feldkamp cone
weight, generalised in three ways:

* the fan angle of every column is computed from the real pixel positions, so any
  curved or piecewise-linear detector with arbitrary column spacing works;
* the filter is applied as an explicit matrix on the (possibly non-uniform) fan-angle
  grid. Gaps (invalid columns) are bridged by trapezoid quadrature, i.e. the missing
  samples are implicitly interpolated linearly. This is an approximation; use the
  iterative *_surface algorithms when the gaps are wide;
* scans shorter than 2*pi are handled with Parker weights.

Requirements (checked, ValueError / NotImplementedError otherwise):
circular source orbit about the z axis at constant height, a detector extruded along
z (pixel x/y independent of the row, z independent of the column and view), and the
same valid columns in every view (whole-column gaps only).
"""
import numpy as np

from tigre.utilities.Ax_surface import (
    _volume_params, _fdk_surface_ext, _devices, _slabs, _slab_offset, _run_slabs)


class _Fan:
    """Per-column / per-row tables of a validated circular-orbit surface."""
    pass


def _analyse(surface):
    src = np.asarray(surface.source_positions_mm, dtype=np.float64)
    pix = np.asarray(surface.pixel_centers_mm, dtype=np.float64)
    mask = np.asarray(surface.valid_mask)
    nv, nrows, ncols = mask.shape

    radius = np.hypot(src[:, 0], src[:, 1])
    if radius.min() <= 0 or np.ptp(radius) > 1e-4 * radius.mean() or np.ptp(src[:, 2]) > 1e-3:
        raise NotImplementedError(
            "fdk_surface needs a circular source orbit about z at constant height")

    col_ok = mask.all(axis=(0, 1))
    if not np.array_equal(mask.any(axis=(0, 1)), col_ok):
        raise NotImplementedError(
            "fdk_surface needs the same valid columns in every view and row (whole-column gaps only)")
    cols = np.flatnonzero(col_ok)
    if cols.size < 2:
        raise ValueError("fdk_surface needs at least two valid columns")

    p = pix[:, :, cols, :]
    h = p[..., :2] - src[:, None, None, :2]
    tol = 1e-3 * radius.mean() * 1e-2
    if np.ptp(h, axis=1).max() > tol or np.ptp(p[..., 2], axis=2).max() > tol \
            or np.ptp(p[..., 2], axis=0).max() > tol:
        raise NotImplementedError(
            "fdk_surface needs an extruded detector: pixel x/y independent of the row and "
            "z independent of the column and view")

    c = -src[:, :2] / radius[:, None]
    h0 = h[:, 0]
    gamma = np.arctan2(c[:, None, 0] * h0[..., 1] - c[:, None, 1] * h0[..., 0],
                       c[:, None, 0] * h0[..., 0] + c[:, None, 1] * h0[..., 1])
    rho = np.hypot(h0[..., 0], h0[..., 1])
    if np.ptp(gamma, axis=0).max() > 2e-5 or np.ptp(rho, axis=0).max() > 1e-4 * rho.mean():
        raise NotImplementedError("fdk_surface needs identical fan-angle tables in every view")
    gamma, rho = gamma[0], rho[0]
    zrow = p[0, :, 0, 2]

    d = np.diff(gamma)
    if np.all(d < 0):
        cols = cols[::-1]
        gamma, rho = gamma[::-1], rho[::-1]
    elif not np.all(d > 0):
        raise ValueError("fan angles must be strictly monotone along the columns")
    flip_rows = nrows > 1 and zrow[-1] < zrow[0]
    if flip_rows:
        zrow = zrow[::-1]
    if nrows > 1 and not np.all(np.diff(zrow) > 0):
        raise ValueError("detector rows must have strictly monotone z")

    fan = _Fan()
    fan.cols, fan.flip_rows = cols, flip_rows
    fan.gamma, fan.rho, fan.zrow = gamma, rho, zrow
    fan.radius, fan.sz, fan.src = radius.mean(), src[0, 2], src
    return fan


def _interp_matrix(src_x, dst_x):
    """Linear interpolation matrix (len(dst_x), len(src_x)) from samples at src_x."""
    idx = np.clip(np.searchsorted(src_x, dst_x) - 1, 0, src_x.size - 2)
    t = np.clip((dst_x - src_x[idx]) / (src_x[idx + 1] - src_x[idx]), 0.0, 1.0)
    out = np.zeros((dst_x.size, src_x.size))
    rows = np.arange(dst_x.size)
    out[rows, idx] = 1.0 - t
    out[rows, idx + 1] += t
    return out


def _ramp_matrix(gamma, cutoff, window):
    """Filter matrix K so that Q = P @ K.T, with equiangular weighting folded in.

    The data are linearly resampled (bridging gaps) onto a uniform fan-angle grid,
    filtered there, and linearly resampled back at the real column angles.
    """
    delta = np.min(np.diff(gamma))
    m = int(np.ceil((gamma[-1] - gamma[0]) / delta - 1e-9)) + 1
    grid = gamma[0] + delta * np.arange(m)
    wc = cutoff / (2.0 * delta)

    def ramp(t):
        return wc ** 2 * (2.0 * np.sinc(2.0 * wc * t) - np.sinc(wc * t) ** 2)

    d = grid[:, None] - grid[None, :]
    h = ramp(d)
    if window == "hann":
        tau = 1.0 / (2.0 * wc)
        h = 0.5 * h + 0.25 * (ramp(d + tau) + ramp(d - tau))
    elif window != "ram-lak":
        raise ValueError("filter must be 'ram-lak' or 'hann'")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(d == 0, 1.0, d / np.sin(d))
    k_uniform = 0.5 * ratio ** 2 * h * delta
    return _interp_matrix(grid, gamma) @ k_uniform @ _interp_matrix(gamma, grid)



def _view_weights(beta):
    """Return (beta_relative >= 0 along the sweep, d_beta per view, sweep direction, span, is_full)."""
    beta = np.unwrap(beta)
    nv = beta.size
    if nv < 2:
        raise ValueError("fdk_surface needs at least two views")
    sweep = beta[-1] - beta[0]
    direction = 1.0 if sweep >= 0 else -1.0
    rel = direction * (beta - beta[0])
    steps = np.diff(rel)
    if np.any(steps <= 0):
        raise ValueError("the source must move monotonically along the orbit")
    span = rel[-1]
    tol = 1e-3
    dbeta = np.zeros(nv)
    if span >= 2 * np.pi - tol:
        dbeta[:-1] += steps / 2
        dbeta[1:] += steps / 2
        return rel, dbeta, direction, span, True
    if span + np.median(steps) >= 2 * np.pi - tol:
        return rel, np.full(nv, np.median(steps)), direction, span, True
    dbeta[:-1] += steps / 2
    dbeta[1:] += steps / 2
    return rel, dbeta, direction, span, False


def parker_weights(beta_rel, gamma_tilde, gamma_max):
    """Parker weights for views beta_rel (n,) and columns gamma_tilde (m,); returns (n, m).

    gamma_tilde is the fan angle in the Parker convention: conjugate rays satisfy
    beta' = beta + pi + 2 gamma_tilde, gamma' = -gamma_tilde, with beta increasing along
    the scan. The weights of conjugate rays sum to 1.
    """
    eps = 1e-6
    b = np.asarray(beta_rel, dtype=np.float64)[:, None]
    g = np.minimum(np.asarray(gamma_tilde, dtype=np.float64), gamma_max - eps)[None, :]
    g = np.maximum(g, -gamma_max + eps)
    a = gamma_max - g
    w1 = np.sin(np.pi / 4 * b / a) ** 2
    w3 = np.sin(np.pi / 4 * (np.pi + 2 * gamma_max - b) / (gamma_max + g)) ** 2
    return np.where(b <= 2 * a, w1, np.where(b <= np.pi - 2 * g, 1.0, w3))


def fdk_surface(proj, geo, surface, filter="ram-lak", cutoff=1.0, parker=True,
                max_short_deficiency=np.radians(2.0), gpuids=None, n_slabs=None):
    """FDK-style reconstruction for a circular orbit and an arbitrary extruded detector.

    proj: (views, rows, cols) measured data aligned with ``surface``.
    geo: tigre geometry (only the volume description is used).
    filter: 'ram-lak' or 'hann'; cutoff: fraction of the Nyquist frequency.
    parker: apply Parker weights when the scan is shorter than 2*pi.
    gpuids: :class:`tigre.utilities.gpu.GpuIds` (default all GPUs); the volume is
    split into z-slabs (``n_slabs``, default one per device) backprojected concurrently.
    Returns the volume (nz, ny, nx), float32.

    Approximations: cone-beam error grows with the cone angle (as in FDK), and
    interior gap columns are bridged by linear interpolation in the filter.
    """
    proj = np.asarray(proj, dtype=np.float32)
    nv, nrows, ncols = surface.valid_mask.shape
    if proj.shape != (nv, nrows, ncols):
        raise ValueError(f"proj must have shape {(nv, nrows, ncols)}; got {proj.shape}")
    if not np.isfinite(proj[surface.valid_mask]).all():
        raise ValueError("proj contains NaN or Inf at valid samples")
    if not (np.isfinite(cutoff) and 0 < cutoff <= 1):
        raise ValueError(f"cutoff must be in (0, 1], got {cutoff!r}")
    fan = _analyse(surface)
    gamma, rho, zrow = fan.gamma, fan.rho, fan.zrow
    data = proj[:, :, fan.cols]
    if fan.flip_rows:
        data = data[:, ::-1, :]

    beta = np.arctan2(fan.src[:, 1], fan.src[:, 0])
    rel, dbeta, direction, span, full = _view_weights(beta)

    pre = (fan.radius * np.cos(gamma))[None, :] * (
        rho[None, :] / np.sqrt(rho[None, :] ** 2 + (zrow[:, None] - fan.sz) ** 2))
    kernel = _ramp_matrix(gamma, cutoff, filter).astype(np.float32)

    view_weight = dbeta.copy()
    if full:
        col_weight = np.ones((nv, gamma.size))
    elif not parker:
        col_weight = np.ones((nv, gamma.size))
        view_weight *= 2.0
    else:
        gamma_m = 0.5 * (span - np.pi)
        deficit = np.abs(gamma).max() - gamma_m
        if deficit > max_short_deficiency:
            raise ValueError(
                f"scan span {np.degrees(span):.2f} deg is {np.degrees(deficit):.2f} deg short of "
                f"the {np.degrees(np.pi + 2 * np.abs(gamma).max()):.2f} deg needed; "
                "pass a larger max_short_deficiency to accept it")
        gamma_m = max(gamma_m, 1e-3)
        col_weight = 2.0 * parker_weights(rel, direction * gamma, gamma_m)

    shape_zyx = tuple(int(v) for v in geo.nVoxel)
    voxel, offset = _volume_params(geo)
    gamma32 = gamma.astype(np.float32)
    rho32 = rho.astype(np.float32)
    z32 = zrow.astype(np.float32)
    src32 = np.ascontiguousarray(fan.src, dtype=np.float32)

    slabs = _slabs(shape_zyx[0], _devices(gpuids), n_slabs)
    slab_offsets = [_slab_offset(offset, voxel, shape_zyx[0], lo, hi) for lo, hi, _ in slabs]
    chunk = max(1, int(2e8 // (nrows * gamma.size * 4)))
    volume = np.zeros(shape_zyx, dtype=np.float32)
    for lo in range(0, nv, chunk):
        hi = min(nv, lo + chunk)
        q = data[lo:hi] * pre[None].astype(np.float32)
        q = (q.reshape(-1, gamma.size) @ kernel.T).reshape(hi - lo, nrows, gamma.size)
        q *= (col_weight[lo:hi] * view_weight[lo:hi, None])[:, None, :].astype(np.float32)
        q = np.ascontiguousarray(q, dtype=np.float32)

        def run(i, lo=lo, hi=hi, q=q):
            z0, z1, dev = slabs[i]
            volume[z0:z1] += _fdk_surface_ext(
                q, (z1 - z0,) + shape_zyx[1:], voxel, slab_offsets[i],
                src32[lo:hi], gamma32, rho32, z32, dev)

        _run_slabs(list(range(len(slabs))), run)
    return volume
