"""Approximate FDK reconstruction for circular or experimental per-view surfaces.

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
same valid columns in every view (whole-column gaps only). ``allow_approximate=True``
opts into a separate, unvalidated per-view filtered-backprojection path.
"""
import warnings

import numpy as np

from tigre.utilities.Ax_surface import (
    _volume_params, _fdk_surface_ext, _devices, _slabs, _slab_offset, _run_slabs)
from tigre.utilities.Ax_surface import Atb_surface


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


def _surface_fan_angles(surface):
    """Return per-pixel fan angles and monotone column order for each view/row."""
    source = surface.source_positions_mm.astype(np.float64)
    pixels = surface.pixel_centers_mm
    mask = surface.valid_mask
    nv, nr, nc = mask.shape
    gamma = np.empty((nv, nr, nc), dtype=np.float32)
    orders = np.empty((nv, nr), dtype=object)

    for view in range(nv):
        view_pixels = pixels[view].astype(np.float64)
        rays = view_pixels - source[view]
        valid_rays = rays[mask[view]]
        if valid_rays.size == 0:
            raise ValueError(f"view {view} has no valid detector pixels")
        axis = valid_rays.mean(axis=0)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 0:
            raise ValueError(f"view {view} has no defined detector central ray")
        axis /= axis_norm

        column_direction = np.mean(
            view_pixels[:, -1] - view_pixels[:, 0], axis=0)
        transverse = column_direction - np.dot(column_direction, axis) * axis
        transverse_norm = np.linalg.norm(transverse)
        if transverse_norm <= 1e-8:
            raise ValueError(f"view {view} has no defined detector column direction")
        transverse /= transverse_norm

        forward = np.einsum("...i,i->...", rays, axis)
        lateral = np.einsum("...i,i->...", rays, transverse)
        if np.any(forward[mask[view]] <= 0):
            raise ValueError(f"view {view} contains detector pixels behind its central ray")
        gamma[view] = np.arctan2(lateral, forward)

        for row in range(nr):
            columns = np.flatnonzero(mask[view, row])
            if columns.size < 2:
                raise ValueError(
                    f"view {view}, row {row} needs at least two valid columns for filtering")
            delta = np.diff(gamma[view, row, columns])
            if np.all(delta > 0):
                orders[view, row] = columns
            elif np.all(delta < 0):
                orders[view, row] = columns[::-1]
            else:
                raise NotImplementedError(
                    f"view {view}, row {row} has non-monotone fan angles; "
                    "a single 1D FDK filter cannot represent this detector row")
    return gamma, orders


def _filter_surface_row(values, gamma, valid_columns, cutoff, window):
    """Filter one row after resampling its valid samples to uniform fan angles."""
    columns = valid_columns
    angles = gamma[columns]
    data = values[columns]
    if angles[0] > angles[-1]:
        angles, data = angles[::-1], data[::-1]
    spacing = float(np.min(np.diff(angles)))
    count = int(np.ceil((angles[-1] - angles[0]) / spacing - 1e-9)) + 1
    grid = angles[0] + spacing * np.arange(count)
    uniform = np.interp(grid, angles, data)

    nfft = 1
    while nfft < 2 * count:
        nfft <<= 1
    frequency = np.fft.rfftfreq(nfft, d=spacing)
    nyquist = 0.5 / spacing
    response = 2.0 * frequency
    cutoff_frequency = cutoff * nyquist
    if window == "hann":
        response *= 0.5 * (1.0 + np.cos(np.pi * frequency / cutoff_frequency))
        response[frequency > cutoff_frequency] = 0.0
    elif window == "ram-lak":
        response[frequency > cutoff_frequency] = 0.0
    else:
        raise ValueError("filter must be 'ram-lak' or 'hann'")
    filtered = np.fft.irfft(np.fft.rfft(uniform, nfft) * response, nfft)[:count]
    result = np.zeros(values.shape, dtype=np.float64)
    result[columns] = np.interp(angles if columns[0] < columns[-1] else angles[::-1],
                                grid, filtered)
    return result


def _fdk_surface_per_view(proj, geo, surface, cutoff, filter, parker,
                          max_short_deficiency, gpuids, n_slabs):
    """Experimental filtered backprojection for per-view detector surfaces."""
    warnings.warn(
        "Per-view fdk_surface is experimental: it uses per-view fan-angle filtering "
        "and a ray-driven surface backprojection. Reconstruction quality is not "
        "validated for arbitrary source or detector trajectories.",
        RuntimeWarning, stacklevel=3)
    source = surface.source_positions_mm.astype(np.float64)
    pixels = surface.pixel_centers_mm
    mask = surface.valid_mask
    nv, nr, nc = mask.shape
    gamma, orders = _surface_fan_angles(surface)

    center = np.asarray(geo.offOrigin, dtype=np.float64)
    if center.ndim == 2:
        if not np.allclose(center, center[0]):
            raise ValueError("offOrigin must be identical for all views")
        center = center[0]
    beta = np.unwrap(np.arctan2(source[:, 1] - center[1],
                                source[:, 0] - center[0]))
    if nv < 2 or np.any(np.diff(beta) == 0) or not (
            np.all(np.diff(beta) > 0) or np.all(np.diff(beta) < 0)):
        raise ValueError(
            "experimental fdk_surface needs at least two views with monotone "
            "source azimuth around the volume centre")
    rel, dbeta, direction, span, full = _view_weights(beta)

    radial = np.hypot(source[:, 0] - center[0], source[:, 1] - center[1])
    circular = (np.ptp(radial) <= 1e-4 * radial.mean()
                and np.ptp(source[:, 2]) <= 1e-3)
    apply_parker = bool(parker and not full and circular)
    if parker and not full and not circular:
        warnings.warn(
            "Parker weighting was skipped because the source trajectory is not "
            "a constant-height circular orbit.",
            RuntimeWarning, stacklevel=3)

    gamma_max = max(float(np.max(np.abs(gamma[v][mask[v]]))) for v in range(nv))
    col_weights = None
    if apply_parker:
        gamma_m = 0.5 * (span - np.pi)
        deficiency = gamma_max - gamma_m
        if deficiency > max_short_deficiency:
            raise ValueError(
                f"scan is {np.degrees(deficiency):.2f} deg short of the "
                f"{np.degrees(np.pi + 2 * gamma_max):.2f} deg needed; "
                "pass a larger max_short_deficiency to accept it")
        gamma_m = max(gamma_m, 1e-3)
        col_weights = np.zeros_like(gamma)
        for view in range(nv):
            for row in range(nr):
                col_weights[view, row] = 2.0 * parker_weights(
                    rel[view:view + 1], direction * gamma[view, row], gamma_m)[0]

    filtered = np.zeros_like(proj, dtype=np.float32)
    for view in range(nv):
        view_pixels = pixels[view].astype(np.float64)
        column_step = np.gradient(view_pixels, axis=1)
        if nr > 1:
            row_step = np.gradient(view_pixels, axis=0)
            pixel_area = np.linalg.norm(np.cross(column_step, row_step), axis=-1)
        else:
            pixel_area = np.linalg.norm(column_step, axis=-1)
        rays = view_pixels - source[view]
        ray_length = np.linalg.norm(rays, axis=-1)
        central = view_pixels.mean(axis=(0, 1)) - source[view]
        central_length = np.linalg.norm(central)
        if central_length <= 0:
            raise ValueError(f"view {view} has no defined detector central ray")
        central /= central_length
        if np.any(ray_length[mask[view]] <= 0):
            raise ValueError(f"view {view} has a valid detector pixel at its source")
        cosine = np.maximum(
            np.einsum("rcd,d->rc", rays / np.maximum(ray_length[..., None], 1e-12),
                      central), 0.0)
        weights = (pixel_area * cosine / np.maximum(ray_length ** 2, 1e-12)
                   * dbeta[view])
        if col_weights is not None:
            weights *= col_weights[view]
        elif not full and circular:
            weights *= 2.0
        for row in range(nr):
            columns = orders[view, row]
            row_filtered = _filter_surface_row(
                proj[view, row], gamma[view, row], columns, cutoff, filter)
            filtered[view, row] = row_filtered * weights[row]
    filtered[~mask] = 0.0
    return Atb_surface(np.ascontiguousarray(filtered, dtype=np.float32), geo, surface,
                       gpuids=gpuids, n_slabs=n_slabs)


def fdk_surface(proj, geo, surface, filter="ram-lak", cutoff=1.0, parker=True,
                max_short_deficiency=np.radians(2.0), gpuids=None, n_slabs=None,
                allow_approximate=False):
    """FDK-style reconstruction for a circular orbit and an arbitrary extruded detector.

    proj: (views, rows, cols) measured data aligned with ``surface``.
    geo: tigre geometry (only the volume description is used).
    filter: 'ram-lak' or 'hann'; cutoff: fraction of the Nyquist frequency.
    parker: apply Parker weights when the scan is shorter than 2*pi.
    gpuids: :class:`tigre.utilities.gpu.GpuIds` (default all GPUs); the volume is
    split into z-slabs (``n_slabs``, default one per device) backprojected concurrently.
    allow_approximate: opt in to an experimental per-view surface path when the
    detector or source does not meet the circular-orbit requirements.
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
    try:
        fan = _analyse(surface)
    except NotImplementedError:
        if not allow_approximate:
            raise
        return _fdk_surface_per_view(
            proj, geo, surface, cutoff, filter, parker,
            max_short_deficiency, gpuids, n_slabs)
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
