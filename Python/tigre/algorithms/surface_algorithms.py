"""Iterative reconstruction on arbitrary detector surfaces.

These use :func:`Ax_surface` / :func:`Atb_surface`, which are an exact adjoint
pair, so CGLS is well defined. Masked (missing) samples are ignored: they are
zeroed in the data and never contribute to the back projection.
"""
import numpy as np
from functools import partial

from tigre.utilities.Ax_surface import Ax_surface as _Ax, Atb_surface as _Atb


def _prepare(proj, surface):
    proj = np.ascontiguousarray(proj, dtype=np.float32)
    if proj.shape != surface.valid_mask.shape:
        raise ValueError(
            f"proj must have shape {surface.valid_mask.shape}, got {proj.shape}")
    # Non-finite values at masked samples are harmless: they are zeroed below.
    if not np.isfinite(proj[surface.valid_mask]).all():
        raise ValueError("proj contains NaN or Inf at valid samples")
    return np.where(surface.valid_mask, proj, 0).astype(np.float32)


def cgls_surface(proj, geo, surface, niter=20, init=None, gpuids=None, return_residuals=False):
    """Conjugate gradient least squares on a detector surface.

    ``gpuids`` is a :class:`tigre.utilities.gpu.GpuIds` (default all GPUs).
    """
    Ax_surface, Atb_surface = partial(_Ax, gpuids=gpuids), partial(_Atb, gpuids=gpuids)
    b = _prepare(proj, surface)
    shape = tuple(int(v) for v in geo.nVoxel)
    x = np.zeros(shape, np.float32) if init is None else np.array(init, np.float32)
    r = b - (Ax_surface(x, geo, surface) if init is not None else 0)
    p = Atb_surface(r.astype(np.float32), geo, surface)
    gamma = float(np.sum(p.astype(np.float64) ** 2))
    residuals = []
    for _ in range(int(niter)):
        if gamma == 0.0:
            break
        q = Ax_surface(p, geo, surface)
        qq = float(np.sum(q.astype(np.float64) ** 2))
        if qq == 0.0:
            break
        alpha = gamma / qq
        x += np.float32(alpha) * p
        r = r - np.float32(alpha) * q
        residuals.append(float(np.linalg.norm(r)))
        s = Atb_surface(r.astype(np.float32), geo, surface)
        gamma_new = float(np.sum(s.astype(np.float64) ** 2))
        p = s + np.float32(gamma_new / gamma) * p
        gamma = gamma_new
    return (x, np.array(residuals)) if return_residuals else x


def os_sart_surface(proj, geo, surface, niter=10, blocksize=20, init=None, lmbda=1.0,
                    nonneg=False, momentum=False, gpuids=None):
    """Ordered-subset SART on a detector surface.

    Views are split into interleaved blocks. With ``momentum=True`` the iterate
    is extrapolated before every pass with the Nesterov schedule used by
    ``tigre.algorithms.fast_os_sart`` (an acceleration heuristic, not a
    convergent method). The per-block back-projection normalisation is
    recomputed each time to keep memory at one extra volume. ``gpuids`` is a
    :class:`tigre.utilities.gpu.GpuIds` (default all GPUs).
    """
    Ax_surface, Atb_surface = partial(_Ax, gpuids=gpuids), partial(_Atb, gpuids=gpuids)
    b = _prepare(proj, surface)
    if int(blocksize) != blocksize or blocksize < 1:
        raise ValueError(f"blocksize must be a positive integer, got {blocksize!r}")
    shape = tuple(int(v) for v in geo.nVoxel)
    x = np.zeros(shape, np.float32) if init is None else np.array(init, np.float32)
    n_views = b.shape[0]
    n_blocks = max(1, -(-n_views // int(blocksize)))
    blocks = [np.arange(k, n_views, n_blocks) for k in range(n_blocks)]
    ones_vol = np.ones(shape, np.float32)

    subs, weights = [], []
    for idx in blocks:
        sub = surface.subset(idx)
        row_sum = Ax_surface(ones_vol, geo, sub)
        w = np.zeros_like(row_sum)
        np.divide(1.0, row_sum, out=w, where=row_sum > 1e-6)
        subs.append(sub)
        weights.append(w)

    t = 1.0
    x_old = x.copy()
    for _ in range(int(niter)):
        if momentum:
            t_old, t = t, (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            beta = np.float32((t_old - 1.0) / t)
            for k in range(x.shape[0]):
                saved = x[k].copy()
                x[k] += beta * (x[k] - x_old[k])
                x_old[k] = saved
        for idx, sub, w in zip(blocks, subs, weights):
            resid = (b[idx] - Ax_surface(x, geo, sub)) * w
            col_sum = Atb_surface(sub.valid_mask.astype(np.float32), geo, sub)
            v = np.zeros_like(col_sum)
            np.divide(1.0, col_sum, out=v, where=col_sum > 1e-6)
            x += np.float32(lmbda) * v * Atb_surface(resid.astype(np.float32), geo, sub)
            if nonneg:
                np.maximum(x, 0, out=x)
    return x


def fast_os_sart_surface(proj, geo, surface, niter=10, **kwargs):
    """OS-SART with Nesterov momentum on a detector surface."""
    return os_sart_surface(proj, geo, surface, niter=niter, momentum=True, **kwargs)


def sirt_surface(proj, geo, surface, niter=50, init=None, lmbda=1.0, gpuids=None,
                 nonneg=False):
    """SIRT with row/column sum normalization from the exact adjoint."""
    Ax_surface, Atb_surface = partial(_Ax, gpuids=gpuids), partial(_Atb, gpuids=gpuids)
    b = _prepare(proj, surface)
    shape = tuple(int(v) for v in geo.nVoxel)
    x = np.zeros(shape, np.float32) if init is None else np.array(init, np.float32)

    ones_vol = np.ones(shape, np.float32)
    row_sum = Ax_surface(ones_vol, geo, surface)
    w = np.zeros_like(row_sum)
    np.divide(1.0, row_sum, out=w, where=row_sum > 1e-6)
    col_sum = Atb_surface(surface.valid_mask.astype(np.float32), geo, surface)
    v = np.zeros_like(col_sum)
    np.divide(1.0, col_sum, out=v, where=col_sum > 1e-6)

    for _ in range(int(niter)):
        resid = (b - Ax_surface(x, geo, surface)) * w
        x += np.float32(lmbda) * v * Atb_surface(resid.astype(np.float32), geo, surface)
        if nonneg:
            np.maximum(x, 0, out=x)
    return x
