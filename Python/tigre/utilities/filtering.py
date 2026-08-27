from __future__ import division
from __future__ import print_function
from tigre.utilities.parkerweight import parkerweight
import numpy as np
import cupy as cp

import warnings

class device_manager:
    def __init__(self, arr):
        try:
            self.device = arr.device
        except AttributeError:
            self.device = "cpu"
            
    def __enter__(self):
        return self.device
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

def per_view_dbeta(angles, dbeta=None):
    """Per-view angular quadrature weight dbeta_i, shape (n,) float64 (host).

    dbeta_i is the rate the viewing direction turns about the isocenter per
    view - the correct integration weight for the FDK angular sum. Priority:

    1. explicit ``dbeta`` argument (per-view array or scalar) - for exotic
       trajectories, compute from true source positions as
       gradient(unwrap(arctan2(src_y, src_x)));
    2. derived from ``angles``: |gradient(unwrap(first Euler column))| -
       exact for any monotone in-plane trajectory (uniform or not);
    3. degenerate fallback (single view / constant angles): 2*pi/n, the
       legacy assumption.
    """
    a = np.asarray(angles, dtype=float)
    if a.ndim > 1:
        a = a[:, 0]
    n = max(1, a.size)
    if dbeta is not None:
        db = np.abs(np.asarray(dbeta, dtype=float)).ravel()
        return np.full(n, db[0]) if db.size == 1 else db
    if n < 2:
        return np.full(n, 2.0 * np.pi / n)
    db = np.abs(np.gradient(np.unwrap(a)))
    if not np.all(np.isfinite(db)) or np.all(db == 0):
        return np.full(n, 2.0 * np.pi / n)
    return db


def per_view_scale(geo, angles, dbeta=None, redundancy_factor=None):
    """Per-view ramp-filter scale, shape (n,) float64 (host).

        scale_i = K * (DSD[i]/DSO[i]) * dbeta_i / (4 * dDetector[1])

    generalizing legacy TIGRE's single scalar
        (DSD[0]/DSO[0]) * (2*pi/n) / (4 * dDetector[0]),
    which silently assumed constant magnification and a uniformly-sampled
    full-2*pi circular scan. For that case the per-view weights are all equal
    to the legacy scalar (for a square-pixel detector), so results are
    unchanged; short scans get their true angular weight and variable-DSD/DSO
    (non-circular) trajectories get per-view magnification. dDetector[1]
    (u/column pitch) is the ramp-filter sample spacing - the filter runs
    along u; legacy's dDetector[0] was an anisotropic-detector bug (fixed
    upstream the same way).

    K (``redundancy_factor``) accounts for ray multiplicity: FDK's classic /4
    embeds a 1/2 for the full-circle double coverage (each ray measured at
    beta and beta+pi). A short scan covers each ray once (with Parker
    weighting normalizing the fan-overlap rays to single coverage), so it
    needs K = 2; a full circle needs K = 1. K = 2/c for coverage
    multiplicity c. Pinned empirically on a uniform-cylinder phantom
    (2026-07-14): full-circle interior mean 1.001, short-scan with true
    per-view dbeta and K=1 gave 0.509 -> K=2 restores parity. When None
    (default), K is derived from the scan arc: 1 for a (near-)full 2*pi
    scan, 2 otherwise (printed when it fires).
    """
    a = np.asarray(angles, dtype=float)
    n = a.shape[0]
    DSD = np.atleast_1d(np.asarray(geo.DSD, dtype=float)).ravel()
    DSO = np.atleast_1d(np.asarray(geo.DSO, dtype=float)).ravel()
    if DSD.size < n:
        DSD = np.full(n, DSD[0])
    if DSO.size < n:
        DSO = np.full(n, DSO[0])
    db = per_view_dbeta(angles, dbeta=dbeta)

    K = redundancy_factor
    if K is None:
        from tigre.utilities.redundancy_weighting import scan_arc
        arc = scan_arc(angles)
        a1 = a[:, 0] if a.ndim > 1 else a
        step = float(np.median(np.abs(np.diff(np.unwrap(a1))))) if a1.size > 1 else 0.0
        if arc is not None and arc < 2.0 * np.pi - max(2.0 * step, 1e-6):
            K = 2.0
            print(f"per_view_scale: short scan (arc {np.degrees(arc):.1f} deg) "
                  "-> redundancy factor K=2 (single ray coverage; pair with "
                  "Parker weighting for the fan-overlap band)")
        else:
            K = 1.0

    return K * (DSD[:n] / DSO[:n]) * db / (4.0 * geo.dDetector[1])


# TODO: Fix parker
def filtering(proj, geo, angles, parker, verbose=False, dbeta=None, legacy_scale=False,
              redundancy_factor=None):
    xp = cp.get_array_module(proj)
    if xp.__name__ == "cupy":
        from cupyx.scipy.fft import fft, ifft
        device = proj.device
    else:
        from scipy.fft import fft, ifft
        device = None

    # Apply Parker weighting if needed
    if parker:
        proj=parkerweight(proj.transpose(0,2,1),geo,angles,parker).transpose(0,2,1)

    # Determine filter length
    filt_len=max(64,2**nextpow2(2*max(geo.nDetector)))
    ramp_kernel=ramp_flat(filt_len)

    with device_manager(proj) as device:
        # Create filter
        d=1
        filt=xp.array(filter(geo.filter,ramp_kernel[0],filt_len,d,verbose=verbose))
        ones = xp.ones((xp.int32(geo.nDetector[0]),1),dtype=xp.float32)
        filt=xp.kron(ones, filt)

        # Padding and per-view scale factor (see per_view_scale). legacy_scale
        # reproduces the historical single-scalar behaviour exactly.
        padding = int((filt_len-geo.nDetector[1])//2 )
        if legacy_scale:
            scale = xp.full(len(angles),
                            (geo.DSD[0]/geo.DSO[0]) * (2 * np.pi/ len(angles)) / ( 4 * geo.dDetector[0] ))
        else:
            scale = xp.asarray(per_view_scale(geo, angles, dbeta=dbeta,
                                              redundancy_factor=redundancy_factor))

        #filter 2 projection at a time packing in to complex container
        fproj=xp.empty((geo.nDetector[0],filt_len),dtype=xp.complex64)
        for i in range(0,angles.shape[0]-1,2):
            fproj.fill(0)
            fproj.real[:,padding:padding+geo.nDetector[1]]=proj[i]
            fproj.imag[:,padding:padding+geo.nDetector[1]]=proj[i+1]

            fproj=fft(fproj,axis=1)
            fproj=fproj*filt
            fproj=ifft(fproj,axis=1)

            proj[i]=fproj.real[:,padding:padding+geo.nDetector[1]] * scale[i]
            proj[i+1]=fproj.imag[:,padding:padding+geo.nDetector[1]] * scale[i+1]

        #if odd number of projections filter last solo
        if angles.shape[0] % 2:
            fproj.fill(0)
            fproj.real[:,padding:padding+geo.nDetector[1]]=proj[angles.shape[0]-1]

            fproj=fft(fproj,axis=1)
            fproj=fproj*filt
            fproj=ifft(fproj,axis=1)
            proj[angles.shape[0]-1]=fproj.real[:,padding:padding+geo.nDetector[1]] * scale[angles.shape[0]-1]

        return proj.get() if xp.__name__ == "cupy" else proj


def fdk_weight_filter_chunked(proj, zgeo, angles, pad_left=0, pad_right=0,
                              wang=None, dbeta=None, legacy_scale=False,
                              redundancy_factor=None,
                              chunk_nangles=None, verbose=False):
    """Fused, chunked FDK pre-weight + ramp filter with bounded GPU memory.

    Performs, per angle-chunk on the GPU (or CPU if no CuPy device):
        [Wang redundancy weight] -> [COR zero-pad] -> cosine (intensity)
        weight -> ramp filter with per-view scale
    writing each finished chunk into ONE host float32 output array of shape
    (n, V, U + pad_left + pad_right) ready for Atb.

    Why: the legacy FDK path materialized up to three full-size copies at
    once on a full-resolution scan (zeropadding's xp.pad duplicate, the
    device stack, and filtering's .get() host copy), pushing a 27 GB stack
    to a ~54 GB peak. Here the device never holds more than the input stack
    (if it already lives there) plus one chunk; a host-resident input is
    streamed chunk-by-chunk so the device peak is chunk-sized.

    Parameters
    ----------
    proj : (n, V, U) float32, NumPy (host) or CuPy (device).
        NOTE: when proj is a device array and no padding is needed, weights
        are applied IN PLACE to save memory (same contract as legacy FDK,
        which also consumed its input).
    zgeo : geometry AFTER check_geo (per-angle fields broadcast), already
        extended by zeropad_geometry() when padding applies. zgeo.nDetector[1]
        must equal U + pad_left + pad_right.
    angles : (n,) or (n, 3) projection angles (for the per-view scale).
    wang : optional (V, U) float32 host Wang weight map (applied to the
        original detector columns, before padding).
    dbeta, legacy_scale : forwarded to the per-view ramp scale (see
        per_view_scale / filtering).
    chunk_nangles : optional chunk length; default sizes to ~40% of free
        device memory, forced even so the 2-projections-per-FFT pairing is
        identical to the unchunked implementation.

    Returns
    -------
    (n, V, U + pad_left + pad_right) float32 C-contiguous host array.
    """
    xp_in = cp.get_array_module(proj)
    n, V, U = proj.shape
    Uo = U + pad_left + pad_right
    if int(zgeo.nDetector[1]) != Uo:
        raise ValueError(f"zgeo.nDetector[1]={int(zgeo.nDetector[1])} != "
                         f"U+pads={Uo}; pass the zeropad_geometry() result")

    # Work on GPU when the input is already there or a device is available.
    use_gpu = xp_in.__name__ == "cupy"
    if not use_gpu:
        try:
            use_gpu = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            use_gpu = False
    xp = cp if use_gpu else np

    # Per-view ramp scale (host), then constants on the compute device.
    if legacy_scale:
        scale_np = np.full(n, (zgeo.DSD[0] / zgeo.DSO[0])
                           * (2 * np.pi / n) / (4 * zgeo.dDetector[0]))
    else:
        scale_np = per_view_scale(zgeo, angles, dbeta=dbeta,
                                  redundancy_factor=redundancy_factor)

    filt_len = max(64, 2 ** nextpow2(2 * max(zgeo.nDetector)))
    ramp_kernel = ramp_flat(filt_len)
    filt = xp.array(filter(getattr(zgeo, "filter", None), ramp_kernel[0],
                           filt_len, 1, verbose=verbose))
    filt = xp.kron(xp.ones((int(V), 1), dtype=xp.float32), filt)
    padding = int((filt_len - Uo) // 2)

    wang_dev = None if wang is None else xp.asarray(wang, dtype=xp.float32)

    R_d = None
    if np.any(np.asarray(zgeo.rotDetector)):
        from scipy.spatial.transform import Rotation
        # Match the CUDA, which is what actually places the rays.
        #
        # The kernels apply R = Rz(dRoll)*Ry(dPitch)*Rx(dYaw) to points ordered
        # (x, y, z) = (beam, u, v), and the bindings map dYaw <- rotDetector[0],
        # dPitch <- rotDetector[1], dRoll <- rotDetector[2] (_types.pyx, and
        # Atb_mex.cpp identically). So rotDetector[0] rotates about the BEAM -
        # the in-plane roll - and rotDetector[2] about v, despite the CUDA's
        # internal field names running the other way round.
        #
        # This weight used to be built as from_euler("XYZ", rotDetector) acting
        # on (v, u, DSD), which silently swaps elements 0 and 2: the roll was
        # applied about v and the yaw about the beam. A beam rotation leaves
        # both the numerator and |xyz| unchanged, so the real roll contributed
        # nothing here while the yaw contributed a spurious gradient.
        rot = np.asarray(zgeo.rotDetector, dtype=float)
        R_d = xp.array(
            Rotation.from_euler("ZYX", rot[..., [2, 1, 0]]).as_matrix())
    DSD = xp.asarray(np.atleast_1d(np.asarray(zgeo.DSD, dtype=float)))
    offDet = np.atleast_2d(zgeo.offDetector)

    # Chunk sizing: even, from free device memory (chunk + FFT scratch + the
    # cosine grids cost roughly 3x the chunk's own bytes).
    if chunk_nangles is None:
        if use_gpu:
            free_b, _total = cp.cuda.runtime.memGetInfo()
            per_view_b = V * Uo * 4
            chunk_nangles = int(free_b * 0.4 / (per_view_b * 3))
        else:
            chunk_nangles = n
    chunk_nangles = max(2, min(n, chunk_nangles // 2 * 2))

    out = np.empty((n, V, Uo), dtype=np.float32)
    if verbose:
        print(f"fdk chunked filter: {n} views in chunks of {chunk_nangles} "
              f"({'GPU' if use_gpu else 'CPU'}; pad L{pad_left}/R{pad_right})")

    if use_gpu:
        from cupyx.scipy.fft import fft, ifft
    else:
        from scipy.fft import fft, ifft

    fproj = xp.empty((int(V), filt_len), dtype=xp.complex64)
    yv = xp.linspace(-V / 2 + 0.5, V / 2 - 0.5, int(V)) * zgeo.dDetector[0]

    for s in range(0, n, chunk_nangles):
        e = min(s + chunk_nangles, n)
        m = e - s

        # --- stage the chunk on the compute device, padded if needed -------
        if pad_left or pad_right:
            c = xp.zeros((m, V, Uo), dtype=xp.float32)
            c[:, :, pad_left:pad_left + U] = xp.asarray(proj[s:e])
            if wang_dev is not None:
                c[:, :, pad_left:pad_left + U] *= wang_dev
        else:
            c = xp.asarray(proj[s:e])  # device view (CuPy in) or H2D copy
            if wang_dev is not None:
                c *= wang_dev          # in place (see docstring note)

        # --- cosine (intensity) weight, per view ---------------------------
        for k in range(m):
            i = s + k
            xv = (xp.linspace(-Uo / 2 + 0.5, Uo / 2 - 0.5, int(Uo))
                  * zgeo.dDetector[1] + offDet[i, 1])
            (yy, xx) = xp.meshgrid(xv, yv + offDet[i, 0])
            zz = yy * 0 + DSD[i if DSD.size > 1 else 0]
            # Stacked as (beam, u, v) - the CUDA's own axis order - so R_d
            # above applies verbatim and the cosine numerator is component 0.
            # xx holds v, yy holds u (np.meshgrid(xv, yv) varies its first
            # output along xv, which is the u axis).
            xyz = xp.vstack((zz.ravel(), yy.ravel(), xx.ravel()))
            if R_d is not None:
                xyz = xp.matmul(R_d[i], xyz)
            c[k] *= (xyz[0, :] / xp.linalg.norm(xyz, axis=0)).reshape(xx.shape)

        # --- ramp filter, 2 views per complex FFT, per-view scale ----------
        for k in range(0, m - 1, 2):
            fproj.fill(0)
            fproj.real[:, padding:padding + Uo] = c[k]
            fproj.imag[:, padding:padding + Uo] = c[k + 1]
            fp = ifft(fft(fproj, axis=1) * filt, axis=1)
            c[k] = fp.real[:, padding:padding + Uo] * scale_np[s + k]
            c[k + 1] = fp.imag[:, padding:padding + Uo] * scale_np[s + k + 1]
        if m % 2:
            fproj.fill(0)
            fproj.real[:, padding:padding + Uo] = c[m - 1]
            fp = ifft(fft(fproj, axis=1) * filt, axis=1)
            c[m - 1] = fp.real[:, padding:padding + Uo] * scale_np[e - 1]

        # --- chunk done -> host output --------------------------------------
        out[s:e] = c.get() if use_gpu else c
        del c

    if use_gpu:
        cp.get_default_memory_pool().free_all_blocks()
    return out


def ramp_flat(n, verbose=False):
    nn = np.arange(-n // 2, n // 2)
    h = np.zeros_like(nn, dtype=np.float32)
    h[n // 2] = 0.25
    odd = nn % 2 == 1
    h[odd] = -1 / (np.pi * nn[odd]) ** 2
    return h, nn


def filter(filter, kernel, order, d, verbose=False):
    f_kernel = abs(np.fft.fft(kernel)) * 2

    filt = f_kernel[: int((order / 2) + 1)]
    w = 2 * np.pi * np.arange(len(filt)) / order

    if filter in {"ram_lak", None}:
        if filter is None and verbose:
            warnings.warn("no filter selected, using default ram_lak")
    elif filter == "shepp_logan":
        filt[1:] *= np.sin(w[1:] / (2 * d)) / (w[1:] / (2 * d))
    elif filter == "cosine":
        filt[1:] *= np.cos(w[1:] / (2 * d))
    elif filter == "hamming":
        filt[1:] *= 0.54 + 0.46 * np.cos(w[1:] / d)
    elif filter == "hann":
        filt[1:] *= (1 + np.cos(w[1:] / d)) / 2
    else:
        raise ValueError("filter not recognized: " + str(filter))

    filt[w > np.pi * d] = 0
    filt = np.hstack((filt, filt[1:-1][::-1]))
    return filt.astype(np.float32)


def nextpow2(n):
    return int(np.ceil(np.log2(n))) if n > 0 else 0
    # i = 1
    # while (2 ** i) < n:
    #     i += 1
    # return i
