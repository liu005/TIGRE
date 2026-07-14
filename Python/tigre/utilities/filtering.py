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
        print(f"Using device: {self.device}")
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


def per_view_scale(geo, angles, dbeta=None):
    """Per-view ramp-filter scale, shape (n,) float64 (host).

        scale_i = (DSD[i]/DSO[i]) * dbeta_i / (4 * dDetector[1])

    generalizing legacy TIGRE's single scalar
        (DSD[0]/DSO[0]) * (2*pi/n) / (4 * dDetector[0]),
    which silently assumed constant magnification and a uniformly-sampled
    full-2*pi circular scan. For that case the per-view weights are all equal
    to the legacy scalar (for a square-pixel detector), so results are
    unchanged; short scans get their true angular weight (span/2*pi effect)
    and variable-DSD/DSO (non-circular) trajectories get per-view
    magnification. dDetector[1] (u/column pitch) is the ramp-filter sample
    spacing - the filter runs along u; legacy's dDetector[0] was an
    anisotropic-detector bug (fixed upstream the same way).

    NOTE (short-scan redundancy constant): with Parker weighting each ray is
    counted once on a short scan but twice on a full circle; whether the
    legacy /4 (which embeds the full-circle half) requires a compensating
    factor ~2 for short scans is pinned empirically by the uniform-phantom
    battery (V2) - see docs/fdk_intensity_normalization.md in the pipeline
    repo.
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
    return (DSD[:n] / DSO[:n]) * db / (4.0 * geo.dDetector[1])


# TODO: Fix parker
def filtering(proj, geo, angles, parker, verbose=False, dbeta=None, legacy_scale=False):
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
            scale = xp.asarray(per_view_scale(geo, angles, dbeta=dbeta))

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
