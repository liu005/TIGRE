import numpy as np
import copy
import cupy as cp


def scan_arc(angles):
    """Total angular arc (radians) covered by a projection-angle array.

    Accepts (n,) or (n, 3) (TIGRE Euler convention - the in-plane rotation is
    the first column). Robust to rotation direction and wrap-around via
    np.unwrap. Returns 0.0 for a single projection or None input.
    """
    if angles is None:
        return None
    a = np.asarray(angles, dtype=float)
    if a.ndim > 1:
        a = a[:, 0]
    if a.size < 2:
        return 0.0
    return float(np.ptp(np.unwrap(a)))


def apply_redudancy_weights(geo, verbose=True, angles=None) -> bool:
    """
    Check if Wang redundancy weights should be applied

    Wang weighting (displaced-detector redundancy weighting) assumes a FULL
    360-degree circular scan: it ramps one side of the detector and relies on
    the opposing (beta + pi) views to complement the ramp back to uniform
    coverage. On a short/partial scan those opposing views do not exist, so
    the ramp survives into the reconstruction as a gross one-sided shading
    ("teardrop"). When ``angles`` is provided, the scan arc is checked and
    Wang weights are skipped for anything meaningfully short of 2*pi.

    Parameters
    ----------
    geo : Object
        The Tigre geometry
    angles : array-like, optional
        Projection angles, shape (n,) or (n, 3). If given, Wang weights are
        only applied for a (near-)full 2*pi scan. If None (legacy callers),
        no span check is performed.

    Returns
    -------
    True if applicable

    """
    if angles is not None:
        arc = scan_arc(angles)
        a = np.asarray(angles, dtype=float)
        a = a[:, 0] if a.ndim > 1 else a
        # tolerance: two median angular steps (also covers the common
        # "endpoint=False" full circle, whose arc is 2*pi - one step)
        step = float(np.median(np.abs(np.diff(np.unwrap(a))))) if a.size > 1 else 0.0
        if arc < 2.0 * np.pi - max(2.0 * step, 1e-6):
            if verbose:
                print('Wang weights: short scan detected '
                      f'(arc {np.degrees(arc):.1f} deg < 360 deg), '
                      'Wang weights not being applied')
            return False

    if (np.atleast_2d(geo.offDetector).shape[0] > 1) and np.ptp(np.atleast_2d(geo.offDetector)[:, 1]) > 0:
        if verbose:
            print('Wang weights: varying offDetector detected, Wang weights not being applied')
        return False
    
    # Centred detector: nothing to weight. The routine common case - stays
    # silent (unlike the informative skip messages above/below).
    if np.atleast_2d(geo.offDetector)[0, 1] == 0:
        return False
    
    if (len(np.atleast_2d(geo.DSO)) > 1) and len(np.unique(geo.DSO)) > 1:
        if verbose:
            print('Wang weights: varying DSO detected, Wang weights not being applied');
        return False

    percent_offset = abs(np.atleast_2d(geo.offDetector)[0, 1] / np.atleast_2d(geo.sDetector)[0, 1]) * 100;    
    if percent_offset > 30:
        print('Wang weights: Detector offset percent: %0.2f is greater than 30 which may result in image artifacts, consider rebinning 360 degree projections to 180 degrees', percent_offset)
    
    return True


def redundancy_weighting(geo, angles=None):
    """
    Preweighting using Wang function
    Ref:
        Wang, Ge. X-ray micro-CT with a displaced detector array. Medical Physics, 2002,29(7):1634-1636.

    angles : optional projection angles, (n,) or (n, 3). When given, the
    weights are only computed for a (near-)full 2*pi scan - see
    apply_redudancy_weights. Short scans get an all-ones weight (no-op).
    """

    if not hasattr(geo,'COR'):
        geo.COR=np.array([0])

    w = np.ones((geo.nDetector[0], geo.nDetector[1]), dtype=np.float32)

    if apply_redudancy_weights(geo, angles=angles):
        offset = np.atleast_2d(geo.offDetector)[0, 1]
        DSD = np.atleast_2d(geo.DSD)[0]
        DSO = np.atleast_2d(geo.DSO)[0]
        offset += (DSD / DSO) * np.atleast_1d(geo.COR)[0]   # added correction
        us = np.linspace(-geo.nDetector[1]/2+0.5, geo.nDetector[1]/2-0.5, geo.nDetector[1]) * geo.dDetector[1] + abs(offset)
        
        us *= DSO / DSD
        theta = (geo.sDetector[1]/2 - abs(offset)) * np.sign(offset)
        abstheta = abs(theta * DSO / DSD)
    
        w = np.where(np.abs(us) <= abstheta,
            0.5 * (np.sin( (np.pi / 2) * np.arctan(us / DSO) / np.arctan(abstheta / DSO) ) + 1),
            np.where(us < -abstheta, 0, w)
            )
#        w=w*2
        if (theta<0):
            w = np.fliplr(w)
            
    return w.astype(np.float32)


def zeropad_geometry(geo):
    """Geometry-only half of zeropadding(): compute the COR/offset-induced
    detector extension WITHOUT touching projection data.

    Returns (zgeo, pad_left, pad_right): a deep-copied geometry with
    nDetector[1] extended by pad_left+pad_right columns and offDetector
    shifted accordingly. Pad side follows zeropadding()'s convention (leading
    edge of axis -1 when the effective offset is positive, trailing edge
    otherwise). pad_left == pad_right == 0 means no padding is needed.

    Splitting this out lets callers pad the projections chunk-by-chunk (the
    original zeropadding() materializes a SECOND full-size padded stack via
    xp.pad - a ~27 GB duplicate on a full-resolution scan). deepcopy is used
    (zeropadding()'s copy.copy shares the offDetector/nDetector arrays with
    the caller's geometry and silently mutates them).
    """
    zgeo = copy.deepcopy(geo)

    offDet1 = np.atleast_2d(geo.offDetector)[0, 1]
    offDet1 += np.atleast_1d(geo.DSD)[0] / np.atleast_1d(geo.DSO)[0] * np.atleast_1d(geo.COR)[0]

    width = int(np.fix(2 * offDet1 / geo.dDetector[1])) + 1
    if np.isscalar(zgeo.DSO):
        zgeo.offDetector[1] = zgeo.offDetector[1] - width / 2 * geo.dDetector[1]
    else:
        zgeo.offDetector[:, 1] = zgeo.offDetector[:, 1] - width / 2 * geo.dDetector[1]

    zgeo.nDetector[1] += abs(width)
    zgeo.sDetector[1] = zgeo.nDetector[1] * zgeo.dDetector[1]

    pad_left, pad_right = (width, 0) if offDet1 > 0 else (0, abs(width))
    return zgeo, pad_left, pad_right


def zeropadding(proj, geo):
    """
    Zero padding the projections and modify geometry accordingly

    Parameters
    ----------
    proj : ndarray or cupy array
        Projections.
    geo : obj
        geometry for reconstruction.

    Returns
    -------
    zproj : ndarray or cupy array
        Zero padding the projections.
    zgeo : obj
        modified geometry for reconstruction.
    theta : ndarray
        Angles of projection.

    """
    xp = cp.get_array_module(proj)
    zgeo = copy.copy(geo)
    
    offDet1 = np.atleast_2d(geo.offDetector)[0, 1]
    offDet1 += np.atleast_1d(geo.DSD)[0] / np.atleast_1d(geo.DSO)[0] * np.atleast_1d(geo.COR)[0]
    
    width = int(np.fix(2 * offDet1 / geo.dDetector[1])) + 1
    if np.isscalar(geo.DSO):
        zgeo.offDetector[1] = zgeo.offDetector[1] - width / 2 * geo.dDetector[1]
    else:
        zgeo.offDetector[:, 1] = zgeo.offDetector[:, 1] - width / 2 * geo.dDetector[1]

    zgeo.nDetector[1] += abs(width)
    zgeo.sDetector[1] = zgeo.nDetector[1] * zgeo.dDetector[1]

    padwidth = ((0, 0), (0, 0), (width, 0)) if offDet1 > 0 \
        else ((0, 0), (0, 0), (0, abs(width)))
    zproj = xp.pad(proj, padwidth, constant_values=0)
        
    return zproj, zgeo
