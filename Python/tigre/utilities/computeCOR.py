# -*- coding: utf-8 -*-
"""Centre-of-rotation estimate from a single sinogram slice.

Code modified from SophiaBeads
(https://github.com/Sophilyplum/sophiabeads-datasets/blob/master/tools/centre_geom.m)
Reference: T. Liu, "Direct central ray determination in computed
microtomography", Optical Engineering, April 2009.

For a candidate centre of rotation, every ray has a conjugate ray half a turn
away that should measure the same line integral. The candidate that makes the
sinogram agree with its own reflected self is the centre.

REWRITTEN 2026-08-24. The previous version could not run - both paths raised
on the first call, which is why nothing but the export ever referenced it:

  * gpu=True  -> TypeError: cp.squeeze() was applied to the caller's NumPy
    array BEFORE the branch that converts it.
  * gpu=False -> ValueError broadcasting (n_det,) against (n_angles,):
    geo.DSD is PER-ANGLE after check_geo, and the fan-angle formula wants a
    scalar.
  * the interpolation was `cv2.remap(test_data, angle_grid, (angles_aux, s2))`,
    which is not cv2.remap's signature at all. The commented-out line beside it
    names the real intent - interpn over the (angle, detector) grid - and
    RegularGridInterpolator was already imported and then never used.

Two further fixes, both of which change results rather than just enabling them:

  * `import cupy` was at module scope, and this module is imported by
    `tigre/__init__.py` - so it made CuPy a hard dependency of `import tigre`.
    It is imported lazily now, inside the gpu branch.
  * the score was normalised by `len(nonzero)`. In the MATLAB original,
    `nonzero = find(test > 0)` is a list of INDICES, so `length(nonzero)` is
    the count of valid pixels - which is what the comment carried over beside
    it says it wants. The port translated `find` into a boolean MASK, where
    `len()` returns the first axis length (the angle count) instead. That is
    constant across candidates, so the search effectively minimised a SUM,
    which rewards a candidate whose reflection throws pixels off the detector:
    every lost pixel drops a positive term. Restored to the mean over valid
    pixels, i.e. to what MATLAB does.

Checked against MATLAB/Utilities/computeCOR.m line by line. The arrays are
transposed relative to it throughout - MATLAB holds (detector, angle) and
projections as (V, U, angle), Python holds (angle, detector) and projections
as (angle, V, U) - so the angular wrap is a concatenation along axis 0 here
against a horizontal one there, and `angles[:, None] + beta[None, :]` is the
transpose of its `repmat(angles', nDet, 1) + repmat(beta, 1, nAngles)`.
Everything else - the 21-point search grid per precision, the seven precision
levels, linear interpolation with 0 outside, and `-midpoint * DSO / DSD` -
matches.
"""

import numpy as np


def computeCOR(data, geo, angles, slc=None, gpu=True):
    """Estimate the centre of rotation, in mm at the rotation axis.

    :param data: projections, (n_angles, n_rows, n_cols)
    :param geo: tigre geometry
    :param angles: projection angles, radians
    :param slc: detector row to use (default: the central row)
    :param gpu: use CuPy for the search
    :return: float, the COR offset in mm - the sign convention TIGRE's
        `geo.COR` expects
    """
    if slc is None:
        slc = int(np.floor(data.shape[1] / 2))

    if gpu:
        # Lazy, and optional: this module is imported by tigre/__init__.py, so a
        # module-scope `import cupy` made CuPy a hard dependency of
        # `import tigre`. Asking for the GPU on a machine without it now costs a
        # notice, not an ImportError.
        try:
            import cupy as cp
            from cupyx.scipy.interpolate import RegularGridInterpolator
            xp = cp
        except ImportError:
            print("computeCOR: CuPy unavailable, running on the CPU")
            gpu = False
    if not gpu:
        from scipy.interpolate import RegularGridInterpolator
        xp = np

    # float64 throughout, as in the MATLAB reference (`double(repmat(...))`):
    # this is one detector row, so the precision is free.
    sino = xp.squeeze(xp.asarray(data)[:, slc, :]).astype(xp.float64)
    angles = xp.asarray(angles).astype(xp.float64).ravel()

    # check_geo() broadcasts these to one value per angle; the fan-angle
    # geometry below is a single fixed source-detector pair.
    DSD = float(np.mean(np.atleast_1d(np.asarray(geo.DSD, dtype=np.float64))))
    DSO = float(np.mean(np.atleast_1d(np.asarray(geo.DSO, dtype=np.float64))))

    # RegularGridInterpolator needs ascending coordinates, and this project
    # flips angles upstream for its rotation-direction convention.
    if float(angles[0]) > float(angles[-1]):
        angles = angles[::-1]
        sino = sino[::-1]

    det = xp.linspace(-geo.sDetector[1] / 2 + geo.dDetector[1] / 2,
                      +geo.sDetector[1] / 2 - geo.dDetector[1] / 2,
                      int(geo.nDetector[1]), dtype=xp.float64)
    gamma = xp.arctan(det / DSD)                # fan angle per column; candidate-invariant

    # Repeat the sinogram either side in angle so a conjugate ray that lands
    # outside [angles[0], angles[-1]] still interpolates instead of vanishing.
    two_pi = 2.0 * np.pi
    ang3 = xp.concatenate((angles - two_pi, angles, angles + two_pi))
    sino3 = xp.concatenate((sino, sino, sino), axis=0)
    interp = RegularGridInterpolator((ang3, det), sino3, method="linear",
                                     bounds_error=False, fill_value=0.0)

    midpoint = 0.0
    for pr in (1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001):
        COR = xp.linspace(midpoint - 10 * pr, midpoint + 10 * pr, 21, dtype=xp.float64)
        M = xp.full((COR.size,), xp.inf, dtype=xp.float64)

        for j in range(COR.size):
            gamma_c = xp.arctan(COR[j] / DSD)
            beta = 2.0 * (gamma - gamma_c) + np.pi          # conjugate-ray angle offset
            s2 = DSD * xp.tan(2.0 * gamma_c - gamma)        # ... and its detector position

            angles_aux = angles[:, None] + beta[None, :]
            pts = xp.stack((angles_aux,
                            xp.broadcast_to(s2[None, :], angles_aux.shape)), axis=-1)
            test = interp(pts.reshape(-1, 2)).reshape(angles_aux.shape)

            valid = test > 0
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue        # leaves inf: a candidate that sees nothing never wins
            M[j] = xp.sum((test[valid] - sino[valid]) ** 2) / n_valid

        midpoint = float(COR[int(xp.argmin(M))])

    return -midpoint * DSO / DSD
