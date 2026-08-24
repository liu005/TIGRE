import numpy as np


def l2norm(x, chunk=1 << 24):
    """Euclidean norm of a float32 array, accumulated in float64.

    ``np.linalg.norm`` sums the squares in the array's OWN dtype. On the arrays
    this toolbox actually handles that is not survivable: the running total
    dwarfs the increments still being added to it and the answer comes back
    LOW. Measured against a chunked float64 reference, on a 3072^2 detector:

        45 views  (4.2e8 elements)   0.4 % low
        180 views (1.7e9 elements)     3 % low
        360 views (3.4e9 elements)    16 % low
        720 views (6.8e9 elements)    26 % low

    A 1024^3 volume (1.1e9 elements) is 1.4 % low. These norms are not
    only reported - they drive step sizes and stopping rules (CGLS's
    ``alpha = gamma / q_norm**2``; the ASD-POCS data-fit and TV-step control),
    so at production sizes the algorithms misbehave rather than merely
    mis-report.

    Two properties of the failure are worth knowing. It is driven by the
    ACCUMULATOR growing against terms that stay small - a sum of squares is
    all-positive, so nothing ever cancels - rather than by N alone. And it is
    BLAS-dependent: the figures above are scipy-openblas 0.3.31 on the Linux
    box where the production runs happen; the same sizes on this project's
    Windows BLAS are ~76x more accurate. Do not assume a platform is safe.

    ``np.sum`` is NOT affected - numpy's own pairwise summation gives 4e-9 at
    1024^3 - so `np.sum(x * x)` was never the problem and does not need this.
    Only reductions that dispatch to BLAS (``np.dot``, ``np.linalg.norm``) do.

    One pass, no full-size copy. Returns float32 so no caller's dtype changes:
    these scalars multiply float32 volumes, and Ax/Atb reject float64.
    """
    flat = np.ravel(x)
    if flat.dtype != np.float32:
        return np.linalg.norm(flat, 2)
    total = 0.0
    for k in range(0, flat.size, chunk):
        c = flat[k:k + chunk].astype(np.float64)
        total += float(np.dot(c, c))
    return np.float32(np.sqrt(total))


def inner(a, b, chunk=1 << 24):
    """Inner product of two float32 arrays, accumulated in float64.

    Same reasoning as l2norm, one step weaker: a dot of two DIFFERENT vectors
    has cancelling signs, so its accumulator does not grow monotonically and
    BLAS holds up better - measured 7.8e-5 relative error at 1024^3 against
    l2norm's 1.4e-2 for the sum of squares. Still four orders worse than a
    float64 accumulation, and these products are Gram-Schmidt coefficients
    whose error compounds across a Krylov basis, so they use it too.

    Returns float32, like l2norm, so no caller's dtype changes.
    """
    fa, fb = np.ravel(a), np.ravel(b)
    if fa.dtype != np.float32 or fb.dtype != np.float32:
        return np.dot(fa, fb)
    total = 0.0
    for k in range(0, fa.size, chunk):
        total += float(np.dot(fa[k:k + chunk].astype(np.float64),
                              fb[k:k + chunk].astype(np.float64)))
    return np.float32(total)


def im3DNORM(img, normind, varargin=None):
    """
    % IMAGE3DNORM computes the desired image norm
    %   IMAGE3DNORM(IMG,NORMIND) computes the norm of image IMG using the norm
    %   defined in NORMIND
    %
    %   IMG         A 3D image
    %   NORMIND     {non-zero int, inf, -inf, 'fro', 'nuc'}
    %               'TV': TV norm
    %
    %
    %--------------------------------------------------------------------------
    %--------------------------------------------------------------------------
    % This file is part of the TIGRE Toolbox
    %
    % Copyright (c) 2015, University of Bath and
    %                     CERN-European Organization for Nuclear Research
    %                     All rights reserved.
    %
    % License:            Open Source under BSD.
    %                     See the full license at
    %                     https://github.com/CERN/TIGRE/license.txt
    %
    % Contact:            tigre.toolbox@gmail.com
    % Codes:              https://github.com/CERN/TIGRE/
    % Coded by:           Ander Biguri
    %--------------------------------------------------------------------------
    """
    if normind is [np.inf, -np.inf, "fro", "nuc"]:
        return np.linalg.norm(img.ravel(), normind)
    if type(normind) is int:
        # The L2 case is the one every caller in this toolbox uses, and the one
        # a float32 accumulator cannot compute at these sizes - see l2norm.
        if normind == 2:
            return l2norm(img)
        return np.linalg.norm(img.ravel(), normind)
    if normind == "TV":
        gx, gy, gz = np.diff(img, axis=2), np.diff(img, axis=1), np.diff(img, axis=0)
        g = np.sum(np.sqrt(gx * gx + gy * gy + gz * gz))
        return g
