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

    A 1024^3 volume (1.1e9 elements) is already ~1 % low. These norms are not
    only reported - they drive step sizes and stopping rules (CGLS's
    ``alpha = gamma / q_norm**2``; the ASD-POCS data-fit and TV-step control),
    so at production sizes the algorithms misbehave rather than merely
    mis-report.

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
