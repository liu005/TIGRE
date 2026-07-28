from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import cupy as cp
from tigre.utilities.Atb import Atb
from tigre.utilities.filtering import filtering, fdk_weight_filter_chunked
from tigre.utilities.redundancy_weighting import redundancy_weighting, zeropadding, zeropad_geometry
from scipy.spatial.transform import Rotation
import gc

class device_manager:
    """
    Context manager, either CPU or nVidia GPU
    """
    def __init__(self, arr):
        try:
            self.device = arr.device
        except AttributeError:
            self.device = "cpu"
            
    def __enter__(self):
        return self.device

    def __exit__(self, exc_type, exc_value, traceback):
        pass
    


def FDK(proj, geo, angles, **kwargs):
    """
    solves CT image reconstruction.

    :param proj: np.array(dtype=float32),
    Data input in the form of 3d

    :param geo: tigre.utilities.geometry.Geometry
    Geometry of detector and image (see examples/Demo code)

    :param angles: np.array(dtype=float32)
    Angles of projection, shape = (nangles,3) or (nangles,)

    :param filter: str
    Type of filter used for backprojection
    opts: "shep_logan"
          "cosine"
          "hamming"
          "hann"

    :param verbose: bool
    Feedback print statements for algorithm progress

    :param kwargs: dict
    keyword arguments

    :return: np.array(dtype=float32)

    Usage:
    -------
    >>> import tigre
    >>> import tigre.algorithms as algs
    >>> import numpy
    >>> from tigre.demos.Test_data import data_loader
    >>> geo = tigre.geometry(mode='cone',default_geo=True,
    >>>                         nVoxel=np.array([64,64,64]))
    >>> angles = np.linspace(0,2*np.pi,100)
    >>> src_img = data_loader.load_head_phantom(geo.nVoxel)
    >>> proj = tigre.Ax(src_img,geo,angles)
    >>> output = algs.FDK(proj,geo,angles)

    tigre.demos.run() to launch ipython notebook file with examples.

    --------------------------------------------------------------------
    This file is part of the TIGRE Toolbox

    Copyright (c) 2015, University of Bath and
                        CERN-European Organization for Nuclear Research
                        All rights reserved.

    License:            Open Source under BSD.
                        See the full license at
                        https://github.com/CERN/TIGRE/license.txt

    Contact:            tigre.toolbox@gmail.com
    Codes:              https://github.com/CERN/TIGRE/
    --------------------------------------------------------------------
    Coded by:           MATLAB (original code): Ander Biguri
                        PYTHON : Reuben Lindroos
    """
    verbose = kwargs["verbose"] if "verbose" in kwargs else False
    gpuids = kwargs["gpuids"] if "gpuids" in kwargs else None
    dowang = kwargs["dowang"] if "dowang" in kwargs else True
    # Per-view ramp-scale controls (see utilities/filtering.py):
    #   dbeta        - explicit per-view angular weights for exotic
    #                  trajectories (else derived from `angles`; the
    #                  ArbitrarySource* helpers may attach geo.dbeta).
    #   legacy_scale - reproduce the historical single-scalar ramp scale
    #                  (constant magnification, uniform full-2*pi assumption).
    dbeta = kwargs.get("dbeta", getattr(geo, "dbeta", None))
    legacy_scale = kwargs.get("legacy_scale", False)
    #   redundancy_factor - ray-multiplicity constant K (K=2/c): None (default)
    #                  auto-derives 1 for a full 2*pi circle, 2 for a short
    #                  scan (single coverage; pair with Parker weighting).
    redundancy_factor = kwargs.get("redundancy_factor", None)


    # Wang redundancy weight map (host, (V, U)). `angles` is passed so the
    # weights are auto-skipped for short/partial scans, where Wang's full-360
    # opposing-view assumption does not hold and the ramp would survive as
    # one-sided shading. An all-ones map is dropped entirely (no-op multiply).
    wang = None
    if dowang:
        if verbose:
            print('FDK: applying detector offset weights')
        w = redundancy_weighting(geo, angles)
        if (w != 1).any():
            wang = w

    geo.check_geo(angles)
    geo.checknans()
    geo.filter = kwargs["filter"] if "filter" in kwargs else None

    # COR/offset-induced detector extension: geometry math only - the actual
    # zero-padding of the data happens chunk-by-chunk inside the fused filter
    # (the old zeropadding() materialized a SECOND full-size padded stack).
    if abs(geo.COR).any() > 0:
        geo, pad_left, pad_right = zeropad_geometry(geo)
    else:
        pad_left = pad_right = 0

    # Fused, chunked pre-weight + ramp filter: [Wang] -> [COR pad] -> cosine
    # intensity weight -> per-view-scaled ramp filter, streamed in angle
    # chunks so the device never holds more than the (possibly resident)
    # input stack plus one chunk. Returns the host float32 stack for Atb.
    with device_manager(proj):
        proj_filt = fdk_weight_filter_chunked(
            proj, geo, angles, pad_left=pad_left, pad_right=pad_right,
            wang=wang, dbeta=dbeta, legacy_scale=legacy_scale,
            redundancy_factor=redundancy_factor, verbose=verbose)

    # clean up gpu memory and reset before running Atb()
    del proj
    gc.collect()

    # FDK back projection
    rec = Atb(proj_filt, geo, geo.angles, "FDK", gpuids=gpuids)
            
    return rec


fdk = FDK


def fbp(proj, geo, angles, **kwargs):  # noqa: D103
    __doc__ = FDK.__doc__  # noqa: F841
    if geo.mode != "parallel":
        raise ValueError("Only use FBP for parallel beam. Check geo.mode.")
    geox = copy.deepcopy(geo)
    geox.check_geo(angles)
    verbose = kwargs["verbose"] if "verbose" in kwargs else False
    gpuids = kwargs["gpuids"] if "gpuids" in kwargs else None
    proj_filt = filtering(copy.deepcopy(proj), geox,
                          angles, parker=False, verbose=verbose)
    if not isinstance(geo.DSO, np.ndarray):
        return Atb(proj_filt, geo, angles, gpuids=gpuids)* geo.DSO / geo.DSD
    else:
        return Atb(proj_filt, geo, angles, gpuids=gpuids)* geo.DSO[0] / geo.DSD[0]
