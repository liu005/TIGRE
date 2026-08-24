#   This file is part of the TIGRE Toolbox

#   Copyright (c) 2015, University of Bath and
#                       CERN-European Organization for Nuclear Research
#                       All rights reserved.

#   License:            Open Source under BSD.
#                       See the full license at
#                       https://github.com/CERN/TIGRE/license.txt

#   Contact:            tigre.toolbox@gmail.com
#   Codes:              https://github.com/CERN/TIGRE/
# --------------------------------------------------------------------------
#   Coded by:          MATLAB (original code): Ander Biguri
#                      PYTHON : Sam Loescher, Reuben Lindroos

cimport numpy as np 
import numpy as np
from tigre.utilities.cuda_interface._gpuUtils cimport GpuIds as c_GpuIds, convert_to_c_gpuids, free_c_gpuids

np.import_array()
from libc.stdlib cimport malloc, free 

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyArray_CLEARFLAGS(np.ndarray arr, int flags)

cdef extern from "GD_AwTV.hpp":
    cdef void aw_pocs_tv(float* img, float* dst, float alpha, long* image_size, int maxiter, float dtc, float dtv, float dtt, c_GpuIds gpuids)
def cuda_raise_errors(error_code):
    if error_code:
        raise ValueError('TIGRE:Call to aw_pocs_tv failed')

def AwminTV(np.ndarray[np.float32_t, ndim=3] src,float alpha = 15.0,int maxiter = 100, delta=-0.005, gpuids=None):
    # gpuids=None reaches convert_to_c_gpuids as count 0 / NULL, which this
    # kernel reports as "no available device(s) that support CUDA" (the
    # projectors instead read count 0 as auto-select-all). Resolve it here to
    # the same GpuIds() every Python caller already substitutes for None.
    # Lazy import: these modules load during tigre's own package init.
    if gpuids is None:
        from tigre.utilities.gpu import GpuIds
        gpuids = GpuIds()
    cdef c_GpuIds* c_gpuids = convert_to_c_gpuids(gpuids)
    if not c_gpuids:
        raise MemoryError()

    cdef np.npy_intp size_img[3]
    size_img[0]= <np.npy_intp> src.shape[0]
    size_img[1]= <np.npy_intp> src.shape[1]
    size_img[2]= <np.npy_intp> src.shape[2]

    cdef float* c_imgout = <float*> malloc(<unsigned long>size_img[0] *size_img[1] *size_img[2]* sizeof(float))

    cdef long imgsize[3]
    imgsize[0] = <long> size_img[2]
    imgsize[1] = <long> size_img[1]
    imgsize[2] = <long> size_img[0]

    src = np.ascontiguousarray(src)

    cdef float* c_src = <float*> src.data
    cdef np.npy_intp c_maxiter = <np.npy_intp> maxiter
    # delta: scalar broadcasts to all three axes, so equal components -- and
    # therefore a plain scalar -- reproduce the old isotropic-in-orientation
    # behaviour exactly. A length-3 sequence gives per-axis thresholds in the
    # caller's numpy axis order (REPORT sec 31).
    cdef float c_delta[3]
    _d = np.asarray(delta, dtype=np.float32).ravel()
    if _d.size == 1:
        c_delta[0] = c_delta[1] = c_delta[2] = <float> _d[0]
    elif _d.size == 3:
        c_delta[0] = <float> _d[0]
        c_delta[1] = <float> _d[1]
        c_delta[2] = <float> _d[2]
    else:
        raise ValueError("delta must be a scalar or a length-3 sequence")
    aw_pocs_tv(c_src, c_imgout, alpha, imgsize, c_maxiter, c_delta[0], c_delta[1], c_delta[2], c_gpuids[0])
    imgout = np.PyArray_SimpleNewFromData(3, size_img, np.NPY_FLOAT32, c_imgout)
    PyArray_ENABLEFLAGS(imgout, np.NPY_ARRAY_OWNDATA)

    return imgout
