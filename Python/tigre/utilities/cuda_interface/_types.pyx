# tigre/utilities/cuda_interface/_types.pyx

from libc.stdlib cimport malloc, free
cimport cython
cimport numpy as cnp
import numpy as np

# Pull in the C struct declaration from the .pxd declarations file
from tigre.utilities.cuda_interface._types cimport Geometry

# ============================================================
# Helpers (define BEFORE use; no forward declarations needed)
# ============================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cnp.ndarray[cnp.float32_t, ndim=1] _as_1d_or_broadcast(object val, int nproj, const char* name):
    """
    Return a contiguous float32 1-D array of length nproj.
    Accepts scalar, length-1, or any array -> flattens/broadcasts.
    """
    arr = np.asarray(val)

    if arr.ndim == 0:
        return np.full(nproj, float(arr), dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1)

    if arr.size == 1:
        return np.full(nproj, float(arr[0]), dtype=np.float32)

    if arr.size != nproj:
        raise ValueError(f"{name} has length {arr.size}, expected {nproj}")

    return np.ascontiguousarray(arr, dtype=np.float32)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cnp.ndarray[cnp.float32_t, ndim=2] _as_n_by_k(object val, int nproj, int k, const char* name):
    """
    Return a contiguous float32 (nproj, k) array.
    Accepts scalar -> broadcast; (k,) or (1,k) -> broadcast; (k,nproj) -> transpose; (nproj,k) -> passthrough.
    """
    arr = np.asarray(val, dtype=np.float32).squeeze()

    if arr.ndim == 0:
        arr = np.full((nproj, k), float(arr), dtype=np.float32)

    elif arr.ndim == 1:
        if arr.size == k:
            arr = np.tile(arr[np.newaxis, :], (nproj, 1))
        elif arr.size == 1:
            arr = np.full((nproj, k), float(arr[0]), dtype=np.float32)
        else:
            raise ValueError(f"{name} has shape {arr.shape}, expected length {k} or 1")

    elif arr.ndim == 2:
        if arr.shape == (nproj, k):
            pass
        elif arr.shape == (k, nproj):
            arr = arr.T
        elif arr.shape[0] == 1 and arr.shape[1] == k:
            arr = np.tile(arr, (nproj, 1))
        else:
            raise ValueError(f"{name} has shape {arr.shape}, expected ({nproj},{k})")
    else:
        raise ValueError(f"{name} has {arr.ndim} dimensions; expected <= 2")

    return np.ascontiguousarray(arr, dtype=np.float32)

# =========================
# Free function (C-level)
# =========================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void free_c_geometry(Geometry* c_geom):
    if not c_geom:
        return
    free(c_geom.offOrigX);  c_geom.offOrigX = <float*>0
    free(c_geom.offOrigY);  c_geom.offOrigY = <float*>0
    free(c_geom.offOrigZ);  c_geom.offOrigZ = <float*>0
    free(c_geom.offDetecU); c_geom.offDetecU = <float*>0
    free(c_geom.offDetecV); c_geom.offDetecV = <float*>0
    free(c_geom.DSO);       c_geom.DSO = <float*>0
    free(c_geom.DSD);       c_geom.DSD = <float*>0
    free(c_geom.dRoll);     c_geom.dRoll = <float*>0
    free(c_geom.dPitch);    c_geom.dPitch = <float*>0
    free(c_geom.dYaw);      c_geom.dYaw = <float*>0
    free(c_geom.COR);       c_geom.COR = <float*>0
    free(c_geom)

# =====================================
# Python -> C struct converter (C-level)
# =====================================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Geometry* convert_to_c_geometry(object p_geometry, int total_projections):
    cdef Geometry* c_geom = <Geometry*>malloc(sizeof(Geometry))
    if not c_geom:
        raise MemoryError("Failed to allocate Geometry")

    # ---- declare ALL typed locals BEFORE executable statements ----
    cdef cnp.ndarray[cnp.float32_t, ndim=2] _offOrigin
    cdef cnp.ndarray[cnp.float32_t, ndim=1] _DSO
    cdef cnp.ndarray[cnp.float32_t, ndim=2] _offDetector
    cdef cnp.ndarray[cnp.float32_t, ndim=1] _DSD
    cdef cnp.ndarray[cnp.float32_t, ndim=2] _rotDetector
    cdef cnp.ndarray[cnp.float32_t, ndim=1] _COR
    cdef int i

    # init pointers to null (no libc NULL needed)
    c_geom.offOrigX = <float*>0
    c_geom.offOrigY = <float*>0
    c_geom.offOrigZ = <float*>0
    c_geom.DSO      = <float*>0
    c_geom.offDetecU= <float*>0
    c_geom.offDetecV= <float*>0
    c_geom.DSD      = <float*>0
    c_geom.dRoll    = <float*>0
    c_geom.dPitch   = <float*>0
    c_geom.dYaw     = <float*>0
    c_geom.COR      = <float*>0

    try:
        # --- image sizes ---
        c_geom.nVoxelX = p_geometry.nVoxel[2]
        c_geom.nVoxelY = p_geometry.nVoxel[1]
        c_geom.nVoxelZ = p_geometry.nVoxel[0]

        c_geom.sVoxelX = p_geometry.sVoxel[2]
        c_geom.sVoxelY = p_geometry.sVoxel[1]
        c_geom.sVoxelZ = p_geometry.sVoxel[0]

        c_geom.dVoxelX = p_geometry.dVoxel[2]
        c_geom.dVoxelY = p_geometry.dVoxel[1]
        c_geom.dVoxelZ = p_geometry.dVoxel[0]

        # --- normalize Python-side arrays ---
        _offOrigin   = _as_n_by_k(getattr(p_geometry, "offOrigin",   0.0), total_projections, 3, b"offOrigin")
        _DSO         = _as_1d_or_broadcast(getattr(p_geometry, "DSO",         1000.0), total_projections, b"DSO")
        _offDetector = _as_n_by_k(getattr(p_geometry, "offDetector", 0.0), total_projections, 2, b"offDetector")
        _DSD         = _as_1d_or_broadcast(getattr(p_geometry, "DSD",         1500.0), total_projections, b"DSD")
        _rotDetector = _as_n_by_k(getattr(p_geometry, "rotDetector", 0.0), total_projections, 3, b"rotDetector")
        _COR         = _as_1d_or_broadcast(getattr(p_geometry, "COR",         0.0),    total_projections, b"COR")

        # --- allocate C arrays ---
        c_geom.offOrigX = <float*>malloc(total_projections * sizeof(float))
        if not c_geom.offOrigX:
            raise MemoryError()
        c_geom.offOrigY = <float*>malloc(total_projections * sizeof(float))
        if not c_geom.offOrigY:
            raise MemoryError()
        c_geom.offOrigZ = <float*>malloc(total_projections * sizeof(float))
        if not c_geom.offOrigZ:
            raise MemoryError()
        c_geom.DSO = <float*>malloc(total_projections * sizeof(float))
        if not c_geom.DSO:
            raise MemoryError()

        c_geom.offDetecU = <float*>malloc(total_projections * sizeof(float))
        if not c_geom.offDetecU:
            raise MemoryError()
        c_geom.offDetecV = <float*>malloc(total_projections * sizeof(float))
        if not c_geom.offDetecV:
            raise MemoryError()
        c_geom.DSD = <float*>malloc(total_projections * sizeof(float))
        if not c_geom.DSD:
            raise MemoryError()

        c_geom.dRoll = <float*>malloc(total_projections * sizeof(float))
        if not c_geom.dRoll:
            raise MemoryError()
        c_geom.dPitch = <float*>malloc(total_projections * sizeof(float))
        if not c_geom.dPitch:
            raise MemoryError()
        c_geom.dYaw = <float*>malloc(total_projections * sizeof(float))
        if not c_geom.dYaw:
            raise MemoryError()

        c_geom.COR = <float*>malloc(total_projections * sizeof(float))
        if not c_geom.COR:
            raise MemoryError()

        # --- copy normalized data ---
        for i in range(total_projections):
            # offOrigin indices per your previous code: [2]=X, [1]=Y, [0]=Z
            c_geom.offOrigX[i] = _offOrigin[i, 2]
            c_geom.offOrigY[i] = _offOrigin[i, 1]
            c_geom.offOrigZ[i] = _offOrigin[i, 0]
            c_geom.DSO[i]      = _DSO[i]

            # offDetector: [1] -> U, [0] -> V
            c_geom.offDetecU[i] = _offDetector[i, 1]
            c_geom.offDetecV[i] = _offDetector[i, 0]
            c_geom.DSD[i]       = _DSD[i]

            # rotDetector: [2]=roll, [1]=pitch, [0]=yaw
            c_geom.dRoll[i]  = _rotDetector[i, 2]
            c_geom.dPitch[i] = _rotDetector[i, 1]
            c_geom.dYaw[i]   = _rotDetector[i, 0]

            c_geom.COR[i]    = _COR[i]

        # --- detector scalars & units ---
        c_geom.nDetecU = p_geometry.nDetector[1]
        c_geom.nDetecV = p_geometry.nDetector[0]
        c_geom.sDetecU = p_geometry.sDetector[1]
        c_geom.sDetecV = p_geometry.sDetector[0]
        c_geom.dDetecU = p_geometry.dDetector[1]
        c_geom.dDetecV = p_geometry.dDetector[0]

        c_geom.unitX = 1.0
        c_geom.unitY = 1.0
        c_geom.unitZ = 1.0

        c_geom.accuracy = p_geometry.accuracy

        return c_geom

    except Exception:
        free_c_geometry(c_geom)
        raise