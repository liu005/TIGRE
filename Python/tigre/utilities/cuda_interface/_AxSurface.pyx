cimport numpy as np
import numpy as np

np.import_array()

cdef extern from "Siddon_surface_projection.hpp":
    cdef int siddon_surface_projection(const float* img,
            int nx, int ny, int nz,
            float dx, float dy, float dz,
            float offx, float offy, float offz,
            int nviews, int rows, int cols,
            const float* source,
            const float* pixels, int pixel_views,
            const unsigned char* mask, int mask_views,
            float* result, int device) nogil
    cdef int siddon_surface_backprojection(const float* proj,
            int nx, int ny, int nz,
            float dx, float dy, float dz,
            float offx, float offy, float offz,
            int nviews, int rows, int cols,
            const float* source,
            const float* pixels, int pixel_views,
            const unsigned char* mask, int mask_views,
            float* volume, int device) nogil
    cdef int fdk_surface_backprojection(const float* q,
            int nx, int ny, int nz,
            float dx, float dy, float dz,
            float offx, float offy, float offz,
            int nviews, int rows, int cols,
            const float* source,
            const float* gamma, const float* rho, const float* zrow,
            float* volume, int device) nogil


def _fdk_surface_ext(np.ndarray[np.float32_t, ndim=3] q,
                     shape_zyx, voxel_size, offset_origin,
                     np.ndarray[np.float32_t, ndim=2] source,
                     np.ndarray[np.float32_t, ndim=1] gamma,
                     np.ndarray[np.float32_t, ndim=1] rho,
                     np.ndarray[np.float32_t, ndim=1] zrow,
                     int device=0):
    """Voxel-driven backprojection of filtered data q (views, rows, cols)."""
    q = np.ascontiguousarray(q)
    source = np.ascontiguousarray(source)
    gamma = np.ascontiguousarray(gamma)
    rho = np.ascontiguousarray(rho)
    zrow = np.ascontiguousarray(zrow)
    cdef int nviews = q.shape[0]
    cdef int rows = q.shape[1]
    cdef int cols = q.shape[2]
    cdef int nz = shape_zyx[0]
    cdef int ny = shape_zyx[1]
    cdef int nx = shape_zyx[2]
    cdef float dx = voxel_size[0]
    cdef float dy = voxel_size[1]
    cdef float dz = voxel_size[2]
    cdef float ox = offset_origin[0]
    cdef float oy = offset_origin[1]
    cdef float oz = offset_origin[2]
    volume = np.zeros((nz, ny, nx), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] volume_c = volume
    cdef int status
    with nogil:
        status = fdk_surface_backprojection(<const float*> q.data, nx, ny, nz, dx, dy, dz, ox, oy, oz,
                                            nviews, rows, cols,
                                            <const float*> source.data,
                                            <const float*> gamma.data, <const float*> rho.data,
                                            <const float*> zrow.data,
                                            <float*> volume_c.data, device)
    if status != 0:
        raise ValueError('TIGRE: Call to fdk_surface backprojection failed')
    return volume


def _Atb_surface_ext(np.ndarray[np.float32_t, ndim=3] proj,
                     shape_zyx, voxel_size, offset_origin,
                     np.ndarray[np.float32_t, ndim=2] source,
                     np.ndarray[np.float32_t, ndim=4] pixels,
                     mask=None, int device=0):
    """Exact adjoint of _Ax_surface_ext. proj: (views, rows, cols); returns (nz, ny, nx)."""
    proj = np.ascontiguousarray(proj)
    source = np.ascontiguousarray(source)
    pixels = np.ascontiguousarray(pixels)
    cdef int nviews = source.shape[0]
    cdef int pixel_views = pixels.shape[0]
    cdef int rows = pixels.shape[1]
    cdef int cols = pixels.shape[2]
    cdef int nz = shape_zyx[0]
    cdef int ny = shape_zyx[1]
    cdef int nx = shape_zyx[2]
    cdef float dx = voxel_size[0]
    cdef float dy = voxel_size[1]
    cdef float dz = voxel_size[2]
    cdef float ox = offset_origin[0]
    cdef float oy = offset_origin[1]
    cdef float oz = offset_origin[2]

    cdef np.ndarray[np.uint8_t, ndim=3] mask_arr
    cdef const unsigned char* c_mask = NULL
    cdef int mask_views = 0
    if mask is not None:
        mask_arr = np.ascontiguousarray(mask, dtype=np.uint8)
        mask_views = mask_arr.shape[0]
        c_mask = <const unsigned char*> mask_arr.data

    volume = np.zeros((nz, ny, nx), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] volume_c = volume
    cdef int status
    with nogil:
        status = siddon_surface_backprojection(<const float*> proj.data, nx, ny, nz, dx, dy, dz, ox, oy, oz,
                                               nviews, rows, cols,
                                               <const float*> source.data,
                                               <const float*> pixels.data, pixel_views,
                                               c_mask, mask_views,
                                               <float*> volume_c.data, device)
    if status != 0:
        raise ValueError('TIGRE: Call to Atb_surface failed')
    return volume


def _Ax_surface_ext(np.ndarray[np.float32_t, ndim=3] img,
                    voxel_size, offset_origin,
                    np.ndarray[np.float32_t, ndim=2] source,
                    np.ndarray[np.float32_t, ndim=4] pixels,
                    mask=None, int device=0):
    """Siddon forward projection with explicit source/pixel coordinates.

    img: (nz, ny, nx); source: (views, 3); pixels: (1 or views, rows, cols, 3);
    mask: None or uint8 (1 or views, rows, cols). Returns (views, rows, cols).
    """
    img = np.ascontiguousarray(img)
    source = np.ascontiguousarray(source)
    pixels = np.ascontiguousarray(pixels)
    cdef int nviews = source.shape[0]
    cdef int pixel_views = pixels.shape[0]
    cdef int rows = pixels.shape[1]
    cdef int cols = pixels.shape[2]
    cdef int nz = img.shape[0]
    cdef int ny = img.shape[1]
    cdef int nx = img.shape[2]
    cdef float dx = voxel_size[0]
    cdef float dy = voxel_size[1]
    cdef float dz = voxel_size[2]
    cdef float ox = offset_origin[0]
    cdef float oy = offset_origin[1]
    cdef float oz = offset_origin[2]

    cdef np.ndarray[np.uint8_t, ndim=3] mask_arr
    cdef const unsigned char* c_mask = NULL
    cdef int mask_views = 0
    if mask is not None:
        mask_arr = np.ascontiguousarray(mask, dtype=np.uint8)
        mask_views = mask_arr.shape[0]
        c_mask = <const unsigned char*> mask_arr.data

    result = np.zeros((nviews, rows, cols), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] result_c = result
    cdef int status
    with nogil:
        status = siddon_surface_projection(<const float*> img.data, nx, ny, nz, dx, dy, dz, ox, oy, oz,
                                           nviews, rows, cols,
                                           <const float*> source.data,
                                           <const float*> pixels.data, pixel_views,
                                           c_mask, mask_views,
                                           <float*> result_c.data, device)
    if status != 0:
        raise ValueError('TIGRE: Call to Ax_surface failed')
    return result
