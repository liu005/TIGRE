#ifndef SIDDON_SURFACE_PROJECTION_HPP
#define SIDDON_SURFACE_PROJECTION_HPP

/* Ray-voxel (Siddon) forward projection for an arbitrary detector surface.
 *
 * All positions are in mm, in the volume-fixed lab frame (x,y,z); the volume is
 * centred at offOrig (usually 0). The image is laid out with x fastest
 * (numpy shape (nz,ny,nx)).
 *
 * source : nviews*3 floats (x,y,z)
 * pixels : pixel_views*rows*cols*3 floats; pixel_views is 1 (shared by all views)
 *          or nviews
 * mask   : NULL or mask_views*rows*cols bytes (0 = missing sample), mask_views 1 or nviews
 * result : nviews*rows*cols floats; masked samples are set to 0
 *
 * Returns 0 on success.
 */
int siddon_surface_projection(const float* img,
        int nx, int ny, int nz,
        float dx, float dy, float dz,
        float offx, float offy, float offz,
        int nviews, int rows, int cols,
        const float* source,
        const float* pixels, int pixel_views,
        const unsigned char* mask, int mask_views,
        float* result, int device);

/* Exact adjoint of siddon_surface_projection: scatters proj (nviews*rows*cols)
 * into volume (nx*ny*nz, overwritten) with the same ray weights. */
int siddon_surface_backprojection(const float* proj,
        int nx, int ny, int nz,
        float dx, float dy, float dz,
        float offx, float offy, float offz,
        int nviews, int rows, int cols,
        const float* source,
        const float* pixels, int pixel_views,
        const unsigned char* mask, int mask_views,
        float* volume, int device);

/* Voxel-driven FDK-style backprojection for a circular orbit about the z axis.
 *
 * q      : nviews*rows*cols filtered data (view angle weights already applied)
 * source : nviews*3
 * gamma  : cols ascending in-plane fan angles (rad) from the source->origin ray,
 *          counter-clockwise positive; identical for all views
 * rho    : cols in-plane source->pixel distances (mm)
 * zrow   : rows ascending detector heights (mm), shared by all columns
 * volume : nx*ny*nz floats, overwritten. Each sample is weighted by 1/L^2, with L
 *          the in-plane source->voxel distance. Rays outside the table add 0. */
int fdk_surface_backprojection(const float* q,
        int nx, int ny, int nz,
        float dx, float dy, float dz,
        float offx, float offy, float offz,
        int nviews, int rows, int cols,
        const float* source,
        const float* gamma, const float* rho, const float* zrow,
        float* volume, int device);

#endif
