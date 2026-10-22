# Detector-surface geometry (curved, modular and U-shaped detectors)

TIGRE's standard geometry describes a flat detector. The *detector-surface* model
describes every measured pixel by its physical position, so curved arrays, modules
with gaps, nonuniform column spacing and U-shaped arrays can be used directly,
without rebinning the data to a flat uniform detector.

A surface is a `DetectorSurfaceGeometry`:

| Field | Shape | Meaning |
|---|---|---|
| `source_positions_mm` | `(views, 3)` | source position per view, world (x, y, z) in mm |
| `pixel_centers_mm` | `(views, rows, cols, 3)` | centre of every pixel per view |
| `valid_mask` | `(views, rows, cols)` | false marks absent pixels (gaps, pixels outside the beam) |

The world frame is the one drawn by `tigre.animate_geometry`: x, y, z, with the
volume centred at `offOrigin`. Volume geometry (`nVoxel`, `dVoxel`, `offOrigin`) still
comes from an ordinary `tigre.geometry()`.

```python
import numpy as np, tigre
from tigre.utilities.detector_geometry import DetectorModule, DetectorSurfaceGeometry

# local (rows, cols, 3) pixel centres of one module at view 0 + a 4x4 pose per view
module = DetectorModule(local_centres, poses, column_offset=0)
surf = DetectorSurfaceGeometry.from_modules(source_positions, [module_a, module_b])
surf = surf.mask_outside_volume(geo)          # drop pixels whose ray misses the volume

data = tigre.Ax_surface(volume, geo, surf)    # forward projection, (views, rows, cols)
vol  = tigre.algorithms.fast_os_sart_surface(data, geo, surf, niter=10, blocksize=20)
vol  = tigre.algorithms.fdk_surface(data, geo, surf)     # circular orbits only
tigre.animate_geometry(geo, surface=surf)     # draw it
```

Data in gap columns is ignored. `surface_from_geometry(geo, angles)` builds the surface
equivalent of an ordinary flat-detector geometry.

## Available functions

| Function | Notes |
|---|---|
| `Ax_surface`, `Atb_surface` | Siddon ray-driven projector and its exact adjoint |
| `cgls_surface`, `sirt_surface`, `os_sart_surface`, `fast_os_sart_surface` | iterative solvers; any source/detector motion |
| `fdk_surface` | analytic, approximate; circular orbits only |
| `DetectorSurfaceGeometry.mask_outside_volume` | removes pixels that see no volume |
| `animate_geometry(..., surface=)` | visualisation |
| `linear_source_surface` | stationary detector, source on a line (limited angle; iterative solvers only) |

## Limits

* **GPU memory / multi-GPU.** All surface routines (`Ax_surface`, `Atb_surface`,
  `fdk_surface` and the iterative solvers) take `gpuids=` (a
  `tigre.utilities.gpu.GpuIds`, default all GPUs, as in `tigre.Ax`) and, for the
  projectors and FDK, `n_slabs`. The volume is split into z-slabs run concurrently
  (forward projections are summed, backprojections concatenated), so the volume need
  only fit per slab.
  Unlike core TIGRE, views are not split across devices and slabs are not chosen
  automatically from free memory. Projection and backprojection
  use `pixel_centers_mm` as supplied: `views x rows x cols x 3` floats on the host.
* **Cone-beam Siddon only.** The projector is a single-ray Siddon model with no
  detector-element footprint and no `accuracy` parameter.
* **One `offOrigin`.** The volume offset must be the same for all views.
* **Backprojection uses atomics.** Results are reproducible only to rounding.
* **Rays are the segment source -> pixel.** A source or pixel inside the volume box
  contributes only the path up to that point.
* **Row sampling.** The row pitch projected to the isocentre should not exceed the
  voxel size, otherwise the problem is under-sampled.
* **Convergence with gaps.** Iterative methods converge more slowly with large gaps
  (a three-module test detector needed 100 CGLS iterations for 5 % error).

### `fdk_surface` (approximate)

An equiangular fan-beam FDK with a Feldkamp cone weight and Parker weights for short
scans (< 2 pi). It requires, and checks:

* circular source orbit about z at constant height;
* detector extruded along z: pixel x/y independent of the row, z independent of
  column and view;
* the same valid columns in every view (whole-column gaps only);
* identical fan-angle tables in every view, strictly monotone along the columns
  (a U-shaped array that rotates with the source qualifies if every ray meets exactly
  one panel).

Approximations: cone-beam error grows with the cone angle, as in any FDK; gap columns
are bridged by linear interpolation of the data before filtering, so wide gaps leave
artefacts; short scans must span at least `pi + 2 * max fan angle` (to within
`max_short_deficiency`). Use the iterative solvers for anything outside these limits
(non-circular orbits, fixed detectors with a linearly moving source, arbitrary shapes).

## Examples and tests

`Python/demos/d_detector_surface_gct.py` reconstructs measured data from a 53-module
arc detector (`--alg fdk|fast_os_sart`). Validated at 4096² against the vendor
reconstruction (NCC 0.85 with FDK in 11 s).

`Python/tests/test_u_detector.py` builds a three-panel U and reconstructs it with the
iterative and FDK methods; `test_surface_algorithms.py` has an arc and a modular
detector with gaps; `test_surface_slabs.py` and `test_surface_validation.py` cover
multi-GPU slab splitting (anisotropic voxels, offsets, adjoint identity) and input
validation. Run them with `python -m pytest tests/test_*surface*.py tests/test_u_detector.py
tests/test_linear_source.py tests/test_detector_geometry.py tests/test_ax_surface.py`
from `Python/` (a CUDA GPU is required).

## Input validation

Geometry arrays must be finite and within float32 range; projection data must be finite
at valid samples (non-finite values at masked samples are ignored); `blocksize` must be
a positive integer and `fdk_surface(cutoff=)` must lie in (0, 1]. Violations raise
`ValueError`. CUDA error paths free all device memory before returning.

`animate_geometry(surface=...)` needs a `valid_mask` that is the same in every view and
accepts `view=(elev, azim)`, e.g. `view=(90, 0)` for a top view of a U-shaped array.
