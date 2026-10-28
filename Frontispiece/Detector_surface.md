# Detector-surface geometry (curved, modular and U-shaped detectors)

The [MATLAB detector builder and animation guide](./TIGRE_CT_Simulation_Builder_MATLAB.md)
covers native surface projection/backprojection, CGLS/SIRT, circular surface
FDK, visualization and YAML/MAT/NPZ interchange. Matching head-phantom examples
are demo 25 (smooth cylinder) and demo 26 (tilted modular panels); see the
[demo comparison](Demo_language_comparison.md).

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
vol  = tigre.algorithms.fdk_surface(data, geo, surf)     # validated circular path
vol  = tigre.algorithms.fdk_surface(data, geo, surf, allow_approximate=True)
tigre.animate_geometry(geo, surface=surf)     # draw it
```

Data in gap columns is ignored. `surface_from_geometry(geo, angles)` builds the surface
equivalent of an ordinary flat-detector geometry.

## Building modular layouts

For equal-sized flat panels, use `cylindrical_modular_detector_surface` for a
cylindrical arc or `piecewise_linear_modular_detector_surface` for flat, faceted,
or U-shaped arrangements. Both return a `DetectorSurfaceGeometry`; source azimuth
sets the per-view gantry rotation, and `column_gap` creates masked columns in the
dense data layout.

```python
import numpy as np

from tigre.utilities.detector_geometry import (
    cylindrical_modular_detector_surface,
    piecewise_linear_modular_detector_surface,
)

arc = cylindrical_modular_detector_surface(
    source_positions_mm, module_shape=(64, 32), pixel_spacing_mm=(1.0, 1.0),
    detector_radius_mm=600.0, n_modules=8, angular_span_rad=np.deg2rad(120),
    column_gap=2,
)

# U-shaped three-panel layout: centers are (normal, transverse, z) offsets
# from the nominal detector plane; panel angles are in radians.
u_shape = piecewise_linear_modular_detector_surface(
    source_positions_mm, module_shape=(64, 32), pixel_spacing_mm=1.0,
    detector_radius_mm=600.0,
    module_centers_mm=[
        [0.0, -32.0, 0.0], [10.0, 0.0, 0.0], [0.0, 32.0, 0.0],
    ],
    panel_angles_rad=[np.deg2rad(45), 0.0, np.deg2rad(-45)],
    column_gap=2,
)
```

`module_shape` is `(rows, columns)`, and `pixel_spacing_mm` is either one pitch
or a `(row, column)` pair. The cylindrical helper distributes module centers
over `angular_span_rad`, with every panel tangent to the cylinder. For
piecewise-linear layouts, `module_centers_mm` gives each panel center relative
to the nominal plane center at `(-detector_radius_mm, 0, detector_center_z_mm)`;
`panel_angles_rad` rotates each panel about z, with zero facing the source.
The layout follows source azimuth around z and assumes the detector is centered
on that rotation axis. `DetectorModule` remains available for unequal module
shapes, custom per-view poses, and other geometry.

To configure a layout interactively and inspect a 3D preview, run
`python -m tigre.utilities.detector_geometry_gui` from the `Python/` project
directory, or run `python tigre/utilities/detector_geometry_gui.py` from that
directory. The window offers cylindrical and piecewise-linear layouts,
including U-shaped panels; rotating scans can specify an orbit start and scan
span and COR. Both layouts support panel x/y tilts; stationary-detector modes
support linear and circular-arc source motion. Panel angles and COR are
constant during each GUI scan. See the
[modular detector GUI manual](./TIGRE_CT_Simulation_Builder_Python.md) for launch
instructions, control definitions, workflows, a TIGRE circular CT example, and
reconstruction usage.

The GUI exports `source_positions_mm`, `pixel_centers_mm`, and `valid_mask` in
a compressed `.npz` file, with optional integer `module_slices` metadata. Its background compatibility check validates the
surface geometry and reports whether it meets `fdk_surface`'s stricter
circular-orbit requirements. This does not test CUDA availability or run a
reconstruction. The GUI supports up to 6,000 views, 4,096 pixels per panel
dimension, 256 panels, and 20,000,000 dense detector samples; these are upper
bounds rather than a guarantee that the host has enough memory.
If a circular-orbit detector is incompatible with `fdk_surface`'s extruded,
view-invariant fan requirements, the GUI can optionally interpolate projection
data onto an idealized circular/equiangular detector. This changes the
measurements and cannot repair a non-circular source path; review the warnings
and validation limits in the [GUI manual](./TIGRE_CT_Simulation_Builder_Python.md#remap-measured-projections).

The GUI builder adds panel-centred rotations using
`Rz(gantry + panel_z) @ Ry(panel_y) @ Rx(panel_x)`. Its x/y inputs are in
degrees and accept one angle or one per panel. GUI labels name x/y/z rotations
roll/pitch/yaw, respectively; the explicit axes and matrix define the
convention, since rotation names vary between tools. COR shifts source and detector
in the rotating transverse direction. Exported coordinates include these
corrections; do not apply them twice. The low-level modular helpers retain
their z-angle-only interfaces. Use `surface_from_geometry(geo, angles)` for
per-view `rotDetector` or COR, or `DetectorModule` for custom poses.

**Multi-panel alignment:** panel tilt is independently configurable, while
COR is a single global scan correction that shifts the shared source and all
panels together. In the GUI, x/y angles accept a shared value or one per panel;
piecewise-linear z angles and transverse/normal/z offsets require one entry
per panel. Cylindrical z angles come from the arc layout. Use panel offsets
and tilts for individual panel calibration, rather than per-panel COR values.
See the [multi-panel alignment instructions](./TIGRE_CT_Simulation_Builder_Python.md#multi-panel-alignment-individual-tilt-global-cor)
for a three-panel example. The GUI provides **From list...** editors for panel
angles and offsets, with the same enabled/disabled rules as the main fields.

The GUI's **Load geometry...** opens surface or standard numeric TIGRE
geometry archives in read-only inspection mode. Versioned modular YAML settings
reopen in editable builder mode instead. **Save geometry YAML...** writes
editable recipes or a small reference to an exact surface NPZ companion. Construction controls are
disabled while playback, saving, and remapping remain available. Return to
builder restores its retained settings. From scripts use
`launch_detector_geometry_gui(geometry=geo, angles=angles)` or
`launch_detector_geometry_gui(surface=surface, geometry=geo)`. See the
[loading instructions and archive schema](./TIGRE_CT_Simulation_Builder_Python.md#load-existing-geometry-for-inspection).
The GUI and `animate_geometry(surface=...)` both use `SurfaceRenderer` for
mesh construction, source/volume display and frame updates. The legacy
standard-geometry animation path remains available.

Use `tigre.animate_geometry("geometry.yaml")` or
`tigre.animate_geometry("surface.npz")` to visualize saved geometry with the
same loader. File inputs provide their own trajectory. Existing geometry-object
calls remain supported. YAML uses explicit mm/degree units and a versioned
rotation convention; standard TIGRE arrays remain in radians. See the
[YAML settings guide](./TIGRE_CT_Simulation_Builder_Python.md#save-and-reload-editable-yaml-settings)
and `Python/demos/geometry_examples/three_panel.yaml` for an editable example.
The shared loader API is `tigre.utilities.geometry_io.load_geometry_file`;
`save_geometry_yaml` saves recipes or exact-surface references.

## Available functions

| Function | Notes |
|---|---|
| `Ax_surface`, `Atb_surface` | Siddon ray-driven projector and its exact adjoint |
| `cgls_surface`, `sirt_surface`, `os_sart_surface`, `fast_os_sart_surface` | iterative solvers; any source/detector motion |
| `fdk_surface` | validated circular path; opt-in experimental per-view path |
| `DetectorSurfaceGeometry.mask_outside_volume` | removes pixels that see no volume |
| `animate_geometry(..., surface=)` | visualisation |
| `cylindrical_modular_detector_surface` | equal-sized flat modules along a cylindrical arc |
| `piecewise_linear_modular_detector_surface` | equal-sized flat, faceted, or U-shaped modules |
| `tigre.utilities.detector_geometry_gui` | interactive layout preview and `.npz` export |
| `linear_source_surface` | stationary detector, source on a line (limited angle; iterative recommended, experimental FDK opt-in) |
| `remap_projections_for_fdk` | optional interpolation of projections onto a compatible circular detector; see limitations below |

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
`max_short_deficiency`). By default, inputs outside these limits are rejected. Passing
`allow_approximate=True` opts into a separate experimental path for per-view curved,
modular, or moving detector surfaces. That path filters each detector row in its
per-view fan-angle coordinate and backprojects with `Atb_surface`; it requires
monotone fan angles along every row. For non-circular source paths it derives angular
weights from source azimuth and skips Parker weighting. This is a filtered
backprojection experiment, **not a validated general-trajectory FDK**; expect artifacts
and use iterative solvers for quantitative results. It emits a runtime warning on use.

`tigre.utilities.detector_geometry_remap.remap_projections_for_fdk(projections,
surface)` can resample measured projection values for a supported detector onto
an idealized equiangular circular surface. It returns the remapped data, target
surface, and a report. The source must already follow a monotone circular orbit
at constant height. This opt-in linear interpolation does not recover missing
measurements, and can blur data or bias results. It rejects partial-column
masks, overlapping module ray coverage, non-monotone segments, and insufficient
common coverage; no extrapolation is performed and gaps remain masked. The
report estimates whether the scan meets the default FDK short-scan tolerance,
but does not test CUDA availability or run a reconstruction. Always validate
remapped reconstructions against results from the original measured geometry.

## Examples and tests

`Python/demos/d_geometry_yaml_reconstruction.py` exercises the complete Python
workflow: load the shipped three-panel YAML unchanged, load the bundled head phantom
with `sample_loader.load_head_phantom` (as in `Python/example.py`), simulate
projections with `Ax_surface`, reconstruct with 100 CGLS iterations,
and save `results.npz`, `metrics.json`, and a three-plane `comparison.png`.
The volume is explicitly chosen as 32 x 96 x 96 mm (z/y/x) for that example;
other input geometries may need a different reconstruction volume. Run from
`Python/`, in `recon` with a compatible CUDA surface build:

```powershell
conda activate recon
python demos/d_geometry_yaml_reconstruction.py --output results/yaml_reconstruction
```

The default head is resampled to 16 x 32 x 32 voxels for a quick demonstration.
`--phantom ellipsoids` selects the original synthetic phantom. Use
`--voxels Z Y X` to change resolution, `--geometry path.yaml` to select geometry,
and `--iterations 100` to set the iteration count. Increasing voxel resolution
without increasing detector/angular sampling may produce undersampling.
The head is fitted into the explicitly selected physical volume; this does
not preserve an anatomical calibration from the source dataset. These are
simulated projections from a data-derived volume, not measured projections. Output includes the exact surface, mask, module boundaries, phantom,
reconstruction, measured/predicted projections, residual history, volume size,
and GPU information. This is a noiseless matched-Siddon simulation; a low error
does not establish measured-data or real-scanner accuracy. The CUDA regression
test is `Python/tests/test_yaml_reconstruction_demo.py`.

The [GUI manual](./TIGRE_CT_Simulation_Builder_Python.md#tigre-circular-ct-example) uses
TIGRE's low-resolution circular geometry from `d03_generateData.py`.
`test_detector_geometry_examples.py` checks all 100 poses from
`d17_DetectorRotation.py` and the geometry in `d01_CreateGeometry.py`, with
additional COR sign checks. `test_detector_geometry.py` covers GUI geometry,
preview and input validation; `test_detector_geometry_remap.py` covers data
remapping. These CPU geometry tests do not run reconstruction. The manual
includes their test command. Use the project's `recon` conda environment.

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
a positive integer and `fdk_surface(cutoff=)` must lie in (0, 1]. The experimental path
also requires at least two valid columns per detector row and monotone fan angles.
Violations raise `ValueError` or `NotImplementedError`. CUDA error paths free all
device memory before returning.

## Visualizing curved and modular detectors

`animate_geometry(surface=...)` accepts changing validity masks and arbitrary
per-view source/pixel coordinates. Use `view=(elev, azim)`, e.g. `view=(90, 0)`
for a top view. The sampled triangular mesh follows curved grids; one-row or
one-column detectors use line segments. Masked pixels are not bridged, and
explicit `module_slices` preserve both row and column tile boundaries, even
when tiles have no masked gap. Without module metadata, contiguous valid
column runs are used; adjacent unmarked tiles cannot be distinguished.
Mesh vertices are pixel centres, not extrapolated physical detector edges.
The mesh samples up to 24 rows and 48 columns per module, and the point cloud
shows up to 4000 valid samples per view. Fine mask holes can suppress a sampled
mesh cell; the pixel cloud still shows valid samples. No projection data or
GPU computation is needed for drawing. Keep the returned animation alive:

```python
import matplotlib.pyplot as plt

ani = tigre.animate_geometry(geo, surface=surf, view=(26, 52))
plt.show()
# Static view:
ax = tigre.animate_geometry(geo, surface=surf, pos=3, animate=False)
```

The source marker, detector mesh, sampled points, and central ray update with
the selected view, including views with no valid pixels. Surface mode supports
`rotation="SD"` only. Origin arrows match the GUI: red x, green y, blue z at
30 percent opacity. For thin scan layouts, the z-axis display range is expanded
to at least half the larger x/y range, preserving equal millimetre scale across
axes rather than stretching the detector. Surface coordinates already contain any COR correction;
a separate COR marker cannot be inferred reliably from arbitrary trajectories.
CPU rendering tests are in `Python/tests/test_animate_surface_mesh.py`.
