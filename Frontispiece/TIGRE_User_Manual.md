**TIGRE**\
**User Manual — Python and MATLAB**

**Manual version: 1.1**\
**Updated: 9 November 2026**\
**Software scope: this development checkout, including explicit detector surfaces and projection operators.**

This manual describes the reconstruction toolbox. The desktop application and
its manuals are maintained in the separate [tigre-geometry-design-tool repository](../../tigre-geometry-design-tool/README.md).
Development APIs described here must not be assumed to exist in a stock upstream release.

# Working with this manual

Start with the installation check and one conventional reconstruction. Move to
explicit detector surfaces when your acquisition needs curved pixels, modular
panels, gaps or independently specified source/detector positions. Keep the
conventional calling style for ordinary flat-detector work.

The examples use small, synthetic volumes. Their purpose is to make array order,
units, API calls and diagnostics reproducible. They are not scanner calibration
procedures or performance benchmarks. A successful numerical test does not
establish quantitative accuracy on measured data.

The canonical edition is this Markdown document. To generate PDF and standalone
HTML editions, activate an environment containing Pandoc and XeLaTeX, then run
`.\Frontispiece\Export_User_Manual.ps1` from the repository root. The
[operator reference](Projection_operator.md), [surface reference](Detector_surface.md)
and [validation guide](Validation.md) provide implementation details.
The [integration audit](Integration_audit_2026-11-06.md) records the current
checkout's verification and limitations.

| If you need to… | Read |
| --- | --- |
| Verify the checkout and environment | Installation and first checks |
| Reconstruct a conventional scan | First reconstruction |
| Work with curved or missing detector samples | Explicit detector surfaces |
| Choose an iterative method | Algorithm selection and reference |
| Use TV, POCS or warm starts | Regularization and initialization |
| Transfer geometry between languages | Geometry files and interchange |
| Diagnose unexpected results | Diagnostics and troubleshooting |
| Change the implementation | Verification and contribution workflow |

# Installation and first checks

## Keep source, environment and binaries together

TIGRE combines Python or MATLAB code with compiled CUDA extensions. A source
checkout alone does not provide compatible native binaries. Changes to native
sources, CUDA/toolchain versions or binding interfaces can require rebuilding.
Do not copy a binary from an unrelated checkout and assume it implements the
current API.

For this workspace, Python execution uses the **`recon` Conda environment**.
Use activation or `conda run -n recon`; selecting an interpreter executable by
path alone can omit required DLL search paths on Windows.

From the TIGRE repository root:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'Python'
conda run -n recon python -c "import tigre; print(tigre.__file__)"
conda run -n recon python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
```

The printed TIGRE path must point to the intended checkout when validating local
changes. An older installed copy can otherwise make a new module appear missing
or make tests exercise a different implementation.

## Python build and dependencies

TIGRE requires **Python 3.8 or newer** on Windows and Linux. The package declares
`python_requires=">=3.8"`; Python 3.7 and earlier are unsupported. Choose dependency
versions compatible with your interpreter, and build native extensions for that
interpreter and toolkit.

This fork uses `Python/setup.py`. Build from its `Python` directory in a prepared
CUDA/compiler environment; the local [installation guide](python_installation.md)
documents the build configuration. A typical editable build command is:

```powershell
conda run -n recon python -m pip install --no-build-isolation -e .\Python
```

That command is a build/install action, not an environment-discovery tool. Resolve
the CUDA toolkit, supported host compiler and dependencies first. Consult the
actual build output if compilation fails. The build name `pytigre` and Python
import name `tigre` are different names for different purposes.

The build filters detected GPU capabilities against its CUDA compatibility table.
In a mixed-device machine it warns about excluded devices and builds targets for
the remaining compatible GPUs. If none can be targeted, it stops with guidance
to use a compatible toolkit. If no GPUs are detected, it builds the table's
toolkit-compatible targets. An excluded GPU is not a supported execution device
for that build. See the [installation guide](python_installation.md) for details.

The bundled head phantom is found in the checkout's `Common/data` directory or
the installed environment's `data` directory, including Linux installations.

The source declares NumPy, SciPy, Matplotlib, Cython, tqdm and PyYAML dependencies.
CuPy and PyTorch are optional capabilities; they are not prerequisites for the
ordinary iterative examples below. See [optional CuPy filtering](Optional_CuPy.md).

## MATLAB setup

Use the repository's MATLAB initialization from the MATLAB directory, or add the
required folders and the matching MEX directory to the path. The following is
convenient for local source validation from the repository root:

```matlab
addpath(genpath('MATLAB'));
which Ax -all
which Surface_mex -all
```

Check for older MEX files earlier on the path. This workspace has an isolated
Blackwell build activated with `UseBlackwellBuild`; it is a local configuration,
not a general installation command for every platform. See
[MATLAB installation](MATLAB_installation.md) and the
[local build guide](MATLAB_Blackwell_build.md).

## GPU selection

Conventional Python calls traditionally use a `GpuIds` object:

```python
from tigre.utilities.gpu import GpuIds
gpuids = GpuIds()
gpuids.devices = [0]
```

`SurfaceOperator` also accepts a device integer or a sequence of device integers.
Device indices are zero-based in both languages. Invalid negative, fractional or
empty selections are rejected by the operator adapters. Device existence is
checked when CUDA is actually used.

MATLAB surface wrappers use `'Device',0`. Python surface projection can use
multiple devices or z-slabs through `gpuids` and `n_slabs`; the current MATLAB
surface interface selects one device. Do not infer multi-GPU MATLAB support from
the Python option.

# Coordinate systems, arrays and units

## Three orders to distinguish

Physical coordinates, array indices and acquisition order are separate concepts.
A point in world space is always expressed as **(x,y,z)**. Python volume arrays
use **(z,y,x)**; MATLAB volume arrays use **(x,y,z)**.

| Quantity | Python | MATLAB |
| --- | --- | --- |
| Volume and `nVoxel`/`dVoxel`/`sVoxel` | z, y, x | x, y, z |
| Projection array | views, rows, columns | rows, columns, views |
| Conventional `nDetector` | rows, columns | u/columns, v/rows |
| Source positions | views × xyz | xyz × views |
| Exact pixel centres | views × rows × columns × xyz | rows × columns × xyz × views |
| Exact valid mask | views × rows × columns | rows × columns × views |
| Subset indices | zero-based | one-based |

For equivalent numerical arrays, not metadata dictionaries:

```python
matlab_volume = python_volume.transpose(2, 1, 0)
matlab_projections = python_projections.transpose(1, 2, 0)
```

Use the geometry-file bridge for complete records rather than manually
transposing some fields and forgetting the others.

## Length, angle and signal conventions

Geometry lengths in the surface model and the examples are millimetres.
Conventional angle arrays and rotation fields are in radians. Headless modular
recipes use explicitly named degree fields; exact pixel/source coordinates no
longer need those construction angles during projection.

`sVoxel` is the full physical volume extent, not a radius or half-width:

```text
sVoxel = nVoxel * dVoxel
sDetector = nDetector * dDetector
```

If attenuation is expressed per millimetre and ray lengths are in millimetres,
the forward projection is a dimensionless line integral. Linear reconstruction
algorithms expect line integrals, not unprocessed detector intensities.

Do not change units in only one part of a geometry record. A millimetre/centimetre
mismatch can produce plausible-looking images with the wrong attenuation scale.

## Geometry fields

`nVoxel`, `dVoxel`, `sVoxel` and `offOrigin` describe the reconstruction volume.
For conventional geometry, detector sampling and `DSO`/`DSD`, detector offsets,
rotations and `COR` describe the acquisition. Some fields may vary per view.
Use TIGRE's validation/conversion functions rather than assuming every field is
a scalar after a geometry has been checked.

For a detector surface, the measured ray endpoints are explicit. Surface
projectors use volume fields from `geo` and the exact source/pixel positions.
Do not apply an additional conventional detector rotation or COR shift to
coordinates that already include those corrections.

# First reconstruction

## Python: a complete conventional example

The following block creates a synthetic
sphere with attenuation 0.02 per millimetre; that value is an example input, not
a material calibration.

```python
# manual-example: python-conventional
import numpy as np
import tigre
import tigre.algorithms as algs

geo = tigre.geometry_default(high_resolution=False)
geo.nVoxel = np.array([32, 32, 32])
geo.dVoxel = np.array([2., 2., 2.])
geo.sVoxel = geo.nVoxel * geo.dVoxel
geo.nDetector = np.array([48, 64])
geo.dDetector = np.array([2., 2.])
geo.sDetector = geo.nDetector * geo.dDetector
geo.DSO, geo.DSD = 1000., 1500.
geo.offOrigin = np.zeros(3)

z, y, x = np.indices(tuple(geo.nVoxel), dtype=float)
phantom = np.zeros(tuple(geo.nVoxel), np.float32)
phantom[(x - 15.5)**2 + (y - 15.5)**2 + (z - 15.5)**2 <= 8**2] = .02
angles = np.linspace(0, 2*np.pi, 90, endpoint=False, dtype=np.float32)
data = tigre.Ax(phantom, geo, angles)
image = algs.cgls(data, geo, angles, niter=12, verbose=False)

assert image.shape == phantom.shape and image.dtype == np.float32
assert np.isfinite(image).all()
```

This is the original calling style: data, geometry, angles, iteration count.
There is no need to construct an operator for this conventional scan.

## MATLAB: the corresponding workflow

```matlab
% manual-example: matlab-conventional
geo = defaultGeometry('nVoxel',[32;32;32]);
geo.dVoxel = [2;2;2]; geo.sVoxel = geo.nVoxel.*geo.dVoxel;
geo.nDetector = [64;48]; geo.dDetector = [2;2];
geo.sDetector = geo.nDetector.*geo.dDetector;
geo.DSO = 1000; geo.DSD = 1500; geo.offOrigin = [0;0;0];
[x,y,z] = ndgrid(0:31,0:31,0:31);
phantom = single(.02)*single((x-15.5).^2+(y-15.5).^2+(z-15.5).^2 <= 8^2);
angles = (0:89)*2*pi/90;
data = Ax(phantom,geo,angles);
image = CGLS(data,geo,angles,12,'Verbose',0);
assert(isequal(size(image),size(phantom)) && all(isfinite(image(:))));
```

The volume axes and detector metadata differ in storage order, as described
above. The physical acquisition is equivalent.

## Examine results before increasing scale

Compare a reconstructed central slice with the synthetic truth. Plot measured
and predicted projections using the same colour scale. Record the residual,
relative volume error where truth exists, reconstruction parameters and units.

Changing the iteration count, sampling, regularization and data model at the
same time makes failures difficult to diagnose. Establish one small working
case, then change one relevant assumption at a time.

# Explicit detector surfaces

## When to use them

Use explicit surfaces for curved detector pixels, tilted modules, gaps,
nonuniform spacing or source/detector motion not adequately represented by the
conventional flat-detector parameters. `DetectorSurfaceGeometry` stores the
source position, pixel centres and measurement mask for every view.

A surface need not describe a circular CT scan. Projection and iterative
reconstruction can use exact rays on other trajectories; this does **not** mean
that analytical FDK is valid for those trajectories.

## Convert a conventional scan and introduce a gap

```python
# manual-example: python-surface
surface = tigre.surface_from_geometry(geo, angles)
mask = surface.valid_mask.copy()
mask[:, :, 31:33] = False
surface = tigre.DetectorSurfaceGeometry(
    surface.source_positions_mm, surface.pixel_centers_mm, mask)

surface_data = tigre.Ax_surface(phantom, geo, surface, gpuids=0)
surface_data[~mask] = np.nan
surface_image, residuals = algs.cgls_surface(
    surface_data, geo, surface, niter=12, gpuids=0,
    return_residuals=True)
assert np.isfinite(surface_image).all()
assert residuals[-1] < np.linalg.norm(np.nan_to_num(surface_data))
```

False mask entries are absent measurements, not zero-valued observations. The
operator solvers remove missing NaN/Inf values before arithmetic. Nonfinite
values at valid samples are rejected. Use the operator or iterative wrappers
for this masking contract; do not pass masked NaNs directly to a native kernel.

```matlab
% manual-example: matlab-surface
surface = surfaceFromGeometry(geo,angles);
surface.valid_mask(:,32:33,:) = false;
surfaceData = Ax_surface(phantom,geo,surface,0);
surfaceData(~surface.valid_mask) = NaN;
[surfaceImage,residuals] = CGLS_surface(surfaceData,geo,surface,12,'Device',0);
assert(all(isfinite(surfaceImage(:))));
```

## Construct rather than convert

Low-level constructors support cylindrical modules, piecewise-linear modules,
explicit source paths and custom per-view module poses. The headless
`build_modular_surface` recipe API adds panel-centred tilts and scan controls.
See [the surface reference](Detector_surface.md) for constructor shapes and
[demo 26](../Python/demos/d26_ModularDetector.py) for a three-panel acquisition.

The recipe builder imposes sample-count limits to prevent accidental oversized
allocations. These are allocation policy limits, not statements about detector
physics or the capacity of a particular computer.

# Calling the shared operator interface

## Three equivalent surface entry styles

For adapted methods, the following entry styles choose the same surface backend:

```python
# manual-example: python-operator
operator = tigre.SurfaceOperator(geo, surface, gpuids=0)
via_operator = algs.lsqr(surface_data, operator=operator, niter=8, verbose=False)
via_wrapper = algs.lsqr_surface(surface_data, geo, surface, niter=8,
                              gpuids=0, verbose=False)
np.testing.assert_allclose(via_operator, via_wrapper, atol=2e-5, rtol=2e-3)
```

The explicit `algs.lsqr_operator(data, operator, ...)` convenience is also
available. When passing `operator=`, omit `geo` and `angles` from the algorithm
call: the selected operator already owns its acquisition state.

```matlab
% manual-example: matlab-operator
op = projectionOperator(geo,surface,'Device',0);
viaOperator = LSQR_operator(surfaceData,op,8);
viaWrapper = LSQR_surface(surfaceData,geo,surface,8,'Device',0);
assert(norm(double(viaOperator(:)-viaWrapper(:))) < 1e-3);
```

Existing MATLAB conventional methods also accept an explicit `'Operator',op`
option with empty geometry/angles. `_operator` and `_surface` are the uniform
entry points for methods that did not previously have a conventional MATLAB
function.

## Contract

An operator exposes `forward`, `adjoint`, `subset`, volume/projection shapes and
a valid mask. It is linear and must not modify caller inputs. Normalized
algebraic methods additionally assume nonnegative ray weights. The conventional
adapter's legacy matched backprojection is approximate; the exact surface
Siddon projector/backprojector pair is a discrete adjoint pair.

Subsets select view indices relative to the current operator. Reordered,
repeated and nested indices are supported. Subset normalization is calculated
from that subset's own rays and mask. A completely missing view contributes
neither data nor sensitivity.

Adapters copy acquisition state when bound. Treat a bound operator's geometry
as fixed. Construct a new operator after changing the scan. Keep GPU selection
on the operator instead of supplying contradictory per-algorithm device options.

# Algorithm selection and reference

## Select a method for the problem you actually have

| Method family | Useful starting point | Important distinction |
| --- | --- | --- |
| CGLS / LSQR | Linear least-squares data fitting | Accurate adjoints matter; noise can be fitted with excessive iterations |
| LSMR | Least-squares/normal-equation residual control with damping | Damping changes the objective; the unregularized residual need not decrease monotonically |
| SIRT | Normalized simultaneous data updates | Slower updates can be easier to inspect; sensitivity is operator dependent |
| SART / OS-SART | Sequential or subset data updates | Subset ordering and relaxation affect the path |
| Fast OS-SART | Extrapolated subset updates | Momentum is a heuristic; compare quality at comparable noise levels |
| ISTA / FISTA | Data fitting with a volume-TV proximal step | Step-size/Lipschitz selection and TV strength must be consistent |
| ASD-POCS / PCSD | Alternating data consistency and TV descent | Requires an explicit data tolerance on the operator path |
| Hybrid / IRN Krylov-TV | Projected or iteratively reweighted regularization | More controls and greater numerical sensitivity; validate on a representative small case |
| MLEM | Nonnegative multiplicative updates | Valid measurements must be nonnegative; it is not interchangeable with least squares |

This table describes mathematical behavior, not a universal ranking. Compare
methods with the same data model, geometry, constraints and diagnostic measures.

## Available iterative APIs

Append `_operator` or `_surface` to each Python name below. MATLAB uses its
existing capitalization and established spellings.

| Python | MATLAB base name | Family |
| --- | --- | --- |
| `cgls` | `CGLS` | Least squares |
| `lsqr` | `LSQR` | Least squares |
| `lsmr` | `LSMR` | Damped least squares |
| `sart` | `SART` | Sequential algebraic |
| `sirt` | `SIRT` | Simultaneous algebraic |
| `os_sart` | `OS_SART` | Ordered subsets |
| `fast_os_sart` | `fast_OS_SART` | Accelerated ordered subsets |
| `sart_tv` | `SART_TV` | Sequential + TV |
| `os_sart_tv` | `OS_SART_TV` | Subsets + TV |
| `iterativereconalg` | `iterativereconalg` | Generic normalized iteration |
| `ista` | `ISTA` | Proximal TV |
| `fista` | `FISTA` | Accelerated proximal TV |
| `mlem` | `MLEM` | Multiplicative statistical update |
| `hybrid_lsqr` | `hybrid_LSQR` | Projected regularization |
| `irn_tv_cgls` | `IRN_TV_CGLS` | Reweighted TV |
| `hybrid_flsqr_tv` | `hybrid_fLSQR_TV` | Flexible Krylov-TV |
| `ab_gmres` | `AB_GMRES` | Projection-space GMRES |
| `ba_gmres` | `BA_GMRES` | Volume-space GMRES |
| `asd_pocs` | `ASD_POCS` | Single-view ASD |
| `os_asd_pocs` | `OS_ASD_POCS` | Subset ASD |
| `awasd_pocs` | `AwASD_POCS` | Weighted-TV ASD |
| `os_awasd_pocs` | `OS_AwASD_POCS` | Subset weighted-TV ASD |
| `pcsd` | `PCSD` | Single-view PCSD |
| `os_pcsd` | `OS_PCSD` | Subset PCSD |
| `aw_pcsd` | `AwPCSD` | Weighted-TV PCSD |
| `os_aw_pcsd` | `OS_AwPCSD` | Subset weighted-TV PCSD |

FDK and FBP remain geometry-specific analytical APIs. `fdk_surface` is a
separate approximate analytical surface reconstruction with additional geometry
requirements. A surface accepted by an iterative method need not be accepted
by surface FDK.

## Outputs are algorithm specific

| Entry | Diagnostics |
| --- | --- |
| Python `cgls_surface(..., return_residuals=True)` | `(image, residual_array)` |
| Python POCS methods with `return_history=True` | `(image, history_dict)` |
| Reused Python class methods with `computel2` or quality options | Existing `(image, errors)` convention |
| MATLAB POCS operator/surface methods | Second output is a history struct |
| Other MATLAB operator/surface methods | Residual history; quality output where supported |

Do not assume every solver accepts `return_history`. Check the
[operator guide](Projection_operator.md) before automating a parameter sweep.

# Regularization and initialization

## Warm starts and constraints

Supply a finite volume of the correct shape as `init`/`Init`. Solvers own their
initial reconstruction buffer; the caller's array is preserved. A zero-iteration
call returns an owned initializer, after the applicable initialization policy.
MLEM uses ones when no initializer is supplied; most other methods start at zero.

Some algebraic/proximal methods enforce nonnegativity during iteration. Krylov
methods must preserve their recurrences; upper bounds and support constraints
are applied at suitable initialization/terminal steps rather than arbitrarily
clipping every search direction. Existing conventional options retain their
historical behavior.

The operator path rejects geometry-specific FDK and multigrid initializer
strings. If a suitable initial reconstruction already exists, compute it
separately and pass the volume. GMRES must use the selected operator's adjoint,
not an unrelated FDK backprojector.

## TV controls and axes

TV operations act on the volume lattice. They do not independently correct an
incorrect acquisition geometry. The current native regularizers use discrete
voxel-axis differences; do not assume automatic physical-spacing compensation
for anisotropic voxels.

`tviter` controls the regularizer's iterations. TV strength parameters differ
between an explicit descent step and an ROF/proximal denoiser. A numeric value
of `alpha` is not interchangeable with a numeric value of `tvlambda`.

Weighted TV uses `delta`/`Delta`, a finite nonzero scalar or three thresholds.
Directional vectors follow the language's volume axes: Python z,y,x; MATLAB
x,y,z. Reverse the vector when exchanging equivalent geometry. Equal entries
reproduce the scalar case. Zero thresholds are rejected before native execution.

Native volume TV requires at least two voxels on each axis in the operator
adapter; constant volumes and zero TV iterations bypass unnecessary denoising.

## POCS maxl2err

`maxl2err` is a scalar **absolute L2 residual tolerance**, not a function. There
is currently no dedicated operator-aware estimator. Missing samples are excluded
from its residual. Conventional FDK-derived defaults cannot be assumed for an
arbitrary trajectory or detector surface.

```python
# manual-example: python-pocs
pocs_image, history = algs.asd_pocs_surface(
    surface_data, geo, surface, niter=4, gpuids=0,
    maxl2err=0., tviter=2, return_history=True)
assert np.isfinite(pocs_image).all()
assert len(history['residual']) <= 4
```

Zero is used here only for a noiseless synthetic demonstration. Select a measured
data tolerance using the data's units, noise/reference model and number of valid
measurements. A residual measured from an arbitrary candidate does not by itself
estimate an appropriate noise tolerance.

POCS history records post-TV `residual`, pre-TV `data_residual`, relaxation
`beta`, `tv_step` and update-direction `cosine`. PCSD uses residual-ratio TV
scaling; ASD uses its alpha/rmax adaptation. The operator PCSD path uses a
consistent norm gate and a strict iteration bound, and guards a zero reference
residual. See the operator guide for differences retained in conventional calls.

# Measured projection data

## Intensity preparation

The ideal transmission model is `I/I0 = exp(-line_integral)`. Constructing line
integrals from real measurements generally requires the detector's dark/flat
corrections and a valid reference intensity. Account for invalid, saturated,
dead or unilluminated samples before reconstruction.

Do not hide a calibration failure by clipping every problematic measurement and
calling the result a valid CT dataset. Preserve an explicit validity mask and
record preprocessing decisions. Negative corrected line integrals can occur in
some workflows; MLEM's nonnegative input requirement makes it unsuitable for
those measurements without a justified data model.

## Geometry must correspond to the data

Verify detector row/column orientation, acquisition order, units, source
trajectory and image origin against calibration or scanner metadata. Compare
forward projections of a simple known object before attempting a large unknown
sample. View-dependent geometry must be subset along with its measurements.

Python and MATLAB Varian/dual-energy workflows have different supported features.
Read the [Varian guide](Varian_loader.md) and [dual-energy guide](Dual_energy.md).
The audit does not establish correctness for real acquisitions unavailable in
this workspace. Measured GCT examples also require external projection files.

## Rebinning and remapping

Remapping measurements onto an idealized detector is a change of the data model,
not a substitute for exact geometry. Interpolation can affect resolution,
correlations and quantitative scaling. Preserve the original measurements and
document the interpolation. Prefer exact surface operators when the acquisition
can be represented directly.

# Geometry files and interchange

## Editable recipes versus exact rays

Recipes describe construction parameters. Exact archives describe the source
and measured pixel positions actually used by reconstruction. They answer
different questions and should not be substituted silently.

```python
# manual-example: python-interchange
from pathlib import Path
from tigre.utilities.geometry_io import save_geometry_yaml, load_geometry_file

folder = Path('manual_results')
folder.mkdir(exist_ok=True)
geometry_path = folder / 'surface.yaml'
save_geometry_yaml(geometry_path, surface=surface, geometry=geo)
loaded = load_geometry_file(geometry_path)
np.testing.assert_array_equal(loaded.surface.valid_mask, surface.valid_mask)
np.testing.assert_array_equal(loaded.surface.pixel_centers_mm, surface.pixel_centers_mm)
np.savez_compressed(folder / 'reconstruction.npz', image=surface_image,
                    residuals=residuals)
```

An exact YAML document can reference a uniquely named NPZ companion. Keep both
files together. Exact numeric archives are loaded without Python pickle. A raw
projection NPZ and a geometry NPZ are different records; distinguish them in
file naming and metadata.

MATLAB uses `geometryFile` for native MAT files and shared interchange. YAML/NPZ
conversion can invoke Python in `recon`; native MAT geometry does not require
the GUI application. The bridge handles axis ordering. A MATLAB recipe that
cannot express a supplied source path must not silently simplify it: use an
exact-surface archive for such acquisitions.

## Record enough information to reproduce a run

Save the geometry or its immutable identifier, measurement preprocessing and
mask, algorithm/options, initializer, iteration controls, environment/native
build and output units. Save useful histories alongside the image. Retain a
link to the original measured data instead of relying on a screenshot.

# Visualization and interpretation

Core plotting and animation are headless-capable scientific utilities. Set the
Matplotlib backend to `Agg` for automated Python runs; use a desktop backend
when interactive windows are intended. MATLAB figures can be hidden during
tests with `set(groot,'defaultFigureVisible','off')`.

Inspect geometry from several directions. A plausible central projection can
hide reversed rows, incorrect source motion or an origin shift. Plot source
positions, detector modules, the reconstruction volume and missing samples.
See [surface visualization](Detector_surface.md#visualizing-curved-and-modular-detectors).

Image contrast settings can hide scaling errors. Compare reconstructions with a
shared display range and evaluate numerical attenuation in a known region where
such a reference exists. A small relative data residual is not proof that every
part of a limited-angle or truncated volume is recoverable.

# Performance and memory planning

## Estimate before allocating

A float32 volume occupies `4 * product(nVoxel)` bytes, and a float32 projection
stack occupies `4 * views * rows * columns`. Exact pixel centres require three
coordinates per sample, or approximately twelve bytes per dense sample before
masks, source positions and metadata.

Iterative methods need additional volumes, projection buffers or basis vectors.
Krylov methods that retain bases can scale with iteration count. An acquisition
that fits in memory for one projector call need not fit in memory for a hybrid
solver. Geometry ownership and cached subsets can add another acquisition-sized
allocation; the audit removed a redundant extra subset copy, not all such costs.

Start with a reduced volume or fewer views, measure actual memory use, then scale.
Reducing views changes the inverse problem; it is a development check, not a
claim of equivalent image quality.

## Devices, slabs and optional accelerators

Python surface projection can split the reconstruction volume into z-slabs and
process them across selected devices. The returned projection sums or assembled
volume must still fit in host memory. Multi-device behavior depends on the
available devices and build; the current audit used one physical GPU.

CuPy filtering is explicit optional acceleration for applicable analytical
filtering paths. It does not select the projector for an iterative method and
does not remove the analytical geometry restrictions of FDK.

# Diagnostics and troubleshooting

| Symptom | Check first |
| --- | --- |
| New module not found | Printed `tigre.__file__`, `PYTHONPATH`, active Conda environment |
| DLL/MEX import failure | Matching environment, CUDA/compiler runtime and binary provenance |
| Wrong shape error | Volume axes, view axis, detector rows/columns and geometry counts |
| NaN/Inf at valid samples | Preprocessing, masks, initializer, parameter validity |
| Immediate Krylov failure on an exact start | Use current audited code; exact stationary starts/breakdowns are handled |
| Plausible image with wrong attenuation | Length units, intensity conversion, geometry and algorithm scaling |
| Different Python/MATLAB result | Axis order, Delta reversal, explicit equivalent options and iteration semantics |
| POCS asks for maxl2err | Supply an absolute masked L2 tolerance; no estimator/default is implied |
| Surface FDK rejects the scan | Check its circular/extruded geometry requirements; use an appropriate iterative model |
| Memory exhaustion | Volume/projection size, exact coordinate arrays, saved bases and cached subsets |

When reporting a problem, provide the smallest reproducer, exact shapes and
units, versions, selected devices, native build and complete error output.
Distinguish a compilation error from a numerical failure and from missing
external scanner data.

# Verification and contribution workflow

## Run the intended source

From the repository root:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'Python'
conda run -n recon python -m pytest Python/tests -q
```

The audit suite includes independent dense references, exact warm-start and
breakdown checks, masks and subset metadata, native reconstructions, wrapper
parity and legacy-call regressions. Some native error-path tests intentionally
exercise CUDA failures; interpret their test assertions and exit status, not an
isolated CUDA message alone.

MATLAB validation regenerates Python reference results for equivalent rays:

```matlab
addpath(genpath('MATLAB'));
UseBlackwellBuild(); % only for this local build configuration
test_operatorAuditContract;
test_allOperatorAlgorithms;
test_operatorPocs;
test_userManualExamples;
test_surfaceProjectors;
test_surfaceFDK;
```

See the [validation guide](Validation.md) and
[operator guide](Projection_operator.md) for results, commands and limits.
A test that expects an unsupported input to fail does not establish
reconstruction support for that input.

## Scope of numerical agreement

The cross-language fixture includes masks, a missing view and explicit
equivalent parameters. Flexible LSQR-TV is particularly sensitive to rounding
and its inner preconditioner; its agreement is tested to a finite tolerance,
not assumed bit-identical on all hardware. The operator guide documents its
bounded mean-free CG preconditioner and other operator-only numerical repairs.

The current audit does not replace clean builds and tests on every supported
platform. Real scanner calibration, multi-GPU scaling and other external-data
workflows require additional evidence.

## Repository boundary

TIGRE owns detector models, recipes, geometry files, native projectors,
reconstruction, diagnostics and scientific rendering. tigre-geometry-design-tool owns desktop
controls, interactive demo 27, script exporters, GUI tests, GUI manuals and
their presentation assets. A core reconstruction must work without importing
`tigre_geometry_design_tool` or Tk. Exported standalone scripts depend on TIGRE, not the GUI that
created them.

For a change to an adapted algorithm, update Python and MATLAB entry points,
surface wrappers, meaningful numerical tests and this documentation together.
Keep pure formatting, machine-specific binaries and unrelated environment
changes out of an upstream correctness patch. Preserve the user's working tree
when preparing isolated review branches.

# Companion references

- [Projection-operator API and numerical details](Projection_operator.md)
- [Detector-surface model](Detector_surface.md)
- [Current validation procedure](Validation.md)
- [Integration audit and remaining limitations](Integration_audit_2026-11-06.md)
- [Python installation](python_installation.md)
- [MATLAB installation](MATLAB_installation.md)
- [Optional CuPy filtering](Optional_CuPy.md)
- [Varian loader](Varian_loader.md)
- [Dual-energy workflow](Dual_energy.md)
- [Python demos](../Python/demos/README.md) and [MATLAB demos](../MATLAB/Demos/README.md)
- [tigre-geometry-design-tool application and its manuals](../../tigre-geometry-design-tool/README.md)
- [License](../LICENSE.txt)
