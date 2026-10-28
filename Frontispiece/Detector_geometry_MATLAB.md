# MATLAB TIGRE CT Simulation Builder and surface animation

For a guided build/save/load workflow, run `d27_DetectorGeometryGUI`.
The Python counterpart uses the same number and panel settings; see the
[content comparison](./Demo_language_comparison.md) for remaining differences.
It configures three panels, saves native MAT settings and exact surface files
in a new temporary folder, then reloads both in the same GUI. Settings reopen
in editable builder mode; exact coordinates reopen in inspection mode with
builder fields disabled. **Return to builder** restores the editable recipe.
Optional YAML recipe and exact YAML + NPZ export/load lines are included.
No CUDA is needed for this demo. See the [demo index](../MATLAB/Demos/README.md)
for requirements and the related curved/modular head reconstruction examples.

For the experimentally verified native R2020a/RTX 5070 Ti GPU build, see the
[Blackwell build instructions](./MATLAB_Blackwell_build.md). That build supports
standard TIGRE reconstruction and native surface CUDA projection/backprojection.

The MATLAB port provides native modular geometry construction, a desktop
builder/inspection GUI, and curved/modular surface animation. Tested with
MATLAB R2020a. These visualization utilities do not invoke CUDA projectors.

From the repository root in MATLAB:

```matlab
addpath(genpath('MATLAB'));
app = detectorGeometryGUI();
```

The window opens at up to 1280 x 720 pixels, centred and reduced to fit the
screen. Source, Detectors, Volume and Movement settings use scrollable pages
beside one shared world-coordinate 3D view. Tab-aware help explains coordinate
order and motion. The window can be resized.

The presets match the Python builder, including panel roll/pitch/yaw, the
exaggerated global COR example, linear/arc tomosynthesis, and GCT last.
GCT is an idealized panel arrangement; it excludes the measured-data demo's
pixel corrections. Full GCT coordinate/mask arrays need about 240 MiB.

Choose a layout and motion, edit fields, then **Confirm**. **List...**
opens a per-panel table; **Apply** updates the field and **Confirm** rebuilds
the scene. **Play** toggles timer-based playback; the slider selects a view.
Disabled fields are ignored by the builder. Piecewise-linear geometry derives
the panel count from the transverse list, reported below the controls.
Changing mode, loading geometry, or loading another preset closes open panel
list editors. Applying a list does not rebuild the preview. Returning from
inspection preserves edits that had not yet been previewed. Playback stops on
mode changes and Confirm; an invalid Confirm retains the previous geometry.
Play starts disabled until the first successful explicit Confirm. Closing or deleting the
main window removes its playback timer and list editor windows.

Panel x/y angles accept a shared value or one per panel. Piecewise-linear z
angles and position offsets require one entry per panel; cylindrical z angles
come from the arc. Rotation is `Rz(gantry+z)*Ry(y)*Rx(x)` about each panel
centre. COR is one global correction shared by the source and all panels;
it remains editable but must be zero for stationary-detector scans. Individual
panel alignment uses offsets and tilts. The known builder COR is shown by an
orange sphere and label. Arbitrary loaded surfaces do not imply an independent
COR marker.

The scene preserves physical scale and gives thin scans additional z-axis
space. Curved surfaces use sampled triangles; single-row/column detectors use
lines. Module boundaries and masks are preserved, including changing masks.
Origin arrows are red x, green y, blue z. R2020a `quiver3` lacks portable alpha
support, so arrows/labels use colors blended at 30% against white.

## Loading and saving

**Save YAML...** writes an editable recipe to `.mat` or `.yaml`.
**Save surface...** stores exact arrays: a native `.mat`, or YAML with an NPZ
companion. **Load geometry...** accepts `.mat`, `.yaml`, `.yml`, and `.npz`.
Recipes reopen in builder mode; explicit geometry opens in inspection mode,
with builder controls disabled. **Return to builder** restores the retained
builder settings. Keep YAML and its NPZ companion together when sharing.

MAT files require no Python. A MAT document contains `settings`, or `surface`,
or standard MATLAB `geo` plus `angles`. Optional `geo.sVoxel` and
`geo.offOrigin` draw a fixed or moving reconstruction volume. Loading MAT files does
not execute MATLAB source.

YAML/NPZ interchange uses the same Python loader and schema as the Python GUI,
through `geometry_matlab_bridge.py`. It defaults to `conda run -n recon python`;
TIGRE, SciPy and PyYAML must be installed in that environment. If needed, set
`TIGRE_PYTHON` to the interpreter of an already activated environment:

```matlab
setenv('TIGRE_PYTHON', 'C:\Users\liu005\miniforge3\envs\recon\python.exe');
% The environment's DLL directories must also be on PATH.
app = detectorGeometryGUI('Python/demos/geometry_examples/three_panel.yaml');
```

YAML geometry angles are degrees. Standard TIGRE angles and `rotDetector`
remain radians. MATLAB native geometry uses `nDetector/dDetector/offDetector`
in `[u;v]` order and volume values in `[x;y;z]`; Python uses `[v,u]` and
`[z,y,x]`. The bridge converts these conventions explicitly.

## Volume motion and reconstruction export

Confirm, editable save and exported scripts check physical transmission at
each volume pose. Sources/detector pixels inside or on the volume remain
invalid. Empty views and panels are allowed, including spiral-scan entry/exit;
the ray-coverage check rejects only scans with no crossing ray anywhere.
Confirmation results report the number of allowed empty views. Reposition
the rig/volume if the entire scan is empty. This check does not guarantee complete sampling,
mechanical clearance or FDK support; exact archives remain inspectable.
After Confirm, the persistent check-results box displays physical PASS/FAILED,
view and crossing-sample counts, CGLS geometry support and the separate FDK
requirements. Failed-confirmation reasons remain visible without replacing
the last confirmed scene. The exported script performs the FDK-specific check.

**Source trajectory** selects Cylindrical (circular orbit about +z) or Linear
(xyz endpoints, fixed detector). It stays synchronized with detector motion.
Changes are pending until **Confirm**; `app.confirm()` applies them, and the
legacy `app.preview()` method remains an alias for explicit confirmation.

Volume counts, full size and start/end positions use MATLAB **x,y,z** order.
The editable voxel-size field uses x,y,z mm. Changing counts or voxel size
updates full size; changing full size updates voxel size with counts fixed.
Their relationship is `sVoxel = nVoxel .* dVoxel`. Confirm updates the scene.
The Volume center and Movement start are shared. If timing and endpoint
movement are off but Start and End differ,
Confirm and editable save ask to enable movement, keep the volume stationary
at Start, or cancel. The same stationary choice is remembered until the
endpoints change. Timing acquires one frame per
pulse: `t=(view-1)/firing_rate_hz`. Readout limits firing to
`1000/readout_period_ms`; rotation period determines turns. Choose signed
z speed or endpoint interpolation. Endpoints derive z speed from travel
divided by last-view time; different endpoints require at least two views.
Playback follows elapsed time and may skip display frames. The source keeps
its structural world trajectory while the volume moves.

YAML/MAT persistence preserves voxel counts, all volume offsets, acquisition
timing and motion mode. The bridge reverses xyz/zyx and transposes every view;
it does not collapse motion to its first pose. Nonlinear loaded volume paths
remain read-only. Save YAML also offers native MAT recipes.

**Export recon script...** is enabled only after a successful YAML save; MAT
saves do not unlock it. The native `.m` script references the last saved YAML
and uses `FDK_surface` and `CGLS_surface`. It uses saved voxel settings and
either simulates `headPhantom` or loads a MAT file containing attenuation
`projections` in rows/columns/views order. Edit `algorithm`, `iterations`,
`projectionsPath`, `voxelCounts` and `outputPath` in the script as needed.
The YAML path is absolute; update it when moving files. Moving rays are
converted to a fixed object frame for CGLS, and unsupported FDK geometry is
reported without enabling approximate FDK. YAML loading uses recon Python;
reconstruction uses native MATLAB CUDA MEX.

For automation, `app.save(path)` saves current controls and volume,
`app.save(path,true)` saves the exact displayed surface, and `app.export(path)`
exports after a successful YAML save. `app.geometry()` returns the confirmed
native xyz volume metadata; `app.settings()` returns the confirmed recipe.

## Animation and programmatic construction

Existing `animateGeometry(geo, angles, pos, rotation, animate, fname)` calls
remain available. New inputs use the shared surface renderer:

```matlab
examples = detectorGeometryExamples();
settings = examples(1).settings;
surface = buildModularSurface(settings);
h = animateGeometry(surface, [], 1, 'SD', false);  % static
h = animateGeometry('Python/demos/geometry_examples/three_panel.yaml');

surface = surfaceFromGeometry(geo, angles);
h = animateGeometry(geo, [], 1, 'SD', true, '', surface);
app = detectorGeometryGUI(geo, angles);
```

Surface mode supports `SD` only; indices are one-based. `fname` writes a GIF.
`animateDetectorSurface` and the GUI share `renderDetectorSurface`.
Standard-geometry legacy animation retains its existing behavior.

Native MATLAB surface arrays are:

| Field | Dimensions |
|---|---|
| `source_positions_mm` | `3 x views`, world xyz |
| `pixel_centers_mm` | `rows x columns x 3 x views`, world xyz |
| `valid_mask` | `rows x columns x views` |
| `module_slices` | `modules x 4`, zero-based half-open row/column bounds, matching the archive format |

Pixel arrays include an xyz coordinate axis. `Ax_surface`/`Atb_surface`
projection data instead use `rows x columns x views`, without that axis.
The bridge preserves singleton dimensions, module boundaries and volume axes.
Native surface CUDA reconstruction is described below. The projection-remapping
workflow remains a Python facility.

## Validation

See the [validation guide](Validation.md) for the full native reconstruction,
quality-history, dual-energy, animation and interchange test commands.

```matlab
test_detectorGeometryGUI;   % presets, geometry, renderer, MAT, GUI
test_detectorGeometryGUI_usability; % callbacks, field states, lists and timers
test_animateGeometry;       % existing eight animation/GIF tests
test_geometryInterchange;   % six MATLAB/Python layout/motion comparisons
```

The interchange tests require the Python `recon` environment. Python tests
in `test_geometry_matlab_bridge.py` cover MAT/YAML round trips, singleton
dimensions and volume-axis conversions. Native CUDA checks are available in `test_surfaceProjectors`.

## Native surface reconstruction

`Compile` now builds `Surface_mex` from the same Siddon CUDA kernels used by
Python. On this machine, build only the added gateway with
`CompileBlackwell('surface')`, then activate `UseBlackwellBuild`.
The native operators require no Python, `gpuArray`, or Statistics toolbox.

```matlab
surface = buildModularSurface(settings);
projections = Ax_surface(image, geo, surface);
backprojection = Atb_surface(projections, geo, surface);
[image, residual] = CGLS_surface(projections, geo, surface, 40);
[imageSIRT, residualSIRT] = SIRT_surface(projections, geo, surface, 40);
% Only for circular, vertically extruded detector surfaces:
imageFDK = FDK_surface(projections, geo, surface, 'Filter', 'hann');
```

Images have MATLAB dimensions `[nx ny nz]`; projections use
`[rows columns views]`. Volume `nVoxel`, `dVoxel`, and `offOrigin` are xyz.
Coordinates are world xyz in mm. A source is required for every view; pixel
coordinates and logical masks may be shared across views or supplied per view.
A single fixed volume is used: per-view `offOrigin` is rejected.
Masked gaps contribute zero to both operators and are excluded from solver
residuals. Per-panel tilt and COR are already encoded in pixel/source positions;
the operators do not apply these corrections a second time.

The matched backprojector is the adjoint of the forward projector. It supports
arbitrary explicit surfaces, including tilted modules, curved panels, stationary
panels, and changing masks. CGLS is unconstrained least squares; SIRT defaults
to nonnegative updates. Both return histories truncated to completed iterations.
The optional operator fourth argument is a zero-based CUDA device number;
solvers accept `'Device', 0`. This initial MATLAB implementation uses one GPU
and requires its working set to fit device memory. Automatic multi-GPU/slab
execution is not ported.
Standard MATLAB FDK remains a flat-detector algorithm.

`d25_CurvedDetector` projects TIGRE's bundled 64-cubed head phantom onto a
smooth curved detector, animates the geometry immediately after `plotProj`,
and reconstructs with native CGLS and optional surface FDK. It requires the
surface CUDA MEX and displays matching input/reconstruction/error slices,
projection residuals and relative volume errors. `test_curvedSurfaceDemo`
checks the full native head demo as well as the analytic sphere-ray utility.
`d26_ModularDetector` is a separate native CUDA demo using TIGRE's bundled
64-cubed head phantom and three individually tilted panels, masked gaps and
a scan-wide COR shift. It forward-projects the voxelized head using `Ax_surface`,
animates the geometry immediately after `plotProj`, and reconstructs using
`CGLS_surface`. Matching input/reconstruction/error slices and residuals are
displayed. Tilted panels require iterative reconstruction rather than FDK.
Run `UseBlackwellBuild; d26_ModularDetector` with the isolated custom build.
`test_modularDetectorDemo` runs the complete native head reconstruction demo.
Run `test_surfaceProjectors` for native GPU checks of analytic box lengths,
adjointness, gaps, singleton/shared coordinates, changing masks, CGLS and SIRT.

`FDK_surface` uses the shared CUDA voxel-driven surface backprojector, with
MATLAB equiangular ramp/Hann filtering, cutoff selection and Parker short-scan
weights. It requires a circular source orbit about world z, constant source
height, a vertically extruded detector, identical fan-angle tables at every
view, and whole-column gaps. It rejects arbitrary panel tilt, incomplete short
scans and nonmonotone orbits. Full scans may include a duplicate endpoint;
endpoint-free full scans require uniform view spacing. Gap filtering interpolates
missing columns, so wide gaps favor iterative reconstruction. FDK is approximate
for cone beams; no experimental per-view FDK fallback is enabled.
The dense fan filter is limited to 8192 samples and may require substantial host
memory; iterative reconstruction avoids that dense filter.
Run `test_surfaceFDK` for full/reversed/Parker scans, sphere amplitude, gap/Hann
filtering and rejection of non-extruded detectors.
