**TIGRE CT Simulation Builder — MATLAB user manual**

**Manual version: 1.1**
**Updated: 28 October 2026**
**Software reference: TIGRE Python package 3.0.0; CT Simulation Builder checkout.**

The manual version identifies this document, not a separate application release.
The PDF edition includes a cover page, contents, and wrapping tables/code.

MATLAB GUI behavior is tested with R2020a. The Python package version above
identifies the shared checkout/interchange reference, not a MATLAB release.

# Start the GUI

From the repository root:

```matlab
addpath(genpath('MATLAB'));
app = detectorGeometryGUI();
```

The desktop GUI, geometry construction, visualization and native MAT saves
do not require CUDA. YAML/NPZ interchange uses Python in conda **recon**
with TIGRE, SciPy and PyYAML. Reconstruction requires compiled native
surface CUDA MEX files; see the [Blackwell build guide](./MATLAB_Blackwell_build.md).

The window is reduced to fit the screen, with four scrollable settings tabs
beside one 3D view. The documented native layout is xyz volumes and
rows/columns/views projections, unlike Python's zyx and views/rows/columns.

# Basic workflow

1. Start the GUI or load an example preset.
2. Configure Source, Detectors, Volume and Movement.
3. Press **Confirm** to validate all fields and apply the candidate scan.
4. Read the persistent check results. If validation fails, the old scene stays active.
5. Use **Play** or the slider to inspect motion.
6. Save an editable YAML/MAT recipe or an exact surface archive.
7. After a successful YAML save, export and run a sample reconstruction script.

Play is initially disabled until an explicit Confirm succeeds. The initial
automatic preview does not enable it. Exact loaded surfaces remain inspectable.
Help changes with the tab; field tooltips identify controls. Closing/deleting
the GUI removes its timer and list-editor windows.

# Use the Source tab

Choose **Cylindrical** for a circular orbit about world +z or **Linear**
for a straight source trajectory. Linear selects a fixed detector. Returning
to Cylindrical retains fixed-detector arc mode until detector motion is changed.

| Control | Meaning |
|---|---|
| Views | Number of acquired positions, shared with timing. |
| Source radius / height | Circular radius and fixed world-z height, in mm. |
| Orbit start / Scan span | Degrees about +z. Manual 360-degree scans omit the duplicate endpoint; short arcs include endpoints. |
| Source start / end | World x,y,z mm coordinates for Linear mode. |

Circular fields are inactive for Linear; source endpoints are inactive for
circular/arc motion. Timing derives angular span and supports multiple turns.
Changing mode closes stale list editors and stops playback; Confirm applies it.

# Use the Detectors tab

MATLAB's **Detector motion** dropdown currently lists the three explicit
combinations: Rotating source + detector, Fixed detector / moving source
(linear), and Fixed detector / arc source. Python presents two detector
choices and combines them with its source selector; saved source-motion
values are shared. The controls remain synchronized.

| Control | Meaning |
|---|---|
| Layout | Cylindrical tangent panels or Piecewise linear individually positioned panels. |
| Module shape | Pixel rows, columns; every panel shares the shape. |
| Pixel pitch | Row, column center-to-center spacing in mm. |
| Detector radius | Nominal rotation-axis to detector distance. |
| Modules / Arc span | Cylindrical count and first-to-last panel-center angle. |
| Transverse / normal / z offsets | Piecewise-linear local y, local x and z panel-center offsets, in mm; one value per panel. |
| Panel roll x / pitch y / yaw z | Degrees around panel axes; x/y allow a shared value, z requires one value per piecewise panel. |
| Global COR | One scan-wide shift of source and all panels; stationary-detector modes require zero. |
| Gap columns | Invalid dense-array columns, masked rather than measured. |

**List...** opens a per-panel table. Apply updates its field; Confirm rebuilds
the scene. The transverse list sets the piecewise panel count. Keep lists in
the same panel order. Rotation about each center is
`Rz(gantry+panel_z)*Ry(panel_y)*Rx(panel_x)`. COR is not a per-panel correction.
Tilt and COR are already included in exported ray coordinates.

# Use the Volume tab

Use **Volume** to define the reconstruction box. All four vectors use
**x, y, z** order; source coordinates also use **x, y, z**.

| Field | Meaning and update rule |
|---|---|
| Voxels (x,y,z) | Three positive integers. Editing counts preserves voxel size and updates full size. |
| Full size (x,y,z mm) | Full positive physical lengths, not half-widths. Editing full size preserves counts and updates voxel size. |
| Voxel size (x,y,z mm) | Positive spacing. Editing spacing preserves counts and updates full size. |
| Start / center (x,y,z mm) | Initial box center; shared with Movement Start. |

The relationship is `sVoxel = nVoxel .* dVoxel`, component by component.
For example, counts `128, 128, 128` with spacing `2, 2, 2` describe a
`256, 256, 256` mm box. Changing counts to `256, 128, 128` preserves spacing
and produces a `512, 256, 256` mm box. To keep physical size fixed while
changing sampling, edit counts and then restore the desired full size.

Linked fields update when a valid edit is committed (Enter or leaving the
field). Programmatic edits through control handles must invoke ValueChangedFcn
when linked-field recalculation is required. Counts/full size remain canonical
when confirming programmatic changes. **Confirm** applies the new box to the
scene. The GUI stores metadata and draws a box; it does not allocate voxel
data. YAML preserves counts and full size and derives spacing on reload.

# Use the Movement tab

Use **Movement** for object translation and acquisition timing. Geometry
stays in the structural world frame: the source follows its configured path
while the volume box moves. Enable **Move between start and end**
for endpoint translation; entering two values alone does not enable motion.

| Control | Purpose |
|---|---|
| Start (x,y,z mm) | Initial volume center, shared with Volume Start / center. |
| End (x,y,z mm) | Final center in endpoint mode; calculated/read-only in timed speed mode. |
| Move between start and end | Interpolate all three coordinates across views; distinct endpoints need at least two views. |
| Use acquisition timing | Use one frame per source pulse. Requires rotating source and detector. |
| Readout period (ms) | Minimum frame interval; firing rate must not exceed `1000 / readout_period_ms`. |
| Firing rate (Hz) | Pulses and acquired views per second. Pulses are not discarded or combined. |
| Rotation period (s/turn) | Seconds per full turn; determines turns over the acquisition. |
| Volume z speed (mm/s) | Signed world-z speed; positive moves toward +z. Derived/read-only in timed endpoint mode. |
| Playback speed (x) | Animation speed relative to acquisition time. |
| Views (in Source tab) | Number of positions/pulses, not number of turns. |

Without timing, endpoint mode interpolates Start to End over the selected
source views. Without timing or endpoint motion, the volume is stationary
at Start and End is ignored. If these inactive endpoints differ, Confirm
and editable save ask whether to enable movement, keep stationary, or cancel.
An explicit stationary choice is remembered for the same endpoint pair.

With timing enabled, one-based frame `k` occurs at `t_k = (k-1) / firing_rate_hz`. Angle is `orbit_start + 360*t_k/rotation_period_s`. For N views,
the acquisition window is `N/rate`, while the last view is at `(N-1)/rate`.
The timing summary reports last-view time; do not confuse it with the full
acquisition window. In speed mode the center is
`start + [0; 0; z_speed*t_k]`; endpoint mode derives z speed from total z
travel divided by last-view time. Travel per turn is `z_speed*rotation_period`.

For a timing example, 1,000 views, 10 ms readout, 100 Hz firing, 1 s/turn,
and 20 mm/s describe a 10-second window and 10 turns. The last sample is at
9.99 s, after 9.99 turns and 199.8 mm travel. Timed mode replaces manual
scan span and permits multiple turns. Fixed-arc and linear sources use
manual sampling; timed acquisition currently requires rotating hardware.

For a simple endpoint spiral, use a circular source, 600 views, Start
`0, 0, -200`, End `0, 0, 200`, and **enable endpoint movement**, then Confirm.
Timing is optional for that example. Empty entry/exit views are allowed as
long as at least one measured ray crosses the volume somewhere in the scan.

# Confirm, check results and animate

After editing any tab, select **Confirm**. It builds a candidate from all
current controls, validates it, and replaces the active scene only on success.
Failed confirmation retains the preceding confirmed scene; the plot therefore
does not show rejected values. Play starts disabled until the first successful
explicit Confirm. Loaded exact geometry can be inspected without editing.

The persistent check-results area reports physical PASS/FAILED,
view and crossing-sample counts, allowed empty views, and reconstruction
support. The MATLAB GUI lists the separate FDK requirements; its exported
script checks the actual trajectory, detector structure and angular coverage,
after fixed-object conversion when needed.
These are geometry checks, not a CUDA execution test or a scan-quality certificate.

The physical check uses finite source-to-pixel segments and the volume box
at each view. Source and valid detector pixels must remain outside the box.
Empty views/panels, including helical entry/exit and overscan, are allowed.
The coverage guard rejects only a scan with **zero crossing samples in total**.
It does not establish complete angular sampling or mechanical clearance.

Select **Play** to animate or pause, or drag the view slider to inspect a pose.
With timing enabled, playback follows elapsed time and may skip display frames;
all acquired views remain saved. The slider selects one-based views; the
timing summary reports last-view time.

# Save and load geometry

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

Editable saves validate current controls and volume; exact saves preserve the
displayed rays. All volume offsets, voxel counts, acquisition timing and motion
mode survive MAT/YAML interchange. Nonlinear volume paths remain read-only.
The bridge reverses xyz/zyx and transposes every view explicitly.

# Export a reconstruction example

**Export recon script...** is disabled until **Save YAML...** succeeds.
Cancelled/failed saves and native MAT saves do not unlock it. Export references
the last successfully saved YAML, not later unsaved edits.

The exported `.m` runs native `FDK_surface` and `CGLS_surface`. At the top, edit:

| Script variable | Purpose |
|---|---|
| geometryPath | Absolute path to saved YAML; update it when moving files. |
| projectionsPath | Optional MAT file containing attenuation `projections`, rows/columns/views. Leave empty to simulate headPhantom. |
| algorithm | `fdk`, `cgls`, or `both`. |
| iterations | CGLS iteration count. |
| voxelCounts | Optional x,y,z override; required for older YAML without counts. |
| outputPath | Destination MAT file. |

The script uses the saved physical volume and validates transmission. It
converts moving rays to a fixed object frame by subtracting each view's center
from both source and detector, without reordering measurements. In both mode,
unsupported FDK geometry is reported and CGLS continues. Other errors are
reported normally; no approximate FDK fallback is automatically enabled.
Results contain available FDK/CGLS volumes, geometry, residuals, and the
phantom when simulated. Keep YAML and exact-surface NPZ companions together.
Python is used for interchange only; reconstruction runs native MATLAB CUDA.

# Examples and presets

Run `d27_DetectorGeometryGUI` for the guided three-panel build/save/load
workflow. It writes fresh temporary native MAT recipes and exact archives,
reloads them into the same GUI, and demonstrates returning from inspection.
`app.preview()` remains a compatibility alias for explicit confirmation.

The preset menu follows the Python geometry examples, including circular CT,
tilted/staggered panels, fixed linear/arc source scans, 3D tilt, global COR
and the large GCT model. The GCT preset has 53 one-row panels and 5701 views;
it is idealized and omits measured-data pixel corrections. Reduce Views for
memory-constrained use; loading an example builds it immediately.

`d25_CurvedDetector` and `d26_ModularDetector` simulate the bundled head phantom,
animate geometry and run native CUDA surface reconstruction. They require the
surface MEX build. For language differences, see the
[demo comparison](./Demo_language_comparison.md).

# Animation and programmatic construction

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

GUI automation APIs:

```matlab
app.confirm();               % Validate and apply all controls.
geo = app.geometry();         % Current confirmed xyz volume metadata.
recipe = app.settings();      % Current confirmed builder recipe.
app.save('geometry.yaml');    % Validate/save current editable controls.
app.save('surface.mat',true); % Save displayed exact rays and volume.
app.export('reconstruct.m');  % Requires a successful YAML save first.
```

Programmatic control edits should invoke their ValueChangedFcn when linked
field recalculation is required. Confirm is still required to update the scene.

# Native surface reconstruction

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

# Limitations and troubleshooting

| Boundary | What it means |
|---|---|
| Physical guard | Some transmission is required across the scan. Empty views are allowed. Passing does not prove complete sampling, sensor-front sensitivity or mechanical clearance. |
| Surface CGLS | Supports arbitrary explicit cone-beam rays, including fixed detectors. Limited-angle data may still give an ill-conditioned reconstruction. |
| Moving volumes | Surface reconstruction APIs require a fixed object frame. Exported scripts subtract each view's center from both source and detector before reconstruction. |
| Surface FDK | Requires its supported circular trajectory, detector structure and angular coverage. Helical FDK is not provided by this GUI. |
| Standard FDK | A flat-detector algorithm; do not apply directly to arbitrary modular/curved detector data. |
| Measured data | Supply calibrated attenuation line integrals in the exact saved view/pixel order; the GUI is not an intensity-calibration pipeline. |
| Resource use | GUI size guards are upper bounds, not RAM/GPU-memory guarantees. Reconstruction/exported scripts need the appropriate compiled CUDA backend. |

Geometry units are mm and builder angles are degrees. Standard TIGRE angles
and `rotDetector` use radians. Panel tilt/COR are already encoded in exported
rays and must not be applied a second time.


MATLAB reconstruction uses one GPU and requires the working set to fit device
memory; Python's multi-GPU/slab execution is not ported. The MATLAB FDK fan
filter is limited to 8192 samples. Builder limits are 6000 views, 4096 pixels
per panel dimension, 256 panels and 20 million dense samples including gaps.

| Symptom | Action |
|---|---|
| Start/End changed but no translation | Enable endpoint movement or choose Enable movement in the confirmation guard. End is ignored in stationary mode. |
| Entire scan misses volume | Check xyz coordinates, volume position, detector/source sides and pixel sampling. Empty individual views are permitted. |
| Source/detector inside volume | Correct the offending view or volume extent; this remains invalid. |
| YAML bridge fails | Activate recon and ensure TIGRE/SciPy/PyYAML are installed; check TIGRE_PYTHON and DLL paths. |
| Play disabled | Complete a successful Confirm; failure leaves the last valid scene. |
| Export disabled | Successfully save YAML, not only a MAT file. |
| FDK unsupported | Use CGLS or a supported circular/extruded geometry with sufficient coverage. |

# Validation

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

Additional builder checks:

* `test_ctTransmission`: partially empty scans and hardware positions.
* `test_ctSimulationBuilder`: linked voxel fields, timing, endpoints and YAML/export.
* `test_ctReconstructionExport`: native CUDA exported scripts.

Run from the repository root with `addpath(genpath('MATLAB'))`.

# Documentation and PDF regeneration

See the [Python manual](./TIGRE_CT_Simulation_Builder_Python.md) and
[handover notes](./CT_Simulation_Builder_Handover.md). From PowerShell at the
repository root, run `./Frontispiece/Export_Manuals.ps1`. It regenerates both
PDFs with versioned covers, contents and wrapping tables/code. Pandoc and
XeLaTeX are required; PDFs are ignored generated artifacts.
