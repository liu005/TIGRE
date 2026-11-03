# TIGRE CT Simulation Builder handover

Updated: 2026-11-03 (Australia/Sydney).

## Final audit

The [project audit](Project_audit_2026-11-03.md) records 451 passing Python
tests and 25 passing MATLAB suites. It also closes whole-suite collection
gaps, protects MATLAB imports from silently replacing piecewise source paths,
normalizes MATLAB segment counts and improves missing-phantom errors.
Dependency consistency and all 98 checked local documentation links pass.
External scanner data, measured GCT and optional PyTorch remain outside the
verified set. Audit changes do not implement an editable MATLAB piecewise
source builder; exact imports open inspection mode.

## Piecewise linear source paths (Python)

Manual/PDF revision 1.7 documents the new Source trajectory option and the
14th Python preset, **Tomosynthesis: piecewise linear source**. MATLAB retains
13 presets and does not yet provide an editable piecewise source mode.

Define at least three world xyz waypoints and one view count (at least two)
per segment. Sampling includes segment endpoints but acquires shared junctions
once: two 21-view segments produce 41 views. The first waypoint establishes
the fixed detector pose; later segments preserve its pixel coordinates, panel
tilts, masked gaps and module boundaries. Views is derived on Confirm.
Orbit timing and nonzero COR are unsupported for this mode. This joins source
paths on a common detector rather than merging independently positioned rigs.

YAML recipes retain waypoints and counts; exact YAML/NPZ preserves sampled
coordinates. Preview, inspection/return-to-builder and reconstruction export
use the combined acquisition. Surface CGLS is supported; this is not a circular
FDK trajectory. Tests exercise invalid recipes, multiple segments, shared
junction sampling, GUI transitions, archives, exported geometry, native CUDA
projection equivalence to two linear scans and CGLS data consistency.

Validation uses `recon-new`. Pytest is supplied locally for this environment;
no environment packages were changed. See [validation commands](Validation.md)
and the [demo run report](Demo_validation_2026-11-03.md).
The final expanded run passed 122 checks without skips, including native CUDA,
MAT bridge round trips, existing rotation and cargo DR workflows.

## Fixed-hardware DR timing

DR mode now enables acquisition timing without hardware rotation. Endpoint
motion is forced on while DR timing is active; its full xyz/zyx vector defines
constant conveyor velocity. Rotation period/z-speed widgets are disabled and
ignored. Fixed detector and identical source endpoints are required. The
existing five-field acquisition schema is retained: rotation_period_s=1 is an
unused compatibility value; table_speed_mm_s is the z component, while the full
velocity is derived from saved offsets and view times. Frame timing, playback,
YAML/MAT/NPZ interchange and exported DR view_times_s are supported.

At 1253 views, 100 Hz, 10 ms readout and 6260 mm y travel, the step is 5 mm,
velocity [0,500,0] mm/s, last sample 12.52 s and window 12.53 s. Tests verify
unchanged source/pixel coordinates, any-axis endpoints, invalid rates/source
motion, YAML reload and timing output. Existing CT timing tests still pass.
Manual/PDF revision is now 1.6.

Final validation on 30 October 2026: 99 affected Python tests passed with no
skips. Six MATLAB R2020a suites passed: seaCargoPreset, ctSimulationBuilder,
objectRotation, ctTransmission, detectorGeometryGUI and GUI usability.
The exported Python DR script ran on CUDA with 1253 timed views and an 8x12x6
uniform attenuation volume: finite outputs, lower/upper air margins and all
frame timestamps verified (last view 12.52 s). This validates the software
workflow, not clinical/cargo detection performance or scanner calibration.

## Complete GUI preset catalogue

Manual revision 1.6 includes all 13 GUI presets in menu order under Examples
and presets / Catalogue of GUI presets. Each entry explains geometry, intended
uses and reconstruction/export limitations. The d18 frame switch, current
18+10 module cargo DR arrangement and one-row GCT limitations are explicit.
PDF contents now include third-level headings so readers can find each preset.
Both demo indexes and the detector-surface reference link to these catalogues.

## Sea-cargo modular DR preset

The preset now uses 1253 endpoint-inclusive views over y=-3130 to +3130 mm:
1252 intervals at exactly 5 mm, with 100 mm air before and after the 6060 mm cargo.

DSO now defaults to 4230 mm (10% below 4700). Both layout helpers accept an
optional DSO. Container motion stays fixed. The detector supports are refitted
to the transverse silhouette and air margins at that DSO; individual module
DSDs can change when fitting the arms. Source-facing normals and projected
seam alignment are recomputed together.

The shared preset **Sea cargo: 20-foot container (2D Digital Radiography)**
now uses 18 vertical-arm and 10 horizontal-arm source-facing modules. Each has
36x4 pixels at 5x5 mm pitch (180x20 mm). `cargo_detector.py` and
`cargoModuleLayout.m` solve adjacent module edge-ray alignment while keeping
centres on vertical/horizontal support lines. The horizontal near edge is
about 319.95 mm above and rearward of the vertical arm's upper edge on the same source
ray. There are no projected scanning-direction seam gaps/overlaps over common
transverse width. Whole modules slightly extend the old support envelope.

The preset stores ordinary piecewise-linear centre/tilt lists; it needs no
external checkout or special projector. Top arm rows and Arm clearance fields
are zero because those bend continuous panels; the modular joint clearance and 300 mm horizontal-support lift are
encoded in the lists. Manual edits do not automatically re-aim modules.
Equal linear-source endpoints keep the legacy converted source position fixed.
Container dimensions, 5 mm voxel grid and 1253-view 6260 mm y motion are unchanged.
The vertical arm is lowered, placing its lowest module centre ray on the
limiting lower corner: the first 18 rows form a half-module air margin.
One extra module is now added at the vertical top (18 total), without moving
the lowest module. The horizontal support is refitted at z=2169.48 mm, with
the same 300 mm relative lift and source-ray joint alignment.
The horizontal arm extends through the upper limiting corner, plus a full
air-only module. Current packing is Python 1253x36x112, MATLAB 36x112x1253,
with 28 module slices.

The volume retains `simulation_mode: dr` across YAML/NPZ/MAT, selecting forward-
only DR export with no FDK/CGLS. The output `dr_image` now samples a central
column from every module and concatenates them in module order (1008x1253).
Both DR exporters save module metadata with projections/transmission. The
result is not a calibrated uniformly resampled conveyor image.

Tests check every module normal/pitch, every adjacent edge ray, the exact 300 mm support lift and projected joint
clearance, volume motion, YAML/NPZ interchange and all-module DR assembly.
Coverage tests exercise 4097 transverse fan directions including box corners
and seam boundaries. CUDA confirms the lower 18 rows and upper 36 rows are
air-only in every view. Conveyor movement provides the longitudinal scan.
Python's exported modular DR script passed a CUDA smoke at 8x12x6 voxels;
MATLAB CPU geometry/export tests passed. Manuals/PDFs are revision 1.6.

## Python and MATLAB rotating-object acquisition

Volume contains the fixed-hardware/rotating-object selector and angle controls.
Movement contains translation/timing and points to Volume for rotation. Supports
inclusive start/end z angles or explicit per-view ZYZ degree triples, rotating
about the volume centre, optionally combined with endpoint translation.
Orbit timing is disabled in this mode. `object_motion.py` shares pose validation
and inverse frame conversion between physical checks and generated recon scripts.
The shared renderer animates the oriented box and an Object +x marker.
YAML volume metadata and NPZ archives retain `object_angles_deg`; builder settings
retain `fixed_hardware`. Python readers accept old files without these keys.
MATLAB has matching controls, a ZYZ table editor and the d18 preset. The bridge
normalizes object poses to rows of ZYZ triples, independent of volume xyz/zyx
conversion. `objectRotationMatrices.m` and `fixedObjectGeometry.m` share the
native renderer/validation/reconstruction convention. MATLAB YAML reloading
also normalizes row-vector voxel size/count metadata when populating controls.

The new d18 preset uses 100 inverse poses `[-z2, -y, -z1]` to reproduce d18's
relative gantry trajectory with fixed hardware. GUI detector sampling is 128x128
over the original physical size. Arbitrary poses need iterative reconstruction;
FDK compatibility remains independently checked after object-frame conversion.
Regression coverage is in `Python/tests/test_object_rotation.py` and
`MATLAB/Tests/test_objectRotation.m`.

The d18 checkbox now switches between object poses and inverse rig poses.
Saved builder settings retain `hardware_angles_deg`; fixed-hardware mode ignores
these during rendering, while unchecked mode rotates both source and pixels.
The inverse ZYZ list is `[-z2,-y,-z1]`, not component-wise negation. Explicit
rig poses disable orbit timing. Tests compare the unchecked trajectory against
the checked scan transformed into the fixed object frame, including YAML reload.

## Manuals and versioned PDFs

Both manuals include an Applications subsection under Basic workflow, covering
medical rotating-gantry CT, industrial rotating-object CT, cargo/baggage layouts
and stationary CT with switched source positions. Guidance maps each application
to GUI controls, distinguishes active-source changes from mechanical motion,
and documents the one-active-source-per-view model and explicit-surface API
needed for arbitrary discrete source locations. Linked background sources are
included in the manuals; the PDF editions contain the same guidance.

The Python and MATLAB manuals are reorganized in user-workflow order: launch,
basic workflow, each of the four tabs, confirmation/animation, save/load,
reconstruction export, examples/API reference, limitations and troubleshooting.
Manual revision **1.6**, dated **30 October 2026**, is separate from the shared
TIGRE Python package reference **3.0.0**. Both PDFs have a dedicated cover,
separate contents pages, numbered chapters and wrapped tables/code. Existing
inbound section anchors and local documentation links are preserved.

Run `Frontispiece/Export_Manuals.ps1` with Pandoc/XeLaTeX available. Its support
files are under `Frontispiece/pdf_support/`. PDFs are ignored generated files;
the Markdown and PDF build sources are the maintained documents.

## Follow-up: linked voxel dimensions

The Volume tab now exposes editable voxel size (Python z/y/x mm; MATLAB
x/y/z mm). Counts and voxel size edits update full size; full-size edits
update voxel size with counts fixed. Invalid/partial input does not overwrite
related fields, and invalid active spacing is rejected before applying.
Loading controls suspends Python traces until saved counts and extent are
restored, then derives spacing. Persistence still uses canonical counts/size;
no YAML schema change is required. `test_voxel_field_sync.py` covers all update
directions, partial typing and load behavior without a desktop window.

## Follow-up: detector choices and physical transmission guard

Forgotten movement selection is now guarded in both GUIs: when timing and
endpoint movement are off but Start differs from End, Confirm/editable save
offers enable movement, keep stationary, or cancel. Stationary intent is
remembered for the same pair of endpoints. Timing mode never prompts on its
calculated endpoint. Python tests cover all choices and cancellation before
any candidate scene is built.

The Python detector selector now explicitly offers Moving with Source and
Fixed Position, synchronized with the cylindrical/linear source selector.
Python and MATLAB Confirm, editable saves and exported reconstructions validate
finite source-to-pixel segments against each view's volume box. Source/detector
pixels inside/on the box remain invalid. Empty views and panels (including
same-side views, helical entry/exit and overscan) are allowed. The ray-coverage
guard rejects only scans with zero crossing samples over the entire scan;
reports include `empty_views`. Errors retain the previous confirmed scene.
Exact inspection permits diagnosis of invalid archives. This is a
transmission check, not collision/sampling completeness or FDK certification.
See `ct_physics.py`, `validateCTTransmission.m` and their dedicated tests.
Confirm now populates a persistent, scrollable results area in each GUI with
physical PASS/FAILED, counts, CGLS geometry support and FDK-specific status or
deferred checks. Failure reasons remain visible; these messages explicitly
distinguish geometric validation from CUDA execution and sampling/clearance.

Follow-up validation: the affected Python suite passed **146 tests** with no
skips. It covers the physical guard, selector synchronization, confirmation
results, timing, YAML/MAT interchange, remapping, exports and visualization.
MATLAB coverage includes `test_ctTransmission` and the existing builder,
usability, timing/YAML and guided-demo checks. A fixed-detector linear-source
export also ran CUDA CGLS successfully (three iterations, residual decreasing
39.04 to 24.53); validated FDK rejected the noncircular trajectory as expected.

## Delivered workflow

Python and MATLAB use the application name **TIGRE CT Simulation Builder**.
Source, Detectors, Volume and Movement settings occupy compact, scrollable
tabs alongside one shared world-coordinate 3D image. The volume moves while
the source keeps its configured structural trajectory. Python tab labels are
larger/bold; MATLAB applies tab font properties when supported by its release.

Source trajectory selects Cylindrical (circular orbit) or Linear (straight
xyz endpoints with a fixed detector). Source and detector motion controls
stay synchronized. Returning to Cylindrical from Linear retains a fixed
detector/arc source; select rotating detector motion when needed.

Confirm validates and applies all builder tabs. Play starts disabled and is
enabled by the first successful explicit confirmation. Initial automatic
preview does not unlock it; inspection of exact loaded geometry permits
playback without editing. Changing detector motion no longer applies pending
edits automatically. A failed confirmation retains the preceding valid scene.
Tab-aware help resets stale field descriptions; Python also supports hover
and keyboard-focus help, with a scrollable help box.

## Timing, motion and coordinates

- Python volumes: z/y/x; MATLAB volumes: x/y/z. Source/pixel coordinates are
  world x/y/z in both languages. Python projections: views/rows/columns;
  MATLAB projections: rows/columns/views. The interchange bridge converts all
  volume offsets, voxel counts and singleton dimensions explicitly.
- One detector frame per source pulse. `t_i = i / firing_rate_hz`, starting
  at zero. Readout capacity requires `readout_period_ms * firing_rate_hz <= 1000`.
- For N views, the acquisition window is `N/rate`; the final sample is at
  `(N-1)/rate`. Orbit angle is `orbit_start + 360*t/rotation_period_s`.
  Multiple rotations are supported with timing enabled.
- Speed mode translates the volume along world z. Endpoint mode interpolates
  all three coordinates and derives z speed from endpoint travel divided by
  last-view time. Distinct endpoints require at least two views.
- Playback follows elapsed time and may skip display frames, without dropping
  acquired/saved views. Axes cover the complete volume travel.
- Timed acquisition currently requires rotating source + detector. Linear
  and fixed-arc source geometry remain available with manual sampling.

## Persistence and reconstruction export

Editable YAML saves current controls, even before Confirm; exact surface
saves the displayed/confirmed rays and their volume metadata. This distinction
is intentional. Exports preserve voxel counts, physical size, all offsets,
timing and `motion_mode` (static/speed/endpoints). The mode is retained on
reload so changing the view count in speed mode continues using the speed.
Nonlinear loaded volume trajectories open read-only rather than being fitted
to a straight path. Old files without voxel counts remain readable.

The version-1 YAML schema has additive optional fields. Readers from older
checkouts may reject new volume/timing keys; use this revision in both languages.
Python YAML replacement is atomic, with failed exact-surface companion files
removed. Per-view offset counts are checked before save/load. Remapped Python
projection archives retain available volume metadata.

Export recon script starts disabled and is enabled only after a successful
YAML save. Cancelled/failed saves do not enable it; MATLAB MAT saves do not
enable it. Export always references the **last successfully saved YAML**, not
subsequent unsaved controls. Keep YAML and exact-surface companions together.

- Python exports a standalone `.py`: `--algorithm fdk|cgls|both`, optional
  measured NPY/NPZ attenuation projections, `--iterations`, `--voxels`,
  `--geometry`, and `--output`. Default input is a simulated head phantom.
  YAML references are relative to the script where possible.
- MATLAB exports a native `.m` with editable variables for the same workflow,
  loading measured attenuation projections from MAT or simulating headPhantom.
  Its saved YAML path is absolute and must be edited if files move.
- Both scripts convert moving rays into a fixed object frame by subtracting
  each view's volume center from **both** source and detector positions.
  They then run surface CGLS. Validated FDK is tried separately and unsupported
  geometry is reported; no automatic approximate/experimental FDK opt-in.
- GUI compatibility reports concern detector rays only, not moving-volume
  reconstruction. Python now makes this distinction explicit in its save dialog.

## Code map

| Component | Files |
| --- | --- |
| Python UI, timing and builder controls | `Python/tigre/utilities/detector_geometry_gui.py` |
| Numeric/YAML persistence | `Python/tigre/utilities/geometry_io.py` |
| Python script generation | `Python/tigre/utilities/geometry_recon_export.py` |
| MATLAB/Python axis and metadata conversion | `Python/tigre/utilities/geometry_matlab_bridge.py` |
| Shared Python moving-volume renderer | `Python/tigre/utilities/visualization/surface_renderer.py` |
| Optional standalone object-relative renderer | `Python/tigre/utilities/visualization/source_trajectory.py` (not displayed in the current UI) |
| MATLAB UI and timing | `MATLAB/Utilities/detectorGeometryGUI.m`, `ctAcquisitionTiming.m` |
| MATLAB construction and renderer | `MATLAB/Utilities/buildModularSurface.m`, `renderDetectorSurface.m` |
| MATLAB script export | `MATLAB/Utilities/exportCTReconstructionScript.m`, `ctReconstructionTemplate.txt` |
| User manuals | `Frontispiece/TIGRE_CT_Simulation_Builder_Python.md`, `TIGRE_CT_Simulation_Builder_MATLAB.md` |

Public launch/API names are retained for compatibility. MATLAB `app.preview()`
is an alias for explicit Confirm, alongside `app.confirm()`, `app.save(path)`,
`app.save(path,true)`, `app.export(path)` and `app.geometry()`.

## Validation performed

Use **conda recon** for all TIGRE Python execution. The broad CPU/GUI regression
run covered detector construction, timing, help, confirmation, persistence,
script export, MATLAB bridge, surface rendering, remapping and animation:
**154 passed, 1 skipped**. The skipped help-navigation test passed independently
in a fresh process. A separate follow-up run covering remapped volume metadata,
GUI YAML reload and speed-mode reload passed **11 tests**.

Representative command (from `Python/`):

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
conda run -n recon --no-capture-output python -m pytest tests/test_acquisition_timing.py tests/test_gui_parameter_help.py tests/test_geometry_recon_export.py tests/test_detector_geometry.py tests/test_geometry_yaml.py tests/test_geometry_inspection.py tests/test_animate_surface_mesh.py tests/test_geometry_matlab_bridge.py tests/test_source_trajectory.py tests/test_detector_geometry_examples.py tests/test_detector_geometry_remap.py tests/test_animate_geometry.py tests/test_animate_surface.py -q --tb=short -rs
```

The local recon Tcl/Tk installation intermittently fails to read its Tcl/theme
files when several Tk sessions are created in one test process. Tests skip
when Tk initialization is unavailable. Run an affected test in a fresh process;
no environment installation changes were made. Disabling pytest plugin autoload
also avoids unrelated installed-plugin startup delays.

MATLAB R2020a checks passed: `test_detectorGeometryGUI`,
`test_detectorGeometryGUI_usability`, `test_ctSimulationBuilder`, and
`test_detectorGeometryGUIDemo` (with `MATLAB/Demos` on the path). The timing test
checks multi-turn timing, endpoint motion, live volume rendering, confirmation,
native MAT behavior, YAML bridge and script export.

Both exported Python scripts were executed on the RTX 5070 Ti Laptop GPU
with small fixed/moving volumes and three CGLS iterations. Static FDK/CGLS
outputs were finite; moving FDK reported unsupported nonconstant source height
and moving CGLS completed. MATLAB `test_ctReconstructionExport` also passed
both fixed/moving native CUDA cases, using the existing isolated build:

```matlab
addpath('MATLAB'); UseBlackwellBuild;
addpath('MATLAB/Tests'); test_ctReconstructionExport;
```

These are smoke tests, not accuracy validation of arbitrary clinical or GCT
acquisitions. Measured inputs still require calibration and line-integral
conversion before use. Surface projectors require a fixed object frame;
helical FDK is not implemented by this work.

## Audit fixes and remaining boundaries

Fixed premature application on detector-mode changes, destructive failed
confirmation, speed-mode loss on reload, missing nVoxel and dropped moving
offsets in the MATLAB bridge, single-panel scalar squeezing, offset/view-count
validation, YAML write replacement, and lost volume metadata after remapping.

The guided demo and manuals use Confirm/Save YAML labels. UI sizing was checked
at 1024x700 and 1280x720 during development; settings remain scrollable.
This work does not change CUDA kernels or rebuild dependencies.

The original builder commit `dab22d6` excluded pre-existing `.pyd` binaries,
`Python/setup.py`, `Python/demos/d_detector_surface_gct.py`, and MATLAB GIF
changes. Those were subsequently committed separately in `3987109`
(`python rebuild stuff`); this follow-up does not modify them.
