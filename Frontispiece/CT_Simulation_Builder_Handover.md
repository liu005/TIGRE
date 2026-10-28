# TIGRE CT Simulation Builder handover

Updated: 2026-10-28 (Australia/Sydney).

## Manuals and versioned PDFs

The Python and MATLAB manuals are reorganized in user-workflow order: launch,
basic workflow, each of the four tabs, confirmation/animation, save/load,
reconstruction export, examples/API reference, limitations and troubleshooting.
Manual revision **1.1**, dated **28 October 2026**, is separate from the shared
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
