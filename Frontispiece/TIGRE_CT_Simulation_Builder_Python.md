**TIGRE CT Simulation Builder — Python user manual**

**Manual version: 1.1**
**Updated: 28 October 2026**
**Software reference: TIGRE Python package 3.0.0; CT Simulation Builder checkout.**

The manual version identifies this document, not a separate application release.
The PDF edition includes a cover page, contents, and wrapping tables/code.

# Start the GUI

Install TIGRE's Python requirements, including Tkinter and Matplotlib, and
use the project's `recon` conda environment (`conda activate recon`). From the repository's
`Python/` directory, launch the module:

```console
python -m tigre.utilities.detector_geometry_gui
```

Alternatively, from that same directory run:

```console
python tigre/utilities/detector_geometry_gui.py
```

On Windows PowerShell, for example:

```powershell
conda activate recon
Set-Location D:\works\software\TIGRE_work\Python
python -m tigre.utilities.detector_geometry_gui
```

The GUI does not require a CUDA reconstruction to draw its scene. Running
exported reconstruction scripts requires the compiled surface CUDA backend.

# Basic workflow

1. Choose **Detector motion** and **Layout**.
2. Enter the scan, panel, and pixel parameters described below. Hover over a
   label to display its help in the **Parameter help** box.
3. Select **Confirm** and read the check results. Invalid entries are reported
   in a dialog; the previous confirmed scene is retained and the rejected
   candidate is not activated.
4. Inspect the 3D detector and source path. Use the **View** slider to select a
   view, or **Play**/**Pause** to animate it.
5. Select **Save surface...**. TIGRE checks the geometry before opening the
   save dialog. Review the iterative-projector and `fdk_surface` messages, then
   confirm to choose the `.npz` destination.

For suitable circular acquisitions, **Remap for FDK...** can resample measured
projections onto a compatible detector. It cannot repair a noncircular source
trajectory or inadequate angular coverage. This optional data conversion does
not alter the preview or original data.

Every preview marks the world coordinate origin `(0, 0, 0)` with three short
arrows: +x in red, +y in green, and +z in blue. The arrows share the origin,
carry x/y/z labels, and scale with the source distance for visibility.
When a rotating scan has nonzero COR, an orange sphere labelled **COR** marks the current
COR translation `COR * (-sin(angle), cos(angle), 0)`, connected to the origin
by an orange line. It updates with the view slider and playback. The legend
states the signed COR in millimetres. The shift is drawn at actual scale;
zoom near the origin to inspect small offsets. This is the applied rig
translation, not a separate rotation axis or a per-panel correction.

The plot shows panel outlines, the source path, and the current source position;
it is a layout preview, not a projection or a scale-independent schematic.

The four settings tabs share one main 3D view. Larger/bold tabs, scrollable
settings and fixed action buttons keep controls accessible on smaller screens.
Parameter help switches with the tab; hover or focus a field for details.

# Use the Source tab

Choose **Source trajectory: Cylindrical** for a circular orbit about +z,
or **Linear** for world x/y/z endpoints. Cylindrical source motion is
independent of cylindrical detector layout. Linear selects a fixed detector;
switching back to Cylindrical retains the fixed detector and uses an arc
source. Choose Moving with Source in Detectors for a rotating rig.

Views is shared with Movement. Radius, height and angular controls apply
to circular/arc sources; start/end coordinates apply to linear sources.
Disabled fields are ignored. Timing mode derives angular span instead.


## Moving source and detector

With **Detector motion** set to **Moving with Source**, the source and panel
layout rotate together around the z axis.

| Control | Meaning |
|---|---|
| **Views** | Number of source/detector positions. |
| **Orbit start (degrees)** | Initial gantry/source angle about +z. |
| **Scan span (degrees)** | Angular travel. A full 360-degree scan does not duplicate its endpoint; a shorter scan includes both endpoints. |
| **Source radius (mm)** | Source-to-rotation-axis radius. |
| **Source height (mm)** | Constant source z coordinate. |

For example, `Views = 50`, `Orbit start = 0`, and `Scan span = 360`
sample a full circular CT orbit at 50 distinct angles. A one-view surface is
placed at the starting angle.

**COR (mm)** applies TIGRE's centre-of-rotation correction in rotating mode:
both source and detector receive the per-view shift
`COR * (-sin(angle), cos(angle), 0)`. It defaults to zero and stays enabled in every example. Stationary detectors
require COR = 0; a nonzero value produces a validation error.

## Stationary detector and moving source

Set **Detector motion** to **Fixed Position** and Source trajectory to **Linear** to hold the panels fixed while the
source travels along a straight line. Enter **Views**, **Source start (x, y, z
mm)**, and **Source end (x, y, z mm)** in TIGRE world coordinates. Both
endpoints are included when there is more than one view. Source radius, source
height, orbit start, and scan span are ignored in this mode.

The **Tomosynthesis: linear source, fixed panel** preset moves the source
from `(1000, -500, 0)` to `(1000, 500, 0)` mm over 41 views, a 1000 mm
linear travel past a fixed 512 x 256 mm panel.

The panel pose is set from the first source position and held fixed. COR stays
editable and must be zero in stationary modes. The validated `fdk_surface` path
rejects a linear source trajectory; use iterative surface methods. The
experimental FDK opt-in is described below.

## Stationary detector and arc source

Select **Fixed Position** in Detector motion and **Cylindrical** in Source
trajectory for an arc source with stationary panels. Arc mode uses **Source radius**,
**Source height**, **Orbit start**, **Scan span**, and **Views**. The detector
faces the midpoint of the acquisition arc and stays fixed as the source moves.
Source start/end fields are ignored. Prefer iterative reconstruction.

# Use the Detectors tab

All distances and pixel pitches are in millimetres. The source and detector
coordinates use TIGRE's world frame.

## Shared panel and detector controls

| Control | Meaning |
|---|---|
| **Module rows / Module columns** | Pixel counts in each panel. Every panel has the same shape. |
| **Row pitch / Column pitch** | Centre-to-centre pixel spacing. |
| **Detector radius** | Distance from the rotation axis to the nominal detector plane. |
| **Panel roll / x and pitch / y (degrees)** | One finite angle for all panels or one per panel; zero by default. |
| **Gap columns** | Number of invalid dense-array columns between adjacent panels. Use zero when measured data has no gap samples. |

The dense projection array has shape `(views, rows, columns)`. Gap samples
occupy columns in that array but are marked false in `valid_mask`; do not
interpret them as measured detector pixels.

**Panel roll / x (degrees)** and **Panel pitch / y (degrees)** add full 3D
orientation to either layout. Enter one value for every panel or a single value
to apply to all panels. Rotations act about each panel centre in this order:
local x (panel normal), local y (column direction), then the existing z panel
angle and gantry rotation. This matches TIGRE's detector rotation matrix:
x corresponds to `rotDetector[0]`, y to `[1]`, and z to `[2]`.
The GUI names x, y, and z rotations **roll**, **pitch**, and **yaw**,
respectively. Roll is about the panel normal, pitch about its local column
direction, and yaw about its nominal local row direction. Rotation naming can
vary across TIGRE code and other tools; the axis labels and matrix define the
convention used here. These controls set angles, not pixel pitch.
The matrix is `Rz(gantry + panel_z) @ Ry(panel_y) @ Rx(panel_x)`.
Angles and COR are constant throughout a GUI scan. For per-view values, use
`surface_from_geometry(geo, angles)` or explicit `DetectorModule` poses.
The direct `rotDetector` mapping applies to a flat, unoffset single panel;
for faceted panels the x/y rotations act in each panel's own frame.
The defaults are zero. Preview outlines and exported pixel coordinates include
these rotations. Out-of-plane tilts can make the validated FDK path unsupported;
inspect the compatibility report.

## Cylindrical layout

Choose **Cylindrical** to distribute equal-sized tangent panels over an arc.
Set **Modules (cylindrical)** to the panel count and **Arc span (degrees)** to
the angle between the first and last panel-centre radial directions. For one
panel, the span has no geometric effect.

## Piecewise-linear layout

Choose **Piecewise linear** for panels with individually specified centres and
orientations. The panel count is inferred from the list lengths; the disabled
**Panels (from lists)** field is informational. Enter one comma-separated value
per panel in each list:

| Control | Meaning |
|---|---|
| **Panel transverse offsets (mm)** | Centre offset in local +y. |
| **Panel normal offsets (mm)** | Centre offset in local +x from the nominal detector plane. Positive is toward the source at view zero. |
| **Panel yaw / z (degrees)** | Rotation about +z relative to the nominal plane; zero faces the source. |
| **Panel z offsets (mm)** | Centre offset in z from the nominal detector centre. |

The four lists must have the same nonzero length. For a three-panel U-shaped
example, the defaults `transverse = -32, 0, 32`, `normal = 0, 10, 0`, and
`angles = 45, 0, -45` create two angled outer panels and a forward centre
panel. Adjust these values to match the scanner's physical panel centres and
orientations.

## Multi-panel alignment: individual tilt, global COR

**Each panel can have its own tilt and position offsets. COR is one global
scan correction shared by the source and every panel; it is not a per-panel
alignment parameter.**

| Adjustment | Scope and input |
|---|---|
| Panel x/y tilt | Either layout: one value for all panels, or one comma-separated value per panel. |
| Panel z angle | Piecewise linear: one value per panel. Cylindrical: determined by the arc layout. |
| Transverse, normal, and z position offsets | Piecewise linear: one value per panel in each list. |
| COR | One scalar for the whole rotating scan; shifts source and all panels together. Editable in all modes; stationary detectors require zero. |

For three panels in **Piecewise linear** layout, for example:

| Field | Example |
|---|---|
| Panel roll / x (degrees) | `5, 0, -5` |
| Panel pitch / y (degrees) | `2, 0, 2` |
| Panel yaw / z (degrees), about z | `45, 0, -45` |
| Panel transverse offsets (mm) | `-256, 0, 256` |
| Panel normal offsets (mm) | `0, 10, 0` |
| Panel z offsets (mm) | `0, 0, 0` |
| COR (mm) | `7` |

Use **List...** beside a panel tilt or position field to open its full
list in a scrollable editor, with one value per line in panel order. A shared
x/y tilt expands to one value per panel. **Apply** validates the count and
finite values and copies the list back. Select **Confirm** to rebuild the
displayed geometry, then **Play** to animate it. **Apply** alone leaves the
current preview unchanged. **Cancel** leaves the field unchanged. The transverse list
in the main window determines the piecewise-linear panel count; edit that
comma-separated field first when adding or removing panels.

List buttons are disabled with their fields: cylindrical layout disables
piecewise-linear offsets and z-angle lists, while x/y tilt lists remain
available. **Panels (from lists)** shows the inferred count and is read-only;
switching back to cylindrical preserves its previous module count.

Entries refer to the same panels in the same order across all lists. X/y lists
may contain a single shared value; z angles and position lists must each have
one entry per panel. These are illustrative alignment values, not a scanner
calibration. Rotations act about each panel's centre and remain constant during
the scan.

To model one misaligned panel, adjust that panel's tilt and position offsets.
Do not assign separate COR values to the panels: that would imply different
source/rotation geometries rather than a panel-only alignment correction.

# Use the Volume tab

Use **Volume** to define the reconstruction box. All four vectors use
**z, y, x** order; source coordinates use **x, y, z** instead.

| Field | Meaning and update rule |
|---|---|
| Voxel counts | Three positive integers. Editing counts preserves voxel size and updates full size. |
| Full size (mm) | Full positive physical lengths, not half-widths. Editing full size preserves counts and updates voxel size. |
| Voxel size (z, y, x mm) | Positive spacing. Editing spacing preserves counts and updates full size. |
| Center offset (mm) | Initial box center; shared with Movement Start position. |

The relationship is `sVoxel = nVoxel * dVoxel`, component by component.
For example, counts `128, 128, 128` with spacing `2, 2, 2` describe a
`256, 256, 256` mm box. Changing counts to `256, 128, 128` preserves spacing
and produces a `512, 256, 256` mm box. To keep physical size fixed while
changing sampling, edit counts and then restore the desired full size.

Related fields update after three valid values are entered. Partial typing
does not overwrite the other fields. **Confirm** applies the new box to the
scene. The GUI stores metadata and draws a box; it does not allocate voxel
data. YAML preserves counts and full size and derives spacing on reload.

# Use the Movement tab

Use **Movement** for object translation and acquisition timing. Geometry
stays in the structural world frame: the source follows its configured path
while the volume box moves. Enable **Move between start and end positions**
for endpoint translation; entering two values alone does not enable motion.

| Control | Purpose |
|---|---|
| Start position (z, y, x mm) | Initial volume center, shared with Volume Center offset. |
| End position (z, y, x mm) | Final center in endpoint mode; calculated/read-only in timed speed mode. |
| Move between start and end positions | Interpolate all three coordinates across views; distinct endpoints need at least two views. |
| Use timing to derive orbit and z motion | Use one frame per source pulse. Requires rotating source and detector. |
| Detector readout period (ms) | Minimum frame interval; firing rate must not exceed `1000 / readout_period_ms`. |
| Source firing rate (Hz) | Pulses and acquired views per second. Pulses are not discarded or combined. |
| Rotation period (s/turn) | Seconds per full turn; determines turns over the acquisition. |
| Volume z speed (mm/s) | Signed world-z speed; positive moves toward +z. Derived/read-only in timed endpoint mode. |
| Playback speed (x real time) | Animation speed relative to acquisition time. |
| Views / source pulses | Same value as Source Views, not number of turns. |

Without timing, endpoint mode interpolates Start to End over the selected
source views. Without timing or endpoint motion, the volume is stationary
at Start and End is ignored. If these inactive endpoints differ, Confirm
and Save YAML ask whether to enable movement, keep stationary, or cancel.
An explicit stationary choice is remembered for the same endpoint pair.

With timing enabled, frame `i` occurs at `t_i = i / firing_rate_hz`, starting
at zero. Angle is `orbit_start + 360*t_i/rotation_period_s`. For N views,
the acquisition window is `N/rate`, while the last view is at `(N-1)/rate`.
The GUI distinguishes these values. In speed mode the center is
`start + [z_speed*t_i, 0, 0]`; endpoint mode derives z speed from total z
travel divided by last-view time. Travel per turn is `z_speed*rotation_period`.

For a timing example, 1,000 views, 10 ms readout, 100 Hz firing, 1 s/turn,
and 20 mm/s describe a 10-second window and 10 turns. The last sample is at
9.99 s, after 9.99 turns and 199.8 mm travel. Timed mode replaces manual
scan span and permits multiple turns. Fixed-arc and linear sources use
manual sampling; timed acquisition currently requires rotating hardware.

For a simple endpoint spiral, use a circular source, 600 views, Start
`-200, 0, 0`, End `200, 0, 0`, and **enable endpoint movement**, then Confirm.
Timing is optional for that example. Empty entry/exit views are allowed as
long as at least one measured ray crosses the volume somewhere in the scan.

# Confirm, check results and animate

After editing any tab, select **Confirm**. It builds a candidate from all
current controls, validates it, and replaces the active scene only on success.
Failed confirmation retains the preceding confirmed scene; the plot therefore
does not show rejected values. Play starts disabled until the first successful
explicit Confirm. Loaded exact geometry can be inspected without editing.

The persistent **Confirmation checks** area reports physical PASS/FAILED,
view and crossing-sample counts, allowed empty views, and reconstruction
support. Python checks FDK geometry for fixed volumes; scan coverage remains
separate. For moving volumes it defers the object-frame FDK check to the
exported script. MATLAB lists the separate FDK requirements for its exporter.
These are geometry checks, not a CUDA execution test or a scan-quality certificate.

The physical check uses finite source-to-pixel segments and the volume box
at each view. Source and valid detector pixels must remain outside the box.
Empty views/panels, including helical entry/exit and overscan, are allowed.
The coverage guard rejects only a scan with **zero crossing samples in total**.
It does not establish complete angular sampling or mechanical clearance.

Select **Play** to animate or pause, or drag the view slider to inspect a pose.
With timing enabled, playback follows elapsed time and may skip display frames;
all acquired views remain saved. The view slider displays sample time.

# Save and load geometry

**Save YAML...** validates and saves current entries, even if not yet
confirmed. **Save surface...** writes the displayed rays/volume, not pending
edits. Save YAML does not run a reconstruction. Motion mode, voxel counts,
full size, offsets and timing are retained. Keep exact-surface companions
with their YAML. Older files without counts require an explicit voxel override
when reconstructing. YAML replacement is atomic.

## Save and reload editable YAML settings

Use **Save YAML...** to save the current builder controls as a
human-readable `.yaml` file. This saves the current entries, even if you have
not pressed **Confirm**, and validates that they can build a geometry.
**Save surface...** remains the exact numeric `.npz` export.

**Load geometry...** reopens modular YAML in builder mode: layout, source
motion, panel positions, roll/pitch/yaw, COR, pitches, dimensions, and scan
sampling are restored. Loading also rebuilds the preview. A full 360-degree
scan uses distinct views; shorter scans include both endpoints.

Try the shipped [three-panel YAML example](../Python/demos/geometry_examples/three_panel.yaml).
It uses independent panel rotations, two gap columns, and zero global COR.
Edit its numeric lists with a text editor or use the GUI's **List...**
controls. YAML angles are in **degrees**, unlike standard TIGRE `rotDetector`
and `angles` arrays, which use radians.

The version 1 document has these fields:

| Field | Meaning |
|---|---|
| `schema`, `version` | `tigre.geometry`, integer `1`. |
| `units` | `length: mm`, `angle: degree`. |
| `rotation_convention` | `Rz(gantry+z) @ Ry(y) @ Rx(x)`. |
| `kind: modular`, `builder` | Geometry-builder parameters; x/y/z panel angles are roll/pitch/yaw in degrees. |
| `kind: surface`, `surface_file` | Exact `.npz` companion, resolved relative to the YAML file. Opens in inspection mode. |
| Optional `volume` | `sVoxel`, optional `nVoxel`, `offOrigin` (one or per-view z/y/x offset), and `motion_mode` (`static`, `speed`, `endpoints`). Sizes/offsets are millimetres. |

In inspection mode, **Save YAML...** writes a small YAML file plus a
uniquely named `.surface-....npz` companion. It preserves exact coordinates,
masks and module boundaries without fitting them to builder settings. Keep
both files together when moving or sharing them. Large pixel arrays are never
embedded in YAML. Builder YAML is a recipe; use `.npz` when exact stored
coordinate arrays are required. Unknown schema versions, units, rotation
conventions and settings are rejected; YAML is read with a safe loader.
PyYAML 6 or newer is included in the Python requirements.

The animation utility accepts the same files:

```python
import tigre
import matplotlib.pyplot as plt

ani = tigre.animate_geometry("geometry.yaml")
plt.show()
# Or inspect the file in the GUI:
from tigre.utilities.detector_geometry_gui import launch_detector_geometry_gui
launch_detector_geometry_gui(geometry="geometry.yaml")
```

For scripts, `tigre.utilities.geometry_io.load_geometry_file(path)` returns a
`GeometryDocument` with `geometry` (optional volume), `surface`, and `settings`
(editable modular parameters or `None` for exact surfaces). Call
`save_geometry_yaml(path, settings=parameters)` or
`save_geometry_yaml(path, surface=surface, geometry=geo)` to save from Python.
`animate_geometry` file inputs carry their own trajectory: do not also pass
`angles` or `surface`; file rendering supports `rotation="SD"`.

## Load existing geometry for inspection

Select **Load geometry...** and choose `.npz`, `.yaml`, or `.yml`. Modular YAML
settings reopen in editable builder mode (see below). Numeric surface archives
and YAML surface references open in
**Inspection mode**, preserving the loaded source positions, pixel coordinates,
validity masks (including per-view changes), and available module boundaries.
The view slider, playback, saving, and optional remapping remain available.
Construction fields, layout/motion controls, list editors, and **Confirm** are
disabled; their retained values do not describe or modify the loaded geometry.

**Return to builder** rebuilds the previous builder settings. **Load example**
also exits inspection mode and creates that preset. Loaded arbitrary geometry
is never converted into editable builder parameters.

Supported archives:

| Archive | Fields |
|---|---|
| Explicit detector surface | `source_positions_mm`, `pixel_centers_mm`; optional `valid_mask` and `module_slices`. |
| Standard TIGRE cone geometry | `DSO`, `DSD`, `nDetector`, `dDetector`, `angles` (radians). Optional `offDetector`, `rotDetector` (radians), `COR`, `nVoxel`, `sVoxel`, `offOrigin`. |

`module_slices` is an integer `(modules, 4)` array of
`[row_start, row_stop, column_start, column_stop]`, with exclusive stops.
New surface exports retain it when available. Older archives still load;
without metadata, neighbouring tiles with no gap cannot be distinguished.
Surface archives may include `sVoxel` and `offOrigin` to draw the reconstruction
volume; otherwise no volume is drawn. Standard parameter archives use TIGRE's
low-resolution default volume unless volume fields are supplied. The displayed
volume accepts a single `offOrigin` or one per view. Nonlinear volume paths
open read-only to preserve exact coordinates. No pickle or Python geometry files are
executed by the loader.

From Python, inspect an existing object directly:

```python
from tigre.utilities.detector_geometry_gui import launch_detector_geometry_gui

launch_detector_geometry_gui(geometry=geo, angles=angles)
# Or inspect an explicit detector surface, with optional volume geometry:
launch_detector_geometry_gui(surface=surface, geometry=geo)
```

To write a standard parameter archive for **Load geometry...**:

```python
import numpy as np

np.savez("geometry.npz", angles=angles, DSO=geo.DSO, DSD=geo.DSD,
         nDetector=geo.nDetector, dDetector=geo.dDetector,
         offDetector=geo.offDetector, rotDetector=geo.rotDetector, COR=geo.COR,
         nVoxel=geo.nVoxel, sVoxel=geo.sVoxel, offOrigin=geo.offOrigin)
```

COR and rotations are already represented in the loaded surface coordinates;
they are not applied again. An independent COR marker is not inferred for
arbitrary loaded trajectories. The GUI and `animate_geometry(surface=...)`
use the same renderer for the detector mesh, source path, volume, sampled
pixels, central ray, and per-view updates. Builder previews additionally draw
physical panel edges and the known COR marker.

# Export a reconstruction example

**Export recon script...** starts disabled and becomes available after
**Save YAML...** succeeds. Cancelled or failed saves do not enable it. The
exported Python script references the last successfully saved YAML, so save
again before exporting if you want to include later edits. Keep that YAML and
any companion surface NPZ files together. The export does not run reconstruction.

Run the script in the TIGRE CUDA environment:

```powershell
conda run -n recon python saved_recon.py --algorithm both
conda run -n recon python saved_recon.py --projections projections.npy --algorithm cgls --iterations 30
```

The script uses the saved voxel counts, physical size, center and surface.
Without measured data it simulates the bundled head phantom. Measured inputs
must be attenuation line integrals in `(views, rows, columns)` order, supplied
as NPY or an NPZ containing `projections`. Use `--voxels Z Y X` when an older
YAML has no voxel counts, or to explicitly override them. `--output` chooses
the result NPZ path, and `--geometry` overrides the saved YAML reference.

`--algorithm both` tries validated `fdk_surface` and then `cgls_surface`; an
unsupported FDK trajectory is reported and recorded while CGLS continues.
Selecting `--algorithm fdk` directly reports the FDK error. Moving volumes are
converted to a fixed object frame by subtracting each view's center from both
source and detector coordinates; CGLS then uses that transformed surface.
Helical or linear paths may be incompatible with validated FDK. The script
does not automatically enable experimental FDK. Results contain reconstructed
volumes, volume metadata and CGLS residuals, plus the phantom for simulated data.

Inspect the script before running it. Generated scripts validate physical
transmission again and leave FDK's trajectory/coverage checks to the algorithm.
The FDK skip reason and CGLS residuals are retained in the output archive.

# Reconstruction compatibility and measured-data remapping

## What restricts `fdk_surface`?

The default `fdk_surface` path implements approximate cone-beam FDK, not a
general arbitrary-trajectory solver. It requires:

* source positions on a circular orbit about z at constant height and moving
  monotonically around the orbit;
* detector coordinates that describe an extruded fan: pixel x/y coordinates
  are independent of detector row, z coordinates are independent of column
  and view, and each fan-angle table is the same for every view;
* identical whole-column validity masks in all views, with fan angles strictly
  monotone along valid columns.

The GUI's check catches its structural geometry restrictions. The algorithm
also checks that a short scan covers the required fan-angle range when
reconstruction is run.

The explicit opt-in `allow_approximate=True` path handles per-view geometry
directly, including curved or modular detector surfaces, when fan angles are
monotone across every detector row. It filters each row using its own fan-angle
coordinates and uses the surface ray-driven backprojector. For non-circular
source paths it uses source azimuth for angular quadrature and skips Parker
weighting. This path is experimental—not a validated general-trajectory FDK—
and can produce substantial artifacts. TIGRE emits a runtime warning when it
runs; use it for design exploration, and prefer iterative surface reconstruction
for quantitative results.



## Remap measured projections

Select **Remap for FDK...** after previewing the geometry. Supported input
formats are:

* `.npy`: one projection array;
* `.npz`: an array named `projections`, or a single-array archive;
* `.raw` / `.bin`: little-endian float32 values, reshaped to match the preview.

The array shape must match `(views, rows, columns)`. For a one-row surface,
two-dimensional arrays `(views, columns)` are accepted. The output is a
separate compressed `.npz` containing `projections`, `source_positions_mm`,
`pixel_centers_mm`, `valid_mask`, and remapping metadata, including the target
radius and a short-scan coverage estimate. Use the data and surface from that
output together; do not pair the remapped projections with the original
surface.

The tool maps detector samples by their fan and vertical ray angles onto an
idealized circular detector at the median of the per-view median
source-to-pixel distances. It chooses an approximately input-resolution
equiangular grid automatically.
Interpolation is linear and does not extrapolate outside the measured
segments. Gaps stay masked. The source orbit must already be circular, at
constant height, and monotone; remapping the detector cannot repair a
non-circular or linearly moving source trajectory. The initial implementation
rejects partial-column masks, overlapping module ray coverage, non-monotone
fan-angle segments, and geometry without sufficient common coverage.

**Warning:** remapping creates derived samples; it is not information recovery.
Interpolation can blur features, create correlations, or introduce geometric
bias, especially with large panel offsets, sparse sampling, or wide gaps.
Validate image quality against an iterative reconstruction from the original
geometry/data before relying on the remapped FDK result. The tool checks the
resulting detector geometry and reports whether the scan meets FDK's default
2-degree short-scan tolerance, but it does not run FDK or test CUDA availability.

To reconstruct from the output archive:

```python
import numpy as np
from tigre.algorithms import fdk_surface
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry

with np.load("remapped_for_fdk.npz", allow_pickle=False) as archive:
    projections = archive["projections"]
    surface = DetectorSurfaceGeometry(
        archive["source_positions_mm"],
        archive["pixel_centers_mm"],
        archive["valid_mask"],
    )

reconstruction = fdk_surface(projections, geo, surface)
```

`geo` is the usual TIGRE reconstruction-volume geometry. Check the
`fdk_short_scan_supported` metadata before reconstruction, especially for a
short scan.

# Examples and presets

## TIGRE circular CT example

Select **TIGRE: circular cone-beam CT** and **Load example**, then **Confirm** and **Play**
to inspect the source and flat detector rotating together around the z axis.
This uses the low-resolution cone-beam geometry from TIGRE's
[`d03_generateData.py`](../Python/demos/d03_generateData.py), defined by
[`ConeGeometryDefault`](../Python/tigre/utilities/geometry_default.py):

| Setting | Value |
|---|---:|
| Detector motion | Moving with Source |
| Layout | Piecewise linear |
| Views | 50 |
| Orbit start / Scan span | 0 / 360 degrees |
| Source radius / height | 1000 mm / 0 mm |
| Module rows / columns | 128 / 128 |
| Row pitch / Column pitch | 3.2 mm / 3.2 mm |
| Detector radius | 536 mm |
| Gap columns | 0 |
| Panel transverse / normal offsets | 0 / 0 mm |
| Panel angle / z offset | 0 degrees / 0 mm |

The single flat panel is 409.6 x 409.6 mm. TIGRE specifies source-to-origin
`DSO = 1000 mm` and source-to-detector `DSD = 1536 mm`, so the GUI's detector
radius is `DSD - DSO = 536 mm`. All panel offsets and angles are zero.

The trajectory is the same full circular source-and-detector motion used in
TIGRE's sample-data demo. The demo's `np.linspace(0, 2 * np.pi, 50)` includes
the duplicate endpoint; the GUI uses
`np.linspace(0, 2 * np.pi, 50, endpoint=False)` for 50 distinct views. Generate
projections with the GUI's angles when pairing data with its exported surface.

This example contains `50 x 128 x 128 = 819,200` detector samples. The float32
pixel coordinates occupy about 9.4 MiB and the validity mask about 0.8 MiB,
with additional working memory needed for construction and saving. Preview
and export check geometry compatibility; they do not simulate projections or
run reconstruction.

### GCT modular scan (last preset)

**GCT: 53-panel circular scan** is the last entry in **Example**. It restores
the idealized GCT layout: 53 panels of 1 x 64 pixels, 2.6172 mm column pitch,
DSO 6100 mm, DSD 10600 mm (detector radius 4500 mm), and 5701 views starting
at 156 degrees over a 228-degree scan. Gaps, x/y tilts, and COR are zero.
Use **List...** to inspect the 53 panel centres and z angles.

This is an idealized uniformly sampled panel model. The measured-data demo
[`d_detector_surface_gct.py`](../Python/demos/d_detector_surface_gct.py) also
uses an asymmetric local pixel index and per-pixel offset correction; use its
geometry when reproducing the measured data exactly.
The preset contains 19,337,792 samples: coordinates and mask alone occupy
about 240 MiB, with additional working memory during construction/export.
Reduce **Views** before previewing on memory-constrained systems.
**Load example** builds the full geometry immediately.

### Tilt and COR presets

Select one of these entries in **Example**, then **Load example**, **Confirm** and **Play**:

| Preset | What it demonstrates |
|---|---|
| **3D detector tilt: x, y, z** | Panel rotations of x = 20, y = 15, z = 10 degrees, held constant during the circular scan. COR is zero. |
| **TIGRE: global COR offset** | The low-resolution TIGRE circular geometry with an exaggerated +150 mm COR for visibility. Source and detector shift together; panel tilt is zero. |

The tilt angles are illustrative and independently adjustable.
The COR preset extends the package geometry with a nonzero correction;
+150 mm is exaggerated for visualization, not a calibration value from a
shipped acquisition example. The sphere centre shows the actual shift; its
radius is scaled for display and has no physical meaning.
Both use 50 views, a 128 x 128 panel, 3.2 mm pitch, DSO 1000 mm and DSD 1536 mm.
The example tests compare both full circular scans against TIGRE's geometry
converter. Use panel offsets and tilts for individual panel alignment; COR
remains one global value.

### Try panel tilt and COR

Load **TIGRE: circular cone-beam CT**, enter **Panel roll / x = 13**,
**Panel pitch / y = -17**, **Panel yaw / z = 23**, and **COR = 7**, then select
**Confirm** and **Play**. The panel tilts about its centre and the rig receives
the COR translation. At zero gantry angle, positive COR shifts both source and
detector toward +y. COR does not move the reconstruction volume.

Saved coordinates already include tilt and COR. Do not apply `geo.rotDetector`
or `geo.COR` again to those arrays. The surface projectors use the saved ray
coordinates; `geo` supplies the reconstruction volume. The archive contains
geometry arrays and available module/volume metadata, not the original GUI
settings. Reloading opens inspection mode rather than editable settings.

### Validation against TIGRE examples

`test_detector_geometry_examples.py` reads geometry assignments from
`d17_DetectorRotation.py` and checks all 100 poses against
`surface_from_geometry` and an independent SciPy Euler rotation reference.
Each pose is checked separately: the GUI does not reproduce the demo's
changing tilt in a single scan. Tests also read `d01_CreateGeometry.py` and
check its zero-COR baseline, plus additional positive and negative COR values
at four cardinal angles. The nonzero COR values are test extensions.
Comparisons allow 0.0005 mm for float32 rounding. These are geometry checks,
not projection or reconstruction-quality tests.

Run the CPU geometry and remapping tests from the repository root:

```powershell
conda activate recon
$env:PYTHONPATH = (Join-Path $PWD 'Python')
python -m pytest Python/tests/test_detector_geometry.py Python/tests/test_detector_geometry_examples.py Python/tests/test_detector_geometry_remap.py -q
```

# Programmatic visualization

Use `tigre.animate_geometry(geo, surface=surface)` after loading the archive
as shown below. The renderer supports curved grids, module boundaries, gaps,
and changing per-view masks. It shares the GUI's surface renderer and coordinate-arrow drawing.
See the [surface visualization guide](./Detector_surface.md#visualizing-curved-and-modular-detectors)
for details. New GUI exports include available `module_slices`; pass them
when reconstructing the surface object to preserve adjoining tile boundaries. The animation utility visualizes supplied geometry;
it does not provide GUI editing or infer the original COR value.

For a full YAML-to-CUDA workflow, see
[`d_geometry_yaml_reconstruction.py`](../Python/demos/d_geometry_yaml_reconstruction.py).
That separate demonstration chooses its own volume settings; the GUI-exported
sample script instead uses saved volume metadata. Detector tilt reference
checks are in `test_detector_geometry_examples.py` and `test_demo_geometry_animations.py`.

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


The GUI currently allows at most:

* 6,000 views;
* 4,096 rows or columns in an individual panel;
* 256 panels;
* 20,000,000 dense detector samples (`views x rows x columns`, including gaps).

Exceeding a limit produces an input error before geometry construction. A valid
layout that still runs slowly or exhausts memory should be reduced to suit the
available system RAM; the limits are upper bounds, not memory guarantees.

Other common issues:

* **Unequal panel-list lengths:** enter the same number of values in all four
  piecewise-linear lists.
* **Unexpected panel orientation:** check the sign of the transverse offset
  and panel angle; positive angles rotate counterclockwise about +z.
* **Unexpected preview or stale geometry:** edit the inputs and select
  **Confirm** again. Saving always checks the currently previewed surface.
* **FDK unsupported:** use an iterative surface algorithm, or adjust the scan
  and detector geometry to satisfy `fdk_surface`'s constraints.
* **Save/check error:** correct the reported invalid geometry and preview
  again; a successful save is not attempted if the compatibility check fails.

# Documentation and PDF regeneration

See the [MATLAB manual](./TIGRE_CT_Simulation_Builder_MATLAB.md),
[handover notes](./CT_Simulation_Builder_Handover.md),
[surface geometry reference](./Detector_surface.md), and
[demo comparison](./Demo_language_comparison.md).

From the repository root, run `./Frontispiece/Export_Manuals.ps1` to regenerate
both PDFs. Pandoc, XeLaTeX and the configured Consolas font are required.
The PDF helper fixes table widths, wraps code, and produces a versioned cover
and separate contents pages. PDFs are generated artifacts ignored by Git.
