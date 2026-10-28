# TIGRE CT Simulation Builder manual

## Export a reconstruction example

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

## Acquisition timing and moving volume

In **Source**, choose **Source trajectory: Cylindrical** for a circular orbit
about world +z, or **Linear** for a straight start-to-end source path. Linear
mode selects a stationary detector and enables source start/end coordinates.
Switching back to Cylindrical retains a fixed detector with an arc source;
use **Detector motion** to rotate the detector with the source. **Confirm**
applies the selected trajectory. This selector is independent of detector layout.

The workspace has **Source**, **Detectors**, **Volume** and **Movement**
settings tabs beside one shared 3D preview, with no top-level workspace tabs.
**Movement** contains scan timing, speed, view count and start/end volume
positions. Settings scroll when needed; Confirm, save/load actions, playback
and the view slider remain accessible.
The compact settings sidebar keeps save/load actions below the settings, while
the playback strip spans the bottom so the main 3D image has more room.
In the builder, **Play** stays disabled until the first successful **Confirm**;
the initial automatic preview does not enable playback.
Parameter help switches to an overview of the selected settings tab. Hover a
parameter label or control, or focus its entry using the keyboard, for its
units, coordinate order and mode-specific behavior. The help box scrolls for
long explanations and identifies inactive controls during geometry inspection.

The shared preview keeps the source trajectory in world coordinates, as defined
by the structural settings, and animates the volume box at its per-view
`offOrigin`. Z translation moves the volume through the rotating hardware;
the displayed source trajectory stays circular rather than becoming an
object-relative helix. Movement controls retain timing and speed readouts,
including the current source position relative to the volume center.

The reconstruction-volume controls use TIGRE **z, y, x** order for voxel counts,
full physical size and center offset. Select **Confirm** to apply changes.
For a timed helical-motion preview, enable **Use timing** and set:

* **Views**: number of detector frames / source pulses to acquire.
* **Detector readout period (ms)**: minimum time between frames.
* **Source firing rate (Hz)**: one frame per pulse; it cannot exceed
  `1000 / readout_period_ms`. This model assumes synchronized triggering and
  does not discard pulses or combine exposures.
* **Rotation period (s/turn)**: time for one complete source/detector turn.
* **Volume z speed (mm/s)**: signed table/volume speed; positive moves along
  world +z. This is independent of the detector readout and source firing rates.
* **Playback speed**: animation time relative to acquisition time.

In **Movement**, **Start position** and **End position** use z, y, x mm.
Start is shared with the structural volume center. Enable **Move between start
and end positions** to interpolate all three coordinates across the views.
With timing enabled, z speed is derived as `(end_z - start_z) / last_view_time`;
the z-speed input becomes read-only. At least two views are needed for different
endpoints. Without timing, the same endpoints define the per-view movement.
Disable the endpoint option to use the entered z speed with acquisition timing;
the end position then displays the calculated final center.

Timed mode replaces the manual scan span. In z-speed mode, starting
at `t = 0`, frame `i` occurs at `i / firing_rate_hz`. Its source angle is
`orbit_start + 360 * t / rotation_period_s`, and its volume center is
`offOrigin_start + [table_speed_mm_s * t, 0, 0]`. Multiple turns are supported.
For `N` views, the acquisition window is `N / firing_rate_hz`, while the final
sample is at `(N - 1) / firing_rate_hz`. The summary distinguishes these values
and shows turns, views per turn, z travel and travel per turn. The plot bounds
cover the whole volume trajectory, and the view slider displays sample time.
Playback may skip display frames to keep pace; all acquired views remain saved.

For example, 1,000 views, a 10 ms readout, 100 Hz firing, 1 s/turn rotation and
20 mm/s volume speed describe a 10 s window with 10 turns. The last view is at
9.99 s, after 9.99 turns and 199.8 mm of volume travel. Use smaller detector
arrays if the GUI's total-sample limit is exceeded.

YAML retains acquisition controls, motion mode and per-view volume offsets. Surface NPZ
archives retain numeric `acquisition_*` fields, `view_times_s`, voxel metadata
and per-view `offOrigin`. Loaded timed archives preserve playback timing.
This simulates rotating hardware with a translating object. The surface
projectors and surface FDK currently require a fixed `offOrigin`; moving-volume
exports require conversion to the object's fixed coordinate frame before
reconstruction, and helical FDK support is not provided by this preview.

Run `Python/demos/d27_DetectorGeometryGUI.py` in `recon` for a guided
three-panel build and editable/exact geometry save/load example. Its MATLAB
counterpart has the same demo number. See the
[demo content comparison](./Demo_language_comparison.md) for matched geometry,
reconstruction and file workflows across the two languages.

The matching reconstruction demos are `d25_CurvedDetector.py` and
`d26_ModularDetector.py`. They plot projections, animate geometry, then
reconstruct the bundled head phantom. `tigre.plotproj` accepts angle arrays
with shape `(views,)`, `(views, 1)` or `(views, 3)`; ZYZ titles display the
first component, alpha. Detector-coordinate slices do not use view angles as
detector indices. The plotter retains its animation during playback.

For the MATLAB port, native array conventions and shared YAML interchange, see
the [MATLAB detector builder guide](./Detector_geometry_MATLAB.md).

The TIGRE CT Simulation Builder is a desktop tool for laying out equal-sized
detector panels, previewing their arrangement, and exporting per-view detector
coordinates. It supports cylindrical panel arcs, arbitrary piecewise-linear
arrangements (including U-shaped layouts), full 3D panel orientation, COR, and
a stationary detector with a linear or circular-arc source trajectory.

The GUI creates a `DetectorSurfaceGeometry`-compatible archive. Its optional
projection-remapping workflow loads measured data and writes a second archive
for approximate `fdk_surface` reconstruction. Volume settings define metadata
and a preview box; the GUI does not allocate voxel data or run reconstruction.
Exported scripts perform simulation and reconstruction separately. For the surface model and
reconstruction APIs, see [Detector-surface geometry](./Detector_surface.md).

## Start the GUI

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

## Basic workflow

1. Choose **Detector motion** and **Layout**.
2. Enter the scan, panel, and pixel parameters described below. Hover over a
   label to display its help in the **Parameter help** box.
3. Select **Confirm**. Invalid entries are reported in a dialog; the previous
   preview is cleared so it cannot be mistaken for the new geometry.
4. Inspect the 3D detector and source path. Use the **View** slider to select a
   view, or **Play**/**Pause** to animate it.
5. Select **Save surface...**. TIGRE checks the geometry before opening the
   save dialog. Review the iterative-projector and `fdk_surface` messages, then
   confirm to choose the `.npz` destination.

If FDK reports that the detector geometry is unsupported, use **Remap data for
FDK...** to resample measured projections onto a compatible detector. This is
an optional data-conversion step and does not alter the preview or original
data.

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

## Detector motion and scan controls

Detector motion offers **Moving with Source** and **Fixed Position**. Fixed
Position follows the Source tab's Cylindrical arc or Linear trajectory;
Moving with Source selects a cylindrical orbit if Linear was active.

After Confirm, the scrollable **Confirmation
checks** panel shows PASS/FAILED, checked views and crossing samples, CGLS
geometry support, and FDK geometry results. FDK scan coverage is separate;
for moving volumes the object-frame FDK check is deferred to the exported
script. Failures remain visible with their reason and preserve the previous
confirmed scene. No success popup interrupts editing.

Confirm and editable YAML save use the actual volume box at every view,
including per-view offOrigin. The source and
valid detector pixels must be outside the volume. Each view needs at least
one sampled source-to-pixel segment crossing it. A detector panel wholly on
the source side and measuring no crossing rays is rejected. Far-side overscan
panels may legitimately miss the object, and valid off-center corner rays
are accepted. A long/full source orbit with a fixed detector will generally
fail these checks; reduce the arc, reposition the detector/volume, or select
Moving with Source. Errors identify the offending view/panel and retain the
previous confirmed scene. Exact archives remain inspectable for diagnosis.

TIGRE's arbitrary-surface Siddon/CGLS APIs can trace these rays, but that alone
does not certify a physical acquisition. The guard checks transmission, not
complete sampling, mechanical clearance, panel-front sensitivity, or FDK
requirements. Exported scripts repeat the physical check before reconstruction.

### Moving source and detector

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

### Stationary detector and moving source

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

### Stationary detector and arc source

Select **Fixed Position** in Detector motion and **Cylindrical** in Source
trajectory for an arc source with stationary panels. Arc mode uses **Source radius**,
**Source height**, **Orbit start**, **Scan span**, and **Views**. The detector
faces the midpoint of the acquisition arc and stays fixed as the source moves.
Source start/end fields are ignored. Prefer iterative reconstruction.

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

## Detector layout controls

All distances and pixel pitches are in millimetres. The source and detector
coordinates use TIGRE's world frame.

### Shared panel and detector controls

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

### Cylindrical layout

Choose **Cylindrical** to distribute equal-sized tangent panels over an arc.
Set **Modules (cylindrical)** to the panel count and **Arc span (degrees)** to
the angle between the first and last panel-centre radial directions. For one
panel, the span has no geometric effect.

### Piecewise-linear layout

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

## TIGRE circular CT example

Select **TIGRE: circular cone-beam CT** and **Load example**, then use **Play**
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

Select one of these entries in **Example**, then **Load example** and **Play**:

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

## Visualize an exported surface from a script

Use `tigre.animate_geometry(geo, surface=surface)` after loading the archive
as shown below. The renderer supports curved grids, module boundaries, gaps,
and changing per-view masks. It shares the GUI's surface renderer and coordinate-arrow drawing.
See the [surface visualization guide](./Detector_surface.md#visualizing-curved-and-modular-detectors)
for details. New GUI exports include available `module_slices`; pass them
when reconstructing the surface object to preserve adjoining tile boundaries. The animation utility visualizes supplied geometry;
it does not provide GUI editing or infer the original COR value.

## Compatibility check and reconstruction

### End-to-end: GUI design, save, load and reconstruct

Run these commands from the repository's `Python/` directory in the
`recon` environment. Reconstruction requires a CUDA GPU and TIGRE's compiled
surface extensions, including `_AxSurface`.

1. Launch the builder:

   ```powershell
   conda activate recon
   python -m tigre.utilities.detector_geometry_gui
   ```

2. Choose a detector layout and scan trajectory, enter the parameters, and
   select **Confirm**. Inspect the source and detector using **Play** or the
   view slider. Start with a small number of views and detector pixels for a
   quick reconstruction trial.
3. Select **Save surface...**, review the compatibility messages, and save
   `my_surface.npz` in `Python/`. This stores the exact source positions,
   detector pixel coordinates, validity mask and available module boundaries.
   To retain editable builder controls too, use **Save YAML...** and
   save `my_geometry.yaml`.
4. Load the exported surface, simulate projections and reconstruct:

   ```powershell
   python demos/d_geometry_yaml_reconstruction.py --geometry my_surface.npz --output results/my_surface --iterations 100
   ```

   The same demo accepts editable YAML recipes or exact YAML surface references:

   ```powershell
   python demos/d_geometry_yaml_reconstruction.py --geometry my_geometry.yaml --output results/my_geometry --iterations 100
   ```

   Keep an exact surface YAML file together with its referenced NPZ companion.
5. Open `results/my_surface/comparison.png` to compare the phantom,
   reconstruction and error slices. `results.npz` contains the phantom,
   reconstruction, simulated and predicted projections, residual history and
   geometry arrays. `metrics.json` records errors, runtime and run settings.

The demo uses `load_geometry_file(...)` to obtain the saved surface, simulates
head-phantom measurements with `tigre.Ax_surface(...)`, then reconstructs with
`cgls_surface(...)`. Use `--phantom ellipsoids` for a synthetic alternative or
`--voxels 16 64 64` to change the volume sampling in z/y/x order.

This is a simulated-data workflow: the demo does not load measured projections.
It preserves the loaded detector surface but defines its own reconstruction
volume, with size `[32, 96, 96]` mm in z/y/x order, zero origin offset, and
default sampling `[16, 32, 32]`. Saved volume metadata is not used by this demo.
Adjust the volume definition in the script for a different field of view or
object position. Structural compatibility does not guarantee adequate scan
coverage or reconstruction quality. For measured data, use the saved surface
with matching projections as described below.

For a runnable YAML-to-projection-to-reconstruction example, see
[`d_geometry_yaml_reconstruction.py`](../Python/demos/d_geometry_yaml_reconstruction.py)
and the [end-to-end instructions](./Detector_surface.md#examples-and-tests).
It uses the shipped three-panel geometry unchanged and saves phantom,
reconstruction, projections, quantitative errors, and comparison slices.

The compatibility check validates finite coordinates, array shapes, and a
non-empty mask, then tests the stricter geometry conditions used by the
validated `fdk_surface` path. A successful iterative status means the geometry
meets the surface API's structural requirements; it does **not** test the
installed CUDA build, GPU memory, projection data, or reconstruction quality.
If the validated FDK path is unsupported, `fdk_surface(...,
allow_approximate=True)` can try an experimental per-view reconstruction on
curved/modular detector pixels and changing source/detector poses.

### What restricts `fdk_surface`?

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

### Remap measured projections

Select **Remap data for FDK...** after previewing the geometry. Supported input
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

After saving, load the arrays and construct the geometry object:

```python
import numpy as np
from tigre.utilities.detector_geometry_gui import load_geometry_archive

volume_metadata, surface = load_geometry_archive("detector_surface.npz")
```

Your projection data must match `(views, rows, columns)` from the saved
geometry. For example:

```python
import tigre
from tigre.algorithms import fast_os_sart_surface

geo = tigre.geometry()
reconstruction = fast_os_sart_surface(
    projections, geo, surface, niter=10, blocksize=20)
```

Masked gap values are ignored by the surface algorithms. The GUI archive stores
surface arrays, module boundaries, available voxel metadata and acquisition
timing. It contains no projections or reconstruction results. Older archives
may lack voxel counts; supply them before reconstructing. Moving-volume rays
require the fixed-object conversion used by exported scripts.

## Input limits and troubleshooting

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
