**TIGRE CT Simulation Builder — Python user manual**

**Manual version: 1.7**
**Updated: 3 November 2026**
**Software reference: TIGRE Python package 3.0.0; CT Simulation Builder checkout.**

The manual version identifies this document, not a separate application release.
The PDF edition includes a cover page, contents, and wrapping tables/code.

# Start the GUI

Install TIGRE's Python requirements, including Tkinter and Matplotlib, and
use the project's `recon-new` conda environment (`conda activate recon-new`). From the repository's
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
conda activate recon-new
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

Hover over or focus any of the three checkboxes (Rotate object, acquisition timing, or start/end movement) to show its purpose, enabled/disabled behavior and constraints in Parameter help.

## Applications

Choose geometry from the relative motion of the source, detector and object,
rather than from the application name alone. The following setups are starting
points; detector coverage and reconstruction suitability still need checking.

### Medical CT: rotating source and detector

Conventional rotating-gantry CT keeps the patient approximately stationary while
the source and detector rotate together. Fan-beam scans can use a line detector;
multislice CT uses several detector rows. Flat-panel cone-beam CT uses a
2D detector panel. These are related acquisition geometries, with different
coverage and reconstruction requirements. See the
[DICOM CT overview](https://dicom.nema.org/medical/dicom/current/output/chtml/part17/sect_UUUU.3.2.5.html).

In the builder, leave **Rotate object (fixed source and detector)** off in
Volume. Select a **Cylindrical** source trajectory and a detector that moves
with the source. Use one detector row for a line-detector model, several rows
for a multirow detector, or one piecewise-linear panel for a flat panel.
For a helical/spiral scan, add volume translation in Movement. Confirm checks
ray intersections; it does not establish sufficient sampling or clinical image quality.

### Industrial CT: rotating object

Laboratory and industrial CT commonly keep the source and detector fixed and
rotate the specimen. This is a common arrangement rather than a requirement
for every industrial scanner. See the
[University of Texas CT principles](https://www.ctlab.geo.utexas.edu/about-ct/essentials-of-computed-tomography/).

Select **Rotate object (fixed source and detector)** in Volume. Use start/end
z angles for a conventional turntable, or a **ZYZ angle list** for arbitrary
object poses. The d18 preset illustrates the latter. Endpoint translation in
Movement can be combined with rotation. Exported scripts transform the rays
into the stationary object frame before reconstruction.

### Tomosynthesis: limited-angle depth imaging

Tomosynthesis reconstructs depth sections from projections over a limited
angular range rather than a complete CT orbit. Digital breast tomosynthesis
is one application. A common arrangement moves a source along an arc while
keeping a flat-panel detector fixed; stationary multisource designs instead
activate distributed emitters electronically. References 12 and 13 describe
stationary breast tomosynthesis and its geometric calibration.

In Volume, leave **Rotate object (fixed source and detector)** off. Load
**Tomosynthesis: arc source, fixed panel** for an arc acquisition, or
**Tomosynthesis: linear source, fixed panel** for a linear approximation.
Use a single piecewise-linear panel to represent a flat detector. Set the
source span or endpoints and the number of views to match the intended
acquisition, then Confirm. A changing active source position can also
approximate sequential emitters; arbitrary source locations require the
explicit-surface API rather than the GUI's regularly sampled paths.

Use iterative reconstruction such as CGLS for these examples. Limited angular
coverage produces direction-dependent resolution and out-of-plane artifacts;
passing the physical geometry check does not remove this incompleteness.
A circular-geometry FDK check alone does not certify tomosynthesis coverage.
The presets illustrate geometry, not a calibrated clinical acquisition.

### Cargo and baggage scanning: geometry varies

Security systems may use rotating CT, stationary-source CT, line scanning or
multiple projection views while cargo travels through the scanner. For example,
Smiths describes a [cargo system combining CT and an optional dual-view line scanner](https://www.smiths.com/news-and-insights/news/2025/smiths-detection-sets-new-standard-with-future-proof-hold-baggage-and-air-cargo-screening-solution).
A few projection views alone do not imply complete CT coverage.

Configure the actual source/detector arrangement and use Movement for conveyor
translation along the appropriate world axis. Piecewise-linear or cylindrical
modules can approximate detector layouts. The GCT preset is another geometry
example, not a validated model of every cargo scanner. Spectral response,
scatter, attenuation calibration and detection performance are outside this
geometry builder's scope.

### Stationary / multisource CT: electronically switched source positions

Stationary CT can use fixed detectors and an array of fixed source spots fired
in sequence. No tube needs to move mechanically: the *active source position*
changes between views. See the [Manchester/Rapiscan stationary-gantry description](https://www.manchester.ac.uk/collaborate/business-engagement-knowledge-exchange/case-studies/rapiscan-and-the-university-of-manchester/).

The terms *stationary CT*, *static CT*, *stationary multisource CT* and
*multi-source stationary CT* are used for related architectures in the
literature. “Multisource CT” alone is broader: multiple sources can also be
mounted on rotating or semi-stationary hardware. Specify what remains fixed
and how emitters are activated rather than treating these terms as exact
synonyms. The source-array studies and medical static-CT system listed in
references 9-11 provide further examples.
Leave object rotation off. Select a fixed detector with a linear or arc source
path to approximate sequential firing along that path. Here “moving source”
means changing source coordinates in the geometry, and can represent successive
physical emitters. The GUI samples these paths regularly; arbitrary discrete
source locations require an explicit detector-surface geometry loaded for
inspection or supplied through the API. One active source is represented per
view; simultaneous-source superposition, independent emitter schedules and
scanner electronics are not simulated. These layouts generally need iterative
reconstruction; check the exported script's FDK assessment separately.

The numbered [References and further reading](#references-and-further-reading) are listed at the end of this manual.

# Use the Source tab

Choose **Source trajectory: Cylindrical** for a circular orbit about +z,
**Linear** for world x/y/z endpoints, or **Piecewise linear** for joined straight
segments through world-coordinate waypoints. Cylindrical source motion is
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

## Piecewise linear source movement

Choose **Source trajectory: Piecewise linear** to join two or more straight
source movements into one acquisition with a fixed detector. The
**Tomosynthesis: piecewise linear source** example provides a starting point.

1. Enter **Source waypoints (x, y, z mm)** as an ordered list, for example
   `[[1000, -300, 0], [1000, 0, 0], [800, 300, 100]]`.
2. Enter **Views per source segment**, for example `21, 21`. Each count includes
   its segment's two endpoints. A shared junction is acquired once, so this
   example has `21 + 21 - 1 = 41` total views.
3. Configure the fixed detector panels and volume, then press **Confirm**.
   **Views** is derived from the segment counts. Use Play or the slider to inspect
   the source moving through the junction.

The detector pose is established from the first waypoint and held fixed for
every segment, including its tilts, gaps and mask. This combines source paths
on the same detector; it does not merge two independently positioned detectors.
Samples are uniformly spaced within each segment. Segment lengths and view
counts may differ; these counts do not specify motion speed or exposure timing.
Disable orbit timing and use COR = 0. Source start/end and circular-orbit fields
are ignored while this mode is selected.

Save geometry YAML preserves the editable waypoints and counts. Exact YAML/NPZ
exports preserve every sampled source position and detector pixel. The exported
reconstruction script uses this combined acquisition; use surface iterative
methods such as CGLS. A bent source path does not supply a circular FDK orbit
or guarantee complete angular coverage.

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
| Top arm rows | Upper rows forming a 90-degree arm; 0 means straight; not used for the modular cargo preset. |
| Arm clearance (mm) | Raise the arm and calculate its backward offset to align projected joint edges; used for continuous folded panels. |

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

## Fixed hardware and rotating object

In Volume, select **Rotate object (fixed source and detector)**.
The source and detector remain at the initial pose configured in Source and
Detectors. The volume rotates about its own center; endpoint translation can
also be enabled. Orbit timing is disabled in this mode.

Choose **Start/end z angles** to interpolate from the start to the end angle,
including both endpoints. Angles are in degrees; decreasing values rotate in
reverse. Different endpoints require at least two views. Positive z rotation
is counterclockwise when viewed from +z.

Choose **ZYZ angle list** for arbitrary orientations using demo d18's Euler
convention. Paste a numeric list such as `[[0, 0, 0], [90, 45, 0], [180, 90, 30]]`
and set Views to 3. Each row is an absolute object pose `[z1, y, z2]` in degrees,
with matrix `Rz(z1) @ Ry(y) @ Rz(z2)`. Supply one row per acquired view, or a
single row to share one orientation across all views. Expressions are not
evaluated, and list poses are not interpolated.

Press **Confirm**, then **Play**. The teal **Object +x** line makes orientation
visible even for a cubic volume. Physical checks use the oriented volume,
and YAML and exact NPZ saves retain the poses. Reconstruction export transforms
both source and detector into the fixed object frame before running algorithms.
Arbitrary poses generally require CGLS; FDK compatibility is checked after
conversion. The MATLAB builder supports the same poses and d18 preset; shared
YAML files preserve ZYZ degree triples without reversing their order.

For a complete preset, choose **TIGRE d18: arbitrary object rotation (ZYZ)**
in the bottom Example menu and select **Load example**, then **Confirm** and
**Play**. It uses 100 inverse object poses to reproduce d18's relative gantry
trajectory. The detector is downsampled to 128 by 128 while preserving the
409.6 mm physical size; the volume is 256/1.5 mm per axis. Use CGLS for this
example; its trajectory is outside FDK's supported geometry.

For the d18 preset, unchecking **Rotate object (fixed source and detector)**
keeps the object stationary and applies the inverse orientations to the source
and detector together. Check it again to return to rotating-object mode. For
ZYZ angles, the inverse of `[z1, y, z2]` is `[-z2, -y, -z1]`; reversing the
order is essential. Press Confirm after switching. The equivalent hardware
trajectory is retained by geometry saves and reconstruction export. This
preset-specific switch preserves d18's relative geometry with its centered,
stationary object; it does not add acquisition timing to arbitrary ZYZ paths.
# Use the Movement tab

Use **Movement** for object translation and acquisition timing. Object rotation
is configured in **Volume**. Geometry
stays in the structural world frame: the source follows its configured path
while the volume box moves. Enable **Move between start and end positions**
for endpoint translation; entering two values alone does not enable motion.

| Control | Purpose |
|---|---|
| Start position (z, y, x mm) | Initial volume center, shared with Volume Center offset. |
| End position (z, y, x mm) | Final center in endpoint mode; calculated/read-only in timed speed mode. |
| Move between start and end positions | Interpolate all three coordinates across views; distinct endpoints need at least two views. |
| Use timing to derive orbit and z motion | One frame per pulse; rotating hardware for CT, fixed hardware and endpoints for DR. |
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
manual sampling when the source moves. Fixed-hardware DR supports endpoint
acquisition timing as described below.

For a simple endpoint spiral, use a circular source, 600 views, Start
`-200, 0, 0`, End `200, 0, 0`, and **enable endpoint movement**, then Confirm.
Timing is optional for that example. Empty entry/exit views are allowed as
long as at least one measured ray crosses the volume somewhere in the scan.

## Fixed-hardware DR acquisition timing

Keep conveyor movement on its physical axis; the cargo preset uses world y.
In Movement select **Use timing for DR conveyor motion**. The GUI enables
endpoint motion automatically and keeps it enabled while DR timing is active.
Set Start and End in z/y/x order for Python or x/y/z for MATLAB. Motion may
follow any straight 3D endpoint displacement; it is not restricted to z.

Readout period limits the source firing rate: `rate * readout_ms <= 1000`.
One view is acquired per pulse at `t_i=i/rate`. For N views the last view is
at `(N-1)/rate`, and the acquisition window is `N/rate`. The endpoint velocity
vector is `(end-start)/last_view_time`. The GUI reports its components, total
speed and per-view displacement. Source and detector remain stationary.

For the cargo defaults, 1253 views from y=-3130 to +3130 mm at 100 Hz and a
10 ms readout period give **5 mm per view**, **500 mm/s along y**, **12.52 s**
between first and last views and a **12.53 s** acquisition window. A slower
firing rate changes the physical conveyor speed for the same views/endpoints;
it does not change the 5 mm sampling step. Readout must accommodate the rate.

Rotation period and Volume z speed are inactive in DR timing: no orbit is
created and conveyor speed is derived from the complete endpoint vector.
Source start/end must be identical and the detector fixed; moving-source DR
timing is rejected. Arbitrary object/rig rotation timing remains unsupported.
Press Confirm after edits, then Play; playback follows the saved frame times.
Save YAML again before exporting DR simulation. YAML/MAT/NPZ retain timing;
DR simulation outputs include `view_times_s` when timing is enabled.
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

# Export CT reconstruction or DR simulation

The export type follows the **last successfully saved YAML**. CT scans use
**Export recon script...**. The sea-cargo DR preset uses **Export DR simulation
script...**, with forward projection only and no 3D reconstruction. Save again
after edits before exporting. Export writes a script; it does not run CUDA.
The saved `volume.simulation_mode` is `dr` for the cargo preset and `ct` for
ordinary CT setups, and is retained by YAML/MAT/NPZ interchange.

## CT reconstruction export

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

## DR simulation export

1. Load **Sea cargo: 20-foot container (2D Digital Radiography)**.
2. Adjust the detector arms in Detectors and the conveyor endpoints in Movement.
3. Press **Confirm**, inspect the DR-only check message, then **Save YAML...**.
4. Select **Export DR simulation script...** and save, for example, `cargo_dr.py`.
5. Run the script in the TIGRE CUDA environment:

```powershell
conda run -n recon python cargo_dr.py --uniform-mu 0.001 --voxels 64 128 64
conda run -n recon python cargo_dr.py --volume attenuation.npy --output cargo_dr.npz
```

`attenuation.npy` must be a finite, nonnegative z/y/x linear-attenuation array
in 1/mm. Its shape determines voxel counts; physical size comes from the YAML.
Without an input volume, the script creates a uniform object. The default
64x128x64 grid avoids allocating the full 306-million-voxel cargo volume.
`--geometry` overrides the YAML path and `--output` selects the output NPZ.

Output `projections` has shape views/rows/columns and contains attenuation
line integrals; `transmission=exp(-projections)` is normalized primary-beam
intensity. `dr_image` samples a central column from each module and concatenates
the row strips in module order; `module_slices` describes that order, and
`valid_mask` identifies measured samples. This is a sampled 2D
DR view, not a uniform-pixel conveyor mosaic or a reconstructed 3D volume.
The L-shaped arms have different ray directions and magnification, and 100
views over 6.2 m do not by themselves guarantee dense conveyor sampling.
Geometry-dependent resampling, calibration, scatter, spectra and noise need
separate treatment. The script contains no FDK or CGLS calls.

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

## Catalogue of GUI presets

The Python Example menu contains the following 14 presets, in this order.
MATLAB has the original 13; the piecewise linear source preset is currently
Python-only. Choose a name and select **Load example** to replace the
current controls and build its preview. Press **Confirm** before playback.
Presets are editable starting points, not calibrated or validated commercial
scanner models. Export uses the last saved YAML, so save after making changes.

### Circular CT: curved modular detector

**What it is:** A full circular source-detector orbit with eight flat modules
arranged along a cylindrical arc. Physical separation and masked array gaps
illustrate a modular curved detector rather than one continuous flat panel.

**Used for:** Learning circular CT geometry, inspecting module boundaries and
ray coverage, and exploring how curved detector layouts differ from flat ones.

**Reconstruction:** CT export supports surface iterative reconstruction. FDK
requires the separate geometry and scan-coverage checks; gaps do not become
measured data merely because the preview looks continuous.

### Short scan: curved detector

**What it is:** A 240-degree orbit with a narrower 40-degree detector arc.
The source and curved modular detector rotate together.

**Used for:** Exploring short-scan geometry and comparing reduced angular
coverage with the full circular preset. The required scan range depends on
fan angle; a partial orbit is not automatically a sufficient short scan.

**Reconstruction:** Use CT export and review the FDK compatibility/coverage
assessment. Iterative methods can use the original surface, but cannot create
information absent from incomplete sampling.

### TIGRE: circular cone-beam CT

**What it is:** The standard low-resolution TIGRE cone-beam arrangement: one
128x128 flat panel, 3.2 mm detector pitch, DSO 1000 mm, DSD 1536 mm, and 50
views around a full circle.

**Used for:** A simple baseline for learning source/detector/volume settings,
comparing standard and explicit-surface geometry, and testing save/export.

**Reconstruction:** The natural starting point for FDK and CGLS examples.
The small number of default views is for demonstration; it does not establish
adequate sampling or application-quality reconstructed images.

### Three tilted panels

**What it is:** Three 256x256-pixel panels with transverse offsets, independent
panel angles and a small normal displacement. The rig follows a circular scan.

**Used for:** Inspecting faceted detector assembly, physical gaps, panel joins
and the effects of module position/orientation on measured rays.

**Reconstruction:** Surface iterative reconstruction is the general option.
Read the FDK compatibility report before applying a circular FDK method or
attempting supported data remapping; panel tilt is already in the coordinates.

### Staggered panels

**What it is:** Three panels whose centres are offset in z as well as across
the detector, producing a staggered, non-extruded detector surface.

**Used for:** Studying detector assemblies with modules at different heights
and seeing why a single regular detector grid may not describe their geometry.

**Reconstruction:** Prefer surface iterative reconstruction. Vertical staggering
is not automatically compatible with the circular surface FDK assumptions;
check compatibility and do not infer complete coverage from the panel count.

### Tomosynthesis: arc source, fixed panel

**What it is:** A source tracing a 60-degree circular arc from -30 to +30
degrees across 61 views, while one flat panel remains stationary.

**Used for:** Limited-angle depth imaging and tomosynthesis geometry studies.
It also approximates sequential source positions on an arc with fixed detectors.

**Reconstruction:** Use iterative methods such as CGLS for depth-section studies.
Limited-angle sampling causes direction-dependent resolution and out-of-plane
artifacts; this is not the rotating source-detector geometry required by FDK.

### Tomosynthesis: linear source, fixed panel

**What it is:** A source moving from world y=-500 to +500 mm in 41 views at
fixed x and z, in front of a stationary 512-column flat panel.

**Used for:** Linear tomosynthesis and experiments with linearly distributed
source spots. Source-coordinate changes may represent electronic source
switching rather than literal mechanical travel.

**Reconstruction:** Use surface iterative reconstruction. The linear trajectory
and limited angular range do not form a conventional circular FDK scan.

### Tomosynthesis: piecewise linear source

**What it is:** Two straight source segments through `(1000, -300, 0)`,
`(1000, 0, 0)` and `(800, 300, 100)` mm with one fixed panel. Each segment
has 21 views; the shared junction is counted once, giving 41 total views.

**Used for:** Exploring bent source travel or joined linear source arrays.
Edit the waypoints and segment counts to define additional straight segments.

**Reconstruction:** Use surface iterative reconstruction. The detector pose
stays fixed throughout; limited angular coverage can leave reconstruction
artifacts. See [piecewise linear movement](#piecewise-linear-source-movement).

### Tomosynthesis: three fixed panels, linear source

**What it is:** Three fixed, tilted panels with different transverse/normal
positions. The source travels on a straight 3D line, including a height change.

**Used for:** Exploring multi-panel limited-angle systems, noncoplanar source
motion, and the combined effects of detector joins and source position.

**Reconstruction:** Use surface iterative reconstruction and assess limited-angle
artifacts. This is a geometry experiment, not a calibrated clinical protocol
or a complete circular CT acquisition.

### 3D detector tilt: x, y, z

**What it is:** A circular scan with one panel tilted about all three local
axes: x=20, y=15 and z=10 degrees. The tilts remain constant during the scan.

**Used for:** Understanding roll/pitch/yaw, checking local versus world frames,
and studying geometric misalignment or deliberately tilted detector designs.

**Reconstruction:** Use surface iterative reconstruction. Do not apply the tilt
a second time to exported coordinates; standard FDK assumptions may not hold.

### TIGRE: global COR offset

**What it is:** The circular cone-beam setup with an exaggerated +150 mm
global centre-of-rotation correction, shifting source and detector together.
The orange COR marker shows the actual per-view displacement.

**Used for:** Understanding TIGRE's COR convention, its sign and units, and
how a global rig correction differs from individual panel misalignment.

**Reconstruction:** CT export retains the corrected coordinates. Review the
compatibility and coverage results; the large offset is chosen for visibility,
not as a recommended calibration value or a clinical scan configuration.

### TIGRE d18: arbitrary object rotation (ZYZ)

**What it is:** One hundred ZYZ object poses adapted from demo d18. By default
the object rotates with fixed hardware. Turning object rotation off applies
the inverse orientations to the source-detector rig instead. The GUI uses a
128x128 detector while preserving the demo's physical panel size.

**Used for:** Arbitrary-axis industrial CT studies, understanding object-frame
versus rig-frame descriptions, and checking Euler-angle order and inversion.

**Reconstruction:** Use CGLS through fixed-object frame conversion. Its general
3D trajectory is outside FDK's supported circular geometry. The checkbox switch
preserves the centred preset's relative geometry; save after switching modes.

### Sea cargo: 20-foot container (2D Digital Radiography)

**What it is:** A fixed source at DSO 4230 mm, a translating 20-foot container,
and a source-facing L-shaped detector: 18 vertical-arm and 10 horizontal-arm
modules, each 36x4 pixels at 5x5 mm pitch. The horizontal support has a 300 mm
lift relative to its unlifted layout. Projected seams are aligned, with a
half-module lower air margin and one full upper air module.

**Used for:** Simulating conveyor cargo DR, testing piecewise-linear detector
modules, projected coverage and air margins, and generating attenuation and
normalized transmission images from a known 3D attenuation volume.

**Export:** DR forward simulation only, with no FDK/CGLS or 3D reconstruction.
The 1253-view conveyor movement supplies the length scan at 5 mm steps. The sampled DR image
contains all modules, but is not a calibrated uniformly resampled mosaic.
Detailed geometry, data dimensions and export instructions follow below.

### GCT: 53-panel circular scan

**What it is:** An idealized large circular acquisition with 53 one-row,
64-column panels and 5701 views over 228 degrees. It uses a long-distance,
large-radius scanner geometry and remains the final menu entry.

**Used for:** Inspecting a many-module fan detector and reproducing the nominal
GCT acquisition layout before applying measured-data-specific corrections.
Reduce Views for initial interactive inspection because the dense geometry
alone occupies about 240 MiB.

**Reconstruction:** Use appropriate slice/fan-beam or surface methods after
checking geometry and data. A one-row detector does not provide a complete
3D cone-beam volume acquisition. This preset is idealized: detector corrections,
measurement calibration and sufficient sampling must be assessed separately.

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

## Sea-cargo digital radiography example

Choose **Sea cargo: 20-foot container (2D Digital Radiography)**, select
**Load example**, **Confirm** and **Play**. This refines plot 5 of the legacy
TIGRE_MOD `Python/tests/test_animate_geometry.py` into a source-facing modular
L-shaped detector. The external checkout is not needed to run the preset.

| Setting | Value |
|---|---|
| Vertical arm | 18 modules, with centres on a vertical support line. |
| Horizontal arm | 10 modules; centre support raised 300 mm, now at world z about 2169.48 mm. |
| Each module | 36 rows x 4 columns; 5 x 5 mm pitch; 180 x 20 mm physical size. |
| Module orientation | Its centre normal points directly toward the fixed source. |
| Arm joint | About 319.95 mm vertical edge separation after the support lift; source-ray alignment retained. |
| Source distance | DSO 4230 mm, reduced 10% from 4700 mm; source fixed while the detector arms are fitted for coverage. |
| Container motion | 1253 views; world y from -3130 to +3130 mm; x=0, z=100 mm. |
| Conveyor sampling | 6260 mm travel / 1252 intervals = 5 mm; 100 mm air at each end of the 6060 mm cargo. |
| Physical container | 2590 mm high, 6060 mm long, 2440 mm wide. |
| Purpose | 2D DR forward simulation only; no 3D reconstruction coverage. |

Tilting modules to face the source changes their projected coverage. The preset
therefore uses 18+10 complete modules rather than dividing the previous arm
lengths by 180 mm. Module positions are solved so each upper edge and the next
lower edge lie on the same ray plane through the source. Full modules slightly
extend the former arm envelope. At the arm joint, the horizontal module's near
edge is now about 319.95 mm above and farther back than the vertical arm's
upper edge. The horizontal module-centre support has moved exactly 300 mm up;
its backward position and module tilts are refitted to retain the joint ray.
The added vertical top module is retained; source position, container
trajectory and lower/upper air margins are unchanged.
Neighbouring edges have no projected seam gap or overlap in the scanning
direction over their common transverse coverage. Physical gaps may remain,
and finite outer side edges can have different projected widths.

The detector is fitted to the full container x-z silhouette at DSO 4230 mm.
The vertical arm is lowered so that the centre ray of its lowest module grazes
the limiting lower corner: its first 18 pixel rows (half a module, 90 mm of
detector length) are air-only. The vertical arm now includes one additional module at its top, giving 18
modules while leaving the lowest module and its half-module air margin fixed.
The horizontal arm is refitted to the new upper joint, retaining its 300 mm
lift relative to the unlifted layout.
The horizontal arm extends to cover the upper corner, followed by one complete
air-only module, giving 10 modules on that arm. Tests cover the full fan-angle
interval and all seams; CUDA forward simulation confirms both air margins over
all 1253 views. Conveyor movement supplies the length scan. This fan-coverage
check is not a claim of complete instantaneous 3D coverage by the narrow strip.
In Detectors, the piecewise-linear lists contain all 28 module centres and
pitch angles. Module rows/columns are **36/4**, pixel pitches are **5/5 mm**,
and **Gap columns** is 0. The saved array packs modules along the column
axis: Python projections are **1253x36x112**, MATLAB projections **36x112x1253**.
The 28 module boundaries are retained in `module_slices`.

**Top arm rows** and **Arm clearance (mm)** are 0 for this modular preset;
those fields fold a single continuous panel. The modular joint clearance is
encoded by the panel centres and orientations instead. The layout generators
`cargo_module_layout()` (Python) and `cargoModuleLayout()` (MATLAB) default to
a 20 mm base joint clearance and a 300 mm horizontal-support lift. The final
physical edge separation is about 319.95 mm because the source-facing tilt is
also refitted. Position and angle lists are calculated together when
the preset loads. Manual list edits do not automatically re-aim modules or
restore projected seam alignment; Confirm's physical checks do not certify it.

The source is at `[4230*cos(13 deg), 0, -4230*sin(13 deg)]` in world xyz.
DSO is reduced from 4700 to 4230 mm (10%). The source stays fixed while
the vertical arm is lowered and the horizontal arm is extended for coverage.
The final module coordinates define their individual source-to-detector
distances; these change when the arm positions are refitted.
Equal source start/end
coordinates keep it stationary. Endpoint container motion is enabled; timing is initially off and can be
enabled with **Use timing for DR conveyor motion**. Object rotation stays off. The saved voxel grid remains 518/1212/488 in
Python z/y/x, or 488/1212/518 in MATLAB xyz, at 5 mm voxel size.

After saving YAML, **Export DR simulation script...** exports forward
projection only. It contains no FDK/CGLS calls. The full cargo grid has about
306 million voxels; the exported uniform-object example defaults to a smaller
64x128x64 grid while preserving physical size. See [DR simulation export](#dr-simulation-export)
for attenuation inputs and commands.

The DR image samples one central column per module and concatenates all 36-row
module strips in detector order: **1008x1253** for this preset. It contains both
arms. `module_slices` is saved with the per-view projections and transmission.
The first 18 output rows are the lower air margin; the last 36 rows are the
upper air module. This is a sampled detector-coordinate DR image, not a uniformly resampled
conveyor mosaic. Magnification, nonuniform angular/pixel spacing, conveyor
sampling, calibration, scatter, spectra and noise need separate treatment.

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

See the [validation guide](./Validation.md) for current regression commands.
`test_sea_cargo_preset.py` covers the modular DR layout, projected seams and air
margins, 5 mm conveyor steps, fixed-hardware timing and DR-only export.
`test_acquisition_timing.py` retains CT timing checks, while
`test_object_rotation.py` covers d18 poses and frame switching.

See the [MATLAB manual](./TIGRE_CT_Simulation_Builder_MATLAB.md),
[handover notes](./CT_Simulation_Builder_Handover.md),
[surface geometry reference](./Detector_surface.md), and
[demo comparison](./Demo_language_comparison.md).

From the repository root, run `./Frontispiece/Export_Manuals.ps1` to regenerate
both PDFs. Pandoc, XeLaTeX and the configured Consolas font are required.
The PDF helper fixes table widths, wraps code, and produces a versioned cover
and separate contents pages. PDFs are generated artifacts ignored by Git.

# References and further reading

References checked on 29 October 2026. The GUI setup suggestions above are
interpretations of these acquisition principles, not vendor-validated scanner
recipes. Manufacturer pages illustrate particular systems rather than all
systems in that application category.

1. **DICOM Standards Committee.** [Computed Tomography, PS3.17, section UUUU.3.2.5](https://dicom.nema.org/medical/dicom/current/output/chtml/part17/sect_UUUU.3.2.5.html).
   Background on medical source/detector rotation, patient motion and acquisition
   modes. This is explanatory material, not a reconstruction algorithm reference.

2. **DICOM Standards Committee.** [Dental CT and Cone-Beam CT, PS3.17, section UUUU.3.2.6.5](https://dicom.nema.org/medical/dicom/current/output/chtml/part17/sect_uuuu.3.2.6.5.html).
   Describes rotating source/detector gantries and the relationship between
   dental CT and wide-cone acquisitions.

3. **U.S. Food and Drug Administration.** [Dental Cone-beam Computed Tomography](https://www.fda.gov/radiation-emitting-products/medical-x-ray-imaging/dental-cone-beam-computed-tomography).
   Accessible description of cone-beam acquisition around a patient and its
   clinical applications. Useful context for the medical cone-beam example.

4. **University of Texas High-Resolution X-ray CT Facility.** [Essentials of Computed Tomography](https://www.ctlab.geo.utexas.edu/about-ct/essentials-of-computed-tomography/).
   Explains scanner generations and relative motion: rotating the specimen is
   common in industrial CT, whereas medical systems commonly rotate the rig.

5. **Ferrucci, M. and Ametova, E. (2021).** [Charting the course towards dimensional measurement traceability by X-ray computed tomography](https://www.nist.gov/publications/charting-course-towards-dimensional-measurement-traceability-x-ray-computed-tomography).
   *Measurement Science and Technology*, 32. [DOI: 10.1088/1361-6501/abf058](https://doi.org/10.1088/1361-6501/abf058).
   Explains industrial CT metrology, calibration, uncertainty and traceability.
   These topics go beyond the builder's geometry-intersection checks.

6. **Smiths Detection.** [SDX 10080 SCT product information](https://www.smithsdetection.com/products/sdx-10080-sct/).
   Manufacturer example combining baggage/air-cargo CT with an optional
   dual-view line scanner. Illustrates why CT and projection-only scanning
   should be distinguished when choosing an acquisition model.

7. **University of Manchester.** [Rapiscan and The University of Manchester](https://www.manchester.ac.uk/collaborate/business-engagement-knowledge-exchange/case-studies/rapiscan-and-the-university-of-manchester/).
   Describes a stationary-gantry security scanner using fixed X-ray sources
   fired sequentially. Supports the distinction between mechanical source
   motion and a changing active source position.

8. **Warnett, J. M. et al. (2016).** [Towards in-process x-ray CT for dimensional metrology](https://wrap.warwick.ac.uk/id/eprint/76972/).
   *Measurement Science and Technology*, 27(3), 035401.
   [DOI: 10.1088/0957-0233/27/3/035401](https://doi.org/10.1088/0957-0233/27/3/035401).
   Peer-reviewed evaluation of an airport-baggage real-time tomography scanner
   for industrial dimensional measurements. The university repository provides
   the published paper, connecting stationary-source security CT with industrial
   inspection and its measurement limitations.

9. **Spronk, D. et al. (2021).** [Evaluation of carbon nanotube x-ray source array for stationary head computed tomography](https://pubmed.ncbi.nlm.nih.gov/33382470/).
   *Medical Physics*, 48(3), 1089-1099.
   [DOI: 10.1002/mp.14696](https://doi.org/10.1002/mp.14696).
   Experimental source-array study supporting stationary head CT; relevant to
   representing successive emitters as different source positions per view.

10. **Simulation on system configuration for stationary head CT using linear carbon nanotube x-ray source arrays (2021).**
    [Open-access paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8526838/).
    Investigates one-, two- and three-array stationary source configurations,
    stationary detectors, angular coverage and iterative reconstruction. Useful
    background for the GUI's linear-source/fixed-detector approximation and
    the need to check coverage independently of physical ray intersections.

11. **Ding, H. et al. (2026).** [Development and clinical application of a high-performance medical static computed tomography system](https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.70456).
    *Journal of Applied Clinical Medical Physics*, 27(1), e70456.
    [DOI: 10.1002/acm2.70456](https://doi.org/10.1002/acm2.70456).
    Describes a circular multisource/detector architecture with sequential
    exposures. Its methods also include small gantry-angle shifts: the term
    “static” in a paper title does not always mean every component stays
    completely motionless. Its specialized reconstruction is not implemented
    by this builder.

12. **Qian, X. et al. (2012).** [High resolution stationary digital breast tomosynthesis using distributed carbon nanotube x-ray source array](https://pmc.ncbi.nlm.nih.gov/articles/PMC3321055/).
    *Medical Physics*, 39(4), 2090-2099.
    [DOI: 10.1118/1.3694667](https://doi.org/10.1118/1.3694667).
    Experimental stationary breast-tomosynthesis system using electronically
    activated source positions instead of mechanical tube motion. Provides
    an application example linking limited-angle imaging and multisource arrays.

13. **Jiang, C., Zhang, N., Gao, J. and Hu, Z. (2017).** [Geometric calibration of a stationary digital breast tomosynthesis system based on distributed carbon nanotube X-ray source arrays](https://doi.org/10.1371/journal.pone.0188367).
    *PLOS ONE*. DOI: 10.1371/journal.pone.0188367.
    Describes geometric calibration for distributed-source tomosynthesis.
    Useful context for why source and detector coordinates require calibration
    beyond entering an idealized GUI layout.
