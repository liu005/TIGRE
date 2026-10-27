# MATLAB detector builder and surface animation

The MATLAB port provides native modular geometry construction, a desktop
builder/inspection GUI, and curved/modular surface animation. Tested with
MATLAB R2020a. These visualization utilities do not invoke CUDA projectors.

From the repository root in MATLAB:

```matlab
addpath(genpath('MATLAB'));
app = detectorGeometryGUI();
```

The window opens at up to 1280 x 720 pixels, centred and reduced to fit the
screen. Controls use compact spacing; the left panel remains scrollable on
smaller displays, and the window can be resized.

The presets match the Python builder, including panel roll/pitch/yaw, the
exaggerated global COR example, linear/arc tomosynthesis, and GCT last.
GCT is an idealized panel arrangement; it excludes the measured-data demo's
pixel corrections. Full GCT coordinate/mask arrays need about 240 MiB.

Choose a layout and motion, edit fields, then **Preview**. **From list...**
opens a per-panel table; **Apply** updates the field and **Preview** rebuilds
the scene. **Play** toggles timer-based playback; the slider selects a view.
Disabled fields are ignored by the builder. Piecewise-linear geometry derives
the panel count from the transverse list, reported below the controls.
Changing mode, loading geometry, or loading another preset closes open panel
list editors. Applying a list does not rebuild the preview. Returning from
inspection preserves edits that had not yet been previewed. Playback stops on
mode changes and Preview; an invalid Preview marks the previous display as
out of date and disables playback until corrected. Closing or deleting the
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

**Save settings...** writes an editable recipe to `.mat` or `.yaml`.
**Save surface...** stores exact arrays: a native `.mat`, or YAML with an NPZ
companion. **Load geometry...** accepts `.mat`, `.yaml`, `.yml`, and `.npz`.
Recipes reopen in builder mode; explicit geometry opens in inspection mode,
with builder controls disabled. **Return to builder** restores the retained
builder settings. Keep YAML and its NPZ companion together when sharing.

MAT files require no Python. A MAT document contains `settings`, or `surface`,
or standard MATLAB `geo` plus `angles`. Optional `geo.sVoxel` and
`geo.offOrigin` draw a fixed reconstruction volume. Loading MAT files does
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

These are visualization geometry arrays, not MATLAB projection-data layout.
The bridge preserves singleton dimensions, module boundaries and volume axes.
The MATLAB port does not add MATLAB surface reconstruction kernels or the
Python GUI's projection-remapping workflow; those remain Python facilities.

## Validation

```matlab
test_detectorGeometryGUI;   % presets, geometry, renderer, MAT, GUI
test_detectorGeometryGUI_usability; % callbacks, field states, lists and timers
test_animateGeometry;       % existing eight animation/GIF tests
test_geometryInterchange;   % six MATLAB/Python layout/motion comparisons
```

The interchange tests require the Python `recon` environment. Python tests
in `test_geometry_matlab_bridge.py` cover MAT/YAML round trips, singleton
dimensions and volume-axis conversions. No CUDA reconstruction is tested.
