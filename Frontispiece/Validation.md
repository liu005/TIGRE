# Validation of the detector and demo workflows

Use the project's `recon-new` Conda environment for current Python validation;
older validation commands below retain the original `recon` environment. From the
repository root, set `PYTHONPATH` to the checkout's `Python` folder. An older
installed TIGRE copy may not contain the new modules. Scanner demo 21 and new
demos 23/25/26/27 also select the checkout when launched directly from an IDE.

## Python regression checks

The [final 3 November project audit](Project_audit_2026-11-03.md) records
451 passing Python tests, 25 passing MATLAB suites, dependency checks,
documentation checks, fixed issues and remaining external-data limitations.

For the Python piecewise linear source feature, use `recon-new` and run from
the repository root with pytest available:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'Python'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
conda run -n recon-new python -m pytest `
  Python/tests/test_piecewise_source_gui.py Python/tests/test_gui_parameter_help.py `
  Python/tests/test_geometry_yaml.py Python/tests/test_detector_geometry_examples.py `
  Python/tests/test_geometry_recon_export.py Python/tests/test_acquisition_timing.py -q -rs
```

This set covers 83 checks, including actual Tk workflows and CUDA surface
projection/reconstruction of a bent source path. The native test also compares
the combined projection stack with two separate linear-source scans sharing
the same fixed detector. Segment junctions are acquired once. Recipes, exact
archives and exported reconstruction geometry preserve the complete path.
MATLAB's builder does not yet implement this editable mode.

The final expanded check on 3 November 2026 passed 122 tests with no skips,
including MATLAB bridge round trips, object rotation, endpoint guards, voxel
synchronization, cargo DR and CT transmission checks. Empty optional waypoint
fields are normalized from MATLAB arrays back to absent settings so existing
non-piecewise recipes remain compatible. The run log is
`.codex-results/piecewise_source_commit_final.log`.

The local `recon-new` environment has no pytest installation; the recorded run
uses a workspace-local copy of pytest and its dependencies, with a custom
workspace `tmp_path` fixture because sandbox temporary-directory permissions
prevent normal pytest fixtures. Numerical libraries and CUDA extensions are
loaded from `recon-new`. The previous demo runs in `recon` and `recon-new` are
recorded in [the 3 November report](Demo_validation_2026-11-03.md).

For the CT Simulation Builder's current CT/DR workflows, run from `Python/`:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
conda run -n recon --no-capture-output python -m pytest `
  tests/test_sea_cargo_preset.py tests/test_acquisition_timing.py `
  tests/test_endpoint_motion_guard.py tests/test_object_rotation.py `
  tests/test_voxel_field_sync.py tests/test_ct_physics.py `
  tests/test_geometry_yaml.py tests/test_geometry_recon_export.py `
  tests/test_geometry_matlab_bridge.py tests/test_detector_geometry_examples.py -q -rs
```

The cargo check verifies the 18+10 source-facing modules, every projected
joint, transverse fan coverage, half-module lower/full-module upper air margins,
1253 views at 5 mm conveyor steps, 100 mm end margins, DR-only export and
timing. At 100 Hz its last sample is 12.52 s and y velocity is 500 mm/s;
source/detector coordinates stay fixed. Arbitrary-axis endpoints, excessive
firing rates and moving-source DR timing are also checked. CUDA forward runs
are a separate smoke check; CPU tests mock the GPU call in exported scripts.

If a batch skips a Tk test because a later interpreter cannot initialize,
rerun that test in a fresh process. A skipped GUI test is not a pass.

```powershell
$env:PYTHONPATH = Join-Path $PWD 'Python'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
conda run -n recon python -m pytest `
  Python/tests/test_varian_loader.py Python/tests/test_dual_energy.py `
  Python/tests/test_demo_language_alignment.py Python/tests/test_demo_geometry_animations.py `
  Python/tests/test_detector_geometry_examples.py Python/tests/test_quality_history_plot.py `
  Python/tests/test_plotproj_angles.py Python/tests/test_geometry_yaml.py `
  Python/tests/test_geometry_matlab_bridge.py Python/tests/test_geometry_inspection.py `
  Python/tests/test_detector_geometry.py Python/tests/test_detector_geometry_remap.py `
  Python/tests/test_surface_validation.py Python/tests/test_animate_surface.py `
  Python/tests/test_animate_surface_mesh.py -q -rs
```

These checks exercise actual GUI save/load transitions, disabled controls,
YAML/NPZ/MAT interchange, all supported projection-angle shapes, animation
callbacks, surface validation, independent quality-history lengths, XIM/XML
loading and dual-energy calculations. Desktop-dependent tests may skip when Tk
is unavailable. The overflow-rejection test intentionally supplies coordinates
outside float32 range and may emit a NumPy cast warning before rejecting them.

Native GPU smoke runs additionally exercise head reconstruction in demos 25/26
and the dual-energy MAT-import/VM/FDK/OS-SART path. Synthetic dual-energy inputs
exercise software plumbing, not scanner calibration accuracy.

## MATLAB checks

CPU builder checks, including DR timing and retained CT behavior:

```matlab
addpath(genpath('MATLAB'));
test_seaCargoPreset;
test_ctSimulationBuilder;
test_objectRotation;
test_ctTransmission;
test_detectorGeometryGUI;
test_detectorGeometryGUI_usability;
```

These require MATLAB desktop/UI support and the Python `recon` bridge for
YAML interchange. They do not require native CUDA reconstruction MEX files.

Final builder validation, 30 October 2026: the targeted Python command above
passed 99 tests without skips, and all six MATLAB builder suites passed on
R2020a. A separate real CUDA run of the exported timed cargo DR script used
1253 views and an 8x12x6 uniform attenuation grid. Projection/transmission
outputs were finite, both specified air margins remained clear, and the saved
time vector ended at 12.52 s. These are workflow/geometry checks, not scanner
calibration or detection-performance certification.

For the local custom toolchain, build/activate the isolated native MEX files as
described in the [build guide](MATLAB_Blackwell_build.md), then:

```matlab
addpath(genpath('MATLAB'));
UseBlackwellBuild;
test_compileConfiguration;
test_blackwellBuild;
test_qualityMeasures;
test_qualityMeasuresGPU;
test_qualityEarlyStop;
test_FISTA_defaults;
test_VarianLoaderValidation;
test_dualEnergy;
test_surfaceProjectors;
test_surfaceFDK;
test_detectorGeometryGUI;
test_detectorGeometryGUI_usability;
test_detectorGeometryGUIDemo;
test_geometryInterchange;
test_demoGeometryAnimations;
test_curvedSurfaceDemo;
test_modularDetectorDemo;
```

`test_animateGeometry` separately checks the legacy animation/GIF API. MATLAB
tests that use YAML invoke Python conversion in `recon`; ensure Conda is on PATH
and the checkout is selected. Tests can hide standard figures with
`set(groot,'defaultFigureVisible','off')`; UI callbacks are tested directly.

## Scope of the local validation

The listed Python regression set passed 187 checks on 27 October 2026. The
listed native MATLAB checks passed with R2020a and the custom CUDA build on the
RTX 5070 Ti Laptop GPU. Geometry comparisons checked actual MATLAB/Python
coordinates for 100 detector-rotation/arbitrary-orbit poses and modular gap
masks. Dual-energy comparisons checked both kVp coefficient sets and VM spline
values; XIM comparisons used MATLAB's native decoder on the same fixtures.

This is not an exhaustive run of every legacy algorithm or every hardware
configuration. No real paired dual-energy or Varian acquisition is available
locally. Python Varian DPS, object-scatter and beam-hardening corrections remain
unported, and MATLAB surface reconstruction currently uses one GPU. See the
[Varian](Varian_loader.md), [dual-energy](Dual_energy.md) and
[MATLAB surface](TIGRE_CT_Simulation_Builder_MATLAB.md) manuals for those limits.
