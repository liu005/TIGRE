# Validation of the detector and demo workflows

Use the project's `recon` Conda environment for Python commands. From the
repository root, set `PYTHONPATH` to the checkout's `Python` folder. An older
installed TIGRE copy may not contain the new modules. Scanner demo 21 and new
demos 23/25/26/27 also select the checkout when launched directly from an IDE.

## Python regression checks

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
