# Validation of the detector and demo workflows

From the repository root, set `PYTHONPATH` to the checkout's `Python` folder.
An older installed TIGRE copy may not contain the new modules. The commands
below use the local `recon` Conda environment; other installations should use
their own environment and matching native extensions.

## Python regression checks

The [final 3 November project audit](Project_audit_2026-11-03.md) records
451 passing Python tests, 25 passing MATLAB suites, dependency checks,
documentation checks, fixed issues and remaining external-data limitations.
The [6 November integration audit](Integration_audit_2026-11-06.md) records
the current split and operator checks.

Run the core suite from the repository root:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'Python'
conda run -n recon python -m pytest Python/tests -q -rs
```

To focus on the operator migration and the headless repository boundary:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'Python'
conda run -n recon python -m pytest `
  Python/tests/test_projection_operator.py Python/tests/test_all_operator_algorithms.py `
  Python/tests/test_operator_audit_contract.py Python/tests/test_operator_dense_oracles.py `
  Python/tests/test_operator_pocs.py Python/tests/test_operator_pcsd.py `
  Python/tests/test_headless_repository_boundary.py Python/tests/test_user_manual_examples.py -q -rs
```

These tests cover operator contracts, dense numerical references and geometry
round trips with GUI and Tk imports forbidden. Native tests require compatible
CUDA extensions and hardware. Results and limitations for the operator fixture
are recorded in the [operator guide](Projection_operator.md).

Desktop builder tests, exported-script tests and their dependencies now belong
to the separate TIGRE-GUI checkout. Follow its README and validation guide to
run them with a matching TIGRE checkout. Historical 3 November runs, before
the extraction, are recorded in [the project audit](Project_audit_2026-11-03.md)
and [demo report](Demo_validation_2026-11-03.md); their test names and counts
describe that earlier tree.

Native GPU smoke runs additionally exercise head reconstruction in demos 25/26
and the dual-energy MAT-import/VM/FDK/OS-SART path. Synthetic dual-energy inputs
exercise software plumbing, not scanner calibration accuracy.

## MATLAB checks

Core geometry checks:

```matlab
addpath(genpath('MATLAB'));
test_objectRotation;
test_ctTransmission;
```

GUI tests require the separate repository on the MATLAB path; see its
validation guide. MATLAB YAML interchange also uses the Python `recon` bridge.

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
test_operatorAuditContract;
test_allOperatorAlgorithms;
test_operatorPocs;
test_userManualExamples;
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
[MATLAB surface](../../TIGRE-GUI/Frontispiece/TIGRE_CT_Simulation_Builder_MATLAB.md) manuals for those limits.
