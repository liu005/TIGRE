# Validation of the detector and demo workflows

Python requires **3.8 or newer**. Choose compatible dependency versions and
native extensions for the selected interpreter. The latest audit fixes and
validation results are recorded in the [9 November follow-up](Audit_fixes_2026-11-09.md).
The [clean-install integration report](Clean_install_integration_2026-11-09.md)
records the user's `recon-new` core results and a separate agent-run GUI
save/export followed by installed GPU reconstruction with GUI imports blocked.

From the repository root, set `PYTHONPATH` to the checkout's `Python` folder.
An older installed TIGRE copy may not contain the new modules. The commands
below use the local `recon` Conda environment; other installations should use
their own environment and matching native extensions.

## Python regression checks

### Automated CPU and headless checks

`.github/workflows/cpu-headless.yml` runs on pushes, pull requests and manual
dispatch, with Ubuntu and Python 3.10/3.12. It installs CPU test dependencies
and runs a curated suite covering build configuration, phantom loading, the
public design API, GUI-free file round trips, detector geometry/remapping,
MATLAB file conversion and dense solver references. JUnit reports are uploaded
for each Python version.

Reproduce locally (the runner requires Python 3.10 or newer):

```powershell
conda run -n recon-new python Python/tests/run_cpu_checks.py --junitxml=cpu-results.xml
```

The runner stages Python source without native binaries in a temporary folder.
Test-only import guards allow the current eager CUDA imports to resolve but
raise on any native call. GUI imports are also forbidden, including in child
processes. A preflight verifies the staged package path and both guards. These
checks do not establish CUDA build/install correctness or GPU coverage; run
the full suite below in a native environment as well. Production imports and
package build behavior are unchanged.

Local validation on 9 November 2026 in `recon-new`: **111 passed**. This is a
Windows/Python 3.12 local run; the hosted Ubuntu matrix has not yet run.

### Full and native checks

Installation regression tests exercise installed phantom lookup, CUDA version
parsing, mixed-device target filtering, headless builds, Python 3.8 syntax and
minimum-version metadata without invoking a CUDA compiler:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'Python'
conda run -n recon python -m pytest Python/tests/test_build_configuration.py Python/tests/test_sample_loader.py -q
```

Syntax checks and annotation checks do not replace a native build and full
runtime validation on Python 3.8.

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
to the separate tigre-geometry-design-tool checkout. Follow its README and validation guide to
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
[MATLAB surface](../../tigre-geometry-design-tool/Frontispiece/TIGRE_Geometry_Design_Tool_MATLAB.md) manuals for those limits.
