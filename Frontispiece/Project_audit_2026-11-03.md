# Project audit: 3 November 2026

The audit covered the CT Simulation Builder, explicit detector geometry,
source/object motion, numerical regression tests, reconstruction exports,
MAT/YAML/NPZ interchange, demo validation and user documentation. Issues found
were fixed and checked. This records software behavior on the tested machine;
it does not establish scanner calibration or acquisition completeness.

## Findings and fixes

| Finding | Fix and evidence |
| --- | --- |
| MATLAB rebuilt Python piecewise source recipes with its straight-line builder, silently changing the acquisition. | `geometryFile` preserves the bridge's exact surface and removes editable settings so the GUI opens inspection mode. Direct native builds and recipes missing an exact surface are rejected. The new MATLAB suite checks MAT import, actual Python/YAML bridge import, coordinate preservation and explicit rejection. |
| Whole-directory pytest collection failed on obsolete phantom imports, Python 2 `StringIO`, and import-time access to generated legacy configurations. | Updated imports and stdout capture, deferred configuration loading until script execution, and renamed test factories so pytest does not mistake their arguments for fixtures. Legacy configuration files are still required when running that separate generated algorithm matrix. |
| Some GUI regression fixtures predated new controls and help entries. | Updated their expected fields and mock state. These now execute as part of the full suite rather than being absent from the targeted feature checks. |
| MATLAB uses floating-point arrays for integer-valued settings. | The Python bridge validates and normalizes segment view counts before rebuilding a piecewise recipe. Round trips test both SciPy integer arrays and MATLAB-style double arrays. |
| A missing bundled head dataset produced an uninitialized-variable error, and resampling used a deprecated SciPy namespace. | Missing data now raises an actionable `FileNotFoundError`; resampling uses `scipy.ndimage.zoom` with unchanged numerical options. Added missing-data and anisotropic-shape tests. |
| Development requirements omitted pytest and GPU markers were unregistered. | Added pytest to development requirements and registered the GPU marker. No environment packages were installed or upgraded during the audit. |
| FAQ links pointed at nonexistent installation paths; pCT documentation embedded absent historical PDFs. | Corrected installation links and identified the pCT supplements as unbundled instead of presenting broken image links. All checked local file links now resolve. |

## Verification

| Check | Result |
| --- | --- |
| Complete Python pytest collection and execution | **451 passed, no skips**, in `recon-new`; about 104 seconds. Includes CUDA failure propagation/memory checks, missed-ray multiprocessing checks, CPU/GPU geometry and quality checks, iterative algorithms, GUI state, exports, scanner fixtures and interchange. |
| MATLAB suites | **25 passed** on R2020a with the isolated native CUDA build, including projector/FDK, algorithms, quality, scanner fixtures, demo 25/26, animations, exports, all builder suites and the new piecewise import checks. |
| Python installed dependency consistency | `pip check`: no broken requirements. |
| Local documentation file links | **98 checked; no missing targets** across Frontispiece Markdown and the Python/MATLAB demo indexes. External URLs and anchor targets were not validated. |
| Manual sources and PDFs | Updated interchange behavior in both user manuals; regenerated the version 1.7 PDF editions. PDFs remain local under the existing Git ignore rule. |

Python verification used the checkout's package and `recon-new` numerical/CUDA
libraries. Because pytest is absent from that environment, the audit supplied
a workspace-local copy of pytest and its pure-Python dependencies and used
workspace temporary fixtures. The audit runner was guarded for Windows
multiprocessing so child imports did not restart the suite.

MATLAB suites were isolated in separate processes to avoid one UI issue
preventing later suites from executing. `test_ctSimulationBuilder` initially
timed out and then reported a MATLAB web-window file-write failure under the
sandbox. It passed in a fresh process outside the sandbox. The combined
interchange/object-rotation batch stalled; its individual suites passed.
These initial attempts are not counted as successful executions.

The four Python warnings in the final test summary arise from deliberate
experimental FDK, overflow and invalid-coordinate tests. A separate Matplotlib
warning notes an animation deleted without rendering during headless tests.
Interactive visual appearance was not manually certified.

## Remaining limits

- External scanner demo 21 and measured GCT reconstruction still need real
  acquisition files. PyTorch is not installed in `recon-new`. The earlier
  [demo report](Demo_validation_2026-11-03.md) records 26 passing demos and these
  three unmet requirements; the full demo set was not rerun during this audit.
- Python Varian preprocessing remains a limited baseline. Synthetic fixtures
  and MATLAB agreement do not validate real scanner corrections.
- MATLAB can inspect exact piecewise source acquisitions but cannot edit their
  waypoints in its native builder. Independently positioned detector rigs are
  not merged by the Python piecewise source feature.
- The separate legacy generated configuration/XML-report matrix, pCT kernels,
  optional PyTorch training, multi-GPU behavior and other OS/toolchain/GPU
  combinations were not validated by this audit.
- Tests used the existing compiled Python extensions and isolated MATLAB MEX
  build. Compiler-configuration dry runs passed; a clean rebuild of every
  native binary was not performed.

## Local evidence and reproduction

Full Python results: `.codex-results/project_audit_python_final.log`.
MATLAB suite results: `.codex-results/matlab_project_audit/results.json` and
per-suite logs. The successful desktop retry is under
`.codex-results/matlab_project_audit_desktop/`; the final expanded piecewise
import test is under `.codex-results/matlab_project_audit_piecewise_final/`.
The separate six-layout/motion interchange check is recorded in
`.codex-results/project_audit_matlab_core.log`.

With pytest available, run from the repository root:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'Python'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
conda run --no-capture-output -n recon-new python -m pytest Python/tests -q -rs
conda run -n recon-new python -m pip check
```

MATLAB setup is `InitTIGRE; UseBlackwellBuild;` from `MATLAB/`, followed by the
`test_*` functions in `MATLAB/Tests/`. Run UI suites in fresh processes with
desktop file access. See [Validation](Validation.md) for individual workflows.
