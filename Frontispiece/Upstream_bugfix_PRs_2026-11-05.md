# Upstream bug-fix PRs prepared on 5 November 2026

Refreshed `CERN/TIGRE` and the fork before extracting changes. All four branches
start independently from upstream master `69bdd9f728e02095f38d27a9746a88eb79fc5e09` and contain one focused commit.
They are prepared locally; no branches were pushed and no GitHub PRs were opened.

## Ready for review

| PR | Branch | Commit | Validation |
| --- | --- | --- | --- |
| [Fix noisy inputs and metric rows in the Python quality demo](../.codex-results/upstream_prs_20261105/01-python-quality-demo-PR.md) | `fix/python-quality-demo-inputs-and-metrics` | `387c2aa` | 2 failed before; 2 passed after |
| [Fix projection plotting for one-dimensional and ZYZ angle arrays](../.codex-results/upstream_prs_20261105/02-plotproj-PR.md) | `fix/python-plotproj-angle-shapes` | `5ef6c4a` | 16 failed / 6 passed before; 22 passed after |
| [Report missing bundled head phantom data clearly](../.codex-results/upstream_prs_20261105/03-sample-loader-PR.md) | `fix/python-head-phantom-missing-data` | `9e00c4e` | 1 failed / 1 passed before; 2 passed after |
| [Plot each MATLAB quality history using its actual iteration count](../.codex-results/upstream_prs_20261105/04-matlab-quality-demo-PR.md) | `fix/matlab-quality-demo-history-lengths` | `123630c` | MATLAB concatenation error before; plotting regression passed after |

The complete patch files, PR descriptions, manifest, baseline failures and passing
logs are in `.codex-results/upstream_prs_20261105/`. Each named subdirectory is a
separate Git worktree; the `.patch` files can also be applied with `git am`.
The main `cupy-fdk` checkout and its pre-existing setup.py/ToDo.md edits are retained.

## What changed since the earlier audit

- [#779](https://github.com/CERN/TIGRE/pull/779), the Siddon missed-ray fix, merged on 28 October 2026.
- [#778](https://github.com/CERN/TIGRE/pull/778), the Wang short-scan gate, merged on 28 October 2026.
- [#790](https://github.com/CERN/TIGRE/pull/790) added threaded CPU filtering. Any CuPy benchmark must use this version.
- [#801](https://github.com/CERN/TIGRE/pull/801) updated per-angle indexing. Its changes are preserved on every branch.
- Noise defaults, MSSIM and UQI have also received upstream fixes (#781, #783, #796).
  Older local implementations must not overwrite these updates.

The previous demo-consistency report is useful extraction evidence, but several
local fixes are not upstream defects:

- Upstream demo 01 still uses the supported `plot_geometry(geo, angle=...)` API.
  The local replacement with `angles=` belongs to the fork's new plotting API.
- Upstream demo 02 does not reference NumPy, so the local missing NumPy import
  arose only after adding explicit acquisition dimensions.
- The duplicate nVoxel archive keyword belongs to the fork-only YAML demo.
- Upstream already contains the corrected Python helical offOrigin axis and
  endpoint-free periodic angle sampling. Local demo alignment must preserve those.
- Upstream ArbitrarySourceDetMoveGeo already converts detector rotations from
  degrees to radians; that historical fork regression should not be resubmitted.
- Renaming demos, matching phantom sizes, changing iteration counts or swapping
  algorithms are not included in the correctness PRs. Python demo 10 keeps its
  upstream filename, public `ossart` name and 20 iterations.

## Validation scope and reproduction

The combined CPU Python suite passes **26 tests** in the `recon` environment.
Tests execute the source files from their individual upstream worktrees rather
than importing the installed or development-fork TIGRE package. Projection plots
use the Agg backend; the demo uses deterministic operator stand-ins. The MATLAB
R2020a plotting regression also passes live. No new native/CUDA changes are present,
and a full upstream GPU-demo rerun was not performed.

From the main checkout:

```powershell
conda run -n recon python -m pytest `
  .codex-results/upstream_prs_20261105/python-quality-demo/Python/tests/test_quality_demo.py `
  .codex-results/upstream_prs_20261105/plotproj/Python/tests/test_plotproj_angles.py `
  .codex-results/upstream_prs_20261105/sample-loader/Python/tests/test_sample_loader.py -q
```

From the MATLAB worktree:

```powershell
& 'C:\Program Files\MATLAB\R2020a\bin\matlab.exe' -batch "addpath('MATLAB/Tests'); test_qualityDemoHistories"
```

The original tracked plotting/demo files use CRLF endings; whitespace checks use
`git -c core.whitespace=cr-at-eol diff --check` to retain their existing format.
The brief individual conda runs encountered an OpenCL activation temp-file race
when launched concurrently; the final combined run was sequential and clean.

## Remaining candidates

FDK cosine-weight pivot/convention, full-scan intensity, arbitrary geometry frame
corrections and CUDA error handling remain separate candidates. They need current
upstream numerical reproducers and, for native changes, clean Python/MATLAB builds.
They are not represented as ready PRs here. Surface detectors, GUI features and
CuPy acceleration are outside this bug-fix batch.

Submission can use each PR description with its corresponding branch, targeting
`CERN/TIGRE:master`. Each branch is independent; no merge order is required.
