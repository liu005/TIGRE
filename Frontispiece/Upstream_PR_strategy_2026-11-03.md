# CERN/TIGRE comparison and PR strategy

Prepared 3 November 2026. This is a submission plan; no branches, commits,
pushes, PRs or maintainer messages were created.

## Comparison baseline and limits

- Working branch: `cupy-fdk`, HEAD `06a0bfb`.
- Fork remote: `https://github.com/yliu88au/TIGRE.git`.
- Upstream remote: `https://github.com/CERN/TIGRE.git`, default branch `master`.
- Local upstream reference: `aa5a92d`, dated 7 September 2026.
- Merge base: `41a1bfea8878e31c9bb785f69df4483ea5925cb0`.
- Local graph: 239 upstream-only and 68 local-only commits. These are ancestry
  counts, not counts of missing features: upstream already contains adapted
  versions of several local changes.
- Tip-to-tip difference: 356 files, 27,618 insertions, 7,463 deletions.
- Merge-base-to-local difference: 318 files, 27,163 insertions, 2,120 deletions.
  This also includes work already accepted upstream under different commits.
- Uncommitted `Python/setup.py`: version 3.0.0 to 3.2.0 plus line-ending changes.
  Keep this user change out of extracted PRs unless explicitly needed.

Network fetch and direct remote-head lookup failed. Exact statistics describe
the local upstream reference, not a verified current GitHub tip. Web PR pages
were consulted, but their cached state must be refreshed before submission.

## What differs

| Area | Local development | Upstream integration consequence |
| --- | --- | --- |
| Packaging | `Python/setup.py`, package name `pytigre`, local CUDA/toolchain changes | Upstream uses root `setup.py`, root `pyproject.toml`, name `tigre`, version 3.1.3 in the inspected reference. Port only necessary build changes into that layout. |
| CI | Local tree has no `.github/workflows` directory | Upstream has build/release workflows. Preserve them on new branches. |
| FDK | CuPy filtering, chunking, angular quadrature, redundancy and detector-rotation changes | Separate mathematical corrections from acceleration and dependency policy. |
| Iterative algorithms | Constraints, MLEM stabilization, momentum and numerical changes | Some corrections already merged; isolate remaining behavior and convergence effects. |
| Native code | Error cleanup, TV changes, explicit detector-surface projector | Shared code must build and run through both Python and MATLAB. |
| Geometry | Euler/ASDMG fixes, tilted-axis helpers, explicit modular/curved surfaces | Submit coordinate corrections before proposing the new public geometry model. |
| Applications | GUI builder, animation, motion, acquisition timing, cargo DR, exports | Layer these over accepted geometry APIs; avoid one application-sized PR. |
| Data utilities | Varian/XIM, dual energy, sample loading, quality plots | Reconcile upstream loader work; separate fixes from new scientific functionality. |
| Repository artifacts | Eight tracked Windows `.pyd` files, local IDE additions and generated GIFs | Exclude new machine-specific binaries and incidental artifacts from PRs. |

Tip-to-tip deletions such as upstream OSEM, plotSinogram, PICCS bindings, Varian
modules, bibliography entries and packaging files are a warning about branch
age. They must not be submitted as intentional removals.

## Reconcile earlier contributions first

The inspected upstream history contains these merges:

| Contribution | Upstream evidence | Action |
| --- | --- | --- |
| POCS relaxation decay/default | #766 | Do not resubmit; inspect remaining settings changes separately. |
| Parker fan axis/broadcast | #767 | Do not resubmit; evaluate follow-up Parker activation independently. |
| FISTA automatic hyperparameter | #769 | Do not resubmit the local implementation over the accepted adaptation. |
| TV/PICCS device reset and Krylov dtype | #770 | Preserve accepted behavior; extract only demonstrably missing changes. |
| Python parity/computeCOR/default GPU IDs | #773 | Compare semantics with current upstream before extracting. |
| Krylov norms/restarts | #774 | Do not replace the upstream file with the local version. |
| CUDA error propagation | #765, reverted by #777 | Treat as a revised contribution requiring fresh cross-binding builds. |

`git cherry` finds only two exact equivalent patches; this understates accepted
work because upstream adapted and split changes. Use code comparison and PR
history alongside patch IDs.

Existing PRs shown open by the consulted web pages:

- [#779: Siddon missed-ray return](https://github.com/CERN/TIGRE/pull/779).
  Update this PR if still open; do not create a duplicate.
- [#778: Wang short-scan gate](https://github.com/CERN/TIGRE/pull/778).
  Keep its scope distinct from full-scan intensity and filtering changes.

[#777](https://github.com/CERN/TIGRE/pull/777) explicitly says the error-handling
change broke MATLAB compilation. The existing fork branch
`origin/fix/cuda-error-propagation-v2` ends at `9a7c4a2`, which adds the common
source to Atb MEX builds and supplies fallback error text. It is useful source
material, but does not by itself prove the complete revised patch is portable.
Verify whether a replacement PR already exists before opening another.

## Submission sequence

Candidate commits below are extraction sources, not instructions to cherry-pick
their complete patches. Dependencies and tests must be adapted to upstream.

### Wave 1: finish focused correctness work

Keep at most two or three PRs actively awaiting substantive review. Start with
existing #779 and #778, then select the next independent correction.

| Candidate PR | Source material | Required review evidence |
| --- | --- | --- |
| Complete Siddon missed-ray fix in existing #779 | `5aa3c1e`, `0d2a3c8`; native kernels and `test_siddon_missing_ray.py` | Reproducer in a child process with timeout; clean native rebuild; unchanged ordinary rays. Distinguish reproduced cone hang from inspected parallel hazard. |
| Complete Wang short-scan gate in existing #778 | `9622cfb`, existing `origin/fix/wang-short-scan-gate` | Full/short scans, angle layouts and direction; offset cylinder comparison; MATLAB run. Resolve demo-number collisions against current upstream. |
| Correct Euler and arbitrary-source/detector geometry | `b1daf0a`, `3249c6a`, `51195a2`; focused parts of `common_geometry.py` | Independent source/detector coordinate oracle; nonzero yaw/pitch/roll; degree conversion; rotation covariance; unchanged ordinary geometry. Split independent defects if that makes review easier. |
| Fix FDK detector rotation convention and pivot | `616af34`, `89bfee7`; two FDK rotation tests | Pixel vectors agree with native kernels; zero rotation unchanged; tilted-detector phantom comparison. Keep CuPy rewrite out. |
| Fix full-scan Wang intensity normalization | `9f3ff9b`, `test_wang_intensity.py` | Absolute attenuation, centered/offset full scans, positive/negative and per-view offsets; no regression to short-scan gating. |
| Restore safe CUDA error propagation | `8584e78`, `3c9f71a`, `9a7c4a2`; cleanup header and error tests | Clean Python and MATLAB builds, success/failure smoke tests, repeated error recovery, no process termination and no growing device-memory use. Port into current build scripts. |

Review remaining TV upload/default-GPU changes (`6862931`, `6e350dd`) against
upstream before deciding whether either merits a PR. Do not reopen accepted
device-reset work by accident.

### Wave 2: small independent improvements

1. Sample-loader missing-data/resampling fix and focused regression tests.
2. Plot geometry/plot projection corrections; keep animation additions separate.
3. Iterative box/support constraints (`3f5595d`) with documented defaults,
   initialization ownership, every algorithm override and dtype behavior.
4. MLEM numerical guards (`8350792`, `33d4f60`) as an explicit algorithm-policy
   proposal. Separate safe division/invalid-input handling from the fixed
   `1e-6` floor and x10 cap. Establish units, parameter choices, bias and
   convergence behavior before changing defaults. Finite output alone is
   insufficient evidence of correctness.
5. Fast OS-SART momentum work (`5294668`, `1a520c2`) after checking upstream
   implementation and related PRs; compare convergence and peak memory.
6. Varian and dual-energy work as separate topics. Reconcile upstream #762 and
   its loader modules, preserve licenses and validate real data before claiming
   scanner correction accuracy. Dual energy needs scientific references.

These items can progress independently where they do not share code.

### Wave 3: opt-in performance and toolchain support

Propose bounded-memory CuPy FDK filtering separately from correctness fixes.
Use `88d6c34`, `120ce06` and relevant pieces of `89230d6` as source material.
The current local `filtering.py` imports CuPy unconditionally, while
`Python/setup.py` does not declare it. Agree on an optional dependency/fallback
policy and preserve import/install behavior before submission.

Benchmark against current upstream on identical geometries: numerical error,
absolute scaling, warm/cold timing, peak GPU memory, chunk boundaries,
nonuniform angles and per-view DSD/DSO. State hardware, CUDA, NumPy and CuPy
versions. Do not combine performance claims with a changed weighting formula
without separately measuring each change.

Submit Blackwell/CUDA build support independently. Port minimal discovery and
architecture changes to upstream's root build and CI; preserve existing targets
and MATLAB legacy paths. Do not submit the machine-specific 237-line environment
freeze or compiled binaries as installation requirements.

### Wave 4: propose the new geometry API, then layer features

First prepare a design issue or draft PR for maintainer feedback. The central
decision is whether TIGRE should expose explicit source positions and per-pixel
detector centers alongside its existing parameterized geometry.

Recommended dependency order:

1. Tilted-axis helper and coordinate validation, with zero tilt an identity;
   select from `b2d9923`, `95a1505`, `f5e40a0`, `5338e1e`, `848e1f7`.
2. Explicit detector-surface data model, masks and modular/curved constructors;
   extract from `d747b1d`. Document coordinate frames, array order, units and
   compatibility before changing package exports.
3. Native surface projection and Python/MATLAB bindings, with clean builds,
   analytic-ray tests, flat-detector equivalence, masking and memory tests.
4. Backprojection and iterative surface algorithms, with adjointness and
   reconstruction tests. Dependencies must not introduce circular imports.
5. Surface FDK as a separate scientific contribution. Distinguish circular
   supported geometry from explicitly approximate per-view trajectories;
   do not imply exact reconstruction for arbitrary paths.
6. Geometry serialization and Python/MATLAB interchange: schema version,
   validation, units, round trips and handling of unsupported editable recipes.
7. Geometry visualization and animations, then the GUI and script exporter.
8. Object rotation, acquisition timing, CT/DR workflows, cargo preset and
   piecewise source paths as individually documented application features.

Large commits `dfef35b`, `86dbc60`, `dab22d6`, `5651298` and `34dda57` span
multiple layers. Extract cohesive patches instead of making those commits
the PR boundaries. Ship documentation and examples with the relevant layer;
retain local handover/audit material outside upstream feature patches.

## Validation and branch workflow

The local audit log inspected during this comparison ends with 451 passed and
four warnings. The audit document records 25 MATLAB suites passed, existing
compiled extensions, no clean rebuild of all binaries, and outstanding real
scanner/GCT/PyTorch/multi-GPU/platform validation. Those are previous local
results, not tests of the extracted branches, and no tests were rerun for this
planning task.

For every extracted PR:

1. Refresh `upstream/master` and recheck merged/open/reverted PRs.
2. Create an isolated worktree and topic branch from that upstream tip. Preserve
   the current working tree and its uncommitted setup change.
3. Apply the smallest coherent source patch plus necessary tests and docs.
   Check the three-dot diff for unrelated files and upstream deletions.
4. Demonstrate the reported bug on the upstream baseline and its resolution on
   the topic branch. Rebuild native extensions when C++, CUDA or Cython changes.
5. Run focused and affected regression checks against the topic checkout.
   Record actual import paths to avoid testing installed or prebuilt local code.
   Use the user's required `recon` Conda environment for Python execution unless
   the user changes that instruction; older audit reports used `recon-new`.
6. Preserve upstream packaging/CI. Verify clean install/import and existing
   Python/MATLAB behavior relevant to the patch. CPU tests should be available
   without a GPU; GPU tests must report skips honestly.
7. Push to the verified fork and open a draft PR against `CERN:master` when
   submission is authorized. Existing PR pages reference `liu005` whereas local
   origin uses `yliu88au`; verify the fork identity/redirect before updating.

Example branch preparation after connectivity is restored (commands are not
executed by this plan):

```powershell
git fetch upstream
git worktree add .codex-results/pr-euler -b fix/euler-rotation upstream/master
# Port the focused geometry change and its regression test in this worktree.
git -C .codex-results/pr-euler diff --check
git -C .codex-results/pr-euler diff --stat upstream/master...HEAD
```

Before committing, inspect the working diff as well; the three-dot command
only includes committed changes. Avoid a blanket rebase of `cupy-fdk` or copying
whole old modules over current upstream.

PR descriptions should state the concrete trigger, current/changed behavior,
minimal reproducer, scope, evidence, compatibility and any untested cases. For
new numerical methods, include the relevant paper and update TIGRE.bib as
[upstream requests](https://github.com/CERN/TIGRE/blob/master/contributing.md).
Keep pure formatting and version bumps out of correctness PRs. Address review
on existing PRs before expanding the number of submissions.

## CuPy isolation comes first

The user prioritized isolating CuPy before other upstream work. The working
tree now defaults FDK/FBP filtering to NumPy/SciPy, with explicit
`filter_backend="cupy"` acceleration. Imports and GPU-specific execution live
in `_cupy_backend.py`; the common numerical filtering code is shared.
See [Optional CuPy filtering](Optional_CuPy.md). Validate this boundary first,
then extract correctness PRs without the optional accelerator. The earlier
comparison table describes the branch before this isolation change.

## Recommended next action

Finish validating CuPy isolation, refresh upstream when network access works,
reconcile #778/#779 and any revised
CUDA PR, then prepare one isolated Euler/ASDMG correction branch with an
independent coordinate test. In parallel conceptually, prepare the explicit
surface API design for later maintainer discussion. The first deliverable is
one small patch demonstrated against current upstream, not the entire fork.
