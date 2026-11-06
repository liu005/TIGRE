# Shared projection operators

All 26 public iterative methods accept explicit operators in Python, with
matching MATLAB operator and surface entry points. Conventional calls retain
their established implementations. Use the operator interface when needed,
such as reconstruction on curved/modular detector surfaces with missing pixels.

## Python calls

```python
import tigre
import tigre.algorithms as algs

# Existing conventional call:
image = algs.lsqr(data, geo, angles, niter=20)

# Surface convenience, available for every adapted method:
image = algs.lsqr_surface(data, geo, surface, niter=20, gpuids=gpuids)
image = algs.os_aw_pcsd_surface(data, geo, surface, niter=20,
    maxl2err=tolerance, blocksize=20, delta=-0.005, gpuids=gpuids)

# Explicit interface; omit geo and angles from the algorithm call:
operator = tigre.SurfaceOperator(geo, surface, gpuids=gpuids)
image = algs.hybrid_lsqr(data, operator=operator, niter=20, lmbda=1.)
image = algs.hybrid_lsqr_operator(data, operator, niter=20, lmbda=1.)
```

POCS `_surface` wrappers require `maxl2err` (a keyword), and all surface wrappers
accept `gpuids` and optional `n_slabs`. Python names are:

| Family | Public algorithm names (append `_operator` or `_surface`) |
| --- | --- |
| Algebraic | `sart`, `sirt`, `os_sart`, `fast_os_sart`, `sart_tv`, `os_sart_tv`, `iterativereconalg` |
| Krylov | `cgls`, `lsqr`, `lsmr`, `hybrid_lsqr`, `irn_tv_cgls`, `hybrid_flsqr_tv`, `ab_gmres`, `ba_gmres` |
| Proximal/statistical | `ista`, `fista`, `mlem` |
| ASD-POCS | `asd_pocs`, `os_asd_pocs`, `awasd_pocs`, `os_awasd_pocs` |
| PCSD | `pcsd`, `os_pcsd`, `aw_pcsd`, `os_aw_pcsd` |

`TigreOperator(geo, angles, gpuids=...)` is available for explicit conventional
backend comparisons. Normal conventional calls do not construct it. Its Siddon
forward / legacy matched backprojector pair is approximate; exact adjointness
and exact least-squares interpretation are guaranteed only for the surface pair.
FDK/FBP require their geometry-specific APIs and explicitly reject `operator=`.

## Contract and numerical conventions

`ProjectionOperator` is a structural Python protocol with `forward(volume)`,
`adjoint(projections)` and `subset(view_indices)`, plus `volume_shape`,
`projection_shape` and boolean `valid_mask` (or `None` for no missing samples).
Operators must be linear; normalized algebraic methods assume nonnegative ray
weights. Inputs must not be modified. Results are finite float32 arrays.

Python volumes use `(z,y,x)`; projections use `(views,rows,columns)`. MATLAB
volumes use `(x,y,z)` and projections `(rows,columns,views)`. Subset indices are
relative to the current operator: Python zero-based, MATLAB one-based.
Reordered/repeated/nested subsets preserve rays and masks. Conventional adapters
also subset per-view source distances, COR, offsets and detector rotations.
Surface kernels require a shared volume frame (constant offOrigin).

Adapters own their geometry/device settings. Data and volume initializers are
copied. NaN/Inf at missing samples are zeroed before arithmetic; nonfinite valid
samples fail. Row sums use `forward(ones)` and sensitivity uses
`adjoint(valid_samples)` for the selected subset. Unobservable rows/voxels get
zero reciprocal weight; column weights are recomputed rather than cached per
view as full volumes. Krylov/ISTA/statistical recurrences use the selected
operator for **every** data-consistency operation.

The remaining original algorithm classes are reused through per-instance
projection adapters, not copied into detector-specific recurrences. Internal
view tokens are private indices and never interpreted as physical angles by
native projection routines. Calls without an operator pass their original
projector functions and geometry straight through. MATLAB uses an equivalent
operator binding before interpreting native geometry fields.

## Options and outputs

Existing surface signatures remain available. Kernels for CGLS/SIRT/OS-SART
accept volume `init`, with `nonneg=False` by default for the normalized methods.
CGLS supports `return_residuals=True`; OS-SART supports `blocksize` and the fast
variant uses heuristic Nesterov momentum. Row/column normalization is exact for
the selected operator, not a reproduction of legacy approximate geometry weights.

The newly adapted original classes retain their algorithm options, including
volume `init`, `verbose`, quality metrics and `computel2`, constraints and
algorithm-specific settings. Their return conventions follow the existing class:
image alone, or `(image, errors)` when metrics are requested. SART/TV/proximal
methods default to nonnegativity; Krylov bounds are applied at the appropriate
terminal step. FISTA/ISTA support `hyper="auto"` or a positive Lipschitz parameter,
`tviter` and `tvlambda`. MLEM requires nonnegative valid measurements and uses its
existing guarded multiplicative ratio and sensitivity. Its default initializer
is ones. TV can be disabled with zero iterations, and constant volumes bypass
unnecessary normalization; native volume TV requires at least two voxels per axis.

Operator ordering supports ordered or random indices; angle-distance ordering
has no acquisition-independent meaning and is rejected. GPU settings belong to
the operator. Precomputed geometry-specific W/V and FDK/multigrid initialization
are rejected. GMRES requires the selected adjoint rather than an FDK
backprojector. Unknown operator options fail rather than being silently ignored.

## POCS tolerance and controls

All eight POCS methods require explicit finite nonnegative `maxl2err`, interpreted
as an **absolute masked L2 residual tolerance**. There is no dedicated estimator.
Conventional FDK-derived defaults cannot be assumed for arbitrary surfaces.

ASD methods support `lmbda` (1), `lmbda_red` (.99), `alpha` (.002), `alpha_red`
(.95), `rmax` (.95), `tviter` (20), `nonneg` (True), volume `init`, and
`return_history`. OS variants add `blocksize` (20), weighted variants add `delta`
(-.005). Delta is a finite nonzero scalar or three-value vector in volume-axis
order; equal vector entries reproduce scalar behavior. The native weighted-TV
kernel squares its ratios, so either sign is accepted.

PCSD variants use the same single-view/subset normalization, relaxation decay
and optional weighted TV, but TV step sizes follow the residual ratio to the
first post-data residual (first step one). They do not accept ASD alpha/rmax
controls. A zero reference residual disables later ratio-based TV steps, and an
initially exact solution skips TV. The operator path compares residual norms to
the tolerance consistently and enforces `niter` as an upper bound. Conventional
mixed squared/unsquared gates and extra-iteration behavior are not transplanted.
MATLAB's conventional PCSD saved-iterate initialization was repaired separately.

POCS history contains `residual` (post-TV), `data_residual` (pre-TV), `beta`,
`tv_step` and `cosine`, one entry per completed iteration. Nonnegativity is
enforced after data and TV updates. Momentum/TV/POCS updates do not promise
monotone residual reduction for arbitrary noisy measurements.

An existing candidate's residual can be evaluated as follows; this is not a
noise-based estimate of an appropriate tolerance:

```python
from tigre.utilities.projection_operator import prepare_projections
import numpy as np
clean = prepare_projections(data, operator)
residual = operator.forward(candidate) - clean
if operator.valid_mask is not None:
    residual = residual[operator.valid_mask]
residual_l2 = np.linalg.norm(residual.astype(np.float64).ravel())
```

A future tolerance helper needs a noise/reference model, units and valid-sample
scaling. It remains a separate adoption task; no arbitrary default was added.

## MATLAB calls and naming

```matlab
[image,history] = OS_AwPCSD_surface(data,geo,surface,20, ...
    'MaxL2err',tolerance,'BlockSize',20,'Delta',[-.03 -.02 -.01],'Device',0);
op = projectionOperator(geo,surface,'Device',0);
[image,residual] = LSQR_operator(data,op,20);
% Existing conventional methods also offer explicit opt-in:
[image,residual] = LSQR(data,[],[],20,'Operator',op);
```

MATLAB names follow existing conventions: `CGLS`, `LSQR`, `LSMR`, `hybrid_LSQR`,
`hybrid_fLSQR_TV`, `IRN_TV_CGLS`, `AB_GMRES`, `BA_GMRES`, `FISTA`, `ISTA`,
`MLEM`, `SART`, `SIRT`, `OS_SART`, `fast_OS_SART`, `SART_TV`, `OS_SART_TV`,
`iterativereconalg`, the four ASD-POCS names and `PCSD`, `AwPCSD`, `OS_PCSD`,
`OS_AwPCSD`. Append `_operator` or `_surface` to each. Methods absent from the
original MATLAB conventional API are provided as these explicit entry points.

MATLAB option names remain native: `lambda`/`lambda_red` for relaxation or
regularization, `Ratio` for ASD rmax, `TViter`, `Nonneg`, `Init`, `MaxL2err`,
`BlockSize`, `Delta`; CGLS/SIRT/OS-SART kernels use `Relaxation`, `Nonnegative`
and optional relative `Tolerance`. Option names are case-insensitive.
PCSD/ASD second outputs are history structs; other operator solvers return
residual histories and, where supported, quality arrays. Existing MATLAB
`CGLS_surface`/`SIRT_surface` keep their historical tolerance 1e-6 and
`SIRT_surface` keeps nonnegativity enabled. Set options explicitly for numerical
comparisons. IRN niter is the number of inner iterations per outer pass, default
four outer passes, matching Python. Damped LSMR and hybrid methods default to
regularization one on the operator path; conventional defaults are unchanged.

Reverse directional delta when exchanging Python zyx and MATLAB xyz volumes.
The MATLAB AwminTV gateway was extended and locally rebuilt to accept xyz
vectors and map them to the native gradient order; scalar behavior is retained.
CUDA projector and TV kernels are unchanged.

## Operator-only numerical corrections

The operator path fixes issues found during cross-language/dense validation:
hybrid LSQR uses scalar inner products for reorthogonalization; flexible LSQR-TV
uses the validated finite-difference/transpose pair and refreshes its weights.
Its TV preconditioner is a bounded mean-free CG inversion of D' W^2 D (50 steps,
relative residual target 1e-6), shared across languages, avoiding different inner
LSQR stopping behavior. These are algorithm-specific operator adaptations, not
substitution of another reconstruction method. Flexible Krylov/TV methods remain
sensitive to rounding in ill-conditioned problems.

MATLAB operator LSQR/LSMR respect the requested iteration limit, and operator
BA-GMRES retains the full Arnoldi basis in its final reconstruction. Damped LSMR
does not restart merely because the **unregularized** residual increases: that
restart changes the damped least-squares problem. Conventional recurrences and
options retain their existing paths.

## Validation

On 6 November 2026 the complete Python suite passed **615 tests** with four
numerical warnings, before the manual-example test was added; that example test
passed separately. All 26 native Python reconstruction cases and
surface wrappers passed. All 26 MATLAB operator/surface and Python-reference
comparisons passed; the largest observed relative difference in the latest run
was about 4.3e-4 for flexible LSQR-TV, with other methods below 4e-7. MATLAB
POCS, native projector and surface FDK regression suites also passed. These
checks used `recon`, MATLAB R2020a and the local RTX 5070 Ti builds.
An earlier concurrent GPU run failed a flexible LSQR-TV assertion; see the
[integration audit](Integration_audit_2026-11-06.md) for the isolated reruns
and limitations.

Tests cover all 26 native Python reconstructions and surface-wrapper parity,
all 26 MATLAB operator/surface counterparts, and common exact-ray references
with a missing view and masked NaNs. Independent dense tests cover LSQR/GMRES
least squares, damped LSMR's augmented system, ISTA/FISTA data steps, MLEM,
subset normalization and adaptive POCS schedules. Other tests cover native TV,
directional weighting, initialization, zero iterations and conventional calls.

The cross-language test uses explicit equivalent parameters (including Hyper
for ISTA/FISTA), a 2e-3 relative tolerance and the local RTX 5070 Ti native builds.
Most methods agree much more closely; flexible LSQR-TV has larger rounding
sensitivity. This is fixture-based numerical evidence, not measured-scanner or
multi-platform validation. FDK/FBP are tested separately.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'Python'
conda run -n recon python -m pytest Python/tests -q
```

```matlab
addpath(genpath('MATLAB'));
UseBlackwellBuild(); % local helper; use the matching build on other systems
test_allOperatorAlgorithms; % regenerates Python references in recon
test_operatorPocs;
```
