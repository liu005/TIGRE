# Shared projection-operator adoption

All **26 public Python iterative entry points** now support explicit operators.
The same solver set has MATLAB `_operator` and `_surface` entry points. Every
adapted Python method also has `_operator` and `_surface` conveniences.
See [API, options, numerical scope and validation](Projection_operator.md).

Validation: **615 Python tests passed** (four warnings), with the manual examples
also tested separately; all 26 MATLAB
operator/surface/Python numerical comparisons and native MATLAB surface
projector/FDK regressions passed. Flexible LSQR-TV is the most rounding-sensitive
case; see the guide for its bounded preconditioner and test tolerances.

Conventional calls retain their original paths. The operator path owns masks,
subsets, row/column normalization and initialization; geometry-specific FDK or
multigrid initialization and FDK preconditioners are rejected. FDK/FBP stay
separate analytical APIs.

Completed families:

- SART/SIRT/OS-SART/fast OS-SART, SART-TV and OS-SART-TV, generic iterative base.
- CGLS, LSQR, damped LSMR, hybrid LSQR, IRN-TV-CGLS, hybrid flexible LSQR-TV,
  AB-GMRES and BA-GMRES.
- ISTA, FISTA and MLEM.
- All ASD-POCS and PCSD variants, including weighted and ordered-subset forms.

Remaining adoption follow-ups:

1. Review the contract and extraction scope with Ander.
2. `maxl2err` is a parameter, **not an estimation function**. There is still no
   dedicated operator-aware tolerance helper. All operator POCS methods require
   an explicit absolute masked L2 tolerance. Design a noise/reference-based
   helper separately, including measurement units and valid-sample scaling.
3. Maintain both languages, operator/surface wrappers and meaningful numerical
   tests for any future algorithms. Preserve conventional call behavior.
4. Extend validation beyond the local GPU and synthetic/exact-ray fixtures to
   measured scanner data and additional hardware/platforms.
5. Keep analytical FDK weighting/filtering and geometry-specific initialization
   outside the linear-operator contract.

PCSD operator behavior deliberately uses an absolute L2 gate, a strict requested
iteration limit and guarded zero reference residuals. Its conventional mixed
norm gate/off-by-one behavior is retained; MATLAB's pre-existing uninitialized
saved iterate was repaired. See the API guide for other operator-only numerical
corrections and the bounded flexible-TV preconditioner.

## Original proposal

Other iterative methods need adaptation to use the surface projectors. We should avoid maintaining a separate copy of every algorithm.

The fork already provides these in [surface_algorithms.py](D:/works/software/TIGRE_work/Python/tigre/algorithms/surface_algorithms.py):
- cgls_surface
- sirt_surface
- os_sart_surface
- fast_os_sart_surface
Existing standard methods call Ax/Atb and use conventional detector geometry. Passing surface projection data into them won’t make them use the explicit detector coordinates.

The better long-term design is a shared projection-operator interface:
operator.forward(volume)
operator.adjoint(projections)
operator.subset(view_indices)
Each iterative algorithm could then work with either conventional geometry or detector surfaces. Supporting this requires more than replacing two function calls: subset handling, normalization weights, missing-pixel masks, and initialization must also use the selected operator.

For upstream contribution, I recommend:
1. Introduce surface projectors with a few validated methods—CGLS and SIRT are good starting points.
2. Agree with Ander on the operator interface.
3. Adapt the existing algorithm families progressively, keeping _surface functions as convenient wrappers if useful.
TV regularization can largely be reused because it operates on the reconstructed volume. Its data-consistency steps still need the surface operator. FDK needs separate treatment, since its analytical weighting and filtering depend on the acquisition geometry.

  
