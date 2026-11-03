# Optional CuPy filtering

FDK and FBP use NumPy/SciPy for pre-weighting and FFT filtering by default.
Importing TIGRE and running these filtering paths does not import CuPy, inspect
CUDA through CuPy, or require a CuPy installation. TIGRE's native CUDA
projection/backprojection extensions are still required for reconstruction.

```python
volume = tigre.algorithms.fdk(projections, geo, angles)
# Optional acceleration, after installing CuPy for your CUDA version:
volume = tigre.algorithms.fdk(projections, geo, angles, filter_backend="cupy")
```

FBP accepts the same keyword. Acceleration is explicit: installing CuPy does
not change the default. A requested but unavailable CuPy backend raises an
actionable error rather than silently falling back. Device projection arrays
require the CuPy option; call `.get()` first to use NumPy filtering instead.
Both filtering paths return NumPy arrays for the native backprojector.

`utilities/_cupy_backend.py` contains the CuPy/CuPyX imports, device context,
FFT functions, memory sizing and host transfer. `utilities/filtering.py`
contains the shared numerical implementation; it does not import CuPy.
The low-level `filtering` and `zeropadding` utilities continue to recognize
device input, without loading CuPy when given host arrays.

`computeCOR(..., gpu=True)` retains its existing optional GPU behavior and
CPU fallback when CuPy cannot be imported. Its GPU imports also live in the
optional module; `gpu=False` uses SciPy directly.

For upstream review, submit NumPy correctness changes and their regression
tests first. Offer the optional CuPy module, dispatch and performance evidence
as a separate PR after maintainers agree to the dependency policy. No package
version bump or mandatory CuPy requirement is needed for this separation.

Validation on 3 November 2026: 34 focused tests passed in `recon`, including
a fresh interpreter with CuPy imports blocked, an independent per-view NumPy
FFT reference, host/device CuPy parity, odd view counts, padding, detector
rotation, FBP host staging, COR estimation and native Wang reconstruction.
The final run used desktop access because the sandbox blocked CuPy's compiler
temporary files. No native source was changed or rebuilt for this separation.
