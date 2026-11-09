# Installation audit fixes — 9 November 2026

This follow-up fixes the three installation findings from the 9 November audit
of the current development checkout. Earlier audit reports describe historical
runs and retain their original results.

## Changes

- The head phantom loader now checks `sys.prefix/data/head.mat` after its
  existing checkout and package-relative locations. An installed-layout test
  writes a real MAT dataset outside `lib/python3.8/site-packages` and loads it
  through the public API.
- Detected GPU targets and headless fallback targets now share CUDA compatibility
  filtering. Mixed-device builds warn about excluded capabilities, and both
  native and PTX targets use compatible capabilities. A build with no compatible
  detected targets stops with an actionable error. An empty device query uses
  the headless fallback instead of inventing a capability 7.5 device.
- CUDA version parsing preserves complete major/minor components and compares
  integer tuples; for example, `12.10` is not truncated to `12.1`.
- The minimum supported interpreter is **Python 3.8**, declared in package
  metadata as `Requires-Python: >=3.8`. The geometry document's `dict | None`
  annotation was replaced with `Optional[dict]` so it can be evaluated on 3.8.
- README, installation instructions, FAQ, demo index, validation guide and
  documentation-build Python selection now reflect Python 3.8+. The user manual
  is version 1.1, dated 9 November 2026; HTML and PDF editions were regenerated.

The CUDA table was corrected against NVIDIA's documented compiler boundaries:
[CUDA 9 Fermi removal](https://docs.nvidia.com/cuda/archive/9.0/cuda-toolkit-release-notes/index.html),
[CUDA 11 removal of capabilities 3.0/3.2](https://docs.nvidia.com/cuda/archive/11.0_GA/cuda-toolkit-release-notes/index.html),
[CUDA 11.8 compiler targets](https://docs.nvidia.com/cuda/archive/11.8.0/pdf/CUDA_Compiler_Driver_NVCC.pdf),
[CUDA 12.8 Blackwell support](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html),
and [CUDA 13 removal of targets below 7.5](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/).
Unknown future capabilities still require an explicit table update.

## Verification

Run the full Python suite in the required `recon` environment:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'Python'
conda run -n recon python -m pytest Python/tests -q --disable-warnings --maxfail=5
```

The final complete regression run passed **651 tests with four warnings in
92.02 seconds**. This includes the added installed-layout, architecture/version
boundary and Python minimum-version regressions. CUDA error-path tests emitted
expected diagnostic messages while passing.

The real `setup.py egg_info` command completed and emitted
`Requires-Python: >=3.8`. It detected local capability 12.0 and selected build
targets successfully. Metadata generation does not compile native sources.

The manual export script completed both editions. `pdfinfo` reports a 16-page
A4 PDF; extracted PDF text confirms version 1.1, the updated date and Python 3.8
minimum. PDFs are ignored by Git, so the generated PDF remains a local artifact;
the Markdown source and export script reproduce it.

## Validation limits

Runtime tests used Python 3.12 in `recon` and the environment's existing CUDA
extensions with checkout Python sources. Python 3.8 syntax and affected
annotations have regression checks, but an actual Python 3.8 native build/runtime
was not executed. Neither a clean CUDA build, Linux runtime execution,
multi-GPU execution nor MATLAB suites were run for these Python installation
changes. The simulated installed-layout test is portable and does not establish
a complete Linux wheel build.
