# Experimental native MATLAB build for the RTX 5070 Ti

A custom build was compiled and executed locally using MATLAB R2020a Update 8,
Visual Studio 2022 Community (MSVC 14.44.35207), CUDA Toolkit 13.3.73, NVIDIA
driver 591.86, and an RTX 5070 Ti Laptop GPU. This is an experimentally tested
combination, not a MathWorks-supported R2020a compiler/toolkit configuration.
No MATLAB upgrade, Python reconstruction, `gpuArray`, or MATLAB CUDA-toolbox
integration is used by this build. TIGRE's native MEX gateways call CUDA directly.

## Activate the binaries already built on this computer

From MATLAB:

```matlab
cd('D:\works\software\TIGRE_work');
addpath('MATLAB');
UseBlackwellBuild();

% Optional native GPU validation:
test_blackwellBuild;
```

Activation lasts for the current MATLAB session and does not call `savepath`.
It gives the isolated binaries precedence over `MATLAB/Mex_files/win64` and
adds the external toolkit's DLL directory to the process PATH. Existing MEX
binaries are not overwritten. The output directory is ignored by Git: another
checkout or computer must rebuild rather than assuming the binaries exist.

## Rebuild using the normal Compile entry point

The original `MATLAB/Compile.m` now supports this computer directly:

```matlab
cd('D:\works\software\TIGRE_work');
addpath('MATLAB');
Compile('-dry_run');  % report detected CUDA, native GPU target and destination
Compile;             % rebuild all ten modules into MATLAB/Mex_files/win64
```

On Windows with CUDA 12.8 or newer it detects native targets using `nvidia-smi`
and uses the external compiler configuration. The default configuration is
VS2022; `-mex_xml` can select another template. The detected target must be
supported by the selected `nvcc`. Older-toolkit and non-Windows builds retain
the legacy route, with the shared CUDA error-support source now linked.
The experimental R2020a compatibility qualification still applies.

Optional overrides:

```matlab
Compile('-gpu_arch','sm_120', ...
        '-cuda_path','C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3', ...
        '-outdir','.codex-results/matlab_compile_entry');
```

Output paths are relative to the caller's working directory. `Compile` restores
that directory and MATLAB search path on return or failure. The existing
`-no_pinned_mem` and `-mex_xml` options remain supported. Custom output does
not automatically become active; add that directory at the beginning of the
MATLAB path after compilation. `test_blackwellBuild(outputDirectory)` checks
that its Ax/Atb binaries are the ones actually executed.

The updated original entry point was tested by rebuilding all ten modules in
an isolated output folder and running native GPU validation against those new
binaries. Existing normal-install binaries were left intact during that test.

## Rebuild the isolated convenience configuration

```matlab
cd('D:\works\software\TIGRE_work');
addpath('MATLAB');
setenv('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3');
CompileBlackwell('all');
UseBlackwellBuild();
test_blackwellBuild;
```

For incremental investigation, `CompileBlackwell('gpu')` builds enumeration
only; `'projectors'` also builds Ax/Atb. `'all'` builds the ten standard MEX
modules, including TV, noise, PICCS, and XIM reading. Use `'all'` before
`UseBlackwellBuild`, which checks that all ten outputs exist.

The build uses a private copy of the VS2022 compiler XML with native
`compute_120` / `sm_120`, C++17, and CUDA per-thread default streams. It bypasses
the legacy architecture-generation helper and does not change saved `mex`
compiler preferences. `TIGRE_common.cpp` is linked explicitly for the shared
CUDA error-reporting symbols. `Compile.m` delegates modern Windows builds to
this implementation; the compiler XML templates themselves remain unchanged.

Artifacts and build logs are under `.codex-results/matlab_blackwell/`.
The compiler may print existing size-conversion warnings and redundant temporary
file-deletion messages; all ten MEX builds completed successfully in this run.

## What was checked

- Native enumeration identifies the RTX 5070 Ti from R2020a.
- Siddon projection of a uniform box agrees with its analytic central-ray
  length to approximately 0.00032 mm.
- Forward projection is linear for the tested input; zero forward/backprojection
  inputs produce zero outputs.
- Matched backprojection, interpolated forward projection, and FDK execute with
  finite outputs.
- Forty CGLS iterations on a small synthetic phantom yield relative image error
  about 0.171 and projection residual about 0.0081.
- Sixty CGLS iterations reconstruct MATLAB's bundled 32-cubed head phantom with
  relative image error about 0.134.
- `minTV`, `AwminTV`, `tvDenoise`, `AddNoise`, and `minPICCS` execute finite-output
  smoke tests. `mexReadXim` compiles; decoding a real XIM file was not tested.

Metrics are saved in `validation.json`/`validation.mat`; the head-phantom input,
reconstruction and projections are in `head_validation.mat`. These small
noiseless tests establish native execution, not clinical image quality, exhaustive
algorithm validation, multi-GPU behavior, or support for other toolchain versions.

This builds standard MATLAB TIGRE plus native detector-surface projection,
matched backprojection and FDK backprojection. See the surface section below.

## Geometry and quality-measurement demos

`showGeoCBCTDiagram` now draws its schematic locally instead of downloading an
external image, so `d01_CreateGeometry` does not depend on HTTPS availability.
Pearson CC uses base MATLAB's `corrcoef`, avoiding the Statistics Toolbox
dependency encountered by `d10_QualityMeasurements`. SIRT, SART, and OS-SART
also handle lists of quality metrics with a scalar condition. Constant-image
correlation is mathematically undefined: the first comparison against a zero
initial image may legitimately show NaN. `test_qualityMeasures` checks the
CPU metrics; `test_qualityMeasuresGPU` checks all three reconstruction paths
on a small head phantom, including ground-truth comparisons.
When SIRT, SART or OS-SART stops early, residual and quality histories contain
only completed iterations, including the iteration that triggered stopping.
`d10_QualityMeasurements` uses `plotQualityMeasures` to plot each algorithm's
own iteration range, without padding or trailing zeros. `test_qualityEarlyStop`
exercises all three stop branches and unequal-length plots; legitimate zero
metric values are preserved.

`d21_ScannerDataLoader` needs external scanner data; its example paths are
placeholders. The Varian acquisition folder must contain `Scan.xml`. Missing
folders/XML are now reported before any MATLAB GPU initialization. Optional
flags use name/value pairs, for example `VarianDataLoader(folder,'acdc',false)`.
Varian corrections/log normalization still use MATLAB `gpuArray`/`gpuDevice`:
the native TIGRE MEX build does not upgrade R2020a's built-in GPU libraries or
establish support for those preprocessing stages on Blackwell. Validation here
covers missing-data handling only, not a real Varian dataset.

## Explicit detector surface kernels

`Compile` and `CompileBlackwell('all')` also build `Surface_mex` and `FDKSurface_mex`.
For an existing custom build, run `CompileBlackwell('surface')` to add it,
then `UseBlackwellBuild`, `test_surfaceProjectors` and `test_surfaceFDK`.
`Ax_surface` and `Atb_surface` share the native CUDA Siddon surface kernels
with Python; `CGLS_surface` and `SIRT_surface` reconstruct directly from
explicit pixel positions and logical masks. `FDK_surface` provides approximate
circular/extruded surface FDK with geometry validation. See
[the MATLAB surface manual](TIGRE_CT_Simulation_Builder_MATLAB.md#native-surface-reconstruction)
for layouts and limits. This path does not use a Python reconstruction bridge.

Run `d25_CurvedDetector` for the smooth-cylinder head-phantom example, or
`d26_ModularDetector` for native head-phantom projection/reconstruction with
three individually tilted panels, gaps and COR. `test_modularDetectorDemo`
checks the full head-phantom demo on the active native CUDA build.
