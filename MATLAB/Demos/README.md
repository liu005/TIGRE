# MATLAB demos

Add TIGRE's MATLAB folders to the path with `addpath(genpath('MATLAB'))`
from the repository root, then run a demo by its filename without `.m`.
GPU examples require compiled native MEX files. On the local custom Blackwell
build, run `UseBlackwellBuild` first; see the
[build instructions](../../Frontispiece/MATLAB_Blackwell_build.md).

## Geometry builder and file round trips

Run `d27_DetectorGeometryGUI` to configure three panels, preview their tilt and
COR, save an editable settings MAT file and an exact surface MAT file, and load
both in the same GUI. It leaves the GUI in inspection mode and prints a new
temporary output directory. Run sections individually to interact between steps.
Use **Return to builder** to edit again. **Save settings...**, **Save surface...**
and **Load geometry...** provide the same operations interactively.

MAT files need no Python or CUDA. Optional YAML lines demonstrate an editable
recipe and an exact YAML + NPZ pair; that interchange requires Python `recon`
with TIGRE, SciPy and PyYAML. Keep an exact archive's YAML and NPZ files together.
The [MATLAB manual](../../Frontispiece/TIGRE_CT_Simulation_Builder_MATLAB.md) describes
coordinate ordering, masks, panel tilts, global COR and inspection mode.

## Choosing a demo

The common geometry/phantom setup is repeated deliberately so each introductory
script can run independently. Demos 4 and 6 compare basic reconstruction and FDK
options; demo 15 changes the beam model to parallel geometry. Demos 9 and 24
compare TV regularization without and with anatomical priors. Demo 25 uses a
smooth cylindrical detector and supports surface FDK; demo 26 uses individually
tilted modular panels and matched iterative reconstruction. Demo 27 demonstrates
editing and file interchange without projection/reconstruction.

Run scanner examples 21 and 23 section by section after replacing their paths.
They require external acquisitions and are not bundled-data smoke tests. The
generic-loading sections are commented templates. Demos may create figures,
clear workspace variables and, where explicitly requested, write animated GIFs.
Several legacy examples use large volumes: reduce sizes before a quick GPU test.

| Script | Purpose | Requirements |
| --- | --- | --- |
| [d01_CreateGeometry.m](d01_CreateGeometry.m) | Describe CT geometry | MATLAB graphics; no CUDA or external data. |
| [d02_SampleData.m](d02_SampleData.m) | Load bundled sample data | MATLAB graphics and bundled sample data; no CUDA. |
| [d03_generateData.m](d03_generateData.m) | Generate projections and CT noise | native TIGRE CUDA MEX and bundled or generated data. |
| [d04_SimpleReconstruction.m](d04_SimpleReconstruction.m) | Simple FDK and OS-SART reconstruction | native TIGRE CUDA MEX and bundled or generated data. |
| [d05_plottingFunctions.m](d05_plottingFunctions.m) | Plot images and projections | native TIGRE CUDA MEX and bundled or generated data. |
| [d06_Algorithms01.m](d06_Algorithms01.m) | FDK options | native TIGRE CUDA MEX and bundled or generated data. |
| [d07_Algorithms02.m](d07_Algorithms02.m) | SART-family algorithms | native TIGRE CUDA MEX and bundled or generated data. |
| [d08_Algorithms03.m](d08_Algorithms03.m) | Krylov subspace algorithms | native TIGRE CUDA MEX and bundled or generated data. |
| [d09_Algorithms04.m](d09_Algorithms04.m) | Total variation algorithms | native TIGRE CUDA MEX and bundled or generated data. |
| [d10_QualityMeasurements.m](d10_QualityMeasurements.m) | Image quality measurements | native TIGRE CUDA MEX and bundled or generated data. |
| [d11_PostProcessing.m](d11_PostProcessing.m) | Image postprocessing | native TIGRE CUDA MEX and bundled or generated data. |
| [d12_ProjectionOperators.m](d12_ProjectionOperators.m) | Forward and backprojection operators | native TIGRE CUDA MEX and bundled or generated data. |
| [d13_HelicalGeometry.m](d13_HelicalGeometry.m) | Helical scan geometry | native TIGRE CUDA MEX and bundled or generated data. |
| [d14_Offsets.m](d14_Offsets.m) | Volume and detector offsets | native TIGRE CUDA MEX and bundled or generated data. |
| [d15_3Dparallel.m](d15_3Dparallel.m) | 3D parallel tomography | native TIGRE CUDA MEX and bundled or generated data. |
| [d16_2Dtomography.m](d16_2Dtomography.m) | 2D tomography | native TIGRE CUDA MEX and bundled or generated data. |
| [d17_DetectorRotation.m](d17_DetectorRotation.m) | Detector rotation | native TIGRE CUDA MEX and bundled or generated data. |
| [d18_ArbitraryAxisOfRotation.m](d18_ArbitraryAxisOfRotation.m) | Arbitrary axis of rotation | native TIGRE CUDA MEX and bundled or generated data. |
| [d19_DBTexample.m](d19_DBTexample.m) | Digital breast tomosynthesis | native TIGRE CUDA MEX and bundled or generated data. |
| [d20_Algorithms05.m](d20_Algorithms05.m) | FISTA with a TV proximal operator | native TIGRE CUDA MEX and bundled or generated data. |
| [d21_ScannerDataLoader.m](d21_ScannerDataLoader.m) | Load external scanner data | external scanner data and native CUDA; run one scanner section at a time. |
| [d22_ListGPUs.m](d22_ListGPUs.m) | List and select available GPUs | native GPU enumeration MEX; no projection or reconstruction. |
| [d23_DualEnergy.m](d23_DualEnergy.m) | Dual-energy CBCT | two external energy datasets, calibration files and native CUDA. |
| [d24_Algorithms06.m](d24_Algorithms06.m) | Anatomical priors and PICCS | native TIGRE CUDA MEX and bundled or generated data. |
| [d25_CurvedDetector.m](d25_CurvedDetector.m) | Head phantom with a smooth curved detector | native surface CUDA and the bundled head phantom. |
| [d26_ModularDetector.m](d26_ModularDetector.m) | Head phantom with tilted modular detectors | native surface CUDA and the bundled head phantom. |
| [d27_DetectorGeometryGUI.m](d27_DetectorGeometryGUI.m) | Build, save and load detector geometry with the GUI | MATLAB desktop graphics; no CUDA or external data. |

Additional examples:

- [pRad_demo.m](pRad_demo.m): bundled proton radiography data; requires the
  separately built `pCTCubicSpline_mex` and Image Processing Toolbox for `imshow`.
- [SophiaBeads_demo.m](SophiaBeads_demo.m): external SophiaBeads data and code;
  follow its preparation instructions and preserve the imported workspace.

## Python counterparts and validation

The [validation guide](../../Frontispiece/Validation.md) provides the full
Python/native MATLAB regression commands and records their coverage limits.

[Python demos](../../Python/demos) now use the same basenames for numbered
counterparts, including dual energy (23), curved detectors (25), modular
detectors (26) and the GUI workflow (27). PICCS demo 24 remains MATLAB-only;
Python `d_pytorch.py` demonstrates optional PyTorch bindings. The
[content comparison](../../Frontispiece/Demo_language_comparison.md) records
physical acquisitions, algorithm choices, outputs and remaining differences.
Python's named GCT example requires measured projection data; its
YAML reconstruction example uses bundled geometry and simulated projections.
Use `conda run -n recon python ...` for Python in this project and ensure the
checkout's `Python` directory is on `PYTHONPATH` when running from the root.

The audit checked headings, numbering, renamed references, repeated content,
Python syntax and algorithm exports, and MATLAB Code Analyzer diagnostics.
The common standalone setup and geometry-specific comparisons are intentional;
overwriting alternative reconstructions and stale algorithm calls are corrected.
This does not establish numerical correctness for every legacy algorithm or
validate external scanner data that are not available locally.

Relevant MATLAB checks:

```matlab
test_detectorGeometryGUIDemo  % CPU-only GUI, MAT round trips and mode changes
test_detectorGeometryGUI      % presets, surface construction and native loading
test_curvedSurfaceDemo       % native head reconstruction, smooth cylinder
test_modularDetectorDemo     % native head reconstruction, tilted modules
test_surfaceProjectors       % matched operators, masks, COR and iterative solvers
test_surfaceFDK              % supported circular FDK scans and rejected geometries
test_dualEnergy              % CPU material decomposition, angle matching and VM
```
