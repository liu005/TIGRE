# MATLAB and Python demo contents

Matching filenames identify the same teaching topic. The comparison below is
based on executable geometry, input data, trajectories, algorithm calls and
outputs, not just headings. Python stores volumes as z/y/x and projections as
views/v/u; MATLAB uses volumes x/y/z and projections v/u/views. Detector
offsets change from MATLAB u/v to Python v/u; local detector rotation angles
remain x/y/z. Geometry lengths are mm and angular inputs are radians.

## Content comparison

| Demo | Common content after alignment | Remaining differences |
| --- | --- | --- |
| 01 CreateGeometry | Explicit cone geometry, offsets, COR, detector rotations and default geometry alternatives. | Default factory APIs and the illustrative alternative defaults differ. |
| 02 SampleData | Bundled head and Shepp-Logan examples; 256-cubed head geometry. | Phantom generators and resampling methods differ. |
| 03 generateData | Head, 100 full-circle views, matched physical cone geometry and CT noise. | Noise draws and implementations are independent. |
| 04 SimpleReconstruction | Head, 100 views, FDK and 50 OS-SART iterations. | Language-specific plotting and output layouts. |
| 05 plottingFunctions | Head reconstructions, image/projection display options and GIF examples. | Figure APIs and montage directions differ. |
| 06 Algorithms01 | Head, 200 views, Hann versus Ram-Lak FDK. | Filter spelling is `ram-lak` in MATLAB and `ram_lak` in Python. |
| 07 Algorithms02 | SIRT/SART/OS-SART, 20 iterations, RMSE and residual comparisons. | MATLAB returns residual and quality histories separately. |
| 08 Algorithms03 | Noisy head, 100 views, 30-iteration SIRT/CGLS/LSQR/LSMR/hybrid/GMRES comparisons. | MATLAB also returns ground-truth error histories; plotting differs. |
| 09 Algorithms04 | Noisy head, 30 endpoint-free views, 60 OS-SART and 50 main ASD-POCS iterations. | Stopping-threshold selection differs; Python includes additional 10-iteration TV variants. |
| 10 QualityMeasurements | Reconstruct the noisy head data for 30 iterations with SIRT/SART/OS-SART; OS block size 22; RMSE/CC/MSSIM/UQI. | MATLAB early stopping may shorten histories; each plot uses its own history length. |
| 11 PostProcessing | Noisy-head FDK followed by TV denoising (100 iterations, weight 15). | MATLAB additionally demonstrates cone-support cropping; Python has no equivalent crop demo. |
| 12 ProjectionOperators | Shepp-Logan, Siddon/interpolated operators and sampling-accuracy comparisons. | Phantom implementations and montage orientation differ. |
| 13 HelicalGeometry | Same 1024-mm axial volume, three 100-view turns, axial offset sweep, OS-SART 30/CGLS 20. | Axial motion is MATLAB offOrigin(3,:) versus Python offOrigin[:,0]. |
| 14 Offsets | Static and view-varying detector offsets, smaller moving volume and 10 ordered SART iterations for the final case. | Python reverses the coordinate components and stores per-view vectors in rows. |
| 15 3Dparallel | 128-cubed head, 512-mm extent, 128-square detector at 5.5 mm, 360 views, FBP/OS-SART 40. | Plot montage layout differs. |
| 16 2Dtomography | Single-slice parallel and cone examples, 100 views and OS-SART 40. | MATLAB uses Modified Shepp-Logan and ASD-POCS; Python retains a head slice and CGLS. This pair is not numerically equivalent. |
| 17 DetectorRotation | Same physical acquisition and all 100 per-view x/y/z detector rotations; OS-SART 50 and geometry animation. | Floating-point precision and plotting conventions differ. |
| 18 ArbitraryAxisOfRotation | Same physical volume/detector and 100 ZYZ source/detector poses; geometry animation. | Language-specific reconstruction/plot interfaces. |
| 19 DBTexample | Same physical acquisition, nine views over 25 degrees, fixed detector, FDK and two ordered SART iterations. | Python explicitly reverses xyz/u-v dimensions; display conventions differ. |
| 20 Algorithms05 | Noisy head, 30 endpoint-free views, four 100-iteration FISTA/TV configurations including modified momentum. | Python `tvlambda` corresponds to MATLAB `lambda`; operator-norm estimates need not agree exactly. |
| 21 ScannerDataLoader | Section-by-section external scanner-loading examples and geometry outputs. | Loader coverage differs. Python Varian is an explicitly limited CPU baseline; MATLAB also has calibrated DPS/scatter/BH preprocessing. |
| 22 ListGPUs | Query installed models and select valid zero-based device IDs. | Native binding and object names differ. |
| 23 DualEnergy | Same shared calibration/attenuation JSON files, cubic material transfer, angle matching, 60-keV VM synthesis and optional FDK/OS-SART 100. | MATLAB uses its scanner pipeline. Python accepts the exact preprocessed MAT export, or raw folders through its limited Varian baseline. |
| 24 Algorithms06 | Anatomical priors/PICCS. | MATLAB only; Python has no corresponding PICCS implementation to demonstrate. |
| 25 CurvedDetector | 64-cubed head, 90 views, smooth cylindrical pixels, 50 CGLS iterations, Hann surface FDK and animation after projections. | Python cubic phantom resampling differs from MATLAB nearest-neighbor resampling. |
| 26 ModularDetector | Same three panels, tilt lists, 2-mm COR, masked gaps, head geometry, 90 views, CGLS 50 and animation. | Same phantom-resampling difference; no surface FDK for these tilted panels. |
| 27 DetectorGeometryGUI | Same three-panel settings, preview, editable recipe/exact surface save/load, inspection mode and return to builder. | MATLAB uses native MAT and optional YAML/NPZ; Python uses YAML/NPZ. Tk and MATLAB desktop controls differ. |

The shared introductory cone acquisitions now specify their sizes and physical
detector/volume extents explicitly. Factory defaults alone are not equivalent:
Python's former low-resolution default was 64-cubed, whereas many MATLAB
examples used 128-cubed with a different detector sampling.

The head dataset is shared, but MATLAB `headPhantom` uses nearest-neighbor
resampling while Python `load_head_phantom` uses cubic resampling without
prefiltering. Independent noise and numerical implementations also prevent
pixelwise reconstruction equality. Common settings do not imply equal error
metrics or quantitative equivalence of scanner preprocessing.

## Validation and language-specific examples

Live MATLAB/Python comparisons check the dual-energy polynomial and VM spline,
all 100 detector-rotation/arbitrary-orbit poses, and modular source/pixel arrays
and masks. CPU GUI tests check actual recipe/exact round trips and disabled
inspection fields. Reduced-size native GPU runs check the new Python surface
head demos. Tests also cover XIM decoding and complete synthetic Varian folders.
External acquisitions are not bundled and have not been validated locally.

PyTorch remains [d_pytorch.py](../Python/demos/d_pytorch.py), without a MATLAB
counterpart. The named Python GCT and YAML reconstruction examples, MATLAB's
proton-radiography example and external SophiaBeads workflow remain separate.
Old Python names `d10_MeasureQuality.py`, `d12_ProjectionOperations.py`,
`d24_CurvedDetector.py` and `d25_Pytorch.py` were replaced by the aligned names;
there are no duplicate wrapper demos.

See the [MATLAB index](../MATLAB/Demos/README.md),
[Python index](../Python/demos/README.md),
[dual-energy guide](Dual_energy.md) and [Varian loader guide](Varian_loader.md).
