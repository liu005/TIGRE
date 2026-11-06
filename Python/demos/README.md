# Python demos

Use the [core user manual](../../Frontispiece/TIGRE_User_Manual.md) for the complete
workflow and [validation guide](../../Frontispiece/Validation.md) for reproducible checks.
Run from a checkout with matching native binaries. Python execution uses `recon`;
MATLAB uses its normal TIGRE path and compiled MEX files. GPU reconstruction
examples require CUDA. Inspect each script before running large volume sizes.

| Example | Topic |
| --- | --- |
| [d01_CreateGeometry.py](d01_CreateGeometry.py) | CreateGeometry |
| [d02_SampleData.py](d02_SampleData.py) | SampleData |
| [d03_generateData.py](d03_generateData.py) | generateData |
| [d04_SimpleReconstruction.py](d04_SimpleReconstruction.py) | SimpleReconstruction |
| [d05_plottingFunctions.py](d05_plottingFunctions.py) | plottingFunctions |
| [d06_Algorithms01.py](d06_Algorithms01.py) | Algorithms01 |
| [d07_Algorithms02.py](d07_Algorithms02.py) | Algorithms02 |
| [d08_Algorithms03.py](d08_Algorithms03.py) | Algorithms03 |
| [d09_Algorithms04.py](d09_Algorithms04.py) | Algorithms04 |
| [d10_QualityMeasurements.py](d10_QualityMeasurements.py) | QualityMeasurements |
| [d11_PostProcessing.py](d11_PostProcessing.py) | PostProcessing |
| [d12_ProjectionOperators.py](d12_ProjectionOperators.py) | ProjectionOperators |
| [d13_HelicalGeometry.py](d13_HelicalGeometry.py) | HelicalGeometry |
| [d14_Offsets.py](d14_Offsets.py) | Offsets |
| [d15_3Dparallel.py](d15_3Dparallel.py) | 3Dparallel |
| [d16_2Dtomography.py](d16_2Dtomography.py) | 2Dtomography |
| [d17_DetectorRotation.py](d17_DetectorRotation.py) | DetectorRotation |
| [d18_ArbitraryAxisOfRotation.py](d18_ArbitraryAxisOfRotation.py) | ArbitraryAxisOfRotation |
| [d19_DBTexample.py](d19_DBTexample.py) | DBTexample |
| [d20_Algorithms05.py](d20_Algorithms05.py) | Algorithms05 |
| [d21_ScannerDataLoader.py](d21_ScannerDataLoader.py) | ScannerDataLoader |
| [d22_ListGPUs.py](d22_ListGPUs.py) | ListGPUs |
| [d23_DualEnergy.py](d23_DualEnergy.py) | DualEnergy |
| [d25_CurvedDetector.py](d25_CurvedDetector.py) | CurvedDetector |
| [d26_ModularDetector.py](d26_ModularDetector.py) | ModularDetector |

Demos 25 and 26 use exact curved/modular surfaces and native reconstruction.
Scanner loading and measured GCT examples require external input data; check
their headers and the loader guides. YAML reconstruction examples use shared
headless geometry-file APIs.

The interactive demo 27, desktop tests, GUI preset catalogue and script exporters
are maintained in [TIGRE-GUI](../../../TIGRE-GUI/README.md).
