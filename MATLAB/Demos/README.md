# MATLAB demos

Use the [core user manual](../../Frontispiece/TIGRE_User_Manual.md) for the complete
workflow and [validation guide](../../Frontispiece/Validation.md) for reproducible checks.
Run from a checkout with matching native binaries. Python execution uses `recon`;
MATLAB uses its normal TIGRE path and compiled MEX files. GPU reconstruction
examples require CUDA. Inspect each script before running large volume sizes.

| Example | Topic |
| --- | --- |
| [d01_CreateGeometry.m](d01_CreateGeometry.m) | CreateGeometry |
| [d02_SampleData.m](d02_SampleData.m) | SampleData |
| [d03_generateData.m](d03_generateData.m) | generateData |
| [d04_SimpleReconstruction.m](d04_SimpleReconstruction.m) | SimpleReconstruction |
| [d05_plottingFunctions.m](d05_plottingFunctions.m) | plottingFunctions |
| [d06_Algorithms01.m](d06_Algorithms01.m) | Algorithms01 |
| [d07_Algorithms02.m](d07_Algorithms02.m) | Algorithms02 |
| [d08_Algorithms03.m](d08_Algorithms03.m) | Algorithms03 |
| [d09_Algorithms04.m](d09_Algorithms04.m) | Algorithms04 |
| [d10_QualityMeasurements.m](d10_QualityMeasurements.m) | QualityMeasurements |
| [d11_PostProcessing.m](d11_PostProcessing.m) | PostProcessing |
| [d12_ProjectionOperators.m](d12_ProjectionOperators.m) | ProjectionOperators |
| [d13_HelicalGeometry.m](d13_HelicalGeometry.m) | HelicalGeometry |
| [d14_Offsets.m](d14_Offsets.m) | Offsets |
| [d15_3Dparallel.m](d15_3Dparallel.m) | 3Dparallel |
| [d16_2Dtomography.m](d16_2Dtomography.m) | 2Dtomography |
| [d17_DetectorRotation.m](d17_DetectorRotation.m) | DetectorRotation |
| [d18_ArbitraryAxisOfRotation.m](d18_ArbitraryAxisOfRotation.m) | ArbitraryAxisOfRotation |
| [d19_DBTexample.m](d19_DBTexample.m) | DBTexample |
| [d20_Algorithms05.m](d20_Algorithms05.m) | Algorithms05 |
| [d21_ScannerDataLoader.m](d21_ScannerDataLoader.m) | ScannerDataLoader |
| [d22_ListGPUs.m](d22_ListGPUs.m) | ListGPUs |
| [d23_DualEnergy.m](d23_DualEnergy.m) | DualEnergy |
| [d24_Algorithms06.m](d24_Algorithms06.m) | Algorithms06 |
| [d25_CurvedDetector.m](d25_CurvedDetector.m) | CurvedDetector |
| [d26_ModularDetector.m](d26_ModularDetector.m) | ModularDetector |

Demos 25 and 26 use exact curved/modular surfaces and native reconstruction.
Scanner loading and measured GCT examples require external input data; check
their headers and the loader guides. YAML reconstruction examples use shared
headless geometry-file APIs.

The interactive demo 27, desktop tests, GUI preset catalogue and script exporters
are maintained in [tigre-geometry-design-tool](../../../tigre-geometry-design-tool/README.md).
