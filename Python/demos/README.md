# Python demos

Run these examples in the project's `recon` Conda environment. From `Python/`:

```powershell
conda run -n recon python demos/d25_CurvedDetector.py
```

Scanner demo 21 and new demos 23/25/26/27 resolve the checkout automatically
when launched directly from an IDE. For other legacy scripts launched from the
repository root, set `PYTHONPATH` to the checkout's `Python` folder so the local
package is used. Reconstruction examples require
TIGRE's compiled CUDA extensions; most introductory scripts use bundled head
data or generated phantoms. Many legacy examples allocate large volumes.

Examples 1–22 cover geometry, sample data, noisy projections, reconstruction,
plotting, algorithm families, quality measures, postprocessing, projection
operators, helical/offset/parallel/2D geometry, detector rotations, arbitrary
orbits, tomosynthesis, FISTA, scanner loading and GPU selection.

- `d21_ScannerDataLoader.py` requires external scanner files. Run one scanner
  section at a time after replacing its path. The generic-loader block is a
  commented template. Varian loading is an explicitly limited CPU baseline;
  see the [loader guide](../../Frontispiece/Varian_loader.md).
- `d23_DualEnergy.py` shares MATLAB d23's calibration/attenuation tables and
  accepts its exact preprocessed MAT export. It also accepts raw low/high
  Varian folders using the CPU baseline; see the
  [dual-energy guide](../../Frontispiece/Dual_energy.md).
- `d25_CurvedDetector.py` projects the bundled head onto a smooth cylindrical
  surface, animates after plotting projections, then runs matched CGLS and
  approximate surface FDK. It uses exact source/pixel coordinates without
  flattening. `main(..., show=False)` supports runs without plots.
- `d26_ModularDetector.py` matches MATLAB's three tilted panels, gaps, COR,
  head phantom, CGLS reconstruction and geometry animation.
- `d27_DetectorGeometryGUI.py` demonstrates the matching build/save/load
  workflow with editable YAML recipes and exact YAML/NPZ archives. It needs
  Tk desktop graphics and PyYAML, with no projection or reconstruction calls.
- `d_pytorch.py` requires optional PyTorch bindings and a compatible PyTorch
  installation.
- `d_detector_surface_gct.py` requires a measured float32 GCT projection file;
  its module docstring specifies the expected acquisition dimensions.
- `d_geometry_yaml_reconstruction.py` loads a bundled YAML geometry, simulates
  head or synthetic measurements, and writes reconstruction comparisons.

Numbered counterparts now share their MATLAB basenames. Demo 24
(`Algorithms06`, anatomical priors/PICCS) remains MATLAB-only because there is
no corresponding Python PICCS implementation. PyTorch remains a named
Python-only example. The [content comparison](../../Frontispiece/Demo_language_comparison.md)
records matched acquisitions, algorithms, outputs and remaining differences.
See the
[MATLAB demo index](../../MATLAB/Demos/README.md) and
[Python GUI manual](../../Frontispiece/TIGRE_CT_Simulation_Builder_Python.md).

The [validation guide](../../Frontispiece/Validation.md) lists current test
commands, plotting-angle regressions, scanner fixtures and hardware limits.
