# Python Varian loader: CPU baseline

```python
from tigre.utilities.io import VarianDataLoader
projections, geo, angles = VarianDataLoader(
    acquisition_folder, dps=False, sc=False, bh=False)
```

The folder is an unzipped TrueBeam acquisition containing `Scan.xml`, recursive
`Proj_*.xim` files and `Calibrations/AIR-*` blank calibration files. The returned
float32 array has shape `(views, detector_rows, detector_columns)`, with angles
in radians and standard Python TIGRE geometry. Loading and normalization do not
initialize or reset a GPU. Reconstruction still requires TIGRE's CUDA operators.

## Implemented stages

- Compressed predictive and uncompressed XIM pixels; scalar/text/array metadata.
- Geometry from `Scan.xml`, with optional recursive `Reconstruction.xml`.
  Set `load_geometry=False` to estimate the volume when reconstruction metadata
  is absent or ambiguous. xyz/u-v ordering is converted to Python zyx/v-u.
- Acceleration/deceleration view selection using 90% of velocity/frame-rate
  angular spacing. Set `acdc=False` to retain every nonempty frame. The spacing
  comparison uses shortest angular distance at wrap boundaries.
- TrueBeam 2.0 single blanks and 2.7 directional bowtie blank selection.
  Blank images interpolate by gantry angle with endpoint clamping; normalization
  retains MATLAB's adjacent blank-chamber mean convention.
- Chamber/air normalization, negative/nonfinite line-integral clipping and
  optional zero-padded horizontal median filtering (`ring_removal=True`,
  `ring_kernel=9`). Set `ring_removal=False` to inspect normalized data directly.

The pixel flip, GantryRtn+90 source-angle convention, lateral-offset sign and
normalization follow MATLAB's loader. Projections are ordered by sorted file
paths, as in its XIM workflow. The low-level `read_xim` returns native file
orientation; the acquisition loader applies the TIGRE/MATLAB horizontal flip.

## Current limits

This is not a complete port of MATLAB Varian preprocessing. `dps`, `sc` and
`bh` default to **False**, whereas MATLAB enables DPS and object-scatter
correction by default. Requesting an unported correction raises
`NotImplementedError`; baseline loading warns that those corrections are
omitted. A future correction port must be validated against real calibration
exports. Use MATLAB's preprocessed MAT export for matching its default pipeline
in the [dual-energy demo](Dual_energy.md).

Automatic blank selection currently recognizes TrueBeam 2.0/2.7 naming.
Other acquisition versions, unknown XIM compression/property types and
ambiguous calibrations fail explicitly. Raw folders must have matching image
dimensions and positive chamber readings. No raw acquisition files have been
tested locally, so supported synthetic schemas do not establish compatibility
with every clinical export or quantitative scanner accuracy.

`test_varian_loader.py` generates compressed/uncompressed frames, signed
predictor residuals, unsigned raw pixels, metadata and complete XML/blank
folders. It checks orientation, geometry axes, view selection, blank
interpolation, normalization, invalid files, unsupported corrections and the
dual-energy paired-folder path. XIM decoding also has a live comparison with
the native MATLAB reader. The original ReadXim license is retained alongside
the Python reader.
