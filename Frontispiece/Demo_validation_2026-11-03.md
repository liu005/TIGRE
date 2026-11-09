# Demo validation: 3 November 2026

All 29 top-level Python demo scripts were attempted using the `recon` Conda
environment and the checkout's Python package on an NVIDIA GeForce RTX 5070 Ti
Laptop GPU. After correcting three demo errors, 26 pass and three require
unavailable inputs or dependencies.

All 44 supporting Python checks passed, covering demo acquisition alignment,
geometry animations, GUI geometry and round trips, dual energy, and YAML
reconstruction with both head and synthetic phantoms.

## Python results

### Recheck in `recon-new`

All 29 demos were attempted again: **26 passed**, with the same three unmet
requirements as `recon` (external scanner data, measured GCT projections,
and missing `torch`). No additional source changes were needed. Demo 23 used
synthetic mode and demo 27 used the automated GUI workflow, as above.
The YAML head reconstruction completed 100 iterations with relative volume
error 0.02768 and relative projection residual 0.0001603.

The recheck uses Python 3.12.12, NumPy 2.4.2 and Matplotlib 3.10.8,
with the checkout's `Python/tigre` package and the same NVIDIA GPU.
All 44 supporting checks pass. The environment has no pytest installation;
the test runner uses a workspace-local copy of pytest and its pure-Python
dependencies from `recon`, while numerical libraries and CUDA bindings come
from `recon-new`. No packages were installed into either environment.

The pose and dual-energy comparisons also pass in `recon-new` against the
MATLAB arrays generated earlier on the same day; MATLAB was not rerun for
this environment recheck.

The initial sandboxed run encountered CuPy temporary-compiler-directory
permission errors. Those reconstruction demos pass when run outside the
sandbox. Final per-demo results are in
`.codex-results/demo_run_20261103_recon_new_unrestricted/results.json`;
the initial sandboxed results are retained separately in
`.codex-results/demo_run_20261103_recon_new/`.
Supporting logs are `.codex-results/demo_checks_recon_new_verified_20261103.log`
and `.codex-results/demo_parity_recon_new_verified_20261103.log`.

### Original `recon` run

| Demos | Result |
| --- | --- |
| 01–20, 22, 25, 26 | Pass with original volume sizes, view counts and iteration counts. |
| 23 DualEnergy | Pass using the documented `--synthetic` mode; measured-data reconstruction was not validated. |
| 27 DetectorGeometryGUI | Pass: recipe/exact save and load, Tk construction, inspection mode, return to builder and preview. The interactive main loop was omitted. |
| YAML reconstruction | Pass: bundled head, shipped three-panel geometry, 100 CGLS iterations and saved reconstruction artifacts. Relative volume error 0.02769; relative projection residual 0.0001606. |
| 21 ScannerDataLoader | Cannot complete: configured example scanner folder does not exist. |
| GCT surface reconstruction | Cannot complete: a measured projection file must be supplied. |
| PyTorch | Cannot complete: `torch` is not installed in `recon`. |

Matplotlib used the Agg backend with nonblocking `show()` calls. Plotting code
and GIF exports ran, but interactive playback and visual appearance were not
manually assessed. Successful script execution alone does not certify numerical
accuracy for every reconstruction algorithm.

Three corrections were made and rerun successfully:

- Demo 01: replace obsolete `angle=` with the supported `angles=` array.
- Demo 02: import NumPy before using `np` in geometry assignments.
- YAML reconstruction: avoid passing `nVoxel` twice to `np.savez_compressed`;
  retain the archive helper's volume metadata and explicit `dVoxel`.

## MATLAB consistency

The executable geometry/algorithm settings were inspected alongside the MATLAB
counterparts. See [the topic-by-topic comparison](Demo_language_comparison.md)
for the full mapping and intentional differences.

Fresh MATLAB R2020a outputs were compared numerically with Python:

- Demo 17: all 100 detector poses agree; maximum pixel-coordinate error
  0.00015259 mm, sampled on a 10-by-12 detector with the original physical extent.
- Demo 18: all 100 arbitrary-orbit poses agree; maximum pixel-coordinate error
  0.00018120 mm on the same detector sampling.
- Demo 23: material decomposition and 67-keV virtual-monoenergetic synthesis
  match fresh MATLAB reference arrays within the numerical tolerances
  (`atol=2e-5` for decomposition and `2e-6` for synthesis, `rtol=1e-6`).

Live MATLAB tests passed for demo 17/18/19 animation placement and geometry,
and native CUDA demos 25 and 26 using `UseBlackwellBuild`. The MATLAB curved
head CGLS relative error was 0.0471; modular head CGLS error was 0.0349.
The MATLAB GUI test did not finish in batch mode and was stopped; it is not
counted as passing. Python's GUI workflow did pass.

Consistency does not mean numerical identity. In particular:

- MATLAB demo 24 (PICCS) has no Python implementation/demo.
- Demo 16 uses different phantoms and some different reconstruction algorithms.
- MATLAB demo 11 additionally demonstrates cone-support cropping.
- MATLAB scanner preprocessing has capabilities beyond Python's limited Varian
  baseline, and no measured acquisitions were available for validation.
- Head resampling, noise draws, array order and plotting conventions differ.
- PyTorch, GCT and YAML reconstruction are Python-specific examples; MATLAB's
  proton-radiography and SophiaBeads examples are separate workflows.

## Reproduction and evidence

The subsequent piecewise linear source GUI change passed 122 targeted tests
in `recon-new`, including native CUDA projection equivalence to two joined
linear scans, CGLS consistency, editable/exact archives, MAT interchange, GUI
transitions and retained CT/DR behavior. See [the validation guide](Validation.md)
and [the version 1.7 Python manual](../../tigre-geometry-design-tool/Frontispiece/TIGRE_Geometry_Design_Tool_Python.md#piecewise-linear-source-movement).
This is additional feature validation; the demo counts above describe the
earlier full demo runs. The final feature log is
`.codex-results/piecewise_source_commit_final.log`.

Local logs and the original-attempt results are in
`.codex-results/demo_run_20261103/`; demos 01, 02 and YAML reconstruction have
separate `rerun.log` files. The original `results.json` preserves pre-fix failures.
The runner is `.codex-results/run_all_demos.py`.

Fresh parity results are in `.codex-results/fresh_demo_parity_20261103.log`.
MATLAB native results are in `.codex-results/matlab_demo_native_20261103.log`;
animation checks are in `.codex-results/matlab_demo_checks_20261103.log`.
Supporting Python checks and their workspace-temporary-path workaround are in
`.codex-results/run_demo_checks_20261103.py` and
`.codex-results/demo_checks_verified_20261103.log`. Default pytest temporary
directories were inaccessible in this sandbox, so the runner substitutes a
workspace-local `tmp_path` fixture and disables third-party plugin autoload.
