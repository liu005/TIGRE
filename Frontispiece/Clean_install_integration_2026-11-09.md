# Clean installation and GUI export integration - 9 November 2026

## Baseline

The user reported the core regression suite in `recon-new` completing with
**652 passed, 2 skipped, 4 warnings** in 102.66 seconds. The two skipped
CUDA error-propagation tests could not load the runtime through `ctypes`.
The CuPy tests passed. This report does not claim a new agent-run core suite.

The user also ran `d_geometry_yaml_reconstruction.py` successfully: 100 CGLS
iterations, phantom relative L2 error 0.027676234, projection relative L2
residual 0.000160252, on an NVIDIA GeForce RTX 5070 Ti Laptop GPU.

Current checkout references at integration validation:

- TIGRE: `a56542b5d127d1cef93c3596b4c66389a18056e9`.
- Geometry design tool: `748b7daf0c4b2dcc1b24ae372b0bc81307f5d593`, version 0.2.0.
- Installed backend: `pytigre` 3.2.0, design API 1.0. The installed package's
  source revision was not independently established from its version number.
- Python 3.12.12, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.10.8, PyYAML 6.0.3.

## Agent-run GUI export integration

The GUI was built and installed with pip into a separate local target under
`.codex-results/gui-integration/gui-package`, without installing dependencies
or changing the `recon-new` environment. Only this GUI target was added to
`PYTHONPATH` for the export process; the TIGRE source directory was not added.

1. Created a withdrawn Tk builder window and loaded the GUI repository's
   `Examples/geometry/three_panel.yaml` through `load_geometry_path`.
2. Set the volume to 16 x 32 x 32 voxels over 32 x 96 x 96 mm.
3. Called the GUI save callback and invoked the export button. File dialogs
   were replaced with predetermined output paths; numerical operations and
   serialization were not mocked. The window was then destroyed.
4. Started a fresh Python process from outside both repositories, with
   `PYTHONPATH` removed and GUI/Tk imports explicitly forbidden. Confirmed
   that the GUI package was undiscoverable and TIGRE loaded from
   `recon-new/Lib/site-packages/tigre`.
5. Ran the exported script with `--algorithm both --iterations 100` and its
   bundled head-phantom simulation, then checked the saved arrays.

Results:

- Exported-script GPU reconstruction: **passed**, finite 16 x 32 x 32 output.
- Phantom relative L2 error: **0.027675413** (2.77%; checked below 5%).
- CGLS residual decreased from 1005.153411 to 0.466276. These are the exported
  script's residual-history values, not the core demo's relative residual.
- FDK was skipped with its explicit unsupported-detector message: the tilted
  panels do not satisfy the extruded-detector requirement. CGLS completed;
  approximate FDK was not enabled.
- Focused GUI exporter/example suite: **8 passed, 1 skipped**. The skipped
  Tk example round-trip passed when rerun alone in a fresh process
  (**1 passed**), indicating a Tk lifecycle limitation in the combined run.
- Staged installed CLI `python -m tigre_geometry_design_tool --check`: passed.

Local review artifacts (ignored by Git):

- `.codex-results/gui-integration/check_integration.py` - validation driver.
- `.codex-results/gui-integration/result.json` - machine-readable results.
- `.codex-results/gui-integration/bundle/three_panel.yaml` - GUI-saved geometry.
- `.codex-results/gui-integration/bundle/reconstruct.py` - exported script.
- `.codex-results/gui-integration/bundle/reconstruction.npz` - GPU output.

The user's environment exports are present at the repository root as
`environment-recon-new.yml` and `environment-recon-new-win64-explicit.txt`.
They were left unchanged.

## Limits

This validates Python GUI save/export and installed headless GPU execution
for one small synthetic geometry. It does not validate interactive visual
appearance, measured data, MATLAB integration in `recon-new`, all FDK paths,
other hardware, or rebuilding native binaries from a pristine toolchain.
The two core CUDA error-propagation skips remain unresolved. No release was
published and no hosted CI ran.
