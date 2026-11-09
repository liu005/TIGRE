# Geometry design tool repository split

The Python and MATLAB desktop builders now live in the sibling `tigre-geometry-design-tool`
repository (`D:\works\software\tigre-geometry-design-tool` for this extraction).

TIGRE retains detector surfaces and modular constructors, native CUDA
projectors, surface reconstruction, geometry serialization, rendering,
animation and the headless detector recipe builder. YAML loading needs no GUI
installation. Applications use these through `tigre.design_api` (see below).

## Supported interface

External applications must import from `tigre.design_api`, not from
`tigre.utilities`. It re-exports the builder, file I/O, validation, remapping
and rendering helpers under stable names and carries `API_VERSION = (major,
minor)`; `require_api(major, minor)` raises if the installed TIGRE is
incompatible. Remove or change a name only with a major version bump, and add
names with a minor bump. `Python/tests/test_design_api.py` checks the facade and
that it does not import a desktop toolkit. The tool declares the API version it
needs (`REQUIRED_DESIGN_API`) instead of a `pytigre` version pin, because a
development build and a stock release can share a version number.

Install the desktop application from the new repository:
```powershell
conda run -n recon python -m pip install --no-deps -e ..\tigre-geometry-design-tool
& ..\tigre-geometry-design-tool\launch.ps1
```

Update Python application imports from
`tigre.utilities.detector_geometry_gui` or `tigre_gui.detector_geometry_gui` to
`tigre_geometry_design_tool.geometry_design_tool`. For numerical recipe construction use
`from tigre.design_api import build_modular_surface`.

For MATLAB, initialize TIGRE as usual, then add the new repository's
`MATLAB/Utilities` directory to the path and call `detectorGeometryGUI`.
The GUI demo 27, UI tests and full manuals moved with the application.
Core demos 25/26 and detector-surface code remain here.

This extraction creates a local repository. No GitHub repository or remote
publication is implied; choose the hosting owner and visibility before publishing.

The existing GUI PDFs were moved locally, but are ignored by Git in the new
repository; regenerate them from the tracked Markdown sources or explicitly
manage their publication. See the [integration audit](Integration_audit_2026-11-06.md)
and the new repository's `VALIDATION.md` for current results and limitations.
