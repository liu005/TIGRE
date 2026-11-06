# GUI repository split

The Python and MATLAB desktop builders now live in the sibling `TIGRE-GUI`
repository (`D:\works\software\TIGRE-GUI` for this extraction).

TIGRE retains detector surfaces and modular constructors, native CUDA
projectors, surface reconstruction, geometry serialization, rendering,
animation and the headless detector recipe builder. YAML loading imports
`tigre.utilities.detector_geometry_builder` and requires no GUI installation.

Install the desktop application from the new repository:

```powershell
conda run -n recon python -m pip install --no-deps -e ..\TIGRE-GUI
& ..\TIGRE-GUI\launch.ps1
```

Update Python application imports from
`tigre.utilities.detector_geometry_gui` to
`tigre_gui.detector_geometry_gui`. For numerical recipe construction use
`from tigre.utilities.detector_geometry_builder import build_modular_surface`.

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
