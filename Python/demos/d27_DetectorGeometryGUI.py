# %% Demo 27: Build, save and load detector geometry with the GUI
"""Match MATLAB d27's editable recipe and exact geometry round trips.

Requires Tk desktop graphics and PyYAML; no CUDA calls. Python uses YAML
recipes and exact YAML/NPZ archives where MATLAB also supports native MAT.
"""
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder, build_modular_surface
from tigre.utilities.geometry_io import load_geometry_file, save_geometry_yaml


def save_examples(output=None):
    """Build three panels, save both representations, and verify round trips."""
    folder = Path(mkdtemp(prefix="tigre_gui_demo_")) if output is None else Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(
        layout="Cylindrical", n_views=36, source_radius_mm=1000.0,
        source_z_mm=0.0, module_shape=(24, 16), pixel_spacing_mm=(2.0, 2.0),
        detector_radius_mm=600.0, column_gap=2, n_modules=3,
        angular_span_deg=40.0, transverse_centers_mm=(0.0,),
        normal_offsets_mm=(0.0,), panel_angles_deg=(0.0,), z_offsets_mm=(0.0,),
        panel_x_angles_deg=(-3.0, 0.0, 3.0), panel_y_angles_deg=(2.0, -2.0, 1.0),
        source_motion="Rotating source + detector", orbit_start_deg=0.0,
        scan_span_deg=360.0, cor_mm=5.0,
    )
    surface = build_modular_surface(**settings)
    geo = SimpleNamespace(sVoxel=np.array([200.0] * 3), offOrigin=np.zeros(3))
    recipe, exact = folder / "detector_settings.yaml", folder / "detector_surface.yaml"
    save_geometry_yaml(recipe, settings=settings, geometry=geo)
    save_geometry_yaml(exact, surface=surface, geometry=geo)
    for path in (recipe, exact):
        loaded = load_geometry_file(path).surface
        np.testing.assert_array_equal(loaded.pixel_centers_mm, surface.pixel_centers_mm)
        np.testing.assert_array_equal(loaded.valid_mask, surface.valid_mask)
    print(f"Geometry files saved in: {folder}")
    return recipe, exact


def open_builder(recipe, exact=None, root=None):
    """Load editable controls, then optionally inspect exact rays in one GUI."""
    app = DetectorGeometryBuilder(root=root)
    app.load_geometry_path(recipe)
    if exact is not None:
        app.load_geometry_path(exact)
    # Use Play/slider to inspect. Return to builder restores editable settings.
    # Apply per-panel lists, then Preview before saving an updated surface.
    # Save geometry YAML... writes a recipe in builder mode, an exact archive
    # in inspection mode. Save surface... also offers a numeric NPZ archive.
    return app


def main(output=None, show=True):
    """Leave one GUI open in inspection mode; retain all generated files."""
    recipe, exact = save_examples(output)
    if show:
        app = open_builder(recipe, exact)
        print("Inspecting detector_surface.yaml. Click Return to builder to edit.")
        app.root.mainloop()
    # Keep the exact YAML and its uniquely named NPZ companion together.
    return recipe, exact


if __name__ == "__main__":
    main()
