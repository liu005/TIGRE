"""Export a standalone reconstruction example tied to a saved geometry YAML."""
import os
from pathlib import Path


def reconstruction_script_text(geometry_path, script_path):
    """Reference the saved YAML relative to the script when possible."""
    geometry_path = Path(geometry_path).resolve()
    script_path = Path(script_path).resolve()
    if not geometry_path.is_file():
        raise ValueError("Save geometry YAML successfully before exporting a script.")
    if geometry_path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("The saved geometry must be a YAML file.")
    if script_path.suffix.lower() != ".py":
        raise ValueError("Reconstruction script filename must end in .py.")
    try:
        reference = os.path.relpath(geometry_path, script_path.parent)
    except ValueError:  # Different Windows drives.
        reference = str(geometry_path)
    from .geometry_io import load_geometry_file
    document = load_geometry_file(geometry_path)
    template = _DR_SCRIPT if getattr(document.geometry, "simulation_mode", "ct") == "dr" else _SCRIPT
    return template.replace("__SAVED_GEOMETRY__", repr(reference))


_SCRIPT = '''"""Reconstruct using geometry saved by the TIGRE GUI.

Run in your TIGRE/CUDA environment, for example:
    conda run -n recon python this_script.py --algorithm both
    conda run -n recon python this_script.py --projections projections.npy --algorithm cgls --iterations 30

Without --projections, simulate the bundled head phantom. Measured projections
must already be attenuation line integrals, shaped (views, rows, columns), in
the saved surface's order. This script does not calibrate intensity data.
Keep the saved YAML and any companion surface NPZ files together.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tigre
from tigre.algorithms import cgls_surface, fdk_surface
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry
from tigre.utilities.geometry_io import load_geometry_file

DEFAULT_GEOMETRY = Path(__file__).resolve().parent / __SAVED_GEOMETRY__


def prepare_geometry(path, voxels=None):
    document = load_geometry_file(path)
    saved, surface = document.geometry, document.surface
    if saved is None:
        raise ValueError("YAML has no volume settings. Set the volume in the GUI and save again.")
    counts = voxels if voxels is not None else getattr(saved, "nVoxel", None)
    if counts is None:
        raise ValueError("No saved voxel counts. Supply --voxels Z Y X or save volume settings in the GUI.")
    counts = np.asarray(counts, dtype=float)
    if (counts.shape != (3,) or not np.isfinite(counts).all() or
            np.any(counts < 1) or np.any(counts != np.floor(counts))):
        raise ValueError("Voxel counts must be three positive integers in z, y, x order.")
    counts = counts.astype(np.int64)
    offsets = np.atleast_2d(np.asarray(saved.offOrigin, dtype=float))
    if len(offsets) not in (1, len(surface.source_positions_mm)):
        raise ValueError("Volume offsets must have one entry or one per view.")
    center = offsets[0].copy()
    if not np.all(offsets == center) or hasattr(saved, "object_angles_deg"):
        from tigre.utilities.object_motion import fixed_object_geometry
        surface, _ = fixed_object_geometry(surface, saved)
        center = np.zeros(3)
        print("Converted moving/rotating volume to a fixed object frame for reconstruction.")
    size = np.asarray(saved.sVoxel, dtype=float)
    geo = SimpleNamespace(nVoxel=counts, sVoxel=size, dVoxel=size / counts, offOrigin=center)
    if hasattr(saved, "view_times_s"):
        geo.view_times_s = np.asarray(saved.view_times_s)
    return geo, surface


def load_projections(path, surface):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        values = loaded
    else:
        with loaded:
            if "projections" not in loaded.files:
                raise ValueError("Projection NPZ must contain a 'projections' array.")
            values = loaded["projections"]
    values = np.asarray(values, dtype=np.float32)
    if values.shape != surface.valid_mask.shape:
        raise ValueError(f"Expected projections {surface.valid_mask.shape}; got {values.shape}.")
    if not np.isfinite(values[surface.valid_mask]).all():
        raise ValueError("Projection data contain non-finite valid samples.")
    return np.ascontiguousarray(np.where(surface.valid_mask, values, 0), dtype=np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--projections", type=Path, help="Attenuation projections: .npy or .npz with projections key")
    parser.add_argument("--algorithm", choices=("fdk", "cgls", "both"), default="both")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--voxels", type=int, nargs=3, metavar=("Z", "Y", "X"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().with_suffix(".results.npz"))
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    geo, surface = prepare_geometry(args.geometry, args.voxels)
    from tigre.utilities.ct_physics import validate_transmission_geometry
    validate_transmission_geometry(surface, geo)
    results = dict(geometry_file=str(args.geometry.resolve()), nVoxel=geo.nVoxel,
                   sVoxel=geo.sVoxel, dVoxel=geo.dVoxel, offOrigin=geo.offOrigin)
    if args.projections is None:
        from tigre.utilities.sample_loader import load_head_phantom
        phantom = np.ascontiguousarray(load_head_phantom(geo.nVoxel), dtype=np.float32)
        projections = tigre.Ax_surface(phantom, geo, surface)
        results["phantom"] = phantom
        print("Simulated head-phantom projections (not measured data).")
    else:
        projections = load_projections(args.projections, surface)
    if args.algorithm in ("fdk", "both"):
        try:
            results["fdk"] = fdk_surface(projections, geo, surface)
        except (NotImplementedError, ValueError) as exc:
            if args.algorithm == "fdk":
                raise
            results["fdk_skip_reason"] = str(exc)
            print(f"FDK skipped: {exc}")
            # No automatic experimental/approximate FDK opt-in.
    if args.algorithm in ("cgls", "both"):
        results["cgls"], results["cgls_residuals"] = cgls_surface(
            projections, geo, surface, niter=args.iterations, return_residuals=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **results)
    print(f"Saved reconstruction: {args.output.resolve()}")


if __name__ == "__main__":
    main()
'''


_DR_SCRIPT = '''"""Cargo DR forward simulation only; no 3D reconstruction.

Run with --volume attenuation.npy (z,y,x array, linear attenuation in 1/mm),
or use the default uniform object with --uniform-mu 0.001.
Default simulation grid is reduced to 64x128x64; physical size is unchanged.
Saved projections are attenuation line integrals; transmission is exp(-p).
This monoenergetic model excludes scatter, detector response and noise.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tigre
from tigre.utilities.geometry_io import load_geometry_file
DEFAULT_GEOMETRY = Path(__file__).resolve().parent / __SAVED_GEOMETRY__

''' + _SCRIPT[_SCRIPT.index("def prepare_geometry("):_SCRIPT.index("def load_projections(")].replace(
    "for reconstruction.", "for DR simulation.") + '''
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--volume", type=Path, help="3D NPY attenuation array in z,y,x; units 1/mm")
    parser.add_argument("--uniform-mu", type=float, default=0.001)
    parser.add_argument("--voxels", type=int, nargs=3, default=[64, 128, 64], metavar=("Z", "Y", "X"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().with_suffix(".dr.npz"))
    args = parser.parse_args(argv)
    if args.volume:
        attenuation = np.asarray(np.load(args.volume, allow_pickle=False), dtype=np.float32)
        if attenuation.ndim != 3:
            parser.error("--volume must be a 3D z,y,x array")
        counts = attenuation.shape
    else:
        if not np.isfinite(args.uniform_mu) or args.uniform_mu < 0:
            parser.error("--uniform-mu must be finite and nonnegative")
        if min(args.voxels) < 1:
            parser.error("--voxels must be positive")
        counts = args.voxels
        attenuation = np.full(counts, args.uniform_mu, dtype=np.float32)
    if not np.isfinite(attenuation).all() or np.any(attenuation < 0):
        parser.error("Attenuation must be finite and nonnegative (1/mm)")
    geo, surface = prepare_geometry(args.geometry, counts)
    from tigre.utilities.ct_physics import validate_transmission_geometry
    validate_transmission_geometry(surface, geo)
    projections = tigre.Ax_surface(np.ascontiguousarray(attenuation), geo, surface)
    transmission = np.exp(-projections)
    modules = getattr(surface, "_module_slices", None) or [(slice(0, projections.shape[1]), slice(0, projections.shape[2]))]
    dr_image = np.concatenate([projections[:, rows, (cols.start+cols.stop-1)//2].T
                               for rows, cols in modules], axis=0)
    module_slices = np.array([[r.start, r.stop, c.start, c.stop] for r, c in modules])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, projections=projections, transmission=transmission,
                        dr_image=dr_image, module_slices=module_slices, valid_mask=surface.valid_mask,
                        view_times_s=getattr(geo, "view_times_s", np.array([])),
                        geometry_file=str(args.geometry.resolve()), nVoxel=geo.nVoxel, sVoxel=geo.sVoxel)
    print(f"Saved DR projections: {args.output.resolve()} (no reconstruction)")

if __name__ == "__main__":
    main()
'''
