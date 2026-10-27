"""Load YAML geometry, simulate a phantom, reconstruct and save comparisons.

Run from Python/ in the recon environment:
    python demos/d_geometry_yaml_reconstruction.py --output results/yaml_reconstruction
Uses the bundled head phantom from Python/example.py by default; use
--phantom ellipsoids for the synthetic alternative. Requires TIGRE CUDA surface
projectors. The saved geometry is used unchanged; projections are simulated.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import tigre
from tigre.algorithms.surface_algorithms import cgls_surface
from tigre.utilities.geometry_io import load_geometry_file, surface_archive_arrays
from tigre.utilities.gpu import GpuIds, getGpuNames


def make_phantom(shape, extent):
    """Deterministic ellipsoids defined in physical z/y/x coordinates."""
    coordinates = [(np.arange(n) - (n - 1) / 2) * length / n
                   for n, length in zip(shape, extent)]
    z, y, x = np.meshgrid(*coordinates, indexing="ij")
    phantom = ((x / 26)**2 + (y / 22)**2 + (z / 9)**2 < 1).astype(np.float32)
    phantom += 0.8 * (((x-9)/7)**2 + ((y+7)/6)**2 + ((z-2)/4)**2 < 1)
    phantom -= 0.5 * (((x+10)/5)**2 + ((y-5)/7)**2 + ((z+2)/4)**2 < 1)
    return phantom


def run(geometry_path, output, iterations=100, phantom="head", voxels=(16, 32, 32)):
    """Execute noiseless, matched-projector CGLS and write reproducible artifacts."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    document = load_geometry_file(geometry_path)
    surface = document.surface
    geo = tigre.geometry()
    shape = np.asarray(voxels)
    if shape.shape != (3,) or shape.dtype.kind not in "iu" or np.any(shape < 1):
        raise ValueError("voxels must contain three positive integers in z/y/x order")
    geo.nVoxel = shape
    geo.sVoxel = np.array([32., 96., 96.])  # z/y/x, mm; chosen for the shipped example
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.zeros(3)
    if phantom == "head":
        from tigre.utilities import sample_loader
        truth = np.ascontiguousarray(sample_loader.load_head_phantom(geo.nVoxel), dtype=np.float32)
        phantom_source = "TIGRE Common/data/head.mat, via sample_loader.load_head_phantom (as in Python/example.py)"
    elif phantom == "ellipsoids":
        truth = make_phantom(geo.nVoxel, geo.sVoxel)
        phantom_source = "Deterministic synthetic ellipsoids"
    else:
        raise ValueError("phantom must be head or ellipsoids")
    devices = GpuIds()
    if not len(devices):
        raise RuntimeError("No CUDA GPU is available for the surface projectors")
    # Use one GPU to keep the demonstration's runtime and rounding predictable.
    devices = devices[0]
    started = time.perf_counter()
    projections = tigre.Ax_surface(truth, geo, surface, gpuids=devices)
    initial_residual = float(np.linalg.norm(projections))
    if initial_residual == 0:
        raise ValueError("The supplied geometry does not measure this phantom")
    reconstruction, residuals = cgls_surface(
        projections, geo, surface, niter=iterations, gpuids=devices, return_residuals=True)
    predicted = tigre.Ax_surface(reconstruction, geo, surface, gpuids=devices)
    valid = surface.valid_mask
    relative_error = float(np.linalg.norm(reconstruction-truth) / np.linalg.norm(truth))
    relative_residual = float(np.linalg.norm((predicted-projections)[valid]) / initial_residual)
    metrics = dict(geometry_file=str(Path(geometry_path).resolve()),
        phantom=phantom, phantom_source=phantom_source,
        algorithm="cgls_surface", requested_iterations=iterations,
        completed_iterations=len(residuals), phantom_relative_l2_error=relative_error,
        projection_relative_l2_residual=relative_residual,
        zero_initialization_relative_l2_error=1.0,
        elapsed_seconds=time.perf_counter()-started, gpu_names=getGpuNames(),
        selected_gpu_devices=devices.devices, projection_shape=list(projections.shape),
        volume_shape=geo.nVoxel.tolist(), volume_size_mm=geo.sVoxel.tolist(),
        valid_samples=int(valid.sum()), total_samples=int(valid.size),
        notes="Noiseless matched Siddon simulation; not validation against measured data.")
    if not np.isfinite(reconstruction).all() or not np.isfinite(relative_residual):
        raise RuntimeError("Reconstruction produced non-finite results")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "results.npz", phantom=truth,
        reconstruction=reconstruction, projections=projections, predicted_projections=predicted,
        residuals=np.r_[initial_residual, residuals], nVoxel=geo.nVoxel, dVoxel=geo.dVoxel,
        **surface_archive_arrays(surface, geo))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    error = reconstruction-truth
    limit = max(float(np.abs(error).max()), 1e-6)
    for row, axis in enumerate((0, 1, 2)):
        index = truth.shape[axis] // 2
        vertical, horizontal = [dimension for dimension in range(3) if dimension != axis]
        extent = [-geo.sVoxel[horizontal]/2, geo.sVoxel[horizontal]/2,
                  -geo.sVoxel[vertical]/2, geo.sVoxel[vertical]/2]
        for col, data in enumerate((truth, reconstruction, error)):
            kwargs = dict(cmap="gray", vmin=0, vmax=float(truth.max())) if col < 2 else dict(cmap="coolwarm", vmin=-limit, vmax=limit)
            artist = axes[row, col].imshow(np.take(data, index, axis=axis), origin="lower",
                                          extent=extent, aspect="equal", **kwargs)
            axes[row, col].set_title(f"{('Truth', 'CGLS', 'Error')[col]}: {('z','y','x')[axis]} slice {index}")
            axes[row, col].set_axis_off()
            if col == 2:
                fig.colorbar(artist, ax=axes[row, col], shrink=.7)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path,
        default=Path(__file__).parent / "geometry_examples" / "three_panel.yaml")
    parser.add_argument("--output", type=Path, default=Path("results/yaml_reconstruction"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--phantom", choices=("head", "ellipsoids"), default="head")
    parser.add_argument("--voxels", type=int, nargs=3, metavar=("Z", "Y", "X"), default=(16, 32, 32))
    arguments = parser.parse_args()
    run(arguments.geometry, arguments.output, arguments.iterations, arguments.phantom, arguments.voxels)
