# %% Demo 25: Head phantom with a smooth curved detector
"""Simulate and reconstruct a smooth cylindrical surface without flattening.

Requires TIGRE surface CUDA extensions and the bundled head phantom. Run in
conda env recon. MATLAB's corresponding example is d25_CurvedDetector.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import tigre
from tigre.algorithms import cgls_surface, fdk_surface
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry
from tigre.utilities.sample_loader import load_head_phantom


def main(n=64, nviews=90, niter=50, show=True):
    """Return head, projections and reconstructions; show plots by default."""
    geo = tigre.geometry()
    geo.nVoxel = np.array([n, n, n])  # Python volume order: z, y, x.
    geo.sVoxel = np.array([64.0, 64.0, 64.0])
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.zeros(3)
    head = load_head_phantom(geo.nVoxel).astype(np.float32)
    angles = np.arange(nviews) * (2 * np.pi / nviews)
    fan, z = np.meshgrid(np.linspace(-0.15, 0.15, 96), (np.arange(80) - 39.5) * 1.5)
    local = np.stack((300 - 500 * np.cos(fan), 500 * np.sin(fan), z), axis=-1)
    pixels = np.empty((nviews, 80, 96, 3), np.float32)
    source = np.zeros((nviews, 3), np.float32)
    for k, angle in enumerate(angles):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pixels[k] = local @ rotation.T
        source[k] = rotation @ np.array([300, 0, 0])
    surface = DetectorSurfaceGeometry(source, pixels)
    projections = tigre.Ax_surface(head, geo, surface)
    animation = None
    if show:
        tigre.plotproj(projections, angles)
        animation = tigre.animate_geometry(geo, surface=surface)
        plt.show()
    image, residual = cgls_surface(projections, geo, surface, niter, return_residuals=True)
    fdk = fdk_surface(projections, geo, surface, filter="hann")
    print(f"Curved head CGLS relative error: {np.linalg.norm(image-head)/np.linalg.norm(head):.4f}")
    if show:
        fig, axes = plt.subplots(1, 3)
        for ax, volume, title in zip(axes, (head, image, fdk), ("Input head", "CGLS", "Surface FDK")):
            ax.imshow(volume[n // 2], cmap="gray", vmin=head.min(), vmax=head.max())
            ax.set_title(title)
        fig.tight_layout()
        plt.figure()
        plt.semilogy(np.arange(1, len(residual) + 1), residual)
        plt.xlabel("Completed iteration")
        plt.ylabel("Projection residual")
        plt.show()
    return head, projections, image, fdk, residual, animation


if __name__ == "__main__":
    main()
