# %% Demo 26: Head phantom with tilted modular detectors
"""Match MATLAB d26: three tilted panels, masked gaps, global COR and CGLS.

Run in recon with TIGRE surface CUDA extensions. Python image arrays are
z/y/x; detector arrays are views/rows/columns. Coordinates remain world xyz.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import tigre
from tigre.algorithms import cgls_surface
from tigre.utilities.detector_geometry_gui import build_modular_surface
from tigre.utilities.sample_loader import load_head_phantom


def build_surface(nviews=90):
    """Use the same physical acquisition parameters as MATLAB demo 26."""
    settings = dict(
        layout="Cylindrical", n_views=nviews, source_radius_mm=300.0,
        source_z_mm=0.0, module_shape=(80, 32), pixel_spacing_mm=(1.5, 1.5),
        detector_radius_mm=200.0, column_gap=2, n_modules=3,
        angular_span_deg=28.0, transverse_centers_mm=(0.0,),
        normal_offsets_mm=(0.0,), panel_angles_deg=(0.0,), z_offsets_mm=(0.0,),
        panel_x_angles_deg=(-3.0, 0.0, 3.0), panel_y_angles_deg=(2.0, -2.0, 1.0),
        source_motion="Rotating source + detector", orbit_start_deg=0.0,
        scan_span_deg=360.0, cor_mm=2.0,
    )
    return settings, build_modular_surface(**settings)


def main(n=64, nviews=90, niter=50, show=True):
    """Project/reconstruct the bundled head and return the resulting arrays."""
    geo = tigre.geometry()
    geo.nVoxel = np.array([n, n, n])
    geo.sVoxel = np.array([64.0, 64.0, 64.0])
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.zeros(3)
    head = load_head_phantom(geo.nVoxel).astype(np.float32)
    settings, surface = build_surface(nviews)
    angles = np.arange(nviews) * (2 * np.pi / nviews)
    projections = tigre.Ax_surface(head, geo, surface)
    assert np.all(projections[~surface.valid_mask] == 0)
    animation = None
    if show:
        tigre.plotproj(projections, angles)
        animation = tigre.animate_geometry(geo, surface=surface)
        plt.show()
    image, residual = cgls_surface(projections, geo, surface, niter, return_residuals=True)
    relative_error = np.linalg.norm(image - head) / np.linalg.norm(head)
    print(f"Modular head CGLS relative error: {relative_error:.4f}")
    # Individual panel tilts violate surface FDK's extrusion requirement.
    if show:
        fig, axes = plt.subplots(1, 3)
        limits = dict(cmap="gray", vmin=head.min(), vmax=head.max())
        axes[0].imshow(head[n // 2], **limits)
        axes[1].imshow(image[n // 2], **limits)
        error_plot = axes[2].imshow(abs(image[n // 2] - head[n // 2]), cmap="gray")
        for ax, title in zip(axes, ("Input head", "Surface CGLS", "Absolute error")):
            ax.set_title(title)
        fig.colorbar(error_plot, ax=axes[2])
        plt.figure()
        plt.semilogy(np.arange(1, len(residual) + 1), residual)
        plt.xlabel("Completed iteration")
        plt.ylabel("Projection residual")
        plt.show()
    return dict(geo=geo, head=head, settings=settings, surface=surface,
                projections=projections, image=image, residual=residual,
                relative_error=relative_error, animation=animation)


if __name__ == "__main__":
    main()
