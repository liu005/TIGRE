"""Reconstruct a curved-module detector scan directly with the detector-surface model.

Real example: 53 modules x 64 pixels on an arc (pitch 2.6172 mm, DSD 10600, DSO 6100),
a 228 degree scan of 5701 views, one detector row. The measured data are used as they
are: no rebinning of the arc onto a flat detector.

Usage:  python d_detector_surface_gct.py PROJECTION_FILE [--n 1024] [--step 4] [--alg fdk]

PROJECTION_FILE is a float32 file of shape (5701, 3392). Conventions follow the
reference reconstruction script: view angles are linspace(0, 228 deg, 5701) + 156 deg, the
module-local pixel index is arange(64) - 32.5, and the detector is shifted by -1.0779 mm.
Results are in the world frame of tigre.animate_geometry; the reference images are
displayed with fliplr(recon) / flipud(reference).
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tigre
from tigre.algorithms import fast_os_sart_surface, fdk_surface
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry

# PITCH is the 2.5 mm crystal plus its wrapping, as used in the recon2D scripts.
DSD, DSO, PITCH, N_PIX, N_MODULES = 10600.0, 6100.0, 2.6172, 64, 53
OFF_U = -1.0779
N_VIEWS, SCAN_DEG, START_DEG = 5701, 228.0, 156.0


def gct_angles(step=1):
    return (np.deg2rad(START_DEG) + np.linspace(0, np.deg2rad(SCAN_DEG), N_VIEWS))[::step]


def arc_pixels(off_u=OFF_U):
    """In-plane (x, y) of every pixel with the source at -DSO, detector shifted by off_u."""
    half = np.arctan2(PITCH * (N_PIX / 2 - 0.5), DSD)
    theta = np.linspace(-half * N_MODULES, half * N_MODULES, N_MODULES)
    k = np.arange(N_PIX) - N_PIX / 2 - 0.5
    dx = np.concatenate([DSD * np.cos(t) - k * PITCH * np.sin(t) - DSO for t in theta])
    dy = np.concatenate([DSD * np.sin(t) + k * PITCH * np.cos(t) for t in theta])
    alpha = np.arctan2(dy, dx + DSO)
    new_alpha = np.arctan((DSD * np.tan(alpha) + off_u) / DSD)
    r = np.hypot(dx + DSO, dy)
    return r * np.cos(new_alpha) - DSO, r * np.sin(new_alpha)


def gct_surface(step=1):
    angles = gct_angles(step)
    dx, dy = arc_pixels()
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    lx, ly = -dx[None, :], dy[None, :]          # source on +x in the TIGRE frame
    pixels = np.zeros((len(angles), 1, N_PIX * N_MODULES, 3), np.float32)
    pixels[:, 0, :, 0] = c * lx - s * ly
    pixels[:, 0, :, 1] = s * lx + c * ly
    source = np.zeros((len(angles), 3), np.float32)
    source[:, 0], source[:, 1] = DSO * c[:, 0], DSO * s[:, 0]
    return DetectorSurfaceGeometry(
        source, pixels,
        module_slices=[(slice(0, 1), slice(m * N_PIX, (m + 1) * N_PIX)) for m in range(N_MODULES)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projections")
    ap.add_argument("--n", type=int, default=1024, help="reconstruction size (n x n)")
    ap.add_argument("--fov", type=float, default=5000.0, help="field of view in mm")
    ap.add_argument("--step", type=int, default=4, help="use every step-th view")
    ap.add_argument("--alg", default="fdk", choices=["fdk", "fast_os_sart"])
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--out", default="gct_recon.npy")
    a = ap.parse_args()

    proj = np.fromfile(a.projections, dtype=np.float32).reshape(N_VIEWS, N_PIX * N_MODULES)
    proj = np.ascontiguousarray(proj[::a.step, None, :])
    surf = gct_surface(a.step)

    geo = tigre.geometry()
    geo.nVoxel = np.array([1, a.n, a.n])
    geo.dVoxel = np.array([a.fov / a.n] * 3)
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.offOrigin = np.zeros(3)

    if a.alg == "fdk":
        rec = fdk_surface(proj, geo, surf)
    else:
        rec = fast_os_sart_surface(proj, geo, surf, niter=a.iters, blocksize=20)
    np.save(a.out, rec[0])
    print("saved", a.out, rec.shape)


if __name__ == "__main__":
    main()
