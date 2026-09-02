"""Headless tests for tigre.utilities.visualization.animate_geometry.

No GPU and no display needed (Agg backend). The animation test renders
every frame through the update function by saving to a temporary file with
whichever matplotlib writer is available (pillow ships with matplotlib's
hard dependency set on most installs; the test skips if none is usable).

Replaces the original commented-out visual inspection script (kept in git
history) with assertions CI can run.
"""
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tigre
from tigre.utilities.visualization.animate_geometry import animate_geometry, calCube


def make_geo():
    """Small cone geometry exercising the offset/rotation code paths."""
    geo = tigre.geometry(mode="cone", default=True)
    geo.nVoxel = np.array([32, 48, 64])
    geo.sVoxel = geo.nVoxel.astype(np.float64)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.array([60, 80])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.offDetector = np.array([10.0, -15.0])          # (v, u)
    geo.rotDetector = np.radians([5.0, -2.0, 1.0])     # (roll, pitch, yaw)
    geo.offOrigin = np.array([0.0, 5.0, -8.0])         # (z, y, x)
    geo.COR = 1.5
    return geo


ANGLES = np.linspace(0, 2 * np.pi, 6, endpoint=False)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


class TestStatic:
    def test_returns_axes_with_scene(self):
        ax = animate_geometry(make_geo(), ANGLES, pos=2, animate=False)
        assert isinstance(ax, Axes3D)
        # the scene carries the detector + object cuboids and a title
        assert len(ax.collections) >= 2
        assert "CBCT geometry" in ax.get_title()

    def test_default_angles(self):
        ax = animate_geometry(make_geo(), None, animate=False)
        assert isinstance(ax, Axes3D)

    def test_pos_out_of_range_falls_back(self):
        ax = animate_geometry(make_geo(), ANGLES, pos=999, animate=False)
        assert isinstance(ax, Axes3D)


class TestAnimation:
    @pytest.mark.parametrize("rotation", ["SD", "obj"])
    def test_returns_animation_and_renders_frames(self, rotation, tmp_path,
                                                  monkeypatch):
        monkeypatch.chdir(tmp_path)
        ani = animate_geometry(make_geo(), ANGLES, rotation=rotation,
                               animate=True)
        assert isinstance(ani, animation.FuncAnimation)
        # render every frame through the update function - this is what
        # actually exercises the per-frame artist updates
        writers = [w for w in ("pillow", "ffmpeg")
                   if animation.writers.is_available(w)]
        if not writers:
            pytest.skip("no matplotlib animation writer available")
        out = tmp_path / f"anim_{rotation}.gif"
        ani.save(str(out), writer=writers[0], fps=10)
        assert out.exists() and out.stat().st_size > 0

    def test_fname_save_fallback_chain(self, tmp_path, monkeypatch):
        """fname triggers the internal ffmpeg->pillow->imagemagick fallback;
        whichever writer exists, the call must return the animation."""
        monkeypatch.chdir(tmp_path)
        ani = animate_geometry(make_geo(), ANGLES, animate=True, fname="t")
        assert isinstance(ani, animation.FuncAnimation)
        saved = list(tmp_path.glob("t_geometry.*"))
        if animation.writers.is_available("ffmpeg") or \
                animation.writers.is_available("pillow"):
            assert saved, "a writer was available but nothing was saved"


class TestCalCube:
    def test_single_cuboid_faces(self):
        verts = calCube(np.zeros(3), np.array([2.0, 4.0, 6.0]))
        assert len(verts) == 6                     # six faces
        v = np.asarray(verts, dtype=float)
        assert v.shape == (6, 4, 3)                # quads in 3D
        # extents match the requested size
        flat = v.reshape(-1, 3)
        assert np.allclose(flat.max(0) - flat.min(0), [2.0, 4.0, 6.0])

    def test_batched_with_rotations(self):
        n = 5
        centres = np.arange(n * 3, dtype=float).reshape(n, 3)
        R = np.stack([np.eye(3)] * n)
        verts = calCube(centres, np.array([2.0, 2.0, 2.0]), R)
        assert len(verts) == n
        for i, f in enumerate(verts):
            assert np.asarray(f).shape == (6, 4, 3)
            assert np.allclose(np.asarray(f).reshape(-1, 3).mean(0), centres[i])
