"""computeCOR: does it run, does it recover a known centre of rotation.

Before 2026-08-24 it did neither - both code paths raised on the first call,
which is why nothing but the export in tigre/__init__.py referenced it:

    gpu=True   TypeError: cp.squeeze() applied to the caller's NumPy array
               before the branch that converts it
    gpu=False  ValueError: (n_det,) against (n_angles,) - geo.DSD is per-angle
               after check_geo, and the fan-angle formula wants a scalar

and the interpolation call was not cv2.remap's signature at all. See the
module docstring for the comparison against MATLAB/Utilities/computeCOR.m.
"""
import numpy as np
import pytest

import tigre

TOL_MM = 0.2


def _problem(cor_mm, nang=120):
    geo = tigre.geometry(mode="cone", nVoxel=np.array([128, 128, 128]), default=True)
    angles = np.linspace(0, 2 * np.pi, nang, endpoint=False, dtype=np.float32)
    geo.check_geo(angles)
    geo.COR = np.ones(nang, dtype=np.float32) * np.float32(cor_mm)
    z, y, x = np.mgrid[:128, :128, :128].astype(np.float32)
    ph = np.zeros((128, 128, 128), dtype=np.float32)
    ph[((x - 70) ** 2 + (y - 60) ** 2 + (z - 64) ** 2) < 26 ** 2] = 1.0
    ph[((x - 50) ** 2 + (y - 74) ** 2 + (z - 64) ** 2) < 11 ** 2] = 2.0
    return geo, angles, tigre.Ax(ph, geo, angles, "Siddon")


@pytest.mark.parametrize("true_cor", [0.0, 0.5, -0.5, 1.5, -2.0])
def test_recovers_a_known_cor(true_cor):
    geo, angles, proj = _problem(true_cor)
    est = float(tigre.computeCOR(proj, geo, angles, gpu=True))
    assert abs(est - true_cor) < TOL_MM, f"true {true_cor}, estimated {est:.4f}"


def test_cpu_and_gpu_agree():
    geo, angles, proj = _problem(0.75)
    a = float(tigre.computeCOR(proj, geo, angles, gpu=True))
    b = float(tigre.computeCOR(proj, geo, angles, gpu=False))
    assert a == pytest.approx(b, abs=1e-6)


def test_runs_without_cupy(monkeypatch):
    """This module is imported by tigre/__init__.py, so a module-scope
    `import cupy` made CuPy a hard dependency of `import tigre`. Asking for the
    GPU where there is none must degrade, not raise."""
    import builtins
    real_import = builtins.__import__

    def no_cupy(name, *args, **kwargs):
        if name.split(".")[0] == "cupy" or name.startswith("cupyx"):
            raise ImportError("simulated: no CuPy on this machine")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cupy)
    geo, angles, proj = _problem(0.5)
    est = float(tigre.computeCOR(proj, geo, angles, gpu=True))
    assert abs(est - 0.5) < TOL_MM
