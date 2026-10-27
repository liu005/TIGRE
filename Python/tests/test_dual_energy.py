"""CPU tests for the MATLAB-compatible dual-energy transfer functions."""
import json
from pathlib import Path
import runpy
import os
import subprocess
import sys
import numpy as np
import pytest
from scipy.io import savemat
from tigre.utilities.dual_energy import (
    decompose_dual_energy, make_vm_projections, interpolate_projection,
)


def test_linear_interpolation_uses_correct_weights_and_accepts_descending_angles():
    scan = np.array([[[2, 4]], [[6, 12]]], dtype=np.float32)
    np.testing.assert_allclose(interpolate_projection(scan, [0, 2], .5, "linear"), [[3, 6]])
    np.testing.assert_allclose(interpolate_projection(scan[::-1], [2, 0], .5, "linear"), [[3, 6]])
    np.testing.assert_array_equal(interpolate_projection(scan, [0, 2], -1, "linear"), scan[0])
    np.testing.assert_array_equal(interpolate_projection(scan, [0, 2], 3, "linear"), scan[1])
    assert interpolate_projection(scan, [0, 2], 1, "skip", .1) is None
    assert interpolate_projection(scan, [0, 2], -.5, "skip", .1) is None


def test_transfer_polynomial_uses_shared_matlab_coefficients():
    low = np.ones((2, 1, 3), np.float32)
    high = np.full_like(low, 2)
    first, second, angles = decompose_dual_energy(low, [0, 1], high, [0, 1])
    data = json.loads((Path(__file__).resolve().parents[2] / "Common/data/dual_energy_calibrations.json").read_text())
    terms = np.array([1, 2, 1, 2, 4, 1, 2, 4, 8])
    for result, material in ((first, "Al"), (second, "PMMA")):
        expected = np.dot(data["VarianTrueBeam"]["V_80_140"][material], terms)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        assert result.dtype == np.float32
    np.testing.assert_array_equal(angles, [0, 1])


def test_skip_truncates_both_outputs_and_angles_including_empty_result():
    low = np.ones((2, 2, 3), np.float32)
    high = np.ones((3, 2, 3), np.float32)
    first, second, angles = decompose_dual_energy(low, [0, 1], high, [0, .5, 1],
                                                interpolation="skip", tolerance=.01)
    assert first.shape == second.shape == (2, 2, 3)
    np.testing.assert_array_equal(angles, [0, 1])
    first, second, angles = decompose_dual_energy(low, [4, 5], high, [0, .5, 1],
                                                interpolation="skip", tolerance=.01)
    assert first.shape == second.shape == (0, 2, 3) and angles.size == 0


def test_vm_uses_shared_attenuation_values_and_corrected_100kev_grid():
    first, second = np.full((2, 3, 4), 2.0), np.full((2, 3, 4), 3.0)
    np.testing.assert_allclose(make_vm_projections(first, second, 60), .075*2+.0228*3)
    np.testing.assert_allclose(make_vm_projections(first, second, 100), .046*2+.0194*3)
    with pytest.warns(RuntimeWarning):
        make_vm_projections(first, second, 200)


@pytest.mark.parametrize("kwargs", [dict(scanner="unknown"), dict(kv_low=81),
                                     dict(material1="Water"), dict(tolerance=-1),
                                     dict(interpolation="cubic")])
def test_bad_calibration_or_matching_options_raise(kwargs):
    with pytest.raises(ValueError):
        decompose_dual_energy(np.ones((2, 1, 1)), [0, 1], np.ones((2, 1, 1)), [0, 1], **kwargs)


def test_invalid_shapes_nonfinite_and_duplicate_linear_angles_raise():
    with pytest.raises(ValueError):
        decompose_dual_energy(np.ones((2, 2, 3)), [0], np.ones((2, 2, 3)), [0, 1])
    with pytest.raises(ValueError):
        interpolate_projection(np.full((2, 1, 1), np.nan), [0, 1], 0)
    with pytest.raises(ValueError):
        interpolate_projection(np.ones((2, 1, 1)), [0, 0], .1, "linear")
    with pytest.raises(ValueError):
        make_vm_projections(np.ones(3), np.ones(4), 60)


def test_demo_loads_actual_matlab_array_and_geometry_conventions(tmp_path):
    demo = runpy.run_path(str(Path(__file__).resolve().parents[1] / "demos/d23_DualEnergy.py"))
    geo = dict(nVoxel=[3, 4, 5], sVoxel=[6, 8, 10], dVoxel=[2, 2, 2],
               nDetector=[3, 2], sDetector=[6, 8], dDetector=[2, 4],
               DSO=100, DSD=150, offOrigin=[1, 2, 3], offDetector=[4, 5],
               rotDetector=[.1, .2, .3], COR=0, accuracy=.5, mode="cone")
    low = np.arange(24, dtype=np.float32).reshape(2, 3, 4)/100
    path = tmp_path / "d23.mat"
    savemat(path, dict(proj_l=low, proj_h=low*.8, angles_l=np.arange(4),
                      angles_h=np.arange(4), geo=geo))
    data = demo["load_matlab_input"](path)
    np.testing.assert_array_equal(data["proj_l"], low.transpose(2, 0, 1))
    np.testing.assert_array_equal(data["geo"].nVoxel, [5, 4, 3])
    np.testing.assert_array_equal(data["geo"].nDetector, [2, 3])
    np.testing.assert_array_equal(data["geo"].offOrigin, [3, 2, 1])
    np.testing.assert_array_equal(data["geo"].offDetector, [5, 4])
    np.testing.assert_allclose(data["geo"].rotDetector, [.1, .2, .3])
    result = demo["process"](data)
    assert result["VM_proj"].shape == (4, 2, 3)


def test_demo_synthetic_cli_runs_without_reconstruction(tmp_path):
    demo = runpy.run_path(str(Path(__file__).resolve().parents[1] / "demos/d23_DualEnergy.py"))
    output = tmp_path / "result.npz"
    result = demo["main"](["--synthetic", "--output", str(output)])
    assert output.is_file() and np.isfinite(result["VM_proj"]).all()


def test_demo_direct_launch_uses_checkout_without_pythonpath(tmp_path):
    demo = Path(__file__).resolve().parents[1] / "demos/d23_DualEnergy.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(demo), "--synthetic"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Processed 6 views" in result.stdout
