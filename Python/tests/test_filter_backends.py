"""Optional acceleration must not change FDK weighting or import requirements."""

import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tigre

from tigre.utilities.filtering import (
    cosine_weight, fdk_weight_filter_chunked, filter, filtering, nextpow2, ramp_flat,
)


def inputs():
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    geo = tigre.geometry_default(high_resolution=False)
    geo.nDetector = np.array([3, 7])
    geo.dDetector = np.array([0.8, 1.1])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.rotDetector = np.array([0.03, -0.04, 0.02])
    geo.offDetector = np.array([0.2, -0.3])
    geo.check_geo(angles)
    geo.filter = "hann"
    proj = np.random.default_rng(19).random((5, 3, 7), dtype=np.float32)
    return proj, geo, angles


def test_import_and_filter_without_cupy():
    # A fresh process detects imports hidden by the parent's module cache.
    script = '''
import builtins
import copy
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('cupy', 'cupyx'):
        raise AssertionError('CPU execution attempted to import ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import tigre
import tigre.algorithms as algorithms
from tigre.utilities.filtering import filtering, fdk_weight_filter_chunked
from tigre.utilities.redundancy_weighting import zeropadding
import numpy as np
geo = tigre.geometry_default(high_resolution=False)
geo.nDetector = np.array([3, 7])
geo.sDetector = geo.nDetector * geo.dDetector
angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
geo.check_geo(angles)
geo.filter = None
proj = np.ones((5, 3, 7), dtype=np.float32)
assert np.isfinite(filtering(proj.copy(), geo, angles, False)).all()
assert np.isfinite(fdk_weight_filter_chunked(proj.copy(), geo, angles)).all()
assert zeropadding(proj, copy.deepcopy(geo))[0].shape[0] == 5
import tigre.algorithms.single_pass_algorithms as single
single.Atb = lambda data, *args, **kwargs: data
assert np.isfinite(algorithms.fdk(proj.copy(), geo, angles)).all()
geo.mode = 'parallel'
assert np.isfinite(single.fbp(proj.copy(), geo, angles)).all()
import sys
assert 'tigre.utilities._cupy_backend' not in sys.modules
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("pads", [(0, 0), (2, 1)])
def test_numpy_filter_matches_independent_per_view_fft(pads):
    from scipy.spatial.transform import Rotation

    proj, geo, angles = inputs()
    left, right = pads
    geo.nDetector[1] += left + right
    geo.sDetector = geo.nDetector * geo.dDetector
    wang = np.linspace(0.5, 1.5, 21, dtype=np.float32).reshape(3, 7)
    actual = fdk_weight_filter_chunked(
        proj.copy(), geo, angles, pad_left=left, pad_right=right, wang=wang,
        legacy_scale=True, chunk_nangles=2)

    width = int(geo.nDetector[1])
    length = max(64, 2 ** nextpow2(2 * max(geo.nDetector)))
    ramp = filter(geo.filter, ramp_flat(length)[0], length, 1)
    offset = (length - width) // 2
    scale = (geo.DSD[0] / geo.DSO[0]) * (2 * np.pi / len(angles)) / (4 * geo.dDetector[0])
    expected = []
    for i, view in enumerate(proj):
        weighted = np.pad(view * wang, ((0, 0), (left, right)))
        u = (np.arange(width) - width / 2 + 0.5) * geo.dDetector[1] + geo.offDetector[i, 1]
        v = (np.arange(3) - 1) * geo.dDetector[0] + geo.offDetector[i, 0]
        uu, vv = np.meshgrid(u, v)
        rotation = Rotation.from_euler("ZYX", geo.rotDetector[i, [2, 1, 0]]).as_matrix()
        weighted *= cosine_weight(uu, vv, geo.DSD[i], rotation)
        padded = np.pad(weighted, ((0, 0), (offset, length - width - offset)))
        filtered = np.fft.ifft(np.fft.fft(padded, axis=1) * ramp, axis=1).real
        expected.append(filtered[:, offset:offset + width] * scale)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous


def test_explicit_cupy_missing_has_actionable_error(monkeypatch):
    import builtins

    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in ("cupy", "cupyx"):
            raise ImportError("blocked for test")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    proj, geo, angles = inputs()
    with pytest.raises(ImportError, match="filter_backend='numpy'"):
        fdk_weight_filter_chunked(proj, geo, angles, backend="cupy")


def test_invalid_backend_rejected():
    proj, geo, angles = inputs()
    with pytest.raises(ValueError, match="backend"):
        fdk_weight_filter_chunked(proj, geo, angles, backend="unknown")


def test_numpy_chunked_filter_preserves_host_projections():
    proj, geo, angles = inputs()
    original = proj.copy()
    fdk_weight_filter_chunked(proj, geo, angles, chunk_nangles=2)
    np.testing.assert_array_equal(proj, original)


@pytest.mark.parametrize("device_input", [False, True])
@pytest.mark.parametrize("pads", [(0, 0), (2, 1)])
def test_cupy_matches_numpy(device_input, pads):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device for CuPy parity")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    proj, geo, angles = inputs()
    left, right = pads
    geo.nDetector[1] += left + right
    geo.sDetector = geo.nDetector * geo.dDetector
    options = dict(pad_left=left, pad_right=right, chunk_nangles=2,
                   wang=np.full((3, 7), 0.75, dtype=np.float32))
    expected = fdk_weight_filter_chunked(proj.copy(), copy.deepcopy(geo), angles, **options)
    source = cp.asarray(proj) if device_input else proj.copy()
    actual = fdk_weight_filter_chunked(source, geo, angles, backend="cupy", **options)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-7)


def test_cupy_fbp_filter_accepts_host_input():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device for CuPy parity")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    proj, geo, angles = inputs()
    expected = filtering(proj.copy(), geo, angles, False, backend="numpy")
    actual = filtering(proj.copy(), geo, angles, False, backend="cupy")
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-7)
