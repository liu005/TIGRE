"""A CUDA error must be reportable and must not leak.

Before this change a CUDA error inside voxel_backprojection() left through
mexErrMsgIdAndTxt(), which under IS_FOR_PYTIGRE called exit(1): the interpreter
was gone, so there was nothing to catch and no way to observe what had been
left behind. Under MATLAB the same call longjmps out, running no cleanup at
all. Either way the device buffers, page-locked host memory, host
registrations, streams and texture objects held at that moment stayed
allocated.

These tests pin the four properties the fix is for:

  1. the success path is unchanged;
  2. repeated calls do not drift GPU memory;
  3. a failure raises something catchable instead of ending the process;
  4. the process still works afterwards.

Free-memory readings come from the CUDA runtime via ctypes rather than a
third-party package, so this adds no dependency: TIGRE already links the CUDA
runtime, and its Python requirements are numpy/scipy/matplotlib/h5py/tqdm. If
the library cannot be located the memory test reports skipped rather than
failing for an unrelated reason.
"""

import ctypes
import unittest

import numpy as np

import tigre
from tigre.utilities.errors import TigreCudaCallError


def _load_cudart():
    """The already-linked CUDA runtime, or None if it cannot be located."""
    names = [
        # Linux
        "libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0",
        # Windows
        "cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", "cudart64_101.dll",
    ]
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


_CUDART = _load_cudart()


def gpu_free_bytes():
    """Free device memory, or None if the runtime is unavailable."""
    if _CUDART is None:
        return None
    free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
    if _CUDART.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
        return None
    return free.value


def make_geometry(nvoxel=128, ndetector=128):
    geo = tigre.geometry(mode="cone", default=True, nVoxel=np.array([nvoxel] * 3))
    geo.nDetector = np.array([ndetector, ndetector])
    geo.dDetector = np.array([0.8, 0.8])
    geo.sDetector = geo.nDetector * geo.dDetector
    return geo


class TestCudaErrorPropagation(unittest.TestCase):
    def setUp(self):
        self.geo = make_geometry()
        self.angles = np.linspace(0, 2 * np.pi, 32, dtype=np.float32)
        self.proj = np.ones(
            (len(self.angles), self.geo.nDetector[0], self.geo.nDetector[1]),
            dtype=np.float32)

    def test_backprojection_still_works(self):
        """The success path must be untouched by the error handling."""
        vol = tigre.Atb(self.proj, self.geo, self.angles)
        self.assertEqual(tuple(vol.shape), tuple(self.geo.nVoxel))
        self.assertTrue(np.all(np.isfinite(vol)))
        self.assertGreater(float(np.abs(vol).max()), 0.0)

    def test_repeated_calls_do_not_leak(self):
        """Each call page-locks the caller's projections and allocates device
        memory, streams and textures. A steady decline in free memory means one
        of those is not being handed back."""
        if gpu_free_bytes() is None:
            self.skipTest("CUDA runtime not loadable via ctypes")

        tigre.Atb(self.proj, self.geo, self.angles)   # absorb one-off context setup
        before = gpu_free_bytes()
        for _ in range(5):
            tigre.Atb(self.proj, self.geo, self.angles)
        after = gpu_free_bytes()

        drift_mb = (before - after) / 1e6
        self.assertLess(abs(drift_mb), 50.0,
                        "GPU memory drifted {:+.1f} MB over 5 calls".format(drift_mb))

    def test_failure_raises_instead_of_exiting(self):
        """Ask for a volume far larger than the card.

        Reaching the assertion at all is part of the result: with exit(1) the
        interpreter would have gone with the error."""
        free = gpu_free_bytes()
        if free is None:
            self.skipTest("CUDA runtime not loadable via ctypes")

        # Comfortably beyond the device, but still an expressible size.
        n = int(((free * 8) / 4) ** (1.0 / 3.0))
        geo = make_geometry(nvoxel=n, ndetector=256)
        angles = np.linspace(0, 2 * np.pi, 16, dtype=np.float32)
        proj = np.ones((len(angles), geo.nDetector[0], geo.nDetector[1]),
                       dtype=np.float32)

        with self.assertRaises((TigreCudaCallError, MemoryError)):
            tigre.Atb(proj, geo, angles)

    def test_usable_after_a_failure(self):
        """A failed call should cost that call, not the session - and a
        half-released state would show up here."""
        free = gpu_free_bytes()
        if free is not None:
            n = int(((free * 8) / 4) ** (1.0 / 3.0))
            geo = make_geometry(nvoxel=n, ndetector=256)
            angles = np.linspace(0, 2 * np.pi, 16, dtype=np.float32)
            proj = np.ones((len(angles), geo.nDetector[0], geo.nDetector[1]),
                           dtype=np.float32)
            try:
                tigre.Atb(proj, geo, angles)
            except Exception:                                     # noqa: BLE001
                pass

        vol = tigre.Atb(self.proj, self.geo, self.angles)
        self.assertTrue(np.all(np.isfinite(vol)))
        self.assertGreater(float(np.abs(vol).max()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
