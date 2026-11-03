"""NumPy execution and lazy access to the optional CuPy implementation."""

from contextlib import nullcontext

import numpy as np
from scipy.fft import fft, ifft


class NumpyBackend:
    xp = np
    fft = staticmethod(fft)
    ifft = staticmethod(ifft)
    is_gpu = False

    def device_context(self):
        return nullcontext()

    def to_numpy(self, array):
        return np.asarray(array)

    def default_chunk_size(self, n, rows, columns):
        return n

    def release_cached_memory(self):
        pass


def get_backend(name="numpy", array=None):
    """Select execution without probing CUDA or importing CuPy for host arrays.

    ``None`` preserves the array's backend for low-level array utilities.
    Reconstruction uses the explicit default ``numpy``; GPU filtering requires
    ``cupy``. This only selects filtering, not TIGRE's native backprojector.
    """
    device_array = hasattr(array, "__cuda_array_interface__")
    if name is None:
        name = "cupy" if device_array else "numpy"
    if name == "numpy":
        if device_array:
            raise TypeError("GPU projection arrays require filter_backend='cupy'; "
                            "convert them to NumPy for CPU filtering")
        return NumpyBackend()
    if name == "cupy":
        from tigre.utilities._cupy_backend import CupyBackend

        return CupyBackend(array)
    raise ValueError("filter backend must be 'numpy' or 'cupy'")
