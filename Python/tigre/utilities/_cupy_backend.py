"""Optional CuPy implementation; imported only when GPU work is requested."""


class CupyBackend:
    is_gpu = True

    def __init__(self, array=None):
        try:
            import cupy
            from cupyx.scipy.fft import fft, ifft
        except ImportError as exc:
            raise ImportError(
                "CuPy filtering was requested but CuPy is unavailable. Install "
                "CuPy for your CUDA version or use filter_backend='numpy'."
            ) from exc
        self.xp = cupy
        self.fft = fft
        self.ifft = ifft
        self.device = (array.device if isinstance(array, cupy.ndarray)
                       else cupy.cuda.Device())

    def device_context(self):
        return self.device

    def to_numpy(self, array):
        return self.xp.asnumpy(array)

    def default_chunk_size(self, n, rows, columns):
        free_bytes, _ = self.xp.cuda.runtime.memGetInfo()
        return int(free_bytes * 0.4 / (rows * columns * 4 * 3))

    def release_cached_memory(self):
        self.xp.get_default_memory_pool().free_all_blocks()


def cor_interpolator():
    """Keep the existing computeCOR GPU option separate from CPU imports."""
    import cupy
    from cupyx.scipy.interpolate import RegularGridInterpolator

    return cupy, RegularGridInterpolator
