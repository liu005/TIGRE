"""Uniform public operator/surface conveniences without copying recurrences."""

NAMES = (
    "sart",
    "sirt",
    "os_sart",
    "fast_os_sart",
    "sart_tv",
    "os_sart_tv",
    "iterativereconalg",
    "cgls",
    "lsqr",
    "hybrid_lsqr",
    "lsmr",
    "irn_tv_cgls",
    "hybrid_flsqr_tv",
    "ab_gmres",
    "ba_gmres",
    "fista",
    "ista",
    "mlem",
    "asd_pocs",
    "os_asd_pocs",
    "awasd_pocs",
    "os_awasd_pocs",
    "pcsd",
    "aw_pcsd",
    "os_pcsd",
    "os_aw_pcsd",
)


def _surface(name, algorithm):
    def wrapped(proj, geo, surface, niter=20, init=None, gpuids=None, *, n_slabs=None, **kwargs):
        from tigre.utilities.projection_operator import SurfaceOperator

        return algorithm(
            proj,
            operator=SurfaceOperator(geo, surface, gpuids, n_slabs),
            niter=niter,
            init=init,
            **kwargs,
        )

    wrapped.__name__ = name + "_surface"
    wrapped.__module__ = "tigre.algorithms.surface_algorithms"
    wrapped.__doc__ = f"Surface convenience for {name}; uses the same operator-aware algorithm."
    return wrapped


def _operator(name, algorithm):
    def wrapped(proj, operator, niter=20, **kwargs):
        return algorithm(proj, operator=operator, niter=niter, **kwargs)

    wrapped.__name__ = name + "_operator"
    wrapped.__doc__ = f"Explicit projection-operator entry point for {name}."
    return wrapped


def install(namespace):
    from . import surface_algorithms

    for name in NAMES:
        algorithm = namespace[name]
        for suffix, factory in [("_surface", _surface), ("_operator", _operator)]:
            key = name + suffix
            if key not in namespace:
                namespace[key] = factory(name, algorithm)
            if key not in namespace["__all__"]:
                namespace["__all__"].append(key)
            if suffix == "_surface":
                setattr(surface_algorithms, key, namespace[key])
            else:
                globals()[key] = namespace[key]
