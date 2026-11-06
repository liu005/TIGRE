"""Convenience wrappers over the shared projection-operator solvers."""

from tigre.utilities.projection_operator import SurfaceOperator
from .operator_algorithms import (
    cgls_operator,
    sirt_operator,
    os_sart_operator,
)
from .operator_pocs import (
    asd_pocs_operator,
    os_asd_pocs_operator,
    awasd_pocs_operator,
    os_awasd_pocs_operator,
    pcsd_operator,
)


def cgls_surface(
    proj, geo, surface, niter=20, init=None, gpuids=None, return_residuals=False, *, n_slabs=None
):
    """CGLS with the exact surface Siddon adjoint pair."""
    return cgls_operator(
        proj,
        SurfaceOperator(geo, surface, gpuids, n_slabs),
        niter,
        init=init,
        return_residuals=return_residuals,
    )


def sirt_surface(
    proj, geo, surface, niter=50, init=None, lmbda=1.0, gpuids=None, nonneg=False, *, n_slabs=None
):
    """SIRT with masked surface row/column normalization."""
    return sirt_operator(
        proj,
        SurfaceOperator(geo, surface, gpuids, n_slabs),
        niter,
        init=init,
        lmbda=lmbda,
        nonneg=nonneg,
    )


def os_sart_surface(
    proj,
    geo,
    surface,
    niter=10,
    blocksize=20,
    init=None,
    lmbda=1.0,
    nonneg=False,
    momentum=False,
    gpuids=None,
    *,
    n_slabs=None
):
    """Interleaved surface OS-SART, optionally with Nesterov momentum."""
    return os_sart_operator(
        proj,
        SurfaceOperator(geo, surface, gpuids, n_slabs),
        niter,
        blocksize=blocksize,
        init=init,
        lmbda=lmbda,
        nonneg=nonneg,
        momentum=momentum,
    )


def fast_os_sart_surface(proj, geo, surface, niter=10, **kwargs):
    """OS-SART with heuristic Nesterov acceleration."""
    return os_sart_surface(proj, geo, surface, niter, momentum=True, **kwargs)


def asd_pocs_surface(
    proj, geo, surface, niter=20, init=None, gpuids=None, *, maxl2err, n_slabs=None, **kwargs
):
    """Surface ASD-POCS; maxl2err is an explicit absolute L2 tolerance."""
    return asd_pocs_operator(
        proj,
        SurfaceOperator(geo, surface, gpuids, n_slabs),
        niter,
        init=init,
        maxl2err=maxl2err,
        **kwargs
    )


def os_asd_pocs_surface(
    proj,
    geo,
    surface,
    niter=20,
    init=None,
    gpuids=None,
    *,
    maxl2err,
    blocksize=20,
    n_slabs=None,
    **kwargs
):
    """Surface OS-ASD-POCS with independently normalized subsets."""
    return os_asd_pocs_operator(
        proj,
        SurfaceOperator(geo, surface, gpuids, n_slabs),
        niter,
        init=init,
        maxl2err=maxl2err,
        blocksize=blocksize,
        **kwargs
    )


def awasd_pocs_surface(
    proj,
    geo,
    surface,
    niter=20,
    init=None,
    gpuids=None,
    *,
    maxl2err,
    delta=-0.005,
    n_slabs=None,
    **kwargs
):
    """Surface AwASD-POCS with scalar or directional weighted TV."""
    return awasd_pocs_operator(
        proj,
        SurfaceOperator(geo, surface, gpuids, n_slabs),
        niter,
        init=init,
        maxl2err=maxl2err,
        delta=delta,
        **kwargs
    )


def os_awasd_pocs_surface(
    proj,
    geo,
    surface,
    niter=20,
    init=None,
    gpuids=None,
    *,
    maxl2err,
    blocksize=20,
    delta=-0.005,
    n_slabs=None,
    **kwargs
):
    """Surface OS-AwASD-POCS with scalar or directional weighted TV."""
    return os_awasd_pocs_operator(
        proj,
        SurfaceOperator(geo, surface, gpuids, n_slabs),
        niter,
        init=init,
        maxl2err=maxl2err,
        blocksize=blocksize,
        delta=delta,
        **kwargs
    )


def pcsd_surface(
    proj, geo, surface, niter=20, init=None, gpuids=None, *, maxl2err, n_slabs=None, **kwargs
):
    """Surface PCSD with an explicit absolute L2 data tolerance."""
    return pcsd_operator(
        proj,
        SurfaceOperator(geo, surface, gpuids, n_slabs),
        niter,
        init=init,
        maxl2err=maxl2err,
        **kwargs
    )
