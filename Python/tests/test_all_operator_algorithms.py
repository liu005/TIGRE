"""Native reconstruction smoke checks for every public iterative method."""
import numpy as np
import pytest
import tigre
from tigre import algorithms as algs
from test_projection_operator import ITERATIVE_NAMES, geometry


@pytest.fixture(scope='module')
def acquisition():
    geo=geometry(); geo.nVoxel=np.array([8,8,8]); geo.dVoxel=np.ones(3); geo.sVoxel=geo.nVoxel.astype(float)
    geo.nDetector=np.array([12,14]); geo.sDetector=geo.nDetector*geo.dDetector
    surface=tigre.surface_from_geometry(geo,np.linspace(0,2*np.pi,12,endpoint=False))
    mask=surface.valid_mask.copy(); mask[:,3,4]=False; mask[2]=False
    surface=tigre.DetectorSurfaceGeometry(surface.source_positions_mm,surface.pixel_centers_mm,mask)
    operator=tigre.SurfaceOperator(geo,surface,gpuids=0)
    image=np.zeros(operator.volume_shape,np.float32); image[2:6,2:6,2:6]=1; image[3,4,5]=1.7
    data=operator.forward(image); data[~mask]=np.nan
    return geo,surface,operator,data


@pytest.mark.parametrize('name',ITERATIVE_NAMES)
def test_all_native_methods(name,acquisition):
    geo,surface,operator,data=acquisition
    options={}
    if name not in ('cgls','sirt','os_sart','fast_os_sart','asd_pocs','os_asd_pocs','awasd_pocs','os_awasd_pocs','pcsd','os_pcsd','aw_pcsd','os_aw_pcsd'):
        options.update(verbose=False)
    if 'pocs' in name or 'pcsd' in name: options.update(maxl2err=0.,tviter=2)
    if name in ('sart_tv','os_sart_tv'): options.update(tviter=2,tvlambda=50.)
    if name in ('fista','ista'): options.update(tviter=2,tvlambda=.1)
    if name=='irn_tv_cgls': options.update(niter_outer=1)
    result=getattr(algs,name)(data,operator=operator,niter=2,**options)
    if isinstance(result,tuple): result=result[0]
    assert result.shape==operator.volume_shape and result.dtype==np.float32
    assert np.isfinite(result).all()
    residual=operator.forward(result)-np.where(operator.valid_mask,data,0)
    assert np.linalg.norm(residual)<np.linalg.norm(np.nan_to_num(data))*1.1
    wrapped=getattr(algs,name+'_surface')(data,geo,surface,niter=2,gpuids=0,**options)
    if isinstance(wrapped,tuple): wrapped=wrapped[0]
    np.testing.assert_allclose(wrapped,result,rtol=2e-3,atol=2e-4)
