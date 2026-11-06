from pathlib import Path
import sys
import numpy as np
from scipy.io import savemat
import tigre
from tigre import algorithms as algs
from tigre.algorithms.operator_exports import NAMES

geo=tigre.geometry_default(high_resolution=False)
geo.nVoxel=np.array([8,8,8]); geo.dVoxel=np.ones(3); geo.sVoxel=geo.nVoxel*geo.dVoxel
geo.nDetector=np.array([12,14]); geo.sDetector=geo.nDetector*geo.dDetector
angles=np.linspace(0,2*np.pi,12,endpoint=False)
surface=tigre.surface_from_geometry(geo,angles)
mask=surface.valid_mask.copy(); mask[:,3,4]=False; mask[2]=False
surface=tigre.DetectorSurfaceGeometry(surface.source_positions_mm,surface.pixel_centers_mm,mask)
op=tigre.SurfaceOperator(geo,surface,gpuids=0)
x=np.zeros(op.volume_shape,np.float32); x[2:6,2:6,2:6]=1; x[3,4,5]=1.7
b=op.forward(x); b[~mask]=np.nan
payload=dict(source=surface.source_positions_mm.T,pixels=surface.pixel_centers_mm.transpose(1,2,3,0),
             mask=mask.transpose(1,2,0),data=b.transpose(1,2,0),image=x.transpose(2,1,0),
             nVoxel=geo.nVoxel[::-1],dVoxel=geo.dVoxel[::-1],offOrigin=np.zeros(3))
shared={'cgls','sirt','os_sart','fast_os_sart','asd_pocs','os_asd_pocs','awasd_pocs','os_awasd_pocs','pcsd','os_pcsd','aw_pcsd','os_aw_pcsd'}
for name in NAMES:
    options=dict(maxl2err=0.,tviter=2,return_history=True) if 'pocs' in name or 'pcsd' in name else {}
    if name not in shared: options['verbose']=False
    if name.startswith('os_'): options['blocksize']=5
    if name=='fast_os_sart': options['blocksize']=5
    if 'aw' in name: options['delta']=[-.01,-.02,-.03]
    if name in ('sart_tv','os_sart_tv'): options.update(tviter=2,tvlambda=50.)
    if name in ('fista','ista'): options.update(tviter=2,tvlambda=.1,hyper=2000.)
    if name=='irn_tv_cgls': options['niter_outer']=1
    output=getattr(algs,name)(b,operator=op,niter=3,**options)
    if 'pocs' in name or 'pcsd' in name: result,history=output
    else:
        result=output
        history=dict(residual=[np.linalg.norm(op.forward(result)-np.nan_to_num(b))])
    payload[name]=result.transpose(2,1,0)
    payload[name+'_residual']=history['residual']
savemat(Path(sys.argv[1]) if len(sys.argv)>1 else Path('.codex-results/gui-split/pocs_reference.mat'),payload)
print('Saved Python POCS reference for MATLAB xyz/rows-columns-views comparison')
