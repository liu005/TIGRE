function [image,residual]=SIRT_surface(data,geo,surface,niter,varargin)
%SIRT_SURFACE Shared operator SIRT; legacy defaults are preserved.
p=inputParser; addParameter(p,'Device',0); addParameter(p,'Tolerance',1e-6);
addParameter(p,'Relaxation',1); addParameter(p,'Nonnegative',true); addParameter(p,'Init',[]);
parse(p,varargin{:}); a=p.Results;
validateattributes(a.Relaxation,{'numeric'},{'scalar','positive','<',2,'finite'});
op=projectionOperator(geo,surface,'Device',a.Device);
[image,residual]=SIRT_operator(data,op,niter,'Tolerance',a.Tolerance,'Init',a.Init, ...
    'Relaxation',a.Relaxation,'Nonnegative',a.Nonnegative);
end
