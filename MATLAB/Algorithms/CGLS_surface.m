function [image,residual]=CGLS_surface(data,geo,surface,niter,varargin)
%CGLS_SURFACE Shared operator CGLS; legacy tolerance default is preserved.
p=inputParser; addParameter(p,'Device',0); addParameter(p,'Tolerance',1e-6); addParameter(p,'Init',[]);
parse(p,varargin{:}); a=p.Results;
op=projectionOperator(geo,surface,'Device',a.Device);
[image,residual]=CGLS_operator(data,op,niter,'Tolerance',a.Tolerance,'Init',a.Init);
end
