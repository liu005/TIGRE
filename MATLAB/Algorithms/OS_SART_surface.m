function [image,residual]=OS_SART_surface(data,geo,surface,niter,varargin)
%OS_SART_SURFACE Operator OS-SART; Device default 0, BlockSize default 20.
[image,residual]=surfaceIterativeRecon(data,geo,surface,niter,'os_sart',varargin{:});
end
