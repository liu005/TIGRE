function [image,residual]=fast_OS_SART_surface(data,geo,surface,niter,varargin)
%FAST_OS_SART_SURFACE Operator OS-SART with heuristic Nesterov momentum.
[image,residual]=surfaceIterativeRecon(data,geo,surface,niter,'fast_os_sart',varargin{:});
end
