function [image,residual,quality]=OS_SART_TV_surface(data,geo,surface,niter,varargin)
% Operator-aware OS_SART_TV; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('OS_SART_TV',data,geo,surface,niter,varargin{:});
end
