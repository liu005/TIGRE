function [image,residual,quality]=OS_PCSD_surface(data,geo,surface,niter,varargin)
% Operator-aware OS_PCSD; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('OS_PCSD',data,geo,surface,niter,varargin{:});
end
