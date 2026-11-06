function [image,residual,quality]=SART_surface(data,geo,surface,niter,varargin)
% Operator-aware SART; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('SART',data,geo,surface,niter,varargin{:});
end
