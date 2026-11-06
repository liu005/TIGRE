function [image,residual,quality]=SART_TV_surface(data,geo,surface,niter,varargin)
% Operator-aware SART_TV; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('SART_TV',data,geo,surface,niter,varargin{:});
end
