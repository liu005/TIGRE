function [image,residual,quality]=FISTA_surface(data,geo,surface,niter,varargin)
% Operator-aware FISTA; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('FISTA',data,geo,surface,niter,varargin{:});
end
