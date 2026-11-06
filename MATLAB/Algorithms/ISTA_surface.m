function [image,residual,quality]=ISTA_surface(data,geo,surface,niter,varargin)
% Operator-aware ISTA; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('ISTA',data,geo,surface,niter,varargin{:});
end
