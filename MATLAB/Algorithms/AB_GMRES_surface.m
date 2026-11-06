function [image,residual,quality]=AB_GMRES_surface(data,geo,surface,niter,varargin)
% Operator-aware AB_GMRES; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('AB_GMRES',data,geo,surface,niter,varargin{:});
end
