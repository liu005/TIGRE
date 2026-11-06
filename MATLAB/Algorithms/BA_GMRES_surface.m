function [image,residual,quality]=BA_GMRES_surface(data,geo,surface,niter,varargin)
% Operator-aware BA_GMRES; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('BA_GMRES',data,geo,surface,niter,varargin{:});
end
