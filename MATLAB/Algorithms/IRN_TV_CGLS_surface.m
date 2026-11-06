function [image,residual,quality]=IRN_TV_CGLS_surface(data,geo,surface,niter,varargin)
% Operator-aware IRN_TV_CGLS; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('IRN_TV_CGLS',data,geo,surface,niter,varargin{:});
end
