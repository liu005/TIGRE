function [image,residual,quality]=hybrid_LSQR_surface(data,geo,surface,niter,varargin)
% Operator-aware hybrid_LSQR; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('hybrid_LSQR',data,geo,surface,niter,varargin{:});
end
