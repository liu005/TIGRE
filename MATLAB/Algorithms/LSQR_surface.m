function [image,residual,quality]=LSQR_surface(data,geo,surface,niter,varargin)
% Operator-aware LSQR; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('LSQR',data,geo,surface,niter,varargin{:});
end
