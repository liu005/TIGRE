function [image,residual,quality]=hybrid_fLSQR_TV_surface(data,geo,surface,niter,varargin)
% Operator-aware hybrid_fLSQR_TV; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('hybrid_fLSQR_TV',data,geo,surface,niter,varargin{:});
end
