function [image,residual,quality]=LSMR_surface(data,geo,surface,niter,varargin)
% Operator-aware LSMR; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('LSMR',data,geo,surface,niter,varargin{:});
end
