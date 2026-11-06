function [image,residual,quality]=MLEM_surface(data,geo,surface,niter,varargin)
% Operator-aware MLEM; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('MLEM',data,geo,surface,niter,varargin{:});
end
