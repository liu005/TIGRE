function [image,residual,quality]=iterativereconalg_surface(data,geo,surface,niter,varargin)
% Operator-aware iterativereconalg; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('iterativereconalg',data,geo,surface,niter,varargin{:});
end
