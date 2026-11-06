function [image,residual,quality]=OS_AwPCSD_surface(data,geo,surface,niter,varargin)
% Operator-aware OS_AwPCSD; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('OS_AwPCSD',data,geo,surface,niter,varargin{:});
end
