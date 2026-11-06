function [image,residual,quality]=AwPCSD_surface(data,geo,surface,niter,varargin)
% Operator-aware AwPCSD; conventional calls are unchanged.
[image,residual,quality]=allSurfaceRecon('AwPCSD',data,geo,surface,niter,varargin{:});
end
