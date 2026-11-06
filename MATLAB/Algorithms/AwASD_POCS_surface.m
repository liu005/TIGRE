function [image,history]=AwASD_POCS_surface(data,geo,surface,niter,varargin)
%AWASD_POCS_SURFACE Surface wrapper; explicit MaxL2err is required.
[image,history]=surfacePocsRecon(data,geo,surface,niter,true,false,varargin{:});
end
