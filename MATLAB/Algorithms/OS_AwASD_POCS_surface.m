function [image,history]=OS_AwASD_POCS_surface(data,geo,surface,niter,varargin)
%OS_AWASD_POCS_SURFACE Surface wrapper; explicit MaxL2err is required.
[image,history]=surfacePocsRecon(data,geo,surface,niter,true,true,varargin{:});
end
