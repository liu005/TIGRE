function [image,history]=OS_ASD_POCS_surface(data,geo,surface,niter,varargin)
%OS_ASD_POCS_SURFACE Surface wrapper; explicit MaxL2err is required.
[image,history]=surfacePocsRecon(data,geo,surface,niter,false,true,varargin{:});
end
