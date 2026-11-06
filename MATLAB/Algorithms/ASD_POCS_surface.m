function [image,history]=ASD_POCS_surface(data,geo,surface,niter,varargin)
%ASD_POCS_SURFACE Surface wrapper; explicit MaxL2err is required.
[image,history]=surfacePocsRecon(data,geo,surface,niter,false,false,varargin{:});
end
