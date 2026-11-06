function [image,history]=OS_AwASD_POCS_operator(data,op,niter,varargin)
%OS_AWASD_POCS_OPERATOR Shared ordered-subset weighted ASD-POCS.
[~,image,history]=operatorPocsDispatch(data,niter,true,true,{'Operator',op,varargin{:}});
end
