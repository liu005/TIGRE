function [image,history]=OS_ASD_POCS_operator(data,op,niter,varargin)
%OS_ASD_POCS_OPERATOR Shared ASD-POCS with interleaved subsets.
[~,image,history]=operatorPocsDispatch(data,niter,false,true,{'Operator',op,varargin{:}});
end
