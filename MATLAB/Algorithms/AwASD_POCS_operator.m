function [image,history]=AwASD_POCS_operator(data,op,niter,varargin)
%AWASD_POCS_OPERATOR Single-view ASD-POCS with weighted volume TV.
[~,image,history]=operatorPocsDispatch(data,niter,true,false,{'Operator',op,varargin{:}});
end
