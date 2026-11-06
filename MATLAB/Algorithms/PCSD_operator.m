function [image,history]=PCSD_operator(data,op,niter,varargin)
%PCSD_OPERATOR Absolute L2 tolerance; finite residual-ratio TV steps.
[~,image,history]=operatorPocsDispatch(data,niter,false,false,{'Operator',op,varargin{:}},true);
end
