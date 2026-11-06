function [image,residual,quality]=OS_PCSD_operator(data,op,niter,varargin)
% Operator-aware OS_PCSD; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('OS_PCSD',data,op,niter,varargin{:});
end
