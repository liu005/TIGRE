function [image,residual,quality]=OS_SART_TV_operator(data,op,niter,varargin)
% Operator-aware OS_SART_TV; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('OS_SART_TV',data,op,niter,varargin{:});
end
