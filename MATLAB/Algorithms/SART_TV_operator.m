function [image,residual,quality]=SART_TV_operator(data,op,niter,varargin)
% Operator-aware SART_TV; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('SART_TV',data,op,niter,varargin{:});
end
