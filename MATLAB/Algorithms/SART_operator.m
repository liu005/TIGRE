function [image,residual,quality]=SART_operator(data,op,niter,varargin)
% Operator-aware SART; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('SART',data,op,niter,varargin{:});
end
