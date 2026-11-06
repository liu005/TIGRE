function [image,residual,quality]=LSQR_operator(data,op,niter,varargin)
% Operator-aware LSQR; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('LSQR',data,op,niter,varargin{:});
end
