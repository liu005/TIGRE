function [image,residual,quality]=hybrid_LSQR_operator(data,op,niter,varargin)
% Operator-aware hybrid_LSQR; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('hybrid_LSQR',data,op,niter,varargin{:});
end
