function [image,residual,quality]=hybrid_fLSQR_TV_operator(data,op,niter,varargin)
% Operator-aware hybrid_fLSQR_TV; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('hybrid_fLSQR_TV',data,op,niter,varargin{:});
end
