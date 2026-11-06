function [image,residual,quality]=IRN_TV_CGLS_operator(data,op,niter,varargin)
% Operator-aware IRN_TV_CGLS; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('IRN_TV_CGLS',data,op,niter,varargin{:});
end
