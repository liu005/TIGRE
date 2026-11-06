function [image,residual,quality]=AB_GMRES_operator(data,op,niter,varargin)
% Operator-aware AB_GMRES; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('AB_GMRES',data,op,niter,varargin{:});
end
