function [image,residual,quality]=BA_GMRES_operator(data,op,niter,varargin)
% Operator-aware BA_GMRES; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('BA_GMRES',data,op,niter,varargin{:});
end
