function [image,residual,quality]=FISTA_operator(data,op,niter,varargin)
% Operator-aware FISTA; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('FISTA',data,op,niter,varargin{:});
end
