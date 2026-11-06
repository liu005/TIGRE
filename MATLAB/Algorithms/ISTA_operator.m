function [image,residual,quality]=ISTA_operator(data,op,niter,varargin)
% Operator-aware ISTA; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('ISTA',data,op,niter,varargin{:});
end
