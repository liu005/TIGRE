function [image,residual,quality]=MLEM_operator(data,op,niter,varargin)
% Operator-aware MLEM; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('MLEM',data,op,niter,varargin{:});
end
