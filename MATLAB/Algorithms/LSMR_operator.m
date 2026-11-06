function [image,residual,quality]=LSMR_operator(data,op,niter,varargin)
% Operator-aware LSMR; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('LSMR',data,op,niter,varargin{:});
end
