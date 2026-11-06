function [image,residual,quality]=OS_AwPCSD_operator(data,op,niter,varargin)
% Operator-aware OS_AwPCSD; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('OS_AwPCSD',data,op,niter,varargin{:});
end
