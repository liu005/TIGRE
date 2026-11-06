function [image,residual,quality]=AwPCSD_operator(data,op,niter,varargin)
% Operator-aware AwPCSD; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('AwPCSD',data,op,niter,varargin{:});
end
