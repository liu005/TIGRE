function [image,residual]=fast_OS_SART_operator(data,operator,niter,varargin)
%FAST_OS_SART_OPERATOR Ordered subsets with heuristic Nesterov momentum.
[image,residual]=iterativeOperatorRecon(data,operator,niter,'fast_os_sart',varargin{:});
end
