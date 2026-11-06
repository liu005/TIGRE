function [image,residual]=OS_SART_operator(data,operator,niter,varargin)
%OS_SART_OPERATOR Interleaved operator-normalized ordered subsets.
[image,residual]=iterativeOperatorRecon(data,operator,niter,'os_sart',varargin{:});
end
