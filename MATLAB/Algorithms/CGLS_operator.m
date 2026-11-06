function [image,residual]=CGLS_operator(data,operator,niter,varargin)
%CGLS_OPERATOR CGLS with a conventional or surface projection operator.
[image,residual]=iterativeOperatorRecon(data,operator,niter,'cgls',varargin{:});
end
