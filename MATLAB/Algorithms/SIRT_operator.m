function [image,residual]=SIRT_operator(data,operator,niter,varargin)
%SIRT_OPERATOR Operator-normalized simultaneous reconstruction.
[image,residual]=iterativeOperatorRecon(data,operator,niter,'sirt',varargin{:});
end
