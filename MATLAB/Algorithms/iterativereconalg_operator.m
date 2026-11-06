function [image,residual,quality]=iterativereconalg_operator(data,op,niter,varargin)
% Operator-aware iterativereconalg; conventional calls are unchanged.
[image,residual,quality]=allOperatorRecon('iterativereconalg',data,op,niter,varargin{:});
end
