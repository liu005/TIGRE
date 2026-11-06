function [image,history]=surfacePocsRecon(data,geo,surface,niter,weighted,ordered,varargin)
% Internal shared factory for POCS surface convenience wrappers.
args=varargin; which=find(strcmpi(args(1:2:end),'Device'));
device=0;
assert(numel(which)<=1,'TIGRE:operator:options','Supply one Device');
if ~isempty(which), k=2*which-1; device=args{k+1}; args(k:k+1)=[]; end
op=projectionOperator(geo,surface,'Device',device);
[~,image,history]=operatorPocsDispatch(data,niter,weighted,ordered,{'Operator',op,args{:}});
end
