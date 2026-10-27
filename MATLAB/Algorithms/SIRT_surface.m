function [image,residual]=SIRT_surface(projections,geo,surface,niter,varargin)
%SIRT_SURFACE Row/column-normalized native surface reconstruction.
% Options: 'Device' (zero-based, default 0), 'Relaxation' (0<value<2,
% default 1), 'Nonnegative' (default true), 'Tolerance' (default 1e-6).
opt=inputParser; addParameter(opt,'Device',0); addParameter(opt,'Relaxation',1);
addParameter(opt,'Nonnegative',true); addParameter(opt,'Tolerance',1e-6);
parse(opt,varargin{:}); a=opt.Results;
validateattributes(niter,{'numeric'},{'scalar','integer','nonnegative','finite'});
validateattributes(a.Relaxation,{'numeric'},{'scalar','positive','<',2,'finite'});
validateattributes(a.Tolerance,{'numeric'},{'scalar','nonnegative','finite'});
validateattributes(a.Nonnegative,{'logical'},{'scalar'});
image=zeros(double(geo.nVoxel(:))','single');
% Validate projections and initialize sensitivity using the masked operator.
Atb_surface(projections,geo,surface,a.Device);
b=single(projections);
if isfield(surface,'valid_mask'), b=b.*single(surface.valid_mask); end
row=Ax_surface(ones(size(image),'single'),geo,surface,a.Device);
column=Atb_surface(ones(size(b),'single'),geo,surface,a.Device);
rowWeight=zeros(size(row),'single'); rowWeight(row>0)=1./row(row>0);
colWeight=zeros(size(column),'single'); colWeight(column>0)=1./column(column>0);
initial=norm(double(b(:))); residual=zeros(1,niter); completed=0;
r=b;
for k=1:niter
    if norm(double(r(:)))<=a.Tolerance*initial, break; end
    image=image+single(a.Relaxation)*colWeight.*Atb_surface(rowWeight.*r,geo,surface,a.Device);
    if a.Nonnegative, image=max(image,0); end
    r=b-Ax_surface(image,geo,surface,a.Device);
    completed=k; residual(k)=norm(double(r(:)));
end
residual=residual(1:completed);
end
