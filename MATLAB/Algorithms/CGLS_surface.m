function [image,residual]=CGLS_surface(projections,geo,surface,niter,varargin)
%CGLS_SURFACE Native least-squares reconstruction for arbitrary detector surfaces.
% Optional 'Device', zero-based CUDA index; 'Tolerance', relative residual
% (default 1e-6). Histories contain completed iterations only.
opt=inputParser; addParameter(opt,'Device',0); addParameter(opt,'Tolerance',1e-6);
parse(opt,varargin{:}); device=opt.Results.Device; tol=opt.Results.Tolerance;
validateattributes(niter,{'numeric'},{'scalar','integer','nonnegative','finite'});
validateattributes(tol,{'numeric'},{'scalar','nonnegative','finite'});
image=zeros(double(geo.nVoxel(:))','single');
% Validate data/geometry through the adjoint before constructing residuals.
s=Atb_surface(projections,geo,surface,device);
r=single(projections);
if isfield(surface,'valid_mask'), r=r.*single(surface.valid_mask); end
p=s; gamma=sum(double(s(:)).^2); initial=norm(double(r(:)));
residual=zeros(1,niter); completed=0;
for k=1:niter
    if gamma==0 || norm(double(r(:)))<=tol*initial, break; end
    q=Ax_surface(p,geo,surface,device); delta=sum(double(q(:)).^2);
    if delta==0, break; end
    alpha=gamma/delta; image=image+single(alpha)*p; r=r-single(alpha)*q;
    completed=k; residual(k)=norm(double(r(:)));
    s=Atb_surface(r,geo,surface,device); next=sum(double(s(:)).^2);
    p=s+single(next/gamma)*p; gamma=next;
end
residual=residual(1:completed);
end
