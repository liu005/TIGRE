function image=FDK_surface(projections,geo,surface,varargin)
%FDK_SURFACE Approximate equiangular FDK for circular extruded surfaces.
% Requires a circular source orbit about world z, constant source height,
% identical fan tables and whole-column gaps. Arbitrarily tilted panels
% must use CGLS_surface/SIRT_surface. Gaps are bridged by linear interpolation.
% Options: 'Device' (0), 'Filter' ('ram-lak' or 'hann'), 'Cutoff' (0,1].
% Full scans and sufficiently long Parker-weighted short scans are supported.
opt=inputParser; addParameter(opt,'Device',0); addParameter(opt,'Filter','ram-lak');
addParameter(opt,'Cutoff',1); parse(opt,varargin{:}); a=opt.Results;
validateattributes(a.Device,{'numeric'},{'scalar','integer','nonnegative','finite'});
validateattributes(a.Cutoff,{'numeric'},{'scalar','positive','<=',1,'finite'});
assert(any(strcmp(a.Filter,{'ram-lak','hann'})),'TIGRE:surface:filter','Use ram-lak or hann.');
validateattributes(projections,{'numeric'},{'real','finite','nonsparse'});
n=geo.nVoxel(:); d=geo.dVoxel(:); o=zeros(3,1);
if isfield(geo,'offOrigin'), o=geo.offOrigin; end
validateattributes(n,{'numeric'},{'numel',3,'integer','positive','finite'});
validateattributes(d,{'numeric'},{'numel',3,'positive','finite'});
validateattributes(o,{'numeric'},{'numel',3,'real','finite'});
s=double(surface.source_positions_mm); p=double(surface.pixel_centers_mm);
validateattributes(s,{'double'},{'2d','nrows',3,'finite','real'});
validateattributes(p,{'double'},{'finite','real'});
v=size(s,2); r=size(p,1); c=size(p,2);
assert(v>=2 && size(p,3)==3 && ndims(p)<=4 && (size(p,4)==1 || size(p,4)==v), ...
    'TIGRE:surface:shape','Invalid surface dimensions.');
if size(p,4)==1, p=repmat(p,1,1,1,v); end
m=true(r,c,v);
if isfield(surface,'valid_mask')
    m=surface.valid_mask;
    assert(islogical(m) && size(m,1)==r && size(m,2)==c && ...
        (size(m,3)==1 || size(m,3)==v) && ndims(m)<=3,'TIGRE:surface:mask','Invalid mask.');
    if size(m,3)==1, m=repmat(m,1,1,v); end
end
assert(size(projections,1)==r && size(projections,2)==c && size(projections,3)==v && ndims(projections)<=3, ...
    'TIGRE:surface:shape','Projections must be rows x columns x views.');
radius=hypot(s(1,:),s(2,:)); R=mean(radius);
require(min(radius)>0 && rangeLocal(radius)<1e-4*R && rangeLocal(s(3,:))<1e-3, ...
    'A circular source orbit at constant height is required.');
ok=reshape(all(all(m,1),3),1,c);
require(isequal(m,repmat(reshape(ok,1,c,1),r,1,v)), ...
    'Only whole-column gaps identical in every view are supported.');
cols=find(ok); require(numel(cols)>=2,'At least two valid columns are required.');
p=p(:,cols,:,:); nc=numel(cols);
tol=1e-5*R;
require(max(abs(reshape(p(:,:,1:2,:)-p(1,:,1:2,:),[],1)))<tol && ...
    max(abs(reshape(p(:,:,3,:)-p(:,1,3,1),[],1)))<tol, ...
    'Detector must be extruded along z; use an iterative solver for tilted panels.');
hx=reshape(p(1,:,1,:),nc,v)-s(1,:);
hy=reshape(p(1,:,2,:),nc,v)-s(2,:);
cx=-s(1,:)./radius; cy=-s(2,:)./radius;
g=atan2(cx.*hy-cy.*hx,cx.*hx+cy.*hy); rho=hypot(hx,hy);
require(max(abs(reshape(g-g(:,1),[],1)))<2e-5 && ...
    max(abs(reshape(rho-rho(:,1),[],1)))<1e-4*mean(rho(:)), ...
    'Fan tables must be identical in every view.');
g=g(:,1); rho=rho(:,1); z=reshape(p(:,1,3,1),r,1);
data=double(projections(:,cols,:));
if all(diff(g)<0), g=flipud(g); rho=flipud(rho); data=data(:,end:-1:1,:); end
require(all(diff(g)>0),'Fan angles must be strictly monotone.');
if r>1 && z(end)<z(1), z=flipud(z); data=data(end:-1:1,:,:); end
require(all(diff(z)>0),'Detector row heights must be strictly monotone.');
beta=unwrap(atan2(s(2,:),s(1,:))); direction=sign(beta(end)-beta(1));
rel=direction*(beta-beta(1)); steps=diff(rel);
require(all(steps>0),'Source angles must advance monotonically.');
span=rel(end); weights=zeros(1,v);
weights(1:end-1)=weights(1:end-1)+steps/2;
weights(2:end)=weights(2:end)+steps/2;
full=span>=2*pi-1e-3;
if ~full && span+median(steps)>=2*pi-1e-3
    % A periodic scan without a duplicate endpoint must have uniform steps.
    require(max(abs(steps-median(steps)))<1e-3,'Periodic endpoint-free views must be uniform.');
    weights(:)=median(steps); full=true;
end
require(span<=2*pi+1e-3,'At most one full orbit is supported.');
colWeight=ones(nc,v);
if ~full
    gm=(span-pi)/2;
    require(gm>=max(abs(g))-1e-6,'Short scan does not cover pi plus twice the fan angle.');
    pg=max(-gm+1e-6,min(gm-1e-6,direction*g));
    for k=1:v
        b=rel(k); w=ones(nc,1); first=b<=2*(gm-pg); last=b>pi-2*pg;
        w(first)=sin(pi/4*b./(gm-pg(first))).^2;
        w(last)=sin(pi/4*(pi+2*gm-b)./(gm+pg(last))).^2;
        colWeight(:,k)=2*w;
    end
end
% Resample to a uniform fan grid, filter, and return to measured angles.
delta=min(diff(g)); count=ceil((g(end)-g(1))/delta-1e-9)+1;
require(count<=8192,'Fan filtering grid is too large for the dense filter; use an iterative solver.');
grid=g(1)+(0:count-1)'*delta; wc=a.Cutoff/(2*delta);
t=grid-grid'; h=ramp(t,wc);
if strcmp(a.Filter,'hann'), h=.5*h+.25*(ramp(t+1/(2*wc),wc)+ramp(t-1/(2*wc),wc)); end
ratio=ones(size(t)); nonzero=t~=0; ratio(nonzero)=t(nonzero)./sin(t(nonzero));
kernel=.5*ratio.^2.*h*delta;
toGrid=interp1(g,eye(nc),min(g(end),max(g(1),grid)),'linear');
fromGrid=interp1(grid,eye(count),min(grid(end),max(grid(1),g)),'linear');
kernel=fromGrid*kernel*toGrid;
pre=(R*cos(g')) .* (rho'./sqrt(rho'.^2+(z-s(3,1)).^2));
q=zeros(nc,r,v,'single');
for k=1:v
    filtered=(data(:,:,k).*pre)*kernel';
    q(:,:,k)=single((filtered.*(colWeight(:,k)'*weights(k)))');
end
image=FDKSurface_mex(q,single(n),single(d),single(o),single(s), ...
    single(g),single(rho),single(z),single(a.Device));
end
function require(condition,message)
assert(condition,'TIGRE:surface:FDKGeometry',message);
end
function value=rangeLocal(x)
value=max(x)-min(x);
end
function h=ramp(t,wc)
h=wc^2*(2*normalizedSinc(2*wc*t)-normalizedSinc(wc*t).^2);
end
function s=normalizedSinc(x)
s=ones(size(x)); nz=x~=0; s(nz)=sin(pi*x(nz))./(pi*x(nz));
end
