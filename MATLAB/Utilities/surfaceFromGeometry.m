function surface=surfaceFromGeometry(geo,angles)
%SURFACEFROMGEOMETRY Express standard MATLAB TIGRE cone geometry as pixels.
% MATLAB nDetector/dDetector/offDetector are [u;v]; sVoxel/offOrigin are xyz.
% rotDetector and angles are radians. Surface arrays use buildModularSurface's
% rows x columns x xyz x views convention, independent of projection storage.
if isvector(angles), angles=reshape(angles,1,[]); end
assert(size(angles,1)==1 || size(angles,1)==3,'TIGRE:surface:angles','Use 1xN or 3xN angles');
if size(angles,1)==1, angles=[angles;zeros(2,size(angles,2))]; end
n=size(angles,2); geo=checkGeo(geo,angles);
assert(strcmp(geo.mode,'cone'),'TIGRE:surface:mode','Only finite-source cone geometry is supported');
nc=geo.nDetector(1); nr=geo.nDetector(2);
[u,v]=meshgrid(((0:nc-1)-(nc-1)/2)*geo.dDetector(1),((0:nr-1)-(nr-1)/2)*geo.dDetector(2));
local=[zeros(1,nr*nc);u(:)';v(:)'];
pixels=zeros(nr,nc,3,n,'single'); source=zeros(3,n,'single');
for k=1:n
    a=angles(:,k); r=geo.rotDetector(:,k);
    G=rz(a(1))*ry(a(2))*rz(a(3)); D=rz(r(3))*ry(r(2))*rx(r(1));
    shift=geo.COR(k)*[-sin(a(1));cos(a(1));0];
    center=[geo.DSO(k)-geo.DSD(k);geo.offDetector(:,k)];
    p=G*(D*local+center)+shift;
    pixels(:,:,:,k)=reshape(single(p'),nr,nc,3);
    source(:,k)=single(G*[geo.DSO(k);0;0]+shift);
end
surface=struct('source_positions_mm',source,'pixel_centers_mm',pixels, ...
    'valid_mask',true(nr,nc,n),'module_slices',[0 nr 0 nc]);
end
function R=rx(a), R=[1 0 0;0 cos(a) -sin(a);0 sin(a) cos(a)]; end
function R=ry(a), R=[cos(a) 0 sin(a);0 1 0;-sin(a) 0 cos(a)]; end
function R=rz(a), R=[cos(a) -sin(a) 0;sin(a) cos(a) 0;0 0 1]; end
