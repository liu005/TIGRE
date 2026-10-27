function test_surfaceFDK()
UseBlackwellBuild();
geo=defaultGeometry('nVoxel',[24;24;16]); geo.dVoxel=[1;1;1];
geo.sVoxel=geo.nVoxel.*geo.dVoxel; geo.offOrigin=[0;0;0];
nv=100; nr=32; nc=81; beta=(0:nv-1)*2*pi/nv;
g=linspace(-.23,.23,nc); z=((0:nr-1)-(nr-1)/2)*1.2;
[gg,zz]=meshgrid(g,z); local=cat(3,100-170*cos(gg),170*sin(gg),zz);
surface=struct('source_positions_mm',zeros(3,nv,'single'), ...
    'pixel_centers_mm',zeros(nr,nc,3,nv,'single'),'valid_mask',true(nr,nc,nv));
for k=1:nv
    R=[cos(beta(k)) -sin(beta(k)) 0; sin(beta(k)) cos(beta(k)) 0; 0 0 1];
    surface.source_positions_mm(:,k)=R*[100;0;0];
    surface.pixel_centers_mm(:,:,:,k)=reshape(single(reshape(local,[],3)*R'),nr,nc,3);
end
b=sphereSurfaceProjections(surface,6);
image=FDK_surface(b,geo,surface);
[x,y,zv]=ndgrid((-11.5:11.5),(-11.5:11.5),(-7.5:7.5));
inside=x.^2+y.^2+zv.^2<4^2;
value=mean(image(inside)); assert(abs(value-1)<.1);
assert(all(isfinite(image(:))));
% Reversed scan order and descending detector rows preserve the result.
reverse=surface;
reverse.source_positions_mm=reverse.source_positions_mm(:,end:-1:1);
reverse.pixel_centers_mm=reverse.pixel_centers_mm(end:-1:1,:,:,end:-1:1);
reverse.valid_mask=reverse.valid_mask(end:-1:1,:,end:-1:1);
reversed=FDK_surface(b(end:-1:1,:,end:-1:1),geo,reverse);
assert(max(abs(reversed(:)-image(:)))<1e-4);
% 216-degree arc covers pi plus twice the fan angle; Parker weighting.
short=surface; keep=1:61;
short.source_positions_mm=short.source_positions_mm(:,keep);
short.pixel_centers_mm=short.pixel_centers_mm(:,:,:,keep);
short.valid_mask=short.valid_mask(:,:,keep);
shortImage=FDK_surface(b(:,:,keep),geo,short);
assert(abs(mean(shortImage(inside))-1)<.1);
% Gap interpolation and Hann filter execute, but wide gaps reduce accuracy.
surface.valid_mask(:,40:41,:)=false; b(:,40:41,:)=0;
gap=FDK_surface(b,geo,surface,'Filter','hann');
assert(all(isfinite(gap(:))) && abs(mean(gap(inside))-1)<.15);
% A tilt that breaks extrusion is rejected before the CUDA call.
surface.pixel_centers_mm(2,2,1,1)=surface.pixel_centers_mm(2,2,1,1)+1;
failed=false;
try, FDK_surface(b,geo,surface); catch e, failed=strcmp(e.identifier,'TIGRE:surface:FDKGeometry'); end
assert(failed);
fprintf('PASS surface FDK CUDA: full %.4f, Parker %.4f, masked Hann %.4f\n',value,mean(shortImage(inside)),mean(gap(inside)));
end
