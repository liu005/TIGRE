function [surface,geo]=fixedObjectGeometry(surface,geo)
%FIXEDOBJECTGEOMETRY Apply inverse object poses to BOTH source and detector.
n=size(surface.source_positions_mm,2); rotations=objectRotationMatrices(geo,n);
offsets=double(geo.offOrigin);
assert(size(offsets,1)==3 && any(size(offsets,2)==[1 n]) && all(isfinite(offsets(:))), ...
    'TIGRE:motion:centres','Volume centre must be finite xyz, one or one per view');
for k=1:n
    centre=offsets(:,min(k,size(offsets,2))); r=rotations(:,:,k);
    surface.source_positions_mm(:,k)=r'*(double(surface.source_positions_mm(:,k))-centre);
    points=reshape(double(surface.pixel_centers_mm(:,:,:,k)),[],3);
    surface.pixel_centers_mm(:,:,:,k)=reshape((points-centre')*r,size(surface.pixel_centers_mm,1),size(surface.pixel_centers_mm,2),3);
end
geo.offOrigin=zeros(3,1);
if isfield(geo,'object_angles_deg'), geo=rmfield(geo,'object_angles_deg'); end
end
