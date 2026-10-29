function rotations=objectRotationMatrices(geo,n)
%OBJECTROTATIONMATRICES Active object poses Rz(z1)*Ry(y)*Rz(z2), degrees.
angles=zeros(1,3);
if ~isempty(geo) && isfield(geo,'object_angles_deg'), angles=double(geo.object_angles_deg); end
assert(ismatrix(angles) && size(angles,2)==3 && any(size(angles,1)==[1 n]) && all(isfinite(angles(:))), ...
    'TIGRE:motion:angles','Object angles require one finite ZYZ triple or one triple per view');
if size(angles,1)==1, angles=repmat(angles,n,1); end
rotations=zeros(3,3,n);
for k=1:n
    a=angles(k,:); c=cosd(a(2)); s=sind(a(2));
    rotations(:,:,k)=rz(a(1))*[c 0 s;0 1 0;-s 0 c]*rz(a(3));
end
end
function r=rz(a), r=[cosd(a) -sind(a) 0;sind(a) cosd(a) 0;0 0 1]; end
