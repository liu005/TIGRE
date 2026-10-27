function projections=sphereSurfaceProjections(surface,radius,centre)
%SPHERESURFACEPROJECTIONS Analytic unit-density sphere integrals, CPU only.
% Uses exact source-to-pixel segments, including arbitrary detector poses and
% masks. Output is (row,column,view), matching the explicit surface model.
if nargin<3, centre=[0;0;0]; end
validateattributes(radius,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(centre,{'numeric'},{'numel',3,'real','finite'});
pixels=surface.pixel_centers_mm; source=surface.source_positions_mm;
rows=size(pixels,1); columns=size(pixels,2); views=size(pixels,4);
assert(size(pixels,3)==3 && isequal(size(source),[3 views]), ...
    'TIGRE:sphere:shape','Invalid source/pixel surface dimensions');
mask=reshape(logical(surface.valid_mask),rows,columns,views);
projections=zeros(rows,columns,views,'single');
for k=1:views
    vector=reshape(double(pixels(:,:,:,k)),[],3)-double(source(:,k)');
    length=sqrt(sum(vector.^2,2));
    valid=mask(:,:,k); valid=valid(:);
    assert(all(isfinite(length(valid))) && all(length(valid)>0), ...
        'TIGRE:sphere:rays','Valid source/pixel segments must be finite and nonzero');
    direction=vector./max(length,eps);
    origin=double(source(:,k))-double(centre(:));
    b=direction*origin;
    discriminant=b.^2-(sum(origin.^2)-radius^2);
    root=sqrt(max(discriminant,0));
    chord=max(min(-b+root,length)-max(-b-root,0),0);
    chord(~valid | discriminant<0)=0;
    projections(:,:,k)=reshape(single(chord),rows,columns);
end
end
