function report=validateCTTransmission(surface,geo)
%VALIDATECTTRANSMISSION Finite measured rays must cross the per-view volume.
% This is a physical transmission check, not an FDK/completeness guarantee.
src=double(surface.source_positions_mm); offsets=double(geo.offOrigin);
half=double(geo.sVoxel(:))/2; n=size(src,2); tolerance=1e-9*max([half;1]);
validateattributes(half,{'numeric'},{'numel',3,'positive','finite'});
assert(size(offsets,1)==3 && any(size(offsets,2)==[1 n]) && all(isfinite(offsets(:))), ...
    'TIGRE:physics:offset','Offsets must be finite xyz, one or per view');
p=surface.pixel_centers_mm; modules=[];
if isfield(surface,'module_slices'), modules=surface.module_slices; end
if isempty(modules), modules=[0 size(p,1) 0 size(p,2)]; end
count=0;
for k=1:n
    center=offsets(:,min(k,size(offsets,2))); low=center-half; high=center+half; s=src(:,k);
    assert(~all(abs(s-center)<=half+tolerance),'TIGRE:physics:source', ...
        'Source is inside or on the volume at view %d',k);
    viewHits=0;
    for panel=1:size(modules,1)
        rr=modules(panel,1)+1:modules(panel,2); cc=modules(panel,3)+1:modules(panel,4);
        pixels=reshape(double(p(rr,cc,:,k)),[],3); mask=surface.valid_mask(rr,cc,k);
        active=0; panelHits=0; sameSide=true;
        for start=1:16384:size(pixels,1)
            stop=min(start+16383,size(pixels,1)); values=pixels(start:stop,:); values=values(mask(start:stop),:);
            if isempty(values), continue; end
            active=active+size(values,1); relative=values-center';
            assert(~any(all(abs(relative)<=half'+tolerance,2)),'TIGRE:physics:detector', ...
                'Detector panel %d is inside or on the volume at view %d',panel,k);
            sameSide=sameSide && all(relative*(s-center)>0);
            enter=zeros(size(values,1),1); leave=ones(size(values,1),1);
            for axis=1:3
                direction=values(:,axis)-s(axis); parallel=direction==0;
                a=(low(axis)-s(axis))./direction; b=(high(axis)-s(axis))./direction;
                near=min(a,b); far=max(a,b); near(parallel)=-inf; far(parallel)=inf;
                if s(axis)<low(axis) || s(axis)>high(axis), near(parallel)=inf; far(parallel)=-inf; end
                enter=max(enter,near); leave=min(leave,far);
            end
            panelHits=panelHits+nnz(leave>enter+1e-12);
        end
        assert(~(active && sameSide && panelHits==0),'TIGRE:physics:sameSide', ...
            'Source and detector panel %d are on the same side at view %d; no measured ray crosses the volume. Reduce the arc or reposition the detector/volume.',panel,k);
        viewHits=viewHits+panelHits;
    end
    assert(viewHits>0,'TIGRE:physics:noRays','No measured ray crosses the volume at view %d',k);
    count=count+viewHits;
end
report=struct('views',n,'crossing_samples',count);
end
