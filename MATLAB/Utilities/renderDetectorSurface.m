function scene=renderDetectorSurface(ax,surface,geo,settings)
%RENDERDETECTORSURFACE Shared curved/modular scene for GUI and animation.
% scene.update(k) updates a one-based view. No GPU computation is performed.
if nargin<3, geo=[]; end
if nargin<4, settings=[]; end
p=surface.pixel_centers_mm; src=double(surface.source_positions_mm);
nr=size(p,1); nc=size(p,2); n=size(p,4);
assert(size(p,3)==3 && isequal(size(src),[3 n]),'TIGRE:surface:shape','Invalid surface array shapes');
assert(all(isfinite(p(:))) && all(isfinite(src(:))),'TIGRE:surface:finite','Coordinates must be finite');
mask=logical(surface.valid_mask);
assert(numel(mask)==nr*nc*n && any(mask(:)),'TIGRE:surface:mask','Invalid or empty validity mask');
mask=reshape(mask,nr,nc,n);
modules=[]; if isfield(surface,'module_slices'), modules=double(surface.module_slices); end
if isempty(modules)
    cols=reshape(any(any(mask,1),3),1,[]); edges=diff([false cols false]);
    starts=find(edges==1); stops=find(edges==-1)-1;
    modules=[zeros(numel(starts),1) repmat(nr,numel(starts),1) starts'-1 stops'];
end
assert(size(modules,2)==4 && all(modules(:)==fix(modules(:))) && ...
    all(modules(:,1)>=0 & modules(:,2)<=nr & modules(:,2)>modules(:,1)) && ...
    all(modules(:,3)>=0 & modules(:,4)<=nc & modules(:,4)>modules(:,3)), ...
    'TIGRE:surface:modules','Invalid module bounds');
cla(ax); hold(ax,'on'); grid(ax,'on');
lo=min([src zeros(3,1)],[],2); hi=max([src zeros(3,1)],[],2);
for k=1:n
    points=reshape(double(p(:,:,:,k)),[],3); valid=points(mask(:,:,k),:);
    if ~isempty(valid), lo=min(lo,min(valid,[],1)'); hi=max(hi,max(valid,[],1)'); end
end
if ~isempty(geo)
    origin=geo.offOrigin(:,1); extent=geo.sVoxel(:);
    assert(all(all(abs(geo.offOrigin-origin)<1e-6)),'TIGRE:surface:volume','Volume offset must be fixed');
    [V,F]=animateGeometryCube(origin,extent);
    patch(ax,'Vertices',V,'Faces',F,'FaceColor','cyan','FaceAlpha',0.15,'EdgeAlpha',0.2);
    lo=min(lo,origin-extent/2); hi=max(hi,origin+extent/2);
end
span=max(hi-lo,1); mid=(lo+hi)/2; span(3)=max(span(3),max(span(1:2))/2);
lo=mid-span*.54; hi=mid+span*.54;
xlim(ax,[lo(1) hi(1)]); ylim(ax,[lo(2) hi(2)]); zlim(ax,[lo(3) hi(3)]);
daspect(ax,[1 1 1]); view(ax,52,26); xlabel(ax,'x (mm)'); ylabel(ax,'y (mm)'); zlabel(ax,'z (mm)');
colors=[1 0 0;0 .6 0;0 0 1]; length=max(span(1:2))*.12;
for d=1:3
    v=zeros(3,1); v(d)=length;
    % quiver3 has no portable alpha property in R2020a: blend against white.
    color=.3*colors(d,:)+.7;
    quiver3(ax,0,0,0,v(1),v(2),v(3),0,'Color',color,'LineWidth',1.5);
    text(ax,v(1)*1.12,v(2)*1.12,v(3)*1.12,char('x'+d-1),'Color',color);
end
plot3(ax,src(1,:),src(2,:),src(3,:),'--','Color',[.8 .2 .3]);
source=scatter3(ax,0,0,0,40,'r','filled');
mesh=patch(ax,'Vertices',zeros(0,3),'Faces',zeros(0,3),'FaceColor',[.4 .6 .9], ...
    'FaceAlpha',.4,'EdgeColor',[.2 .3 .6]);
wire=plot3(ax,nan,nan,nan,'Color',[.2 .3 .6]);
cloud=scatter3(ax,nan,nan,nan,3,[.2 .3 .6],'filled');
ray=plot3(ax,nan,nan,nan,'Color',[1 .6 .7]);
corSphere=[]; corLabel=[]; cor=0;
if ~isempty(settings) && strcmp(settings.source_motion,'Rotating source + detector')
    cor=settings.cor_mm;
end
if cor~=0
    [sx,sy,sz]=sphere(16); radius=max(span)*.012;
    sx=sx*radius; sy=sy*radius; sz=sz*radius;
    corSphere=surf(ax,sx,sy,sz,'FaceColor',[1 .5 0],'EdgeColor','none');
    corLabel=text(ax,0,0,0,sprintf('COR (%+g mm)',cor),'Color',[.9 .4 0]);
end
scene=struct('update',@update,'source',source,'mesh',mesh,'cloud',cloud,'axes',ax);
update(1);
    function update(k)
        assert(k>=1 && k<=n && k==fix(k),'TIGRE:surface:view','Invalid view index');
        points=reshape(double(p(:,:,:,k)),[],3); valid=points(mask(:,:,k),:);
        verts=zeros(0,3); faces=zeros(0,3); segments=zeros(0,3);
        for m=1:size(modules,1)
            rr=unique(round(linspace(modules(m,1)+1,modules(m,2),min(24,modules(m,2)-modules(m,1)))));
            cc=unique(round(linspace(modules(m,3)+1,modules(m,4),min(48,modules(m,4)-modules(m,3)))));
            if numel(rr)==1 || numel(cc)==1
                [r,c]=ndgrid(rr,cc); idx=[r(:) c(:)];
                for q=1:size(idx,1)-1
                    a=idx(q,:); b=idx(q+1,:);
                    if all(all(mask(a(1):b(1),a(2):b(2),k)))
                        segments=[segments; reshape(p(a(1),a(2),:,k),1,3); reshape(p(b(1),b(2),:,k),1,3); nan(1,3)]; %#ok<AGROW>
                    end
                end
            else
                for r=1:numel(rr)-1
                    for c=1:numel(cc)-1
                        if ~all(all(mask(rr(r):rr(r+1),cc(c):cc(c+1),k))), continue; end
                        rows=[rr(r) rr(r) rr(r+1) rr(r+1)]; cols=[cc(c) cc(c+1) cc(c+1) cc(c)];
                        v=zeros(4,3);
                        for q=1:4, v(q,:)=reshape(p(rows(q),cols(q),:,k),1,3); end
                        offset=size(verts,1); verts=[verts;v]; faces=[faces;offset+[1 2 3];offset+[1 3 4]]; %#ok<AGROW>
                    end
                end
            end
        end
        set(mesh,'Vertices',verts,'Faces',faces);
        set(wire,'XData',segments(:,1),'YData',segments(:,2),'ZData',segments(:,3));
        sample=valid(1:max(1,ceil(size(valid,1)/4000)):end,:);
        set(cloud,'XData',sample(:,1),'YData',sample(:,2),'ZData',sample(:,3));
        set(source,'XData',src(1,k),'YData',src(2,k),'ZData',src(3,k));
        if ~isempty(corSphere)
            if settings.scan_span_deg==360, step=360/n;
            else, step=settings.scan_span_deg/max(n-1,1); end
            angle=settings.orbit_start_deg+(k-1)*step;
            shift=cor*[-sind(angle);cosd(angle);0];
            set(corSphere,'XData',sx+shift(1),'YData',sy+shift(2),'ZData',sz);
            set(corLabel,'Position',shift'+[radius*2 radius*2 radius*2]);
        end
        if isempty(valid), beam=nan(3,2); else, beam=[src(:,k) mean(valid,1)']; end
        set(ray,'XData',beam(1,:),'YData',beam(2,:),'ZData',beam(3,:));
        title(ax,sprintf('Detector surface - view %d/%d, %d valid pixels',k,n,size(valid,1)));
    end
end
