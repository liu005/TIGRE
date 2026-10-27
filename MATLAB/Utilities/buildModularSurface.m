function surface = buildModularSurface(s)
%BUILDMODULARSURFACE Build explicit detector coordinates from GUI/YAML settings.
% source_positions_mm: 3 x views; pixel_centers_mm: rows x cols x 3 x views;
% valid_mask: rows x cols x views. Coordinates are world x/y/z, in mm.
% module_slices: zero-based half-open [rowStart rowStop colStart colStop].
% Settings use Python/YAML names and degrees; panel rotation is Rz*Ry*Rx.
validateattributes(s.n_views, {'numeric'}, {'scalar','integer','positive','<=',6000});
validateattributes(s.module_shape, {'numeric'}, {'numel',2,'integer','positive','<=',4096});
validateattributes(s.column_gap, {'numeric'}, {'scalar','integer','nonnegative'});
nr=s.module_shape(1); nc=s.module_shape(2);
pitch=s.pixel_spacing_mm(:); if isscalar(pitch), pitch=[pitch;pitch]; end
validateattributes(pitch,{'numeric'},{'numel',2,'positive','finite'});
validateattributes(s.detector_radius_mm,{'numeric'},{'scalar','positive','finite'});
if strcmp(s.layout,'Cylindrical')
    np=s.n_modules; validateattributes(np,{'numeric'},{'scalar','integer','positive','<=',256});
    validateattributes(s.angular_span_deg,{'numeric'},{'scalar','finite','positive','<=',360});
    az=linspace(s.angular_span_deg/2,-s.angular_span_deg/2,np);
    if np==1, az=0; end
    centres=[s.detector_radius_mm*(1-cosd(az)); -s.detector_radius_mm*sind(az); zeros(1,np)];
elseif strcmp(s.layout,'Piecewise linear')
    np=numel(s.transverse_centers_mm);
    assert(np>=1 && np<=256,'TIGRE:surface:panels','Invalid panel count');
    assert(all([numel(s.normal_offsets_mm),numel(s.z_offsets_mm),numel(s.panel_angles_deg)]==np), ...
        'TIGRE:surface:lists','Panel position/z-angle lists must have one entry per panel');
    centres=[s.normal_offsets_mm(:)';s.transverse_centers_mm(:)';s.z_offsets_mm(:)'];
    az=s.panel_angles_deg(:)';
else
    error('TIGRE:surface:layout','Unknown detector layout');
end
assert(all(isfinite([centres(:);az(:)])),'TIGRE:surface:finite','Panel values must be finite');
rx=expand(s.panel_x_angles_deg,np); ry=expand(s.panel_y_angles_deg,np);
width=np*nc+(np-1)*s.column_gap;
assert(s.n_views*nr*width<=20000000,'TIGRE:surface:size','At most 20 million dense samples are allowed');
validateattributes(s.cor_mm,{'numeric'},{'scalar','finite'});
linear=strcmp(s.source_motion,'Fixed detector, moving source');
arc=strcmp(s.source_motion,'Fixed detector, arc source');
assert(linear || arc || strcmp(s.source_motion,'Rotating source + detector'), ...
    'TIGRE:surface:motion','Unknown source motion');
if linear
    start=s.source_start_mm(:); stop=s.source_end_mm(:);
    validateattributes(start,{'numeric'},{'numel',3,'finite'});
    validateattributes(stop,{'numeric'},{'numel',3,'finite'});
    source=start+(stop-start)*linspace(0,1,s.n_views);
    if s.n_views==1, source=start; end
    orientation=repmat(atan2d(start(2),start(1)),1,s.n_views);
else
    validateattributes(s.source_radius_mm,{'numeric'},{'scalar','positive','finite'});
    validateattributes(s.source_z_mm,{'numeric'},{'scalar','finite'});
    validateattributes(s.orbit_start_deg,{'numeric'},{'scalar','finite'});
    validateattributes(s.scan_span_deg,{'numeric'},{'scalar','positive','<=',360});
    if s.scan_span_deg==360
        theta=s.orbit_start_deg+(0:s.n_views-1)*360/s.n_views;
    else
        theta=s.orbit_start_deg+linspace(0,s.scan_span_deg,s.n_views);
        if s.n_views==1, theta=s.orbit_start_deg; end
    end
    source=[s.source_radius_mm*cosd(theta);s.source_radius_mm*sind(theta);repmat(s.source_z_mm,1,s.n_views)];
    orientation=theta;
    if arc, orientation(:)=s.orbit_start_deg+s.scan_span_deg/2; end
end
assert((~linear && ~arc) || s.cor_mm==0,'TIGRE:surface:COR','Stationary detectors require COR = 0');
[u,v]=meshgrid(((0:nc-1)-(nc-1)/2)*pitch(2),((0:nr-1)-(nr-1)/2)*pitch(1));
local=[zeros(1,nr*nc);u(:)';v(:)'];
pixels=zeros(nr,width,3,s.n_views,'single'); mask=false(nr,width,s.n_views);
modules=zeros(np,4);
for panel=1:np
    col=(panel-1)*(nc+s.column_gap)+(1:nc);
    modules(panel,:)=[0 nr col(1)-1 col(end)];
    for k=1:s.n_views
        a=orientation(k); Rz=rotationZ(a);
        R=rotationZ(a+az(panel))*rotationY(ry(panel))*rotationX(rx(panel));
        center=Rz*([-s.detector_radius_mm;0;0]+centres(:,panel));
        point=R*local+center;
        pixels(:,col,:,k)=reshape(single(point'),nr,nc,3);
        mask(:,col,k)=true;
    end
end
if ~linear && ~arc
    source=source+s.cor_mm*[-sind(orientation);cosd(orientation);zeros(1,s.n_views)];
    for k=1:s.n_views
        shift=s.cor_mm*[-sind(orientation(k));cosd(orientation(k));0];
        pixels(:,:,:,k)=pixels(:,:,:,k)+reshape(single(shift),1,1,3);
    end
end
surface=struct('source_positions_mm',single(source),'pixel_centers_mm',pixels, ...
    'valid_mask',mask,'module_slices',modules);
end
function a=expand(a,n)
a=a(:)'; if isscalar(a), a=repmat(a,1,n); end
assert(numel(a)==n && all(isfinite(a)),'TIGRE:surface:tilt','Tilt requires one finite value or one per panel');
end
function R=rotationX(a), R=[1 0 0;0 cosd(a) -sind(a);0 sind(a) cosd(a)]; end
function R=rotationY(a), R=[cosd(a) 0 sind(a);0 1 0;-sind(a) 0 cosd(a)]; end
function R=rotationZ(a), R=[cosd(a) -sind(a) 0;sind(a) cosd(a) 0;0 0 1]; end
