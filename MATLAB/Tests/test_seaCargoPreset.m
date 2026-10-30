function test_seaCargoPreset()
% Legacy plot 5 sea-cargo world poses and endpoint translation.
app=detectorGeometryGUI(); cleanup=onCleanup(@() delete(app.figure)); app.figure.Visible='off';
c=app.controls; c.preset.Value='Sea cargo: 20-foot container (2D Digital Radiography)';
button=findall(app.figure,'Type','uibutton','Text','Load example'); button.ButtonPushedFcn([],[]); app.confirm();
s=app.surface(); g=app.geometry();
assert(isequal(size(s.pixel_centers_mm),[36 112 3 1253]));
assert(max(abs(double(s.source_positions_mm(:,1))-[4230*cosd(13);0;-4230*sind(13)]))<.001);
assert(isequal(s.pixel_centers_mm(:,:,:,1),s.pixel_centers_mm(:,:,:,end)));
origin=double(s.source_positions_mm(:,1)); edges=zeros(3,2,28);
assert(size(s.module_slices,1)==28);
for m=1:28
    cols=(m-1)*4+(1:4); grid=double(s.pixel_centers_mm(:,cols,:,1));
    centre=mean(reshape(grid,[],3),1)';
    row=reshape(mean(grid(end,:,:)-grid(1,:,:),2),3,1)/35;
    column=reshape(mean(grid(:,end,:)-grid(:,1,:),1),3,1)/3;
    assert(abs(norm(row)-5)<.0001 && abs(norm(column)-5)<.0001);
    normal=cross(column,row); normal=normal/norm(normal); direction=origin-centre; direction=direction/norm(direction);
    assert(norm(normal-direction)<2e-6);
    edges(:,1,m)=reshape(mean(grid(1,:,:),2),3,1)-row/2;
    edges(:,2,m)=reshape(mean(grid(end,:,:),2),3,1)+row/2;
end
for m=1:27
    a=edges(:,2,m)-origin; b=edges(:,1,m+1)-origin;
    assert(norm(a/norm(a)-b/norm(b))<1e-7);
end
assert(edges(1,1,19)<edges(1,2,18));
assert(edges(3,1,19)-edges(3,2,18)>300);
[centres,~,v,h]=cargoModuleLayout(); assert(v==18 && h==10);
[baseline,~,vb]=cargoModuleLayout(20,4230,0);
delta=centres(1:v,:)-baseline(1:vb,:); assert(max(abs(delta(:)))<1e-8);
assert(max(abs(centres(v+1:end,3)-baseline(vb+1,3)-300))<1e-8);
assert(abs(norm(origin)-4230)<.001);
assert(max(abs(centres(1:v,1)-centres(1,1)))<1e-8);
assert(max(abs(centres(v+1:end,3)-centres(v+1,3)))<1e-8);
[~,angles]=cargoModuleLayout();
corners=[-1220 0 -1195;-1220 0 1395;1220 0 -1195;1220 0 1395]-origin';
limits=atan2(corners(:,3),-corners(:,1)); phi=linspace(min(limits),max(limits),4097);
directions=[-cos(phi);zeros(size(phi));sin(phi)]; covered=false(size(phi));
for m=1:size(centres,1)
    normal=[cosd(angles(m));0;-sind(angles(m))]; row=[sind(angles(m));0;cosd(angles(m))];
    distance=dot(centres(m,:)'-origin,normal)./(normal'*directions);
    along=row'*(origin+directions.*distance-centres(m,:)');
    covered=covered | (distance>0 & abs(along)<=90.001);
end
assert(all(covered));
air=s; air.valid_mask(:)=false; air.valid_mask(1:18,1:4,:)=true; assertAir(air,g);
air.valid_mask(:)=false; air.valid_mask(:,end-3:end,:)=true; assertAir(air,g);
assert(strcmp(g.simulation_mode,'dr'));
assert(isequal(g.nVoxel,[488;1212;518])); assert(all(g.dVoxel==5));
assert(max(abs(g.offOrigin(2,:)-linspace(-3130,3130,1253)))<1e-8);
assert(all(g.offOrigin(3,:)==100) && all(g.offOrigin(1,:)==0));
assert(all(abs(diff(g.offOrigin(2,:))-5)<1e-8));
assert(all(abs(abs(g.offOrigin(2,[1 end]))-g.sVoxel(2)/2-100)<1e-8));
assert(c.endpoints.Value && ~c.rotateObject.Value);
report=validateCTTransmission(s,g); assert(report.crossing_samples>0);
c.timed.Value=true; c.timed.ValueChangedFcn(c.timed,[]);
assert(strcmp(c.timing.rotation_period_s.Enable,'off') && strcmp(c.timing.table_speed_mm_s.Enable,'off'));
c.timing.rotation_period_s.Value='unused in DR'; c.timing.table_speed_mm_s.Value='unused in DR';
app.confirm(); g=app.geometry();
assert(abs(g.view_times_s(end)-12.52)<1e-10);
assert(norm((g.offOrigin(:,end)-g.offOrigin(:,1))/g.view_times_s(end)-[0;500;0])<1e-8);
assert(isequal(app.surface().source_positions_mm,s.source_positions_mm));
assert(isequal(app.surface().pixel_centers_mm,s.pixel_centers_mm));
base=tempname; mkdir(base); remove=onCleanup(@() rmdir(base,'s'));
app.save(fullfile(base,'cargo.yaml')); app.export(fullfile(base,'cargo_dr.m'));
code=fileread(fullfile(base,'cargo_dr.m'));
assert(contains(code,'Ax_surface') && ~contains(code,'CGLS_surface') && ~contains(code,'FDK_surface'));
app.load(fullfile(base,'cargo.yaml'));
assert(c.timed.Value && c.endpoints.Value);
assert(abs(app.geometry().view_times_s(end)-12.52)<1e-10);
% General endpoint motion retains all three axes in timed mode.
c.start.Value='20, -3130, 10'; c.start.ValueChangedFcn(c.start,[]); c.end.Value='220, 3130, 110';
app.confirm(); g=app.geometry();
assert(norm(g.offOrigin(:,end)-[220;3130;110])<1e-8);
fprintf('PASS sea-cargo preset: fixed hardware, detector pitch, volume size and conveyor motion\n');
end
function assertAir(surface,geo)
try
    validateCTTransmission(surface,geo);
    error('TIGRE:test:air','Expected air-only rays');
catch ex
    assert(strcmp(ex.identifier,'TIGRE:physics:noRays'));
end
end
