function test_objectRotation()
% Arbitrary object poses, GUI preset, inverse geometry and shared YAML.
examples=detectorGeometryExamples(); s=examples(strcmp({examples.name},'TIGRE d18: arbitrary object rotation (ZYZ)')).settings;
s.n_views=3; s.module_shape=[3 4]; surface=buildModularSurface(s);
assert(isequal(surface.source_positions_mm,repmat(surface.source_positions_mm(:,1),1,3)));
g=struct('nVoxel',[6;4;2],'sVoxel',[60;40;20],'offOrigin',[3;2;1], ...
    'object_angles_deg',[0 0 0;30 90 60;-20 40 90]);
[fixed,gfixed]=fixedObjectGeometry(surface,g); rotations=objectRotationMatrices(g,3);
for k=1:3
    reconstructed=rotations(:,:,k)*double(fixed.source_positions_mm(:,k))+g.offOrigin;
    assert(max(abs(reconstructed-double(surface.source_positions_mm(:,k))))<.001);
end
assert(isequal(gfixed.offOrigin,zeros(3,1)) && ~isfield(gfixed,'object_angles_deg'));
report=validateCTTransmission(surface,g); assert(report.crossing_samples>0);
app=detectorGeometryGUI(); cleanup=onCleanup(@() delete(app.figure)); app.figure.Visible='off'; c=app.controls;
assert(isequal(c.rotateObject.Parent,c.volume.nVoxel.Parent));
assert(~isequal(c.rotateObject.Parent,c.start.Parent));
assert(contains(c.rotateObject.Tooltip,'initial pose'));
assert(contains(c.timed.Tooltip,'one frame per source pulse'));
assert(contains(c.endpoints.Tooltip,'interpolate the object centre'));
c.tabs.SelectedTab=ancestor(c.rotateObject,'uitab');
app.figure.WindowButtonMotionFcn(app.figure,[]);
c.preset.Value='TIGRE d18: arbitrary object rotation (ZYZ)';
button=findall(app.figure,'Type','uibutton','Text','Load example'); button.ButtonPushedFcn([],[]);
assert(c.rotateObject.Value && strcmp(c.objectMode.Value,'ZYZ angle list'));
app.confirm(); assert(any(contains(c.checks.Value,'Physical CT: PASS')));
poses=app.geometry().object_angles_deg; assert(isequal(size(poses),[100 3]));
phase=linspace(0,2*pi,100)'; expected=rad2deg([-phase -phase -pi*sin(phase)]);
assert(max(abs(poses(:)-expected(:)))<1e-10);
[equivalent,~]=fixedObjectGeometry(app.surface(),app.geometry());
c.rotateObject.Value=false; c.rotateObject.ValueChangedFcn(c.rotateObject,[]); app.confirm();
rig=app.surface();
assert(max(abs(rig.source_positions_mm(:)-equivalent.source_positions_mm(:)))<.001);
assert(max(abs(rig.pixel_centers_mm(:)-equivalent.pixel_centers_mm(:)))<.001);
assert(~isfield(app.geometry(),'object_angles_deg'));
c.rotateObject.Value=true; c.rotateObject.ValueChangedFcn(c.rotateObject,[]); app.confirm();
assert(strcmp(c.timed.Enable,'off') && strcmp(c.motion.Enable,'off'));
before=app.surface(); c.slider.Value=26; c.slider.ValueChangedFcn(c.slider,[]);
box=findall(app.axes,'Type','patch','FaceColor','cyan'); vertices=box.Vertices;
c.slider.Value=1; c.slider.ValueChangedFcn(c.slider,[]); assert(max(abs(vertices(:)-box.Vertices(:)))>1);
assert(isequal(app.surface(),before));
base=tempname; mkdir(base); remove=onCleanup(@() rmdir(base,'s'));
path=fullfile(base,'object.yaml'); app.save(path); d=geometryFile('load',path);
assert(max(abs(d.geo.object_angles_deg(:)-poses(:)))<1e-10);
app.load(path); assert(c.rotateObject.Value); assert(strcmp(c.objectMode.Value,'ZYZ angle list'));
c.objectMode.Value='Start/end z angles'; c.objectStart.Value='90'; c.objectEnd.Value='-90'; app.confirm();
z=app.geometry().object_angles_deg; assert(max(abs(z(:,1)-linspace(90,-90,100)'))<1e-10);
app.export(fullfile(base,'recon.m')); assert(contains(fileread(fullfile(base,'recon.m')),'fixedObjectGeometry'));
fprintf('PASS object rotation: fixed hardware, ZYZ d18, rendering, YAML and recon export\n');
end
