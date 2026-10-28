function test_ctSimulationBuilder()
% Timing, moving-volume rendering, confirmation and shared YAML interoperability.
addpath(fileparts(fileparts(mfilename('fullpath'))));
a=struct('readout_period_ms',100,'firing_rate_hz',4,'rotation_period_s',1, ...
    'table_speed_mm_s',10,'playback_rate',1);
[~,t]=ctAcquisitionTiming(9,a); assert(isequal(t,0:.25:2));
bad=a; bad.readout_period_ms=300;
try, ctAcquisitionTiming(9,bad); error('test:missed','Expected readout-capacity rejection');
catch ex, assert(strcmp(ex.identifier,'TIGRE:timing:capacity')); end
examples=detectorGeometryExamples(); s=examples(1).settings;
s.n_views=9; s.module_shape=[2 3]; s.n_modules=1; s.acquisition=a;
surface=buildModularSurface(s);
assert(max(abs(surface.source_positions_mm(:,1)-surface.source_positions_mm(:,9)))<1e-8);
app=detectorGeometryGUI(); cleanup=onCleanup(@() delete(app.figure)); app.figure.Visible='off';
c=app.controls; assert(strcmp(app.figure.Name,'TIGRE CT Simulation Builder'));
c.volume.dVoxel.Value='1, 2, 3'; c.volume.dVoxel.ValueChangedFcn(c.volume.dVoxel,[]);
assert(strcmp(c.volume.sVoxel.Value,'128, 256, 384'));
c.volume.nVoxel.Value='10, 20, 30'; c.volume.nVoxel.ValueChangedFcn(c.volume.nVoxel,[]);
assert(strcmp(c.volume.sVoxel.Value,'10, 40, 90'));
c.volume.sVoxel.Value='40, 60, 120'; c.volume.sVoxel.ValueChangedFcn(c.volume.sVoxel,[]);
assert(strcmp(c.volume.dVoxel.Value,'4, 3, 4'));
assert(strcmp(c.play.Enable,'off')); assert(strcmp(c.export.Enable,'off'));
assert(isequal(sort({c.tabs.Children.Title}),sort({'Movement','Volume','Detectors','Source'})));
c.fields.n_views.Value='9'; c.fields.module_shape.Value='2, 3'; c.fields.n_modules.Value='1';
c.volume.nVoxel.Value='16, 12, 8'; c.volume.sVoxel.Value='32, 24, 40';
c.timed.Value=true;
for name=fieldnames(a)', c.timing.(name{1}).Value=num2str(a.(name{1})); end
c.endpoints.Value=true; c.start.Value='1, 2, -10'; c.start.ValueChangedFcn(c.start,[]);
c.end.Value='1, 2, 10'; app.confirm();
assert(strcmp(c.play.Enable,'on'));
assert(any(contains(c.checks.Value,'Physical CT: PASS')));
g=app.geometry(); assert(isequal(g.offOrigin(:,[1 end]),[1 1;2 2;-10 10]));
assert(g.acquisition.table_speed_mm_s==10);
assert(size(app.surface().source_positions_mm,2)==9);
c.slider.Value=9; c.slider.ValueChangedFcn(c.slider,[]);
box=findall(app.axes,'Type','patch','FaceColor','cyan');
assert(abs(mean(box.Vertices(:,3))-10)<1e-8);
before=app.surface(); c.trajectory.Value='Linear'; c.trajectory.ValueChangedFcn(c.trajectory,[]);
assert(strcmp(c.motion.Value,'Fixed detector, moving source'));
assert(isequal(app.surface(),before)); % Selection is pending until Confirm.
c.trajectory.Value='Cylindrical'; c.trajectory.ValueChangedFcn(c.trajectory,[]);
c.motion.Value='Rotating source + detector'; c.motion.ValueChangedFcn(c.motion,[]);
base=tempname; mkdir(base); remove=onCleanup(@() rmdir(base,'s'));
app.save(fullfile(base,'geometry.mat')); assert(strcmp(c.export.Enable,'off'));
app.save(fullfile(base,'geometry.yaml')); assert(strcmp(c.export.Enable,'on'));
app.export(fullfile(base,'reconstruct.m'));
assert(contains(fileread(fullfile(base,'reconstruct.m')),'CGLS_surface'));
loaded=geometryFile('load',fullfile(base,'geometry.yaml'));
assert(isequal(loaded.geo.offOrigin,g.offOrigin)); assert(isequal(double(loaded.geo.nVoxel(:)),g.nVoxel));
assert(strcmp(loaded.geo.motion_mode,'endpoints'));
assert(max(abs(loaded.surface.source_positions_mm(:)-before.source_positions_mm(:)))<1e-8);
app.figure.Position=[100 100 1024 700]; drawnow;
assert(app.axes.Position(3)>300);
fprintf('PASS CT simulation timing, tabs, Confirm/Play, xyz endpoints, YAML bridge and script export\n');
end
