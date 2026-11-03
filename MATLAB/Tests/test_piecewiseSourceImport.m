function test_piecewiseSourceImport()
% Unsupported editable paths must never silently become a straight line.
examples=detectorGeometryExamples(); settings=examples(1).settings;
settings.n_views=3; settings.module_shape=[2 3]; settings.n_modules=1;
settings.source_waypoints_mm=[1000 -100 0;1000 0 0;900 100 20];
settings.source_segment_views=[2 2];
settings.source_motion='Fixed detector, moving source';
settings.source_start_mm=settings.source_waypoints_mm(1,:);
settings.source_end_mm=settings.source_waypoints_mm(end,:);
native=rmfield(settings,{'source_waypoints_mm','source_segment_views'});
surface=buildModularSurface(native);
surface.source_positions_mm=single(settings.source_waypoints_mm');
base=tempname; path=[base '.mat']; yaml=[base '.yaml'];
cleanup=onCleanup(@() removeFiles({path,yaml}));
geo=[]; save(path,'surface','settings','geo');
document=geometryFile('load',path);
assert(~isfield(document,'settings'));
assert(isequal(document.surface.source_positions_mm,surface.source_positions_mm));
assert(isequal(document.surface.pixel_centers_mm,surface.pixel_centers_mm));
% Exercise the real Python recipe bridge, not only the native MAT fallback.
geometryFile('save',yaml,struct('settings',settings,'geo',[]));
imported=geometryFile('load',yaml);
assert(~isfield(imported,'settings'));
assert(max(abs(imported.surface.source_positions_mm(:)-surface.source_positions_mm(:)))<.002);
assert(max(abs(imported.surface.pixel_centers_mm(:)-surface.pixel_centers_mm(:)))<.002);
try
    buildModularSurface(settings);
    error('TIGRE:test:missing','Expected rejection of an editable piecewise path');
catch ex
    assert(strcmp(ex.identifier,'TIGRE:surface:piecewise'));
end
save(path,'settings','geo');
try
    geometryFile('load',path);
    error('TIGRE:test:missing','Expected rejection without exact coordinates');
catch ex
    assert(strcmp(ex.identifier,'TIGRE:geometryFile:piecewise'));
end
fprintf('PASS piecewise source exact import and unsupported recipe rejection\n');
end
function removeFiles(paths)
for k=1:numel(paths), if exist(paths{k},'file'), delete(paths{k}); end; end
end
