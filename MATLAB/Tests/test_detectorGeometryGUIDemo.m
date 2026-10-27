function test_detectorGeometryGUIDemo()
% Exercise the complete public GUI/file workflow without CUDA or dialogs.
d27_DetectorGeometryGUI;
cleanup=onCleanup(@() cleanupDemo(app,outputFolder));
app.figure.Visible='off';
actual=app.surface();
assert(isequal(actual,detectorSurface));
assert(strcmp(app.controls.preview.Enable,'off'));
assert(strcmp(app.controls.fields.cor_mm.Enable,'off'));
assert(strcmp(app.controls.back.Enable,'on'));
assert(isequal(archive.surface.valid_mask,detectorSurface.valid_mask));
assert(isequal(archive.surface.module_slices,detectorSurface.module_slices));
app.controls.back.ButtonPushedFcn([],[]);
assert(strcmp(app.controls.preview.Enable,'on'));
assert(strcmp(app.controls.fields.cor_mm.Enable,'on'));
restoredSettings=app.settings();
names=fieldnames(builderSettings);
for k=1:numel(names)
    assert(isfield(restoredSettings,names{k}) && isequaln(restoredSettings.(names{k}),builderSettings.(names{k})), ...
        'TIGRE:test:recipe','Builder recipe changed: %s',names{k});
end
app.controls.fields.cor_mm.Value='7'; app.preview();
updated=app.surface(); assert(~isequal(updated.source_positions_mm,detectorSurface.source_positions_mm));
app.load(settingsPath);
restored=app.surface(); assert(isequal(restored,detectorSurface));
fprintf('PASS demo 27: GUI preview, native recipe/exact saves, loading, disabled inspector fields and builder restoration\n');
end
function cleanupDemo(app,folder)
if isvalid(app.figure), delete(app.figure); end
% Only the known demo files in its fresh temporary folder are removed.
for name={'detector_settings.mat','detector_surface.mat'}
    file=fullfile(folder,name{1}); if exist(file,'file'), delete(file); end
end
if exist(folder,'dir'), rmdir(folder); end
end
