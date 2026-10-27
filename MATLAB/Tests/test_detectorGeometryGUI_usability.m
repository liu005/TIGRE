function test_detectorGeometryGUI_usability()
% Exercise real UI callbacks, disabled fields, list edits and timer lifecycle.
examples=detectorGeometryExamples(); s=examples(strcmp({examples.name},'Three tilted panels')).settings;
s.n_views=4; s.module_shape=[3 4]; s.column_gap=1;
path=[tempname '.mat']; settings=s; save(path,'settings');
cleanup=onCleanup(@() finish(path));
app=detectorGeometryGUI(path); app.figure.Visible='off'; c=app.controls;
assert(app.figure.Position(4)<=720);
for layout={'Piecewise linear','Cylindrical'}
    c.layout.Value=layout{1}; c.layout.ValueChangedFcn(c.layout,[]);
    for motion={'Rotating source + detector','Fixed detector, moving source','Fixed detector, arc source'}
        c.motion.Value=motion{1}; c.motion.ValueChangedFcn(c.motion,[]);
        linear=strcmp(motion{1},'Fixed detector, moving source'); cylinder=strcmp(layout{1},'Cylindrical');
        assert(strcmp(c.fields.source_start_mm.Enable,onoff(linear)));
        assert(strcmp(c.fields.orbit_start_deg.Enable,onoff(~linear)));
        assert(strcmp(c.fields.n_modules.Enable,onoff(cylinder)));
        assert(strcmp(c.lists.panel_angles_deg.Enable,onoff(~cylinder)));
        assert(strcmp(c.lists.panel_y_angles_deg.Enable,'on'));
        assert(strcmp(c.fields.cor_mm.Enable,'on'));
    end
end
c.layout.Value='Piecewise linear'; c.motion.Value='Rotating source + detector';
c.layout.ValueChangedFcn(c.layout,[]);
before=app.surface();
c.lists.panel_y_angles_deg.ButtonPushedFcn(c.lists.panel_y_angles_deg,[]);
w=findall(groot,'Tag','TIGREPanelList'); assert(numel(w)==1);
table=findall(w,'Type','uitable'); table.Data=[30;0;0];
button=findall(w,'Text','Apply'); button.ButtonPushedFcn(button,[]);
assert(strcmp(c.fields.panel_y_angles_deg.Value,'30, 0, 0'));
assert(isequal(app.surface().pixel_centers_mm,before.pixel_centers_mm));
app.preview(); assert(~isequal(app.surface().pixel_centers_mm,before.pixel_centers_mm));
% An unpreviewed edit survives an inspection round trip.
c.fields.cor_mm.Value='12';
surface=before; exact=[tempname '.mat']; save(exact,'surface');
removeExact=onCleanup(@() delete(exact)); app.load(exact);
assert(strcmp(c.fields.cor_mm.Enable,'off')); assert(strcmp(c.preview.Enable,'off'));
c.back.ButtonPushedFcn(c.back,[]);
assert(strcmp(c.fields.cor_mm.Value,'12')); assert(app.settings().cor_mm==12);
% Mode changes close stale editors; closing the main window deletes playback.
c.lists.panel_y_angles_deg.ButtonPushedFcn(c.lists.panel_y_angles_deg,[]);
c.motion.Value='Fixed detector, arc source'; c.motion.ValueChangedFcn(c.motion,[]);
assert(isempty(findall(groot,'Tag','TIGREPanelList')));
c.play.ButtonPushedFcn(c.play,[]); pause(.25);
assert(~isempty(timerfindall('Tag','TIGREDetectorPlayback')));
delete(app.figure);
assert(isempty(timerfindall('Tag','TIGREDetectorPlayback')));
fprintf('PASS MATLAB GUI layout, six field-state combinations, lists, inspection and playback cleanup\n');
end
function value=onoff(condition)
if condition, value='on'; else, value='off'; end
end
function finish(path)
if exist(path,'file'), delete(path); end
timers=timerfindall('Tag','TIGREDetectorPlayback');
for item=timers(:)', stop(item); delete(item); end
close('all','force');
end
