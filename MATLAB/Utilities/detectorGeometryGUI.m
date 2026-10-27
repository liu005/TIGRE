function app=detectorGeometryGUI(input,angles)
%DETECTORGEOMETRYGUI Modular detector builder and read-only geometry inspector.
% detectorGeometryGUI(); detectorGeometryGUI('geometry.yaml');
% detectorGeometryGUI(geo,angles); detectorGeometryGUI(surface);
% MATLAB surface ordering and YAML interoperability: Detector_geometry_MATLAB.md.
examples=detectorGeometryExamples(); settings=examples(1).settings;
inspection=false; surface=[]; geo=[]; builderGeo=[]; scene=[]; player=[]; current=1;
screen=get(groot,'ScreenSize');
windowSize=[min(1280,screen(3)-80) min(720,screen(4)-120)];
position=[screen(1:2)+(screen(3:4)-windowSize)/2 windowSize];
f=uifigure('Name','TIGRE Modular Detector Builder','Position',position);
f.CloseRequestFcn=@closeWindow;
main=uigridlayout(f,[1 2]); main.ColumnWidth={470,'1x'};
panel=uipanel(main,'Title','Geometry controls','Scrollable','on');
names={'n_views','source_radius_mm','source_z_mm','orbit_start_deg','scan_span_deg', ...
    'source_start_mm','source_end_mm','cor_mm','module_shape','pixel_spacing_mm', ...
    'detector_radius_mm','column_gap','n_modules','angular_span_deg', ...
    'transverse_centers_mm','normal_offsets_mm','z_offsets_mm', ...
    'panel_x_angles_deg','panel_y_angles_deg','panel_angles_deg'};
labels={'Views','Source radius (mm)','Source height (mm)','Orbit start (degrees)','Scan span (degrees)', ...
    'Source start (x,y,z mm)','Source end (x,y,z mm)','Global COR (mm)', ...
    'Module shape (rows,columns)','Pixel pitch (row,column mm)','Detector radius (mm)', ...
    'Gap columns','Modules (cylindrical)','Arc span (degrees)','Panel transverse offsets (mm)', ...
    'Panel normal offsets (mm)','Panel z offsets (mm)','Panel roll / x (degrees)', ...
    'Panel pitch / y (degrees)','Panel yaw / z (degrees)'};
grid=uigridlayout(panel,[numel(names)+6 3]); grid.ColumnWidth={190,'1x',85};
grid.RowHeight=repmat({23},1,numel(names)+6);
grid.RowSpacing=3; grid.Padding=[6 6 6 6];
uilabel(grid,'Text','Example'); preset=uidropdown(grid,'Items',{examples.name});
uibutton(grid,'Text','Load example','ButtonPushedFcn',@loadExample);
uilabel(grid,'Text','Layout'); layout=uidropdown(grid,'Items',{'Cylindrical','Piecewise linear'},'ValueChangedFcn',@changeMode);
uilabel(grid,'Text','');
uilabel(grid,'Text','Detector motion'); motion=uidropdown(grid,'Items', ...
    {'Rotating source + detector','Fixed detector, moving source','Fixed detector, arc source'},'ValueChangedFcn',@changeMode);
uilabel(grid,'Text','');
edits=struct(); listButtons=struct();
listFields={'transverse_centers_mm','normal_offsets_mm','z_offsets_mm','panel_x_angles_deg','panel_y_angles_deg','panel_angles_deg'};
for j=1:numel(names)
    name=names{j}; uilabel(grid,'Text',labels{j}); edits.(name)=uieditfield(grid,'text');
    if any(strcmp(name,listFields))
        field=name; listButtons.(name)=uibutton(grid,'Text','From list...','ButtonPushedFcn',@(~,~) listEditor(field));
    else
        uilabel(grid,'Text','');
    end
end
preview=uibutton(grid,'Text','Preview','ButtonPushedFcn',@(~,~) previewBuilder());
uibutton(grid,'Text','Save settings...','ButtonPushedFcn',@(~,~) saveFile(false));
uibutton(grid,'Text','Save surface...','ButtonPushedFcn',@(~,~) saveFile(true));
uibutton(grid,'Text','Load geometry...','ButtonPushedFcn',@loadFile);
back=uibutton(grid,'Text','Return to builder','ButtonPushedFcn',@returnBuilder);
uilabel(grid,'Text','');
note=uilabel(grid,'Text','Apply list edits, then press Preview.'); note.Layout.Column=[1 3];
right=uigridlayout(main,[4 1]); right.RowHeight={'1x',30,28,45};
ax=uiaxes(right); slider=uislider(right,'Limits',[1 2],'ValueChangedFcn',@slide);
play=uibutton(right,'Text','Play','ButtonPushedFcn',@togglePlay);
status=uilabel(right,'Text','');
populate(); previewBuilder();
if nargin>=1 && ~isempty(input)
    if ischar(input) || isstring(input), applyDocument(geometryFile('load',char(input)));
    elseif isfield(input,'pixel_centers_mm'), inspect(input,[]);
    else
        assert(nargin>=2,'TIGRE:GUI:angles','Supply projection angles');
        inspect(surfaceFromGeometry(input,angles),input);
    end
end
app=struct('figure',f,'axes',ax,'preview',@previewBuilder,'load',@(path) applyDocument(geometryFile('load',path)), ...
    'surface',@getSurface,'settings',@getSettings);

    function populate()
        layout.Value=settings.layout; motion.Value=settings.source_motion;
        for i=1:numel(names), edits.(names{i}).Value=strtrim(sprintf('%.12g, ',settings.(names{i}))); ...
                edits.(names{i}).Value=regexprep(edits.(names{i}).Value,',\s*$',''); end
        updateState();
    end
    function updateState()
        cylindrical=strcmp(layout.Value,'Cylindrical'); linear=strcmp(motion.Value,'Fixed detector, moving source');
        for i=1:numel(names)
            name=names{i}; enabled=~inspection;
            if any(strcmp(name,{'n_modules','angular_span_deg'})), enabled=enabled&&cylindrical; end
            if any(strcmp(name,{'transverse_centers_mm','normal_offsets_mm','z_offsets_mm','panel_angles_deg'})), enabled=enabled&&~cylindrical; end
            if any(strcmp(name,{'source_start_mm','source_end_mm'})), enabled=enabled&&linear; end
            if any(strcmp(name,{'source_radius_mm','source_z_mm','orbit_start_deg','scan_span_deg'})), enabled=enabled&&~linear; end
            edits.(name).Enable=onoff(enabled);
            if isfield(listButtons,name), listButtons.(name).Enable=onoff(enabled); end
        end
        layout.Enable=onoff(~inspection); motion.Enable=onoff(~inspection);
        preview.Enable=onoff(~inspection); back.Enable=onoff(inspection);
        if ~inspection && ~cylindrical
            try
                count=numel(numbers(edits.transverse_centers_mm.Value));
                note.Text=sprintf('%d panels from lists. Apply, then Preview.',count);
            catch
                note.Text='Invalid transverse list; correct it before Preview.';
            end
        elseif ~inspection
            note.Text='Apply list edits, then press Preview.';
        end
    end
    function changeMode(~,~), updateState(); end
    function values=numbers(text)
        if isempty(strtrim(text)), values=[]; return; end
        values=str2double(regexp(strtrim(text),'[,\s]+','split'));
        assert(all(isfinite(values)),'TIGRE:GUI:number','Values must be finite numbers');
    end
    function s=readSettings()
        s=settings; s.layout=layout.Value; s.source_motion=motion.Value;
        for i=1:numel(names)
            if strcmp(edits.(names{i}).Enable,'on'), s.(names{i})=numbers(edits.(names{i}).Value); end
        end
    end
    function previewBuilder()
        if inspection, return; end
        try
            s=readSettings(); next=buildModularSurface(s);
            stopPlayer(); settings=s; surface=next; drawScene();
            status.Text='Builder mode. List Apply updates fields; Preview rebuilds geometry.';
        catch ex, uialert(f,ex.message,'Invalid geometry'); end
    end
    function drawScene()
        if inspection, scene=renderDetectorSurface(ax,surface,geo);
        else, scene=renderDetectorSurface(ax,surface,geo,settings); end
        current=1; n=size(surface.source_positions_mm,2);
        slider.Limits=[1 max(2,n)]; slider.Value=1; slider.Enable=onoff(n>1);
        play.Enable=onoff(n>1); updateState(); drawnow;
    end
    function slide(~,~)
        current=min(size(surface.source_positions_mm,2),round(slider.Value));
        slider.Value=current; scene.update(current);
    end
    function togglePlay(~,~)
        if ~isempty(player) && isvalid(player) && strcmp(player.Running,'on'), stopPlayer(); return; end
        player=timer('ExecutionMode','fixedSpacing','Period',.1,'BusyMode','drop','TimerFcn',@tick);
        play.Text='Pause'; start(player);
    end
    function tick(~,~)
        if ~isvalid(f), stopPlayer(); return; end
        current=mod(current,size(surface.source_positions_mm,2))+1; slider.Value=current;
        scene.update(current); drawnow limitrate;
    end
    function stopPlayer()
        if ~isempty(player) && isvalid(player), stop(player); delete(player); end
        player=[]; if isvalid(play), play.Text='Play'; end
    end
    function loadExample(~,~)
        stopPlayer(); settings=examples(strcmp(preset.Value,{examples.name})).settings;
        inspection=false; geo=[]; builderGeo=[]; populate(); previewBuilder();
    end
    function listEditor(name)
        try
            if strcmp(layout.Value,'Cylindrical'), count=numbers(edits.n_modules.Value);
            else, count=numel(numbers(edits.transverse_centers_mm.Value)); end
            validateattributes(count,{'numeric'},{'scalar','integer','positive','<=',256});
            values=numbers(edits.(name).Value);
            if any(strcmp(name,{'panel_x_angles_deg','panel_y_angles_deg'})) && isscalar(values), values=repmat(values,1,count); end
            assert(numel(values)==count,'TIGRE:GUI:list','Enter one value per panel');
            w=uifigure('Name',name,'Position',[200 200 400 450]);
            g=uigridlayout(w,[2 2]); g.RowHeight={'1x',30};
            table=uitable(g,'Data',values(:),'ColumnEditable',true,'ColumnName',{'Value'}, ...
                'RowName',arrayfun(@num2str,1:count,'UniformOutput',false)); table.Layout.Column=[1 2];
            uibutton(g,'Text','Apply','ButtonPushedFcn',@applyList);
            uibutton(g,'Text','Cancel','ButtonPushedFcn',@(~,~) delete(w));
        catch ex, uialert(f,ex.message,'Invalid panel list'); end
        function applyList(~,~)
            if inspection || strcmp(edits.(name).Enable,'off') || ~isvalid(f)
                delete(w); return;
            end
            if ~all(isfinite(table.Data(:))), uialert(w,'Enter finite values','Invalid list'); return; end
            edits.(name).Value=regexprep(sprintf('%.12g, ',table.Data(:)),',\s*$',''); delete(w);
        end
    end
    function loadFile(~,~)
        [name,folder]=uigetfile({'*.mat;*.yaml;*.yml;*.npz','Geometry files'}); if isequal(name,0), return; end
        try, applyDocument(geometryFile('load',fullfile(folder,name)));
        catch ex, uialert(f,ex.message,'Cannot load'); end
    end
    function applyDocument(d)
        if isfield(d,'settings')
            stopPlayer(); settings=d.settings; inspection=false; geo=d.geo; builderGeo=geo; populate(); previewBuilder();
        else, inspect(d.surface,d.geo); end
    end
    function inspect(s,g)
        stopPlayer(); surface=s; geo=g; inspection=true; drawScene();
        status.Text='Inspection mode: coordinates and masks are preserved; builder fields are inactive.';
    end
    function returnBuilder(~,~)
        stopPlayer(); inspection=false; geo=builderGeo; populate(); previewBuilder();
    end
    function saveFile(exact)
        [name,folder]=uiputfile({'*.mat','MAT geometry';'*.yaml','YAML geometry'},'Save geometry'); if isequal(name,0), return; end
        try
            if exact || inspection, d=struct('surface',surface,'geo',geo);
            else, s=readSettings(); buildModularSurface(s); d=struct('settings',s,'geo',geo); end
            geometryFile('save',fullfile(folder,name),d);
        catch ex, uialert(f,ex.message,'Cannot save'); end
    end
    function closeWindow(~,~), stopPlayer(); delete(f); end
    function value=getSurface(), value=surface; end
    function value=getSettings(), value=settings; end
end
function value=onoff(condition)
if condition, value='on'; else, value='off'; end
end
