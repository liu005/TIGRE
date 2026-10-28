function app=detectorGeometryGUI(input,angles)
%DETECTORGEOMETRYGUI CT simulation builder and read-only geometry inspector.
% detectorGeometryGUI(); detectorGeometryGUI('geometry.yaml');
% detectorGeometryGUI(geo,angles); detectorGeometryGUI(surface);
% MATLAB surface ordering and YAML interoperability: TIGRE_CT_Simulation_Builder_MATLAB.md.
examples=detectorGeometryExamples(); settings=examples(1).settings;
inspection=false; surface=[]; geo=[]; builderGeo=[]; scene=[]; player=[]; current=1;
confirmed=false; lastSavedYaml=''; playbackClock=[]; playbackStart=1;
lastPhysical=[];
stationaryAcknowledged='';
builderFields=[]; builderLayout=''; builderMotion=''; listWindows=gobjects(0);
builderMotionFields=[];
screen=get(groot,'ScreenSize');
windowSize=[min(1280,screen(3)-80) min(720,screen(4)-120)];
position=[screen(1:2)+(screen(3:4)-windowSize)/2 windowSize];
f=uifigure('Name','TIGRE CT Simulation Builder','Position',position);
f.CloseRequestFcn=@closeWindow;
f.DeleteFcn=@(~,~) cleanupWindow();
main=uigridlayout(f,[1 2]); main.ColumnWidth={410,'1x'};
left=uigridlayout(main,[7 1]); left.RowHeight={28,'1x',80,68,28,28,32};
examplesGrid=uigridlayout(left,[1 3]); examplesGrid.ColumnWidth={55,'1x',90}; examplesGrid.Padding=[0 0 0 0];
tabs=uitabgroup(left,'SelectionChangedFcn',@(~,~) updateHelp());
sourceTab=uitab(tabs,'Title','Source'); detectorTab=uitab(tabs,'Title','Detectors');
volumeTab=uitab(tabs,'Title','Volume'); movementTab=uitab(tabs,'Title','Movement');
if isprop(tabs,'FontSize'), tabs.FontSize=12; end
if isprop(tabs,'FontWeight'), tabs.FontWeight='bold'; end
sourceGrid=page(sourceTab,9); detectorGrid=page(detectorTab,17);
volumeGrid=page(volumeTab,4); movementGrid=page(movementTab,12);
help=uitextarea(left,'Editable','off','Value',{'Select a settings tab for help.'});
checkResults=uitextarea(left,'Editable','off','Value',{'Not confirmed. Press Confirm to see check results.'});
actions=uigridlayout(left,[1 3]); actions.Padding=[0 0 0 0];
files=uigridlayout(left,[1 3]); files.Padding=[0 0 0 0];
exportButton=uibutton(left,'Text','Export recon script...','Enable','off','ButtonPushedFcn',@(~,~) exportScript());
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
uilabel(examplesGrid,'Text','Example'); preset=uidropdown(examplesGrid,'Items',{examples.name});
uibutton(examplesGrid,'Text','Load example','ButtonPushedFcn',@loadExample);
uilabel(sourceGrid,'Text','Source trajectory'); trajectory=uidropdown(sourceGrid, ...
    'Items',{'Cylindrical','Linear'},'ValueChangedFcn',@sourceMode); uilabel(sourceGrid,'Text','');
uilabel(detectorGrid,'Text','Layout'); layout=uidropdown(detectorGrid,'Items',{'Cylindrical','Piecewise linear'},'ValueChangedFcn',@changeMode);
uilabel(detectorGrid,'Text','');
uilabel(detectorGrid,'Text','Detector motion'); motion=uidropdown(detectorGrid,'Items', ...
    {'Rotating source + detector','Fixed detector, moving source','Fixed detector, arc source'},'ValueChangedFcn',@changeMode);
uilabel(detectorGrid,'Text','');
edits=struct(); listButtons=struct();
listFields={'transverse_centers_mm','normal_offsets_mm','z_offsets_mm','panel_x_angles_deg','panel_y_angles_deg','panel_angles_deg'};
for j=1:numel(names)
    name=names{j}; if j<=7, grid=sourceGrid; else, grid=detectorGrid; end
    uilabel(grid,'Text',labels{j}); edits.(name)=uieditfield(grid,'text','Tooltip',labels{j});
    if any(strcmp(name,listFields))
        field=name; listButtons.(name)=uibutton(grid,'Text','List...','ButtonPushedFcn',@(~,~) listEditor(field));
    else
        uilabel(grid,'Text','');
    end
end
volumeEdits=struct(); volumeNames={'nVoxel','sVoxel','offOrigin','dVoxel'};
volumeLabels={'Voxels (x,y,z)','Full size (x,y,z mm)','Start / center (x,y,z mm)','Voxel size (x,y,z mm)'};
volumeDefaults={'128, 128, 128','256, 256, 256','0, 0, 0','2, 2, 2'};
for j=1:numel(volumeNames)
    uilabel(volumeGrid,'Text',volumeLabels{j}); volumeEdits.(volumeNames{j})=uieditfield(volumeGrid,'text','Value',volumeDefaults{j});
    uilabel(volumeGrid,'Text','');
end
for field={'nVoxel','sVoxel','dVoxel'}
    key=field{1}; volumeEdits.(key).ValueChangedFcn=@(~,~) syncVoxels(key);
end
timed=uicheckbox(movementGrid,'Text','Use acquisition timing','ValueChangedFcn',@changeMode); timed.Layout.Column=[1 3];
timingNames={'readout_period_ms','firing_rate_hz','rotation_period_s','table_speed_mm_s','playback_rate'};
timingLabels={'Readout period (ms)','Firing rate (Hz)','Rotation period (s/turn)','Volume z speed (mm/s)','Playback speed (x)'};
timingDefaults={'10','100','1','10','1'}; timingEdits=struct();
for j=1:5
    uilabel(movementGrid,'Text',timingLabels{j}); timingEdits.(timingNames{j})=uieditfield(movementGrid,'text','Value',timingDefaults{j});
    uilabel(movementGrid,'Text','');
end
endpoints=uicheckbox(movementGrid,'Text','Move between start and end','ValueChangedFcn',@changeMode); endpoints.Layout.Column=[1 3];
uilabel(movementGrid,'Text','Start (x,y,z mm)'); startPosition=uieditfield(movementGrid,'text','Value','0, 0, 0', ...
    'ValueChangedFcn',@(src,~) setStart(src.Value)); uilabel(movementGrid,'Text','');
volumeEdits.offOrigin.ValueChangedFcn=@(src,~) setStart(src.Value);
uilabel(movementGrid,'Text','End (x,y,z mm)'); endPosition=uieditfield(movementGrid,'text','Value','0, 0, 0'); uilabel(movementGrid,'Text','');
motionSummary=uitextarea(movementGrid,'Editable','off','Value',{'Confirm to apply motion settings.'});
motionSummary.Layout.Row=[10 12]; motionSummary.Layout.Column=[1 3];
preview=uibutton(actions,'Text','Confirm','ButtonPushedFcn',@(~,~) confirmBuilder());
uibutton(actions,'Text','Save YAML...','ButtonPushedFcn',@(~,~) saveFile(false));
uibutton(actions,'Text','Save surface...','ButtonPushedFcn',@(~,~) saveFile(true));
uibutton(files,'Text','Load geometry...','ButtonPushedFcn',@loadFile);
back=uibutton(files,'Text','Return to builder','ButtonPushedFcn',@returnBuilder);
note=uilabel(files,'Text','Confirm to apply.');
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
app=struct('figure',f,'axes',ax,'preview',@confirmBuilder,'load',@(path) applyDocument(geometryFile('load',path)), ...
    'confirm',@confirmBuilder,'surface',@getSurface,'settings',@getSettings,'geometry',@getGeometry, ...
    'save',@saveToPath,'export',@exportToPath);
app.controls=struct('fields',edits,'lists',listButtons,'layout',layout,'motion',motion, ...
    'preview',preview,'back',back,'play',play,'slider',slider,'preset',preset,'tabs',tabs, ...
    'trajectory',trajectory,'volume',volumeEdits,'timed',timed,'timing',timingEdits, ...
    'endpoints',endpoints,'start',startPosition,'end',endPosition,'export',exportButton,'help',help,'checks',checkResults);

    function g=page(tab,rows)
        container=uigridlayout(tab,[1 1]); container.Padding=[0 0 0 0];
        p=uipanel(container,'Scrollable','on');
        g=uigridlayout(p,[rows 3]); g.ColumnWidth={155,'1x',60};
        g.RowHeight=repmat({30},1,rows); g.Padding=[6 6 6 6];
    end
    function setStart(value), volumeEdits.offOrigin.Value=value; startPosition.Value=value; end
    function syncVoxels(changed)
        try
            counts=numbers(volumeEdits.nVoxel.Value);
            validateattributes(counts,{'numeric'},{'numel',3,'positive','integer','finite'});
            if strcmp(changed,'sVoxel')
                sizeMM=numbers(volumeEdits.sVoxel.Value);
                validateattributes(sizeMM,{'numeric'},{'numel',3,'positive','finite'});
                volumeEdits.dVoxel.Value=vectorText(sizeMM./counts);
            else
                spacing=numbers(volumeEdits.dVoxel.Value);
                validateattributes(spacing,{'numeric'},{'numel',3,'positive','finite'});
                volumeEdits.sVoxel.Value=vectorText(counts.*spacing);
            end
        catch ex, status.Text=['Complete voxel fields: ' ex.message]; end
    end
    function sourceMode(~,~)
        if strcmp(trajectory.Value,'Linear'), motion.Value='Fixed detector, moving source';
        elseif strcmp(motion.Value,'Fixed detector, moving source'), motion.Value='Fixed detector, arc source'; end
        changeMode();
    end
    function updateHelp()
        name=tabs.SelectedTab.Title;
        switch name
            case 'Source', text='Choose Cylindrical (circular orbit about +z) or Linear (world xyz endpoints, fixed detector). Views are shared with timing. Confirm applies edits.';
            case 'Detectors', text='Choose cylindrical or piecewise-linear panels. Shape is rows/columns; pitch is row/column mm. List... edits panel alignment. COR shifts the whole rotating rig.';
            case 'Volume', text='MATLAB volume vectors use x,y,z (Python/YAML uses z,y,x). Counts are positive integers; size is the full extent in mm. Center is shared with Movement Start.';
            otherwise, text='One frame per pulse: t=(view-1)/firing rate. Readout limits firing rate. Rotation period sets turns. Use signed z speed or xyz endpoints. Last view is at (N-1)/rate. Confirm, then Play.';
        end
        if inspection, text=[text ' Inspection mode: builder values are inactive.']; end
        help.Value={text};
    end

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
        trajectory.Value='Cylindrical'; if linear, trajectory.Value='Linear'; end
        trajectory.Enable=onoff(~inspection); timed.Enable=onoff(~inspection); endpoints.Enable=onoff(~inspection);
        for field=volumeNames, volumeEdits.(field{1}).Enable=onoff(~inspection); end
        startPosition.Enable=onoff(~inspection);
        endPosition.Enable=onoff(~inspection && (endpoints.Value || ~timed.Value));
        for field=timingNames, timingEdits.(field{1}).Enable=onoff(~inspection && timed.Value); end
        if timed.Value
            edits.scan_span_deg.Enable='off';
            timingEdits.table_speed_mm_s.Enable=onoff(~inspection && ~endpoints.Value);
        end
        preview.Enable=onoff(~inspection); back.Enable=onoff(inspection);
        if ~inspection && ~cylindrical
            try
                count=numel(numbers(edits.transverse_centers_mm.Value));
                note.Text=sprintf('%d panels. Confirm.',count);
            catch
                note.Text='Correct panel lists.';
            end
        elseif ~inspection
            note.Text='Confirm to apply.';
        end
        updateHelp();
    end
    function changeMode(~,~), stopPlayer(); closeEditors(); updateState(); end
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
        if isfield(s,'acquisition'), s=rmfield(s,'acquisition'); end
        if timed.Value
            assert(strcmp(s.source_motion,'Rotating source + detector'),'TIGRE:timing:motion','Timing requires rotating source + detector');
            s.acquisition=readTiming(s.n_views);
            s.scan_span_deg=360*s.n_views/s.acquisition.firing_rate_hz/s.acquisition.rotation_period_s;
        end
    end
    function a=readTiming(n)
        a=struct();
        for field=timingNames
            if strcmp(field{1},'table_speed_mm_s') && endpoints.Value, a.(field{1})=0;
            else, a.(field{1})=numbers(timingEdits.(field{1}).Value); end
        end
        [a,times]=ctAcquisitionTiming(n,a);
        if endpoints.Value
            first=numbers(startPosition.Value); last=numbers(endPosition.Value);
            validateattributes(first,{'numeric'},{'numel',3,'finite'}); validateattributes(last,{'numeric'},{'numel',3,'finite'});
            assert(n>1 || isequal(first,last),'TIGRE:motion:views','Different endpoints require at least two views');
            if n>1, a.table_speed_mm_s=(last(3)-first(3))/times(end); end
        end
    end
    function g=readVolume(s)
        g=struct(); for field=volumeNames, g.(field{1})=numbers(volumeEdits.(field{1}).Value)'; end
        validateattributes(g.nVoxel,{'numeric'},{'numel',3,'integer','positive','finite'});
        validateattributes(g.sVoxel,{'numeric'},{'numel',3,'positive','finite'});
        validateattributes(g.offOrigin,{'numeric'},{'numel',3,'finite'});
        validateattributes(g.dVoxel,{'numeric'},{'numel',3,'positive','finite'});
        g.dVoxel=g.sVoxel./g.nVoxel; g.motion_mode='static';
        first=g.offOrigin;
        if endpoints.Value
            last=numbers(endPosition.Value)'; validateattributes(last,{'numeric'},{'numel',3,'finite'});
            assert(s.n_views>1 || isequal(first,last),'TIGRE:motion:views','Different endpoints require at least two views');
            g.offOrigin=first+(last-first)*linspace(0,1,s.n_views); g.motion_mode='endpoints';
            if s.n_views==1, g.offOrigin=first; end
        end
        if timed.Value
            [g.acquisition,g.view_times_s]=ctAcquisitionTiming(s.n_views,s.acquisition);
            if ~endpoints.Value
                g.offOrigin=repmat(first,1,s.n_views);
                g.offOrigin(3,:)=first(3)+g.acquisition.table_speed_mm_s*g.view_times_s; g.motion_mode='speed';
            end
        end
    end
    function confirmBuilder()
        if ~resolveEndpointMotion(), return; end
        if previewBuilder()
            confirmed=true; play.Enable=onoff(size(surface.source_positions_mm,2)>1);
            lines={sprintf('Physical CT: PASS - %d views; %d crossing samples.',lastPhysical.views,lastPhysical.crossing_samples), ...
                sprintf('Empty views: %d (allowed; only an entirely empty scan is rejected).',lastPhysical.empty_views), ...
                'CGLS geometry: supported; moving volumes require the exported script''s fixed-object conversion.', ...
                'FDK: separate circular trajectory/detector/scan-coverage requirements; exported script checks these.', ...
                'Geometry checks only; CUDA execution, sampling completeness and clearance are not certified.'};
            checkResults.Value=lines; status.Text='Confirmed. Physical checks passed; see check results.';
        end
    end
    function ok=resolveEndpointMotion()
        ok=true;
        if timed.Value || endpoints.Value, return; end
        try, first=numbers(startPosition.Value); last=numbers(endPosition.Value);
        catch, return; end
        if numel(first)~=3 || numel(last)~=3 || isequal(first,last), return; end
        signature=[vectorText(first) '|' vectorText(last)];
        if strcmp(signature,stationaryAcknowledged), return; end
        choice=uiconfirm(f,['Start and End differ, but movement is OFF. Enable movement, ' ...
            'keep the volume stationary at Start, or cancel without applying changes.'], ...
            'Enable volume movement?','Options',{'Enable movement','Keep stationary','Cancel'}, ...
            'DefaultOption',1,'CancelOption',3);
        if strcmp(choice,'Cancel'), ok=false; status.Text='Cancelled; no changes applied.';
        elseif strcmp(choice,'Enable movement'), endpoints.Value=true; updateState();
        else, stationaryAcknowledged=signature; end
    end
    function ok=previewBuilder()
        ok=false; if inspection, return; end
        stopPlayer();
        try
            s=readSettings(); next=buildModularSurface(s); nextGeo=readVolume(s);
            physical=validateCTTransmission(next,nextGeo);
            lastPhysical=physical;
            stopPlayer(); settings=s; surface=next; geo=nextGeo; builderGeo=geo; drawScene(); ok=true;
            volumeEdits.dVoxel.Value=vectorText(geo.dVoxel);
            if timed.Value
                timingEdits.table_speed_mm_s.Value=num2str(geo.acquisition.table_speed_mm_s,12);
                endPosition.Value=vectorText(geo.offOrigin(:,end));
                motionSummary.Value={sprintf('Last view: %.6g s; turns: %.6g; z speed: %.6g mm/s', ...
                    geo.view_times_s(end),geo.view_times_s(end)/geo.acquisition.rotation_period_s,geo.acquisition.table_speed_mm_s)};
            end
            status.Text='Confirm applies all tabs. Play animates the confirmed geometry.';
        catch ex
            status.Text=['Confirm failed; previous geometry retained: ' ex.message];
            checkResults.Value={'Confirm: FAILED',ex.message,'Previous confirmed scene retained.'};
            uialert(f,ex.message,'Invalid geometry');
        end
    end
    function drawScene()
        checkResults.Value={'Displayed geometry changed. Confirm to refresh check results.'};
        if inspection, checkResults.Value={'Inspection mode: exact loaded geometry. Return to builder and Confirm to check edits; exported scripts validate before reconstruction.'}; end
        if inspection, scene=renderDetectorSurface(ax,surface,geo);
        else, scene=renderDetectorSurface(ax,surface,geo,settings); end
        current=1; n=size(surface.source_positions_mm,2);
        slider.Limits=[1 max(2,n)]; slider.Value=1; slider.Enable=onoff(n>1);
        play.Enable=onoff(n>1 && (confirmed || inspection)); updateState(); drawnow;
    end
    function slide(~,~)
        current=min(size(surface.source_positions_mm,2),round(slider.Value));
        slider.Value=current; scene.update(current);
    end
    function togglePlay(~,~)
        if strcmp(play.Enable,'off'), return; end
        if ~isempty(player) && isvalid(player) && strcmp(player.Running,'on'), stopPlayer(); return; end
        player=timer('ExecutionMode','fixedSpacing','Period',.1,'BusyMode','drop', ...
            'Tag','TIGREDetectorPlayback','TimerFcn',@tick);
        playbackClock=tic; playbackStart=current; play.Text='Pause'; start(player);
    end
    function tick(~,~)
        if ~isvalid(f), stopPlayer(); return; end
        n=size(surface.source_positions_mm,2);
        if ~isempty(geo) && isfield(geo,'acquisition')
            a=geo.acquisition; current=mod(playbackStart-1+floor(toc(playbackClock)*a.playback_rate*a.firing_rate_hz),n)+1;
        else, current=mod(current,n)+1; end
        slider.Value=current;
        scene.update(current); drawnow limitrate;
    end
    function stopPlayer()
        if ~isempty(player) && isvalid(player), stop(player); delete(player); end
        player=[]; if isvalid(play), play.Text='Play'; end
    end
    function loadExample(~,~)
        stopPlayer(); closeEditors(); settings=examples(strcmp(preset.Value,{examples.name})).settings;
        inspection=false; geo=[]; builderGeo=[]; timed.Value=false; endpoints.Value=false;
        for k=1:numel(volumeNames), volumeEdits.(volumeNames{k}).Value=volumeDefaults{k}; end
        setStart('0, 0, 0'); endPosition.Value='0, 0, 0'; populate(); previewBuilder();
    end
    function listEditor(name)
        try
            if strcmp(layout.Value,'Cylindrical'), count=numbers(edits.n_modules.Value);
            else, count=numel(numbers(edits.transverse_centers_mm.Value)); end
            validateattributes(count,{'numeric'},{'scalar','integer','positive','<=',256});
            values=numbers(edits.(name).Value);
            if any(strcmp(name,{'panel_x_angles_deg','panel_y_angles_deg'})) && isscalar(values), values=repmat(values,1,count); end
            assert(numel(values)==count,'TIGRE:GUI:list','Enter one value per panel');
            w=uifigure('Name',labels{strcmp(names,name)},'Position',[200 200 400 450], ...
                'Tag','TIGREPanelList','UserData',name);
            listWindows(end+1)=w;
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
            if strcmp(layout.Value,'Cylindrical'), nowCount=numbers(edits.n_modules.Value);
            else, nowCount=numel(numbers(edits.transverse_centers_mm.Value)); end
            if ~isequal(nowCount,count)
                uialert(w,'Panel count changed. Reopen this list.','List out of date'); return;
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
        closeEditors();
        if ~isempty(d.geo) && size(d.geo.offOrigin,2)>1
            offsets=d.geo.offOrigin; linear=offsets(:,1)+(offsets(:,end)-offsets(:,1))*linspace(0,1,size(offsets,2));
            if any(abs(offsets(:)-linear(:))>1e-6), inspect(d.surface,d.geo); return; end
        end
        if isfield(d,'settings')
            stopPlayer(); settings=d.settings; inspection=false; geo=d.geo; builderGeo=geo;
            populateVolume(geo); populate(); previewBuilder();
        else, inspect(d.surface,d.geo); end
    end
    function populateVolume(g)
        for k=1:numel(volumeNames), volumeEdits.(volumeNames{k}).Value=volumeDefaults{k}; end
        setStart(volumeEdits.offOrigin.Value); endPosition.Value=startPosition.Value;
        timed.Value=isfield(settings,'acquisition') && isstruct(settings.acquisition) && ~isempty(settings.acquisition);
        if timed.Value
            for field=timingNames, timingEdits.(field{1}).Value=num2str(settings.acquisition.(field{1}),12); end
        end
        endpoints.Value=false;
        if isempty(g), return; end
        for field=volumeNames
            if isfield(g,field{1}), volumeEdits.(field{1}).Value=vectorText(g.(field{1})(:,1)); end
        end
        setStart(volumeEdits.offOrigin.Value); endPosition.Value=vectorText(g.offOrigin(:,end));
        syncVoxels('sVoxel');
        if isfield(g,'motion_mode'), endpoints.Value=strcmp(g.motion_mode,'endpoints');
        else, endpoints.Value=size(g.offOrigin,2)>1; end
    end
    function value=vectorText(values), value=regexprep(sprintf('%.12g, ',values),',\s*$',''); end
    function values=captureMotion()
        values=struct('timed',timed.Value,'endpoints',endpoints.Value,'start',startPosition.Value,'end',endPosition.Value);
        for field=volumeNames, values.(field{1})=volumeEdits.(field{1}).Value; end
        for field=timingNames, values.(field{1})=timingEdits.(field{1}).Value; end
    end
    function restoreMotion(values)
        timed.Value=values.timed; endpoints.Value=values.endpoints;
        for field=volumeNames, volumeEdits.(field{1}).Value=values.(field{1}); end
        for field=timingNames, timingEdits.(field{1}).Value=values.(field{1}); end
        setStart(values.start); endPosition.Value=values.end;
    end
    function inspect(s,g)
        if ~inspection
            builderFields=struct();
            for i=1:numel(names), builderFields.(names{i})=edits.(names{i}).Value; end
            builderLayout=layout.Value; builderMotion=motion.Value;
            builderMotionFields=captureMotion();
        end
        closeEditors();
        stopPlayer(); surface=s; geo=g; inspection=true; drawScene();
        status.Text='Inspection mode: coordinates and masks are preserved; builder fields are inactive.';
    end
    function returnBuilder(~,~)
        stopPlayer(); closeEditors(); inspection=false; geo=builderGeo;
        if isempty(builderFields), populate();
        else
            layout.Value=builderLayout; motion.Value=builderMotion;
            for i=1:numel(names), edits.(names{i}).Value=builderFields.(names{i}); end
            if ~isempty(builderMotionFields), restoreMotion(builderMotionFields); end
            updateState();
        end
        previewBuilder();
    end
    function saveFile(exact)
        [name,folder]=uiputfile({'*.yaml','YAML geometry';'*.mat','MAT geometry'},'Save geometry'); if isequal(name,0), return; end
        try
            saveToPath(fullfile(folder,name),exact);
        catch ex, uialert(f,ex.message,'Cannot save'); end
    end
    function saveToPath(path,exact)
        if nargin<2, exact=false; end
        if ~exact && ~inspection && ~resolveEndpointMotion(), return; end
        if exact || inspection, d=struct('surface',surface,'geo',geo);
        else
            s=readSettings(); next=buildModularSurface(s); nextGeo=readVolume(s);
            validateCTTransmission(next,nextGeo); d=struct('settings',s,'geo',nextGeo);
        end
        geometryFile('save',path,d); [~,~,extension]=fileparts(path);
        if any(strcmpi(extension,{'.yaml','.yml'})), lastSavedYaml=path; exportButton.Enable='on'; end
        status.Text=['Saved geometry: ' path];
    end
    function exportScript()
        if isempty(lastSavedYaml), return; end
        [~,name]=fileparts(lastSavedYaml);
        [name,folder]=uiputfile('*.m','Export recon script',[name '_recon.m']); if isequal(name,0), return; end
        try
            exportToPath(fullfile(folder,name));
            status.Text=['Exported script using last saved YAML: ' lastSavedYaml];
        catch ex, uialert(f,ex.message,'Cannot export recon script'); end
    end
    function exportToPath(path)
        assert(~isempty(lastSavedYaml),'TIGRE:export:yaml','Save YAML successfully before exporting');
        exportCTReconstructionScript(lastSavedYaml,path);
    end
    function closeEditors()
        for window=listWindows(:)', if isvalid(window), delete(window); end; end
        listWindows=gobjects(0);
    end
    function closeWindow(~,~), stopPlayer(); closeEditors(); delete(f); end
    function cleanupWindow(), stopPlayer(); closeEditors(); end
    function value=getSurface(), value=surface; end
    function value=getSettings(), value=settings; end
    function value=getGeometry(), value=geo; end
end
function value=onoff(condition)
if condition, value='on'; else, value='off'; end
end
