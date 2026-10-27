%% Demo 27: Build, save and load detector geometry with the GUI
% Requires: MATLAB desktop graphics; no CUDA or external data.
% MAT files work natively. Optional YAML/NPZ interchange uses Python recon.
% The example leaves one GUI open and writes files to a new temporary folder.
% Run sections individually to explore the GUI between save/load steps.

%% Open the builder and configure three panels through its controls
app=detectorGeometryGUI();
app.controls.fields.n_views.Value='36';
app.controls.fields.n_modules.Value='3';
app.controls.fields.module_shape.Value='24, 16';
app.controls.fields.pixel_spacing_mm.Value='2, 2';
app.controls.fields.angular_span_deg.Value='40';
app.controls.fields.column_gap.Value='2';
app.controls.fields.panel_x_angles_deg.Value='-3, 0, 3';
app.controls.fields.panel_y_angles_deg.Value='2, -2, 1';
app.controls.fields.cor_mm.Value='5';
app.preview();
% In the GUI, "From list..." edits each panel's roll/pitch/yaw or offsets.
% Apply a list, then press Preview. COR is a scan-wide shift; individual
% panel alignment is expressed with per-panel offsets and tilt angles.
builderSettings=app.settings();
detectorSurface=app.surface();
geo=defaultGeometry('nVoxel',[64;64;64]);
geo.sVoxel=[200;200;200]; geo.dVoxel=geo.sVoxel./geo.nVoxel;
geo.offOrigin=[0;0;0];

%% Save an editable recipe and an exact surface archive
outputFolder=tempname; mkdir(outputFolder);
settingsPath=fullfile(outputFolder,'detector_settings.mat');
surfacePath=fullfile(outputFolder,'detector_surface.mat');
geometryFile('save',settingsPath,struct('settings',builderSettings,'geo',geo));
geometryFile('save',surfacePath,struct('surface',detectorSurface,'geo',geo));
% Equivalent GUI actions: Save settings... and Save surface....
% Recipe files rebuild coordinates; exact archives preserve pixel positions,
% source positions, valid masks and module boundaries for inspection.

%% Reload the settings in editable builder mode
recipe=geometryFile('load',settingsPath);
assert(isequal(recipe.surface.pixel_centers_mm,detectorSurface.pixel_centers_mm));
app.load(settingsPath);
% Builder fields are enabled for the selected layout/motion. Edit a value
% and press Preview before saving a new exact surface.

%% Reload the exact archive in inspection mode in the same GUI
archive=geometryFile('load',surfacePath);
assert(isequal(archive.surface,detectorSurface));
app.load(surfacePath);
% Builder fields are now disabled because the archive contains exact rays.
% Use Play or the view slider to inspect the loaded geometry.
% Return to builder restores the editable recipe. Load geometry... can also
% load a file interactively; settings files open in editable builder mode.
fprintf('Geometry files saved in: %s\n',outputFolder);
fprintf('GUI is inspecting detector_surface.mat. Click Return to builder to edit.\n');

%% Optional shared YAML interchange (run these lines when needed)
% YAML settings remain editable. An exact YAML archive also creates an NPZ
% companion: keep both files together when moving or sharing the geometry.
% Requires conda env recon with TIGRE, SciPy and PyYAML for conversion only.
% geometryFile('save',fullfile(outputFolder,'settings.yaml'),struct('settings',builderSettings,'geo',geo));
% geometryFile('save',fullfile(outputFolder,'surface.yaml'),struct('surface',detectorSurface,'geo',geo));
% app.load(fullfile(outputFolder,'settings.yaml')); % editable recipe
% app.load(fullfile(outputFolder,'surface.yaml'));  % exact inspection
% Standalone inspection and native animation are also available:
% inspector=detectorGeometryGUI(surfacePath);
% h=animateGeometry(surfacePath,[],1,'SD',true);
