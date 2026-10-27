function test_detectorGeometryGUI()
% CPU-only MATLAB regression checks; no CUDA projectors are invoked.
examples=detectorGeometryExamples();
assert(strcmp(examples(end).name,'GCT: 53-panel circular scan'));
for i=1:numel(examples)
    s=examples(i).settings; s.n_views=3; s.module_shape=[3 4];
    surface=buildModularSurface(s);
    assert(size(surface.source_positions_mm,2)==3);
    assert(all(isfinite(surface.pixel_centers_mm(:))));
    assert(size(surface.module_slices,2)==4);
end
s=examples(strcmp({examples.name},'TIGRE: circular cone-beam CT')).settings; s.n_views=4; s.module_shape=[3 4];
s.panel_x_angles_deg=13; s.panel_y_angles_deg=-17; s.panel_angles_deg=23; s.cor_mm=7;
surface=buildModularSurface(s);
geo=struct('DSO',1000,'DSD',1536,'nDetector',[4;3],'dDetector',[3.2;3.2], ...
    'nVoxel',[8;8;8],'sVoxel',[80;80;80],'offOrigin',[0;0;0], ...
    'offDetector',[0;0],'rotDetector',deg2rad([13;-17;23]),'COR',7, ...
    'mode','cone','accuracy',.5);
geo.sDetector=geo.nDetector.*geo.dDetector; geo.dVoxel=geo.sVoxel./geo.nVoxel;
converted=surfaceFromGeometry(geo,(0:3)*pi/2);
assert(max(abs(surface.pixel_centers_mm(:)-converted.pixel_centers_mm(:)))<.001);
h=animateGeometry(surface,[],2,'SD',false); cleanup=onCleanup(@() close('all'));
scene=getappdata(h,'SurfaceScene'); scene.update(3); drawnow; close(h);
varying=surface; varying.valid_mask(:,:,2)=false;
h=animateGeometry(varying,[],1,'SD',false);
scene=getappdata(h,'SurfaceScene'); scene.update(2); drawnow;
assert(contains(scene.axes.Title.String,'0 valid pixels')); close(h);
base=tempname; path=[base '.mat']; remove=onCleanup(@() deleteIfPresent(path));
geometryFile('save',path,struct('settings',s)); loaded=geometryFile('load',path);
assert(isequal(loaded.surface.pixel_centers_mm,surface.pixel_centers_mm));
app=detectorGeometryGUI(path); app.figure.Visible='off';
assert(isequal(app.surface().pixel_centers_mm,surface.pixel_centers_mm));
delete(app.figure);
settings=s; settings.cor_mm=150;
app=detectorGeometryGUI(surface); app.figure.Visible='off';
assert(~isempty(app.surface())); delete(app.figure);
fprintf('PASS detector builder, standard conversion, shared renderer, MAT and GUI loading\n');
end
function deleteIfPresent(path)
if exist(path,'file'), delete(path); end
end
