function test_demoGeometryAnimations()
% Validate demo 17/18 preview geometry without running their reconstructions.
root=fileparts(fileparts(mfilename('fullpath')));
for number=[17 18]
    if number==17, name='d17_DetectorRotation.m'; marker='%% Load data and generate projections';
    else, name='d18_ArbitraryAxisOfRotation.m'; marker='%% Get Image'; end
    source=fileread(fullfile(root,'Demos',name));
    assert(~isempty(regexp(source,'plotProj[^\n]*\n\s*%% Animate acquisition geometry','once')));
    prefix=extractBefore(source,marker);
    prefix=regexprep(prefix,'(?m)^\s*(clear;|close all;)\s*$','');
    eval(prefix);
    geo.nDetector=[24;24]; geo.sDetector=geo.nDetector.*geo.dDetector;
    surface=surfaceFromGeometry(geo,angles);
    h=animateGeometry(geo,[],1,'SD',false,'',surface);
    scene=getappdata(h,'SurfaceScene');
    for k=[1 50 100]
        scene.update(k); drawnow;
        assert(abs(scene.source.XData-double(surface.source_positions_mm(1,k)))<1e-5);
    end
    close(h);
end
source=fileread(fullfile(root,'Demos','d19_DBTexample.m'));
assert(~isempty(regexp(source,'plotProj[^\n]*\ngeometrySurface','once')));
prefix=extractBefore(source,'%% Adapt DBT projections to TIGRE CT projections');
prefix=regexprep(prefix,'clc;clear;close all',''); eval(prefix);
geo.nDetector=[20;16]; geo.dDetector=geo.sDetector./geo.nDetector;
surface=surfaceFromGeometry(geo,angles);
for k=2:size(angles,2)
    delta=surface.pixel_centers_mm(:,:,:,k)-surface.pixel_centers_mm(:,:,:,1);
    assert(max(abs(delta(:)))<1e-4);
end
h=animateGeometry(geo,[],1,'SD',false,'',surface);
scene=getappdata(h,'SurfaceScene'); scene.update(5); drawnow; close(h);
fprintf('PASS demo 17/18/19 animation placement, per-view poses and fixed DBT detector\n');
end
