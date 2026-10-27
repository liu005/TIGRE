function test_geometryInterchange()
% Check native MATLAB geometry against Python's generated coordinates.
examples=detectorGeometryExamples();
base=tempname; mat=[base '.mat']; yaml=[base '.yaml']; reference=[base '_reference.mat'];
cleanup=onCleanup(@() removeFiles({mat,yaml,reference}));
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
script=fullfile(root,'Python','tigre','utilities','geometry_matlab_bridge.py');
for layout={'Piecewise linear','Cylindrical'}
    for motion={'Rotating source + detector','Fixed detector, moving source','Fixed detector, arc source'}
        s=examples(strcmp({examples.name},'Three tilted panels')).settings;
        s.layout=layout{1}; s.source_motion=motion{1}; s.n_views=3; s.module_shape=[3 4];
        s.n_modules=3; s.angular_span_deg=40; s.panel_x_angles_deg=[5 0 -5]; s.panel_y_angles_deg=[0 15 -10];
        s.source_start_mm=[1000 -200 0]; s.source_end_mm=[1000 300 20];
        if strcmp(motion{1},'Rotating source + detector'), s.cor_mm=7; else, s.cor_mm=0; end
        geometryFile('save',yaml,struct('settings',s,'geo',[]));
        interpreter=getenv('TIGRE_PYTHON');
        if isempty(interpreter), command='conda run --no-capture-output -n recon python';
        else, command=['"' interpreter '"']; end
        [status,message]=system(sprintf('%s "%s" to-mat "%s" "%s"',command,script,yaml,reference));
        assert(status==0,'%s',message); expected=load(reference);
        actual=buildModularSurface(s);
        difference=max(abs(actual.pixel_centers_mm(:)-expected.surface.pixel_centers_mm(:)));
        assert(difference<.002,'TIGRE:test:parity','%s / %s coordinate error %g mm',layout{1},motion{1},difference);
        assert(max(abs(actual.source_positions_mm(:)-expected.surface.source_positions_mm(:)))<.002);
        assert(isequal(actual.valid_mask,logical(expected.surface.valid_mask)));
        assert(isequal(actual.module_slices,expected.surface.module_slices));
    end
end
fprintf('PASS six MATLAB/Python layout and motion geometry comparisons, YAML round trips\n');
end
function removeFiles(files)
for k=1:numel(files), if exist(files{k},'file'), delete(files{k}); end; end
end
