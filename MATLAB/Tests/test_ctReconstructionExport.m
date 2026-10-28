function test_ctReconstructionExport()
% End-to-end native CUDA smoke tests for scripts exported from saved YAML.
assert(exist('Surface_mex','file')==3 && exist('FDKSurface_mex','file')==3, ...
    'TIGRE:test:cuda','Activate a native surface CUDA build first');
folder=tempname; mkdir(folder); cleanup=onCleanup(@() rmdir(folder,'s'));
examples=detectorGeometryExamples(); s=examples(strcmp({examples.name},'TIGRE: circular cone-beam CT')).settings;
s.n_views=24; s.source_radius_mm=100; s.detector_radius_mm=50;
s.module_shape=[16 32]; s.pixel_spacing_mm=[2 2];
g=struct('nVoxel',[16;12;8],'sVoxel',[32;24;16],'offOrigin',zeros(3,1));
g.dVoxel=g.sVoxel./g.nVoxel;
for moving=[false true]
    if moving, g.offOrigin=[zeros(2,24);linspace(-2,2,24)]; end
    yaml=fullfile(folder,'geometry.yaml'); target=fullfile(folder,'reconstruct.m');
    geometryFile('save',yaml,struct('settings',s,'geo',g));
    exportCTReconstructionScript(yaml,target);
    code=strrep(fileread(target),'iterations=20;','iterations=3;');
    file=fopen(target,'w'); fprintf(file,'%s',code); fclose(file);
    run(target);
    saved=load(fullfile(folder,'reconstruction_results.mat'));
    assert(isequal(size(saved.cgls),[16 12 8]) && all(isfinite(saved.cgls(:))));
    if moving, assert(isfield(saved,'fdk_skip_reason'));
    else, assert(isfield(saved,'fdk') && all(isfinite(saved.fdk(:)))); end
    fprintf('PASS MATLAB exported CUDA script moving=%d, CGLS residual %.6g\n',moving,saved.residuals(end));
end
end
