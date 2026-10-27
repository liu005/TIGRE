function metrics=test_blackwellBuild(out)
%TEST_BLACKWELLBUILD Exercise isolated native MEX files on the installed GPU.
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
if nargin<1, out=fullfile(root,'.codex-results','matlab_blackwell'); end
addpath(genpath(fullfile(root,'MATLAB'))); addpath(out,'-begin');
cuda=getenv('CUDA_PATH'); if isempty(cuda), cuda='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'; end
setenv('PATH',[fullfile(cuda,'bin') ';' getenv('PATH')]);
assert(startsWith(which('Ax_mex'),out) && startsWith(which('Atb_mex'),out), ...
    'TIGRE:test:binaries','Isolated Ax and Atb builds are required');
count=getGpuCount_mex(); assert(count>0,'TIGRE:test:GPU','No GPU detected');
name=getGpuName_mex(int32(0)); fprintf('Native MATLAB CUDA GPU: %s\n',name);
geo=struct('DSO',500,'DSD',800,'nDetector',[32;32],'dDetector',[1.5;1.5], ...
    'nVoxel',[16;16;16],'sVoxel',[16;16;16],'offOrigin',[0;0;0], ...
    'offDetector',[0;0],'rotDetector',[0;0;0],'COR',0,'mode','cone','accuracy',.5);
geo.sDetector=geo.nDetector.*geo.dDetector; geo.dVoxel=geo.sVoxel./geo.nVoxel;
axisAngles=(0:3)*pi/2;
box=ones(16,16,16,'single'); data=Ax(box,geo,axisAngles,'Siddon');
assert(all(isfinite(data(:))) && max(data(:))>0);
centre=data(16:17,16:17,:); expected=16*sqrt(800^2+2*.75^2)/800;
boxError=max(abs(double(centre(:))-expected));
assert(boxError<.01,'TIGRE:test:box','Central box ray mismatch: %g',boxError);
zerosData=Ax(zeros(size(box),'single'),geo,axisAngles,'Siddon'); assert(all(zerosData(:)==0));
twice=Ax(2*box,geo,axisAngles,'Siddon');
assert(max(abs(twice(:)-2*data(:)))<1e-4);
back=Atb(data,geo,axisAngles,'matched');
assert(isequal(size(back),[16 16 16]) && all(isfinite(back(:))) && max(back(:))>0);
zeroBack=Atb(zeros(size(data),'single'),geo,axisAngles,'matched'); assert(all(zeroBack(:)==0));
[x,y,z]=ndgrid((0:15)-7.5); truth=single((x/5).^2+(y/4).^2+(z/4).^2<1);
angles=(0:23)*2*pi/24; measured=Ax(truth,geo,angles,'Siddon');
reconstruction=CGLS(measured,geo,angles,40,'verbose',false);
relativeError=norm(double(reconstruction(:)-truth(:)))/norm(double(truth(:)));
predicted=Ax(reconstruction,geo,angles,'Siddon');
relativeResidual=norm(double(predicted(:)-measured(:)))/norm(double(measured(:)));
assert(relativeError<.2 && relativeResidual<.1,'TIGRE:test:CGLS','CGLS did not converge');
% Exercise the interpolated projector and FDK backprojector separately.
interpolated=Ax(truth,geo,angles,'interpolated');
fdk=FDK(interpolated,geo,angles);
assert(all(isfinite(interpolated(:))) && all(isfinite(fdk(:))) && max(fdk(:))>0);
metrics=struct('matlab',version,'gpu',name,'gpu_count',count,'box_ray_absolute_error',boxError, ...
    'cgls_relative_error',relativeError,'cgls_relative_residual',relativeResidual, ...
    'Ax_mex',which('Ax_mex'),'Atb_mex',which('Atb_mex'));
extraNames={'minTV','AwminTV','tvDenoise','AddNoise','minPICCS','mexReadXim'};
if all(cellfun(@(name) startsWith(which(name),out),extraNames))
    a=minTV(truth,.01,2); b=AwminTV(truth,.01,2,.01);
    c=tvDenoise(truth,10,3); d=AddNoise(10*ones(8,8,'single'),0,1);
    e=minPICCS(truth,truth,.01,2,.5,int32(0));
    for value={a,b,c,d,e}, assert(all(isfinite(value{1}(:)))); end
    metrics.extra_mex_smoke_tests=extraNames(1:5);
end
% Use MATLAB's bundled head loader and native MATLAB algorithms only.
head=single(headPhantom([32 32 32])); headGeo=geo;
headGeo.nVoxel=[32;32;32]; headGeo.sVoxel=[32;32;32]; headGeo.dVoxel=[1;1;1];
headGeo.nDetector=[48;48]; headGeo.sDetector=headGeo.nDetector.*headGeo.dDetector;
headAngles=(0:63)*2*pi/64;
headData=Ax(head,headGeo,headAngles,'Siddon');
headReconstruction=CGLS(headData,headGeo,headAngles,60,'verbose',false);
metrics.head_relative_error=norm(double(headReconstruction(:)-head(:)))/norm(double(head(:)));
assert(isfinite(metrics.head_relative_error) && metrics.head_relative_error<.5);
save(fullfile(out,'validation.mat'),'metrics','truth','reconstruction','measured');
save(fullfile(out,'head_validation.mat'),'head','headReconstruction','headGeo','headAngles','headData');
fid=fopen(fullfile(out,'validation.json'),'w'); fprintf(fid,'%s',jsonencode(metrics)); fclose(fid);
disp(metrics); fprintf('PASS native MATLAB GPU enumeration, Ax, Atb and CGLS reconstruction\n');
end
