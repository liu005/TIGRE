function out=UseBlackwellBuild()
%USEBLACKWELLBUILD Activate the isolated custom CUDA build for this session.
% No savepath, MATLAB installation change, gpuArray, or Python is required.
folder=fileparts(mfilename('fullpath')); root=fileparts(folder);
out=fullfile(root,'.codex-results','matlab_blackwell');
required={'Ax_mex','Atb_mex','Surface_mex','FDKSurface_mex','getGpuCount_mex','getGpuName_mex', ...
    'minTV','AwminTV','tvDenoise','AddNoise','minPICCS','mexReadXim'};
for k=1:numel(required)
    assert(exist(fullfile(out,[required{k} '.mexw64']),'file')~=0, ...
        'TIGRE:build:missing','Run CompileBlackwell(''all'') first; missing %s',required{k});
end
cuda=getenv('CUDA_PATH');
if isempty(cuda), cuda='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'; end
setenv('PATH',[fullfile(cuda,'bin') ';' getenv('PATH')]);
addpath(genpath(folder));
clear Ax_mex Atb_mex Surface_mex FDKSurface_mex getGpuCount_mex getGpuName_mex minTV AwminTV tvDenoise AddNoise minPICCS mexReadXim;
addpath(out,'-begin');
fprintf('Using isolated native CUDA MEX files from %s\n',out);
fprintf('GPU: %s\n',getGpuName_mex(int32(0)));
end
