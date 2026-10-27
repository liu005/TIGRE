function out=CompileBlackwell(stage,varargin)
%COMPILEBLACKWELL Experimental isolated R2020a / VS2022 / CUDA 13 build.
% stage='gpu' builds enumeration; 'surface' adds surface kernels;
% 'projectors' adds all projectors; 'all' builds the
% standard TIGRE MEX set, including explicit surface projectors.
% Defaults to isolated output. Compile.m can supply the normal MEX output path.
% Saved mex compiler preferences are not overwritten.
if nargin<1, stage='gpu'; end
assert(any(strcmp(stage,{'gpu','surface','projectors','all'})),'TIGRE:build:stage','Use gpu, surface, projectors or all');
options=inputParser;
addParameter(options,'OutDir',''); addParameter(options,'CUDAPath','');
addParameter(options,'MexXML',''); addParameter(options,'GPUArch','sm_120');
addParameter(options,'NoPinnedMemory',false); addParameter(options,'Verbose',true);
parse(options,varargin{:}); opt=options.Results;
assert(ispc,'TIGRE:build:platform' ,'This experimental configuration is Windows-only');
folder=fileparts(mfilename('fullpath')); previous=pwd; cleanup=onCleanup(@() cd(previous)); cd(folder);
root=fileparts(folder); out=opt.OutDir;
if isempty(out), out=fullfile(root,'.codex-results','matlab_blackwell'); end
if ~exist(out,'dir'), mkdir(out); end
cuda=opt.CUDAPath; if isempty(cuda), cuda=getenv('CUDA_PATH'); end
if isempty(cuda), cuda='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'; end
assert(exist(fullfile(cuda,'bin','nvcc.exe'),'file')==2,'TIGRE:build:cuda','CUDA nvcc not found');
setenv('CUDA_PATH',cuda);
setenv('PATH',[fullfile(cuda,'bin') ';' getenv('PATH')]);
template=opt.MexXML;
if isempty(template), template=fullfile(folder,'mex_CUDA_win64_MVS2022.xml'); end
xml=fileread(template);
architectures=regexp(strtrim(char(opt.GPUArch)),'[,\s]+','split');
[status,supported]=system(['"' fullfile(cuda,'bin','nvcc.exe') '" --list-gpu-code']);
assert(status==0,'TIGRE:build:architectures','Cannot query nvcc GPU targets');
gencode='';
for k=1:numel(architectures)
    target=architectures{k};
    assert(~isempty(regexp(target,'^sm_[0-9]+$','once')) && ...
        any(strcmp(strsplit(strtrim(supported)),target)), ...
        'TIGRE:build:architecture','nvcc does not support target %s',target);
    gencode=[gencode sprintf('-gencode=arch=compute_%s,code=%s ',target(4:end),target)]; %#ok<AGROW>
end
flags=['COMPFLAGS="' gencode '-std=c++17 ' ...
    '--default-stream per-thread --compiler-options=/c,/GR,/W3,/EHsc,/nologo,/MD"'];
xml=regexprep(xml,'COMPFLAGS="[^"]*"',flags,'once');
config=fullfile(out,'mex_blackwell.xml'); fid=fopen(config,'w'); fprintf(fid,'%s',xml); fclose(fid);
common=fullfile(root,'Common','CUDA');
diary(fullfile(out,['build_' stage '.log'])); logger=onCleanup(@() diary('off'));
base={'-largeArrayDims','-f',config,'-outdir',out};
if opt.Verbose, base=[{'-v'} base]; end
define='-DNO_FLAGS'; if opt.NoPinnedMemory, define='-DNO_PINNED_MEMORY'; end
% Release this session's copies before replacing a previously used MEX file.
clear getGpuCount_mex getGpuName_mex;
if any(strcmp(stage,{'projectors','all'})), clear Ax_mex Atb_mex; end
if strcmp(stage,'all'), clear minTV AwminTV tvDenoise AddNoise minPICCS mexReadXim; end
mex(base{:},'./Utilities/GPU/getGpuCount_mex.cpp',fullfile(common,'gpuUtils.cu'));
mex(base{:},'./Utilities/GPU/getGpuName_mex.cpp',fullfile(common,'gpuUtils.cu'));
if any(strcmp(stage,{'projectors','all'}))
    mex(base{:},'./Utilities/cuda_interface/Ax_mex.cpp', ...
        fullfile(common,'ray_interpolated_projection.cu'),fullfile(common,'Siddon_projection.cu'), ...
        fullfile(common,'ray_interpolated_projection_parallel.cu'),fullfile(common,'Siddon_projection_parallel.cu'), ...
        fullfile(common,'GpuIds.cpp'),fullfile(common,'TIGRE_common.cpp'),define);
    mex(base{:},'./Utilities/cuda_interface/Atb_mex.cpp', ...
        fullfile(common,'voxel_backprojection.cu'),fullfile(common,'voxel_backprojection2.cu'), ...
        fullfile(common,'voxel_backprojection_parallel.cu'),fullfile(common,'GpuIds.cpp'), ...
        fullfile(common,'TIGRE_common.cpp'),define);
elseif ~any(strcmp(stage,{'gpu','surface'}))
    error('TIGRE:build:stage','Use gpu, surface, projectors or all');
end
if any(strcmp(stage,{'surface','projectors','all'}))
    clear Surface_mex FDKSurface_mex;
    mex(base{:},'./Utilities/cuda_interface/Surface_mex.cpp', ...
        fullfile(common,'Siddon_surface_projection.cu'));
    mex(base{:},'./Utilities/cuda_interface/FDKSurface_mex.cpp', ...
        fullfile(common,'Siddon_surface_projection.cu'));
end
if strcmp(stage,'all')
    shared={fullfile(common,'GpuIds.cpp'),fullfile(common,'gpuUtils.cu'),fullfile(common,'TIGRE_common.cpp')};
    mex(base{:},'./Utilities/cuda_interface/minTV.cpp',fullfile(common,'GD_TV.cu'),shared{:},define);
    mex(base{:},'./Utilities/cuda_interface/AwminTV.cpp',fullfile(common,'GD_AwTV.cu'),shared{:},define);
    mex(base{:},'./Utilities/cuda_interface/tvDenoise.cpp',fullfile(common,'tv_proximal.cu'),shared{:},define);
    mex(base{:},'./Utilities/cuda_interface/AddNoise.cpp',fullfile(common,'RandomNumberGenerator.cu'),shared{:},define);
    mex(base{:},'./Utilities/cuda_interface/minPICCS.cpp',fullfile(common,'PICCS.cu'),shared{:},define);
    mex(base{:},'./Utilities/IO/VarianCBCT/mexReadXim.cpp');
end
fprintf('MEX build output: %s\n',out);
end
