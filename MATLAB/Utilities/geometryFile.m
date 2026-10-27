function document=geometryFile(operation,path,document)
%GEOMETRYFILE Load/save native MAT geometry or shared version-1 YAML files.
% document.settings is editable; document.surface is exact inspection geometry.
% YAML/NPZ interchange uses Python's shared loader in conda env recon (SciPy,
% PyYAML and TIGRE). Set TIGRE_PYTHON to an alternative activated interpreter.
if strcmp(operation,'load')
    [~,~,extension]=fileparts(path);
    if strcmpi(extension,'.mat')
        document=load(path);
    else
        temporary=[tempname '.mat']; cleanup=onCleanup(@() removeTemp(temporary));
        bridge('to-mat',path,temporary); document=load(temporary);
    end
    if ~isfield(document,'geo'), document.geo=[]; end
    if isfield(document,'settings')
        % SciPy preserves YAML integer scalars as int64. MATLAB geometry
        % arithmetic (including linspace) expects floating-point settings.
        names=fieldnames(document.settings);
        for k=1:numel(names)
            value=document.settings.(names{k});
            if isnumeric(value), document.settings.(names{k})=double(value); end
        end
        document.surface=buildModularSurface(document.settings);
    elseif ~isfield(document,'surface') && isfield(document,'angles') && ~isempty(document.geo)
        document.surface=surfaceFromGeometry(document.geo,document.angles);
    end
    assert(isfield(document,'surface'),'TIGRE:geometryFile:format','File needs settings, surface, or geo plus angles');
elseif strcmp(operation,'save')
    [~,~,extension]=fileparts(path);
    if strcmpi(extension,'.mat')
        save(path,'-struct','document');
    elseif any(strcmpi(extension,{'.yaml','.yml'}))
        temporary=[tempname '.mat']; cleanup=onCleanup(@() removeTemp(temporary));
        save(temporary,'-struct','document'); bridge('from-mat',temporary,path);
    else
        error('TIGRE:geometryFile:format','Save as .mat, .yaml or .yml');
    end
else
    error('TIGRE:geometryFile:operation','Use load or save');
end
end
function bridge(operation,source,target)
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
script=fullfile(root,'Python','tigre','utilities','geometry_matlab_bridge.py');
interpreter=getenv('TIGRE_PYTHON');
if isempty(interpreter), command='conda run --no-capture-output -n recon python';
else, command=['"' interpreter '"']; end
assert(~contains([source target script],'"'),'TIGRE:geometryFile:path','Unsupported quote in filename');
[status,message]=system(sprintf('%s "%s" %s "%s" "%s"',command,script,operation,source,target));
assert(status==0,'TIGRE:geometryFile:bridge','Geometry interchange failed: %s',message);
end
function removeTemp(path)
if exist(path,'file'), delete(path); end
end
