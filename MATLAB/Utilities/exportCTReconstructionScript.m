function exportCTReconstructionScript(yamlPath,scriptPath)
%EXPORTCTRECONSTRUCTIONSCRIPT Export a native FDK/CGLS example for saved YAML.
assert(exist(yamlPath,'file')==2,'TIGRE:export:yaml','Save geometry YAML before exporting');
[~,~,extension]=fileparts(yamlPath);
assert(any(strcmpi(extension,{'.yaml','.yml'})),'TIGRE:export:yaml','Supply a YAML geometry');
[~,~,extension]=fileparts(scriptPath);
assert(strcmpi(extension,'.m'),'TIGRE:export:script','Script must end in .m');
[ok,info]=fileattrib(yamlPath); assert(ok,'TIGRE:export:path','Cannot resolve YAML path');
document=geometryFile('load',yamlPath); templateName='ctReconstructionTemplate.txt';
if isfield(document.geo,'simulation_mode') && strcmp(document.geo.simulation_mode,'dr'), templateName='drSimulationTemplate.txt'; end
template=fileread(fullfile(fileparts(mfilename('fullpath')),templateName));
text=strrep(template,'__SAVED_GEOMETRY__',strrep(info.Name,'''',''''''));
file=fopen(scriptPath,'w'); assert(file>=0,'TIGRE:export:write','Cannot write reconstruction script');
cleanup=onCleanup(@() fclose(file));
assert(fprintf(file,'%s',text)==numel(text),'TIGRE:export:write','Incomplete script write');
end
