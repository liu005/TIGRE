function test_VarianLoaderValidation()
% Missing datasets must fail with useful errors before any GPU access.
missing=tempname;
verify(@() GeometryFromXML(missing),'TIGRE:Varian:DataFolder');
verify(@() VarianDataLoader(missing),'TIGRE:Varian:DataFolder');
folder=tempname; mkdir(folder); cleanup=onCleanup(@() rmdir(folder));
verify(@() GeometryFromXML(folder),'TIGRE:Varian:MissingScanXML');
verify(@() VarianDataLoader(folder),'TIGRE:Varian:MissingScanXML');
fprintf('PASS Varian folder/Scan.xml checks occur before GPU initialization\n');
end
function verify(callback,identifier)
try
    callback();
catch ex
    assert(strcmp(ex.identifier,identifier),'Expected %s, got %s: %s',identifier,ex.identifier,ex.message);
    assert(~contains(ex.message,'fullfile'));
    return;
end
error('Expected a dataset-validation error');
end
