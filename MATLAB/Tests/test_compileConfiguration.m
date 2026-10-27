function test_compileConfiguration()
% Check the modern Compile entry point without invoking compiler mutations.
folder=pwd; oldPath=path;
output=evalc('Compile(''-dry_run'')');
assert(contains(output,'sm_120') && contains(output,fullfile('Mex_files','win64')));
assert(strcmp(pwd,folder) && strcmp(path,oldPath));
target=fullfile(folder,'.codex-results','compile_configuration');
output=evalc('Compile(''-dry_run'',''-gpu_arch'',''sm_120'',''-outdir'',target)');
assert(contains(output,target));
assert(~exist(target,'dir'),'Dry run must not create build directories');
failed=false;
try, Compile('-outdir'); catch, failed=true; end
assert(failed,'Missing flag value must fail');
fprintf('PASS Compile modern GPU detection, output selection, dry-run and context restoration\n');
end
