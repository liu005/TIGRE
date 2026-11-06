function test_userManualExamples()
% Execute the MATLAB workflows in the user manual against the current APIs.
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
manual=fileread(fullfile(root,'Frontispiece','TIGRE_User_Manual.md'));
names={'matlab-conventional','matlab-surface','matlab-operator'};
for k=1:numel(names)
    pattern=['(?s)```matlab\r?\n% manual-example: ' names{k} '\r?\n(.*?)```'];
    block=regexp(manual,pattern,'tokens');
    assert(numel(block)==1,'Expected one manual example named %s',names{k});
    eval(block{1}{1});
end
fprintf('PASS MATLAB user manual examples\n');
end
