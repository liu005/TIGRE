function test_dualEnergy()
% CPU-only regression for shared calibration, interpolation and VM synthesis.
scan=reshape(single([2 4 6 12]),1,2,2);
value=proj_interp(scan,[0 2],.5,'linear',1);
assert(max(abs(value(:)-single([3;6])))<1e-6);
assert(isnan(proj_interp(scan,[0 2],-.5,'skip',.1)));
low=reshape(linspace(.05,1,24),2,3,4);
high=.8*low; lowAngles=[0 .2 .4 .6]; highAngles=[0 .1 .3 .6];
[first,second,angles]=DeDecompose(low,lowAngles,high,highAngles, ...
    'scanner','VarianTrueBeam','kVl',80,'kVh',140,'material1','Al','material2','PMMA','interpolation','linear');
vm=MakeVMproj(first,second,67,'material1','Al','material2','PMMA');
assert(isequal(size(vm),[2 3 4]) && all(isfinite(vm(:))));
known=MakeVMproj(2,3,100,'material1','Al','material2','PMMA');
assert(abs(known-(2*.046+3*.0194))<1e-12);
[other,~,~]=DeDecompose(low,lowAngles,high,highAngles, ...
    'scanner','VarianTrueBeam','kVl',90,'kVh',140,'material1','Al','material2','PMMA','interpolation','linear');
assert(norm(other(:)-first(:))>1);
[empty1,empty2,emptyAngles]=DeDecompose(low,lowAngles+4,high,highAngles, ...
    'scanner','VarianTrueBeam','material1','Al','material2','PMMA','interpolation','skip','tolerance',.01);
assert(isempty(empty1) && isempty(empty2) && isempty(emptyAngles));
% Export a small deterministic oracle for a live Python comparison.
folder=fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'.codex-results');
if exist(folder,'dir')
    save(fullfile(folder,'dual_energy_matlab_reference.mat'), ...
        'low','high','lowAngles','highAngles','first','second','angles','vm','other');
end
fprintf('PASS MATLAB dual-energy cubic transfer, 80/90 kVp selection, interpolation, skip and VM synthesis\n');
end
