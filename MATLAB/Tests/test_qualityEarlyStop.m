function test_qualityEarlyStop()
% Trigger the actual residual-increase branch with a deliberately divergent relaxation.
g=defaultGeometry('nVoxel',[16;16;16],'nDetector',[24;24]);
angles=(0:11)*2*pi/12; truth=single(headPhantom([16 16 16]));
p=Ax(truth,g,angles,'interpolated'); histories=cell(1,3);
for algorithm={@SIRT,@SART,@OS_SART}
    [~,residual,quality]=algorithm{1}(p,g,angles,12,'lambda',-.1,'nonneg',false, ...
        'QualMeas',{'RMSE','CC'},'verbose',false);
    assert(numel(residual)>=2 && numel(residual)<12,'Early stopping was not exercised');
    assert(size(quality,2)==numel(residual),'History lengths must match completed iterations');
    assert(residual(end)>residual(end-1),'The final history sample must include the stopping iteration');
    fprintf('PASS %s stopped at %d with truncated residual/quality histories\n',func2str(algorithm{1}),numel(residual));
end
% Exercise unequal lengths in arbitrary order, including a legitimate zero.
histories={zeros(2,8),ones(2,3),2*ones(2,11)};
ax=plotQualityMeasures(histories,{'SIRT','SART','OS-SART'},{'RMSE','CC'});
cleanup=onCleanup(@() close('all'));
for k=1:numel(ax)
    lines=findobj(ax(k),'Type','line');
    for j=1:numel(lines)
        name=lines(j).DisplayName; index=find(strcmp(name,{'SIRT','SART','OS-SART'}));
        assert(numel(lines(j).YData)==size(histories{index},2));
        assert(isequal(lines(j).XData,1:size(histories{index},2)));
    end
end
fprintf('PASS unequal-length quality plots preserve actual iteration counts and zero values\n');
end
