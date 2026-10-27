function test_qualityMeasuresGPU()
% Small native-GPU reproduction of d10's metric options, without plotting.
g=defaultGeometry('nVoxel',[16;16;16],'nDetector',[24;24]);
angles=(0:11)*2*pi/12; truth=single(headPhantom([16 16 16]));
p=Ax(truth,g,angles,'interpolated');
for algorithm={@SIRT,@SART,@OS_SART}
    [r,~,q]=algorithm{1}(p,g,angles,2,'QualMeas',{'RMSE','CC','MSSIM','UQI'},'verbose',false);
    assert(all(isfinite(r(:))) && size(q,1)==4);
    % Comparing the first iterate against a constant zero initial image has
    % undefined Pearson correlation; later nonconstant comparisons are finite.
    assert(isnan(q(2,1)) && isfinite(q(2,end)));
    [~,~,q]=algorithm{1}(p,g,angles,2,'QualMeas',{'CC'},'groundTruth',truth,'verbose',false);
    assert(size(q,1)==2 && all(isfinite(q(:))));
end
fprintf('PASS native GPU SIRT, SART and OS-SART metric lists and ground-truth comparisons\n');
end
