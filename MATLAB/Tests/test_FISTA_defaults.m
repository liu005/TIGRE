function test_FISTA_defaults()
% Default hyper estimation must be executable, preserve lambda, and honor GPUs.
geo=defaultGeometry('nVoxel',[16;16;16],'nDetector',[24;24]);
angles=(0:11)*2*pi/12; truth=single(headPhantom([16 16 16]));
devices=GpuIds(); projections=Ax(truth,geo,angles,'Siddon','gpuids',devices);
rng(17); sigma=power_method(geo,angles,devices);
assert(isfinite(sigma) && sigma>0);
hyper=2.02*double(sigma)^2;
rng(17);
[automatic,residuals,quality]=FISTA(projections,geo,angles,3,'verbose',false, ...
    'tviter',2,'lambda',.05,'QualMeas',{'RMSE','CC'},'gpuids',devices);
[explicit,~,~]=FISTA(projections,geo,angles,3,'verbose',false,'tviter',2, ...
    'lambda',.05,'hyper',hyper,'gpuids',devices);
assert(all(isfinite(automatic(:))) && all(isfinite(explicit(:))));
assert(norm(double(automatic(:)-explicit(:)))/max(norm(double(explicit(:))),eps)<1e-4);
assert(size(quality,2)==numel(residuals));
assert(isnan(quality(2,1)) && all(isfinite(quality(1,:))));
% Also exercise the no-options path used by d20_Algorithms05.
result=FISTA(projections,geo,angles,2);
assert(all(isfinite(result(:))));
failed=false;
try, FISTA(projections,geo,angles,1,'hyper',0,'verbose',false); catch, failed=true; end
assert(failed,'Invalid explicit hyper must be rejected');
fprintf('PASS FISTA automatic norm, explicit hyper equivalence, quality histories and default demo call\n');
end
