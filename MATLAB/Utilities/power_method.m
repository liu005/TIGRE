function [sigma] = power_method(geo,angles,gpuids)
% Estimate sigma_max(A) with the same Siddon/matched pair used by FISTA.
if nargin<3, gpuids=GpuIds(); end
tol=1e-6;
maxiter=100;
arr_old=randn(geo.nVoxel','single');
error = tol + 1;
i = 0;
while error >= tol

    % very verbose and inefficient for now
    omega = Ax(arr_old,geo,angles,'Siddon','gpuids',gpuids);
    alpha = im3Dnorm(omega,'L2');
    assert(isfinite(alpha) && alpha>0,'TIGRE:power_method:operator','Forward operator has zero/nonfinite norm');
    u = (1.0 / alpha) * omega;
    z = Atb(u,geo,angles,'matched','gpuids',gpuids);
    beta = im3Dnorm(z,'L2');
    assert(isfinite(beta) && beta>0,'TIGRE:power_method:operator','Backprojector has zero/nonfinite norm');
    arr = (1.0 / beta) * z;
    error = im3Dnorm(Ax(arr,geo,angles,'Siddon','gpuids',gpuids) - beta * u,'L2');
    sigma = beta;
    arr_old = arr;
    i = i+1;
    if i >= maxiter
        return
    end
end

end

