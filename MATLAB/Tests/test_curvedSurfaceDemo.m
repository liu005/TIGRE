function test_curvedSurfaceDemo()
surface=struct('source_positions_mm',single([100;0;0]), ...
    'pixel_centers_mm',reshape(single([-100 0 0]),1,1,3), ...
    'valid_mask',true);
assert(abs(sphereSurfaceProjections(surface,10)-20)<1e-6);
surface.source_positions_mm(:)=0;
surface.pixel_centers_mm=reshape(single([100 0 0]),1,1,3);
assert(abs(sphereSurfaceProjections(surface,10)-10)<1e-6);
surface.valid_mask=false;
assert(sphereSurfaceProjections(surface,10)==0);
% Run the complete native head-phantom demo.
d25_CurvedDetector;
assert(isequal(size(head),[64 64 64]));
assert(isequal(size(smoothProjections),[80 96 90]));
assert(all(isfinite(smoothProjections(:))));
assert(max(smoothProjections(:))>0);
assert(all(isfinite(smoothImage(:))));
assert(numel(smoothResidual)==50 && relativeError<.3);
assert(smoothResidual(end)<smoothResidual(1));
if exist('FDKSurface_mex','file')==3
    assert(all(isfinite(smoothFDK(:))) && fdkRelativeError<.5);
end
close all;
fprintf('PASS analytic ray utility and complete curved head demo 25: CGLS error %.4f\n',relativeError);
end
