function test_modularDetectorDemo()
% Run the actual head-phantom demo with native CUDA; callers may hide figures.
d26_ModularDetector;
assert(isequal(size(head),[64 64 64]));
assert(isequal(size(modularProjections),[80 100 90]));
assert(size(modularSurface.module_slices,1)==3);
assert(nnz(~modularSurface.valid_mask)>0 && settings.cor_mm==2);
assert(all(modularProjections(~modularSurface.valid_mask)==0));
assert(all(isfinite(modularImage(:))));
assert(numel(modularResidual)==50 && modularResidual(end)<modularResidual(1));
assert(relativeError<.3);
close all;
fprintf('PASS demo 26: head phantom, tilted panels, gaps, COR, native CGLS error %.4f\n',relativeError);
end
