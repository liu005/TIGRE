function image=Atb_surface(projections,geo,surface,device)
%ATB_SURFACE Exact adjoint of Ax_surface, including its valid_mask.
% projections: [rows columns views]; result: single [nx ny nz].
% This is a matched backprojector, not an FDK reconstruction.
if nargin<4, device=0; end
image=surfaceProject(true,projections,geo,surface,device);
end
