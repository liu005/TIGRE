%% Demo 25: Head phantom with a smooth curved detector
% Requires: native surface CUDA and the bundled head phantom.
% Native MATLAB geometry, visualization and CUDA surface reconstruction.
% Each ray is defined by a source and a measured pixel centre in world xyz.
% The bundled head is projected and reconstructed using native surface CUDA;
% do not send explicit surface arrays to the standard flat-detector Ax/FDK.
clear;
close all;
assert(exist('Surface_mex','file')==3,'TIGRE:demo:surfaceBuild', ...
    'Build surface kernels with Compile, or CompileBlackwell(''surface'') then UseBlackwellBuild.');
geo=defaultGeometry('nVoxel',[64;64;64]);
geo.sVoxel=[64;64;64]; geo.dVoxel=geo.sVoxel./geo.nVoxel;
geo.offOrigin=[0;0;0];
head=single(headPhantom(double(geo.nVoxel(:))'));
angles=(0:89)*2*pi/90;
rows=80; columns=96; dso=300; dsd=500;

%% Smooth curved detector: a cylinder centred on the source at each view
% The fan and row coverage include the full head at the scan magnification.
fan=linspace(-.15,.15,columns);
z=((0:rows-1)-(rows-1)/2)*1.5;
[fanGrid,zGrid]=meshgrid(fan,z);
local=cat(3,dso-dsd*cos(fanGrid),dsd*sin(fanGrid),zGrid);
smoothSurface=struct('source_positions_mm',zeros(3,numel(angles),'single'), ...
    'pixel_centers_mm',zeros(rows,columns,3,numel(angles),'single'), ...
    'valid_mask',true(rows,columns,numel(angles)), ...
    'module_slices',[0 rows 0 columns]);
for k=1:numel(angles)
    a=angles(k); R=[cos(a) -sin(a) 0;sin(a) cos(a) 0;0 0 1];
    smoothSurface.source_positions_mm(:,k)=single(R*[dso;0;0]);
    smoothSurface.pixel_centers_mm(:,:,:,k)=reshape(single(reshape(local,[],3)*R'),rows,columns,3);
end
smoothProjections=Ax_surface(head,geo,smoothSurface);
plotProj(smoothProjections,angles);
smoothAnimation=animateGeometry(geo,[],1,'SD',true,'',smoothSurface);

%% Native reconstruction: use every measured ray without flattening panels
[smoothImage,smoothResidual]=CGLS_surface(smoothProjections,geo,smoothSurface,50);
relativeError=norm(double(smoothImage(:)-head(:)))/norm(double(head(:)));
fprintf('Curved head CGLS: %d iterations, relative volume error %.4f\n', ...
    numel(smoothResidual),relativeError);
slice=ceil(geo.nVoxel(3)/2);
limits=[double(min(head(:))) double(max(head(:)))];
figure('Name','Smooth curved detector: head phantom reconstruction');
subplot(1,3,1); imagesc(head(:,:,slice),limits); axis image; title('Input head');
subplot(1,3,2); imagesc(smoothImage(:,:,slice),limits); axis image; title('Surface CGLS');
subplot(1,3,3); imagesc(abs(smoothImage(:,:,slice)-head(:,:,slice))); axis image;
title('Absolute error'); colorbar;
colormap gray;
figure('Name','Surface reconstruction residuals');
semilogy(1:numel(smoothResidual),smoothResidual); grid on;
xlabel('Completed iteration'); ylabel('Projection residual');
if exist('FDKSurface_mex','file')==3
    % FDK accepts this smooth extruded cylinder.
    smoothFDK=FDK_surface(smoothProjections,geo,smoothSurface,'Filter','hann');
    figure('Name','Surface FDK: smooth cylinder');
    subplot(1,2,1); imagesc(head(:,:,slice),limits); axis image; title('Input head');
    subplot(1,2,2); imagesc(smoothFDK(:,:,slice),limits); axis image;
    title('Approximate curved-detector FDK'); colormap gray;
    fdkRelativeError=norm(double(smoothFDK(:)-head(:)))/norm(double(head(:)));
    fprintf('Curved head FDK: relative volume error %.4f\n',fdkRelativeError);
end

%% Inspect arrays and reuse/export the geometry
% source_positions_mm: 3 x views
% pixel_centers_mm: rows x columns x 3 x views
% valid_mask: rows x columns x views (false for module gaps)
% module_slices: zero-based half-open [rowStart rowStop columnStart columnStop]
% No flattening/rebinning was performed: each measurement uses its own ray.
% Optional interactive inspection:
% app=detectorGeometryGUI(smoothSurface);
% Optional exact archive, readable by the MATLAB GUI:
% geometryFile('save','curved_surface.mat',struct('surface',smoothSurface,'geo',geo));
% Optional shared YAML + NPZ export (requires Python recon for file conversion):
% geometryFile('save','curved_surface.yaml',struct('surface',smoothSurface,'geo',geo));
% See d26_ModularDetector for tilted panels and head-phantom reconstruction.
fprintf('Smooth cylinder: %d valid samples.\n',nnz(smoothSurface.valid_mask));
