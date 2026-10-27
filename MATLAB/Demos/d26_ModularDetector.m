%% Demo 26: Head phantom with tilted modular detectors
% Requires: native surface CUDA and the bundled head phantom.
% Native surface CUDA projection and CGLS reconstruction; no flattening,
% Python reconstruction bridge, scanner data or gpuArray is required.
% Build with Compile; for the isolated custom build first UseBlackwellBuild.
clear;
close all;
assert(exist('Surface_mex','file')==3,'TIGRE:demo:surfaceBuild', ...
    'Build surface kernels with Compile, or CompileBlackwell(''surface'') then UseBlackwellBuild.');

%% TIGRE head phantom and a fixed reconstruction volume (world xyz, mm)
geo=defaultGeometry('nVoxel',[64;64;64]);
geo.sVoxel=[64;64;64]; geo.dVoxel=geo.sVoxel./geo.nVoxel;
geo.offOrigin=[0;0;0];
head=single(headPhantom(double(geo.nVoxel(:))'));

%% Three tiled panels on a rotating cylindrical array
examples=detectorGeometryExamples();
settings=examples(strcmp({examples.name},'Circular CT: curved modular detector')).settings;
settings.n_views=90; settings.scan_span_deg=360; settings.orbit_start_deg=0;
settings.source_radius_mm=300; settings.source_z_mm=0;
settings.detector_radius_mm=200; settings.n_modules=3;
settings.module_shape=[80 32]; settings.pixel_spacing_mm=[1.5 1.5];
settings.angular_span_deg=28; settings.column_gap=2;
% Local x/y tilts (degrees) are specified independently for each panel.
% The cylindrical placement also sets each panel's local z rotation.
settings.panel_x_angles_deg=[-3 0 3];
settings.panel_y_angles_deg=[2 -2 1];
% COR is a single scan-wide shift, already encoded in the explicit rays.
settings.cor_mm=2;
modularSurface=buildModularSurface(settings);
angles=(0:settings.n_views-1)*2*pi/settings.n_views;

%% Project the voxelized head directly onto the measured pixel positions
modularProjections=Ax_surface(head,geo,modularSurface);
assert(all(modularProjections(~modularSurface.valid_mask)==0));
plotProj(modularProjections,angles);
modularAnimation=animateGeometry(geo,[],1,'SD',true,'',modularSurface);

%% Matched native reconstruction: masked gaps never become measured rays
[modularImage,modularResidual]=CGLS_surface(modularProjections,geo,modularSurface,50);
relativeError=norm(double(modularImage(:)-head(:)))/norm(double(head(:)));
fprintf('Modular head CGLS: %d iterations, relative volume error %.4f\n', ...
    numel(modularResidual),relativeError);
% Individually tilted panels violate surface FDK's extrusion requirement.
% Use CGLS_surface or SIRT_surface for this geometry.

%% Compare matching central xy slices with a common intensity scale
slice=ceil(geo.nVoxel(3)/2); limits=[double(min(head(:))) double(max(head(:)))];
figure('Name','Modular detector: head phantom reconstruction');
subplot(1,3,1); imagesc(head(:,:,slice),limits); axis image; title('Input head');
subplot(1,3,2); imagesc(modularImage(:,:,slice),limits); axis image; title('Surface CGLS');
subplot(1,3,3); imagesc(abs(modularImage(:,:,slice)-head(:,:,slice))); axis image;
title('Absolute error'); colorbar; colormap gray;
figure('Name','Modular head: projection residual');
semilogy(1:numel(modularResidual),modularResidual); grid on;
xlabel('Completed iteration'); ylabel('Projection residual');

%% Optional inspection and exact geometry export
% app=detectorGeometryGUI(modularSurface);
% geometryFile('save','modular_head.mat',struct('surface',modularSurface,'geo',geo));
% YAML + NPZ interchange uses Python recon for file conversion only:
% geometryFile('save','modular_head.yaml',struct('surface',modularSurface,'geo',geo));
