%% Demo 20: FISTA with a TV proximal operator
% Requires: native TIGRE CUDA MEX and bundled or generated data.
%
%
%  This demo presents the FISTA algorithm in TIGRE. using a total variation proximal.
%  Total variation algorithms try to minimize the variation (gradient) of the
%  image, assuming its piecewise smooth, as most things in nature are (i.e.
%  human body). FISTA attempts to perform this with much faster convergence
%  than standard gradient algorithms such as SIRT.
% 
% 
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% This file is part of the TIGRE Toolbox
% 
% Copyright (c) 2015, University of Bath and 
%                     CERN-European Organization for Nuclear Research
%                     All rights reserved.
%
% License:            Open Source under BSD. 
%                     See the full license at
%                     https://github.com/CERN/TIGRE/blob/master/LICENSE
%
% Contact:            tigre.toolbox@gmail.com
% Codes:              https://github.com/CERN/TIGRE/
% Coded by:           Ander Biguri, Reuben Lindroos
%--------------------------------------------------------------------------
%% Initialize
clear;
close all;

%% Define Geometry
geo=defaultGeometry('nVoxel',[128;128;128]);                     

%% Load data and generate projections 
% see previous demo for explanation
angles=linspace(0,2*pi-2*pi/30,30);
head=headPhantom(geo.nVoxel);
projections=Ax(head,geo,angles,'interpolated');
noise_projections=addCTnoise(projections);

%% FISTA is a quadratically converging algorithm.

% 'hyper': Gradient step denominator. When omitted, FISTA estimates
%          2.02*sigma_max(A)^2 with power_method on the selected GPUs.
%          The computed value is printed and can be reused for the same
%          geometry to avoid repeating the estimation. Explicit hyper values
%          below are illustrative and are not universal defaults.
% 'tviter':  Number of iterations of Im3ddenoise to use. Default: 20
% 'lambda':  Multiplier for the tvlambda used, which is proportional to 
%            L (hyper). Default: 0.1
% 'verbose': get feedback or not. Default: 1
%--------------------------------------------------------------------------
imgFISTA_default = FISTA(noise_projections,geo,angles,100);

%% Adding a different hyper parameter
imgFISTA_hyper = FISTA(noise_projections,geo,angles,100,'hyper' ,2.e6);

%% Recon more significant tv parameters
imgFISTA_hightv = FISTA(noise_projections,geo,angles,100,'hyper',2.e6,'tviter',100,'lambda',20);

%% Faster converging FISTA
imgFISTA_mod = FISTA(noise_projections,geo,angles,100,'hyper',1.e5,'tviter',100,'lambda',20,'fista_q',1/50,'fista_p',1/20);

%% Plot results
plotImg([imgFISTA_default,imgFISTA_hyper,imgFISTA_hightv,imgFISTA_mod])