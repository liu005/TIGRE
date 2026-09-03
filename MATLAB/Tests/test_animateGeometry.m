function test_animateGeometry()
%TEST_ANIMATEGEOMETRY Smoke tests for animateGeometry (no GPU).
%
%   Checks: an animated 'SD' run writes a GIF and returns a live figure;
%   a static 'obj' run with 3xN Euler angles works; defaults (no angles)
%   work; the rotation argument is validated.
%
%   Run:  cd MATLAB/Tests; test_animateGeometry
%--------------------------------------------------------------------------
% This file is part of the TIGRE Toolbox. License: BSD, see
% https://github.com/CERN/TIGRE/blob/master/LICENSE
% Coded by:           Yi Liu
%--------------------------------------------------------------------------
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'Utilities'));

geo.DSD = 1536;  geo.DSO = 1000;
geo.nDetector = [512; 512];  geo.dDetector = [0.8; 0.8];
geo.sDetector = geo.nDetector .* geo.dDetector;
geo.nVoxel = [128; 128; 128];  geo.sVoxel = [256; 256; 256] / 2;
geo.dVoxel = geo.sVoxel ./ geo.nVoxel;
geo.offOrigin = [10; -5; 0];  geo.offDetector = [10; -5];
geo.rotDetector = [0.1; 0.05; 0.02];  geo.COR = 2;
geo.accuracy = 0.5;  geo.mode = 'cone';
angles = linspace(0, 2*pi, 10);

% 1. animated, source/detector moving, GIF written
base = tempname;
h = animateGeometry(geo, angles, 3, 'SD', true, base);
assert(isgraphics(h), 'no figure handle returned');
gif = [base, '_geometry.gif'];
assert(exist(gif, 'file') == 2, 'GIF was not written');
info = imfinfo(gif);
assert(numel(info) == numel(angles), 'GIF has %d frames, expected %d', numel(info), numel(angles));
delete(gif);  close(h);
fprintf('PASS 1: SD animation, %d-frame GIF written\n', numel(angles));

% 2. static, object rotating, 3xN Euler angles
h = animateGeometry(geo, [angles; 0.1 * angles; zeros(1, 10)], 4, 'obj', false);
assert(isgraphics(h));  close(h);
fprintf('PASS 2: static obj plot with 3xN angles\n');

% 3. defaults
h = animateGeometry(geo, linspace(0, 2*pi, 5));
assert(isgraphics(h));  close(h);
fprintf('PASS 3: defaults\n');

% 4. argument validation
threw = false;
try
    animateGeometry(geo, angles, 1, 'bogus');
catch e
    threw = strcmp(e.identifier, 'TIGRE:animateGeometry:rotation');
end
assert(threw, 'bad rotation string was not rejected');
close all;
fprintf('PASS 4: rotation argument validated\n');

fprintf('test_animateGeometry: ALL PASSED\n');
end
