function test_animateGeometry()
%TEST_ANIMATEGEOMETRY Headless tests for animateGeometry / animateGeometryCube.
%
%   One-for-one port of Python/tests/test_animate_geometry.py (8 tests):
%     static:    returns a figure whose scene has the two cuboids and a
%                'CBCT geometry' title; default angles; out-of-range pos
%                falls back;
%     animation: 'SD' and 'obj' render every frame to a GIF; fname save;
%     cube:      single cuboid faces/extents; batched with rotations.
%   No GPU or display needed (runs under matlab -batch).
%
%   Run:  cd MATLAB/Tests; test_animateGeometry
%--------------------------------------------------------------------------
% This file is part of the TIGRE Toolbox. License: BSD, see
% https://github.com/CERN/TIGRE/blob/master/LICENSE
% Coded by:           Yi Liu
%--------------------------------------------------------------------------
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'Utilities'));

% make_geo(): small cone geometry exercising the offset/rotation paths.
% Python (z,y,x)/(v,u) values transposed to MATLAB (x,y,z)/(u,v).
geo.DSD = 1536;  geo.DSO = 1000;
geo.nVoxel = [64; 48; 32];  geo.sVoxel = [64; 48; 32];
geo.dVoxel = geo.sVoxel ./ geo.nVoxel;
geo.nDetector = [80; 60];  geo.dDetector = [0.8; 0.8];
geo.sDetector = geo.nDetector .* geo.dDetector;
geo.offDetector = [-15; 10];                       % [u; v]
geo.rotDetector = deg2rad([5; -2; 1]);
geo.offOrigin = [-8; 5; 0];                        % [x; y; z]
geo.COR = 1.5;
geo.accuracy = 0.5;  geo.mode = 'cone';
ANGLES = linspace(0, 2*pi, 7);  ANGLES = ANGLES(1:6);   % endpoint=False

cleanup = onCleanup(@() close('all'));

%% TestStatic ------------------------------------------------------------
% test_returns_axes_with_scene
h = animateGeometry(geo, ANGLES, 3, 'SD', false);
assert(isgraphics(h, 'figure'), 'static call must return a figure');
ax = findobj(h, 'Type', 'axes');
assert(numel(findobj(ax, 'Type', 'patch')) >= 2, 'scene lacks the two cuboids');
ttl = string(get(get(ax, 'Title'), 'String'));       % multi-line -> string array
assert(any(contains(ttl, 'CBCT geometry')), 'title missing');
close(h);
fprintf('PASS static: returns figure with scene\n');

% test_default_angles
h = animateGeometry(geo, [], [], [], false);
assert(isgraphics(h, 'figure'));  close(h);
fprintf('PASS static: default angles\n');

% test_pos_out_of_range_falls_back
h = animateGeometry(geo, ANGLES, 999, 'SD', false);
assert(isgraphics(h, 'figure'));  close(h);
fprintf('PASS static: pos out of range falls back\n');

%% TestAnimation ----------------------------------------------------------
% test_returns_animation_and_renders_frames[SD|obj]: every frame goes
% through the per-frame update path and lands in the GIF.
for rot = {'SD', 'obj'}
    base = tempname;
    h = animateGeometry(geo, ANGLES, 1, rot{1}, true, base);
    assert(isgraphics(h, 'figure'));
    gif = [base, '_geometry.gif'];
    assert(exist(gif, 'file') == 2, '%s: GIF not written', rot{1});
    d = dir(gif);
    assert(d.bytes > 0, '%s: GIF is empty', rot{1});
    info = imfinfo(gif);
    assert(numel(info) == numel(ANGLES), '%s: %d frames, expected %d', ...
           rot{1}, numel(info), numel(ANGLES));
    delete(gif);  close(h);
    fprintf('PASS animation: %s renders %d frames\n', rot{1}, numel(ANGLES));
end

% test_fname_save_fallback_chain: fname without extension -> <fname>_geometry.gif
base = fullfile(tempdir, ['t_', datestr(now, 'HHMMSSFFF')]);
h = animateGeometry(geo, ANGLES, 1, 'SD', true, base);
assert(isgraphics(h, 'figure'));
saved = dir([base, '_geometry.*']);
assert(~isempty(saved), 'fname given but nothing was saved');
delete(fullfile(tempdir, saved(1).name));  close(h);
fprintf('PASS animation: fname save\n');

%% TestCalCube ------------------------------------------------------------
% test_single_cuboid_faces
[V, F] = animateGeometryCube([0; 0; 0], [2; 4; 6]);
assert(isequal(size(F), [6, 4]), 'six quad faces expected');
assert(isequal(size(V), [8, 3]), 'eight corners expected');
assert(max(F(:)) <= 8 && min(F(:)) >= 1, 'face indices out of range');
assert(all(abs((max(V) - min(V)) - [2, 4, 6]) < 1e-12), 'extents do not match size');
fprintf('PASS cube: single cuboid faces and extents\n');

% test_batched_with_rotations
n = 5;
centres = reshape(0:n*3-1, 3, n);
R = repmat(eye(3), [1, 1, n]);
V = animateGeometryCube(centres, [2; 2; 2], R);
assert(isequal(size(V), [8, 3, n]));
for i = 1:n
    assert(all(abs(mean(V(:, :, i), 1) - centres(:, i)') < 1e-12), ...
           'cuboid %d not centred', i);
end
fprintf('PASS cube: batched with rotations\n');

fprintf('test_animateGeometry: ALL 8 PASSED\n');
end
