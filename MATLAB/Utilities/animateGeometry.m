function h = animateGeometry(geo, angles, pos, rotation, animate, fname)
%ANIMATEGEOMETRY Plot an animated (or static) TIGRE cone-beam geometry.
%
%   H = ANIMATEGEOMETRY(GEO, ANGLES, POS, ROTATION, ANIMATE, FNAME) draws the
%   source, detector (with its rotation rotDetector), object cube (with its
%   offset offOrigin / COR) and beam profile IN SCALE for geometry GEO at
%   the projection ANGLES, and steps through every view.
%
%   ANGLES   1xN or 3xN (ZYZ Euler, as Ax accepts). Default linspace(0,2*pi,100).
%   POS      index of the view drawn in a static plot / first frame (1-based).
%            Default 1.
%   ROTATION 'SD'  - object fixed, source and detector move (default);
%            'obj' - source and detector fixed, object rotates (only
%                    meaningful when source and detector do not move
%                    relative to each other).
%   ANIMATE  true (default) steps through all views; false draws view POS.
%   FNAME    if given, the animation is written to [FNAME '_geometry.gif'].
%
%   Returns the figure handle.
%
%   MATLAB port of tigre.utilities.visualization.animate_geometry. Uses
%   only base-MATLAB graphics (plot3/patch/quiver3/imwrite), no toolboxes.
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
% Coded by:           Ander Biguri (plotgeometry), animated port by Yi Liu
%--------------------------------------------------------------------------

%% Arguments
if nargin < 2 || isempty(angles)
    angles = linspace(0, 2*pi, 100);
end
if nargin < 3 || isempty(pos), pos = 1; end
if nargin < 4 || isempty(rotation), rotation = 'SD'; end
if nargin < 5 || isempty(animate), animate = true; end
if nargin < 6, fname = ''; end
assert(any(strcmp(rotation, {'SD', 'obj'})), ...
    'TIGRE:animateGeometry:rotation', 'rotation must be ''SD'' or ''obj''');

angles = double(angles);
if size(angles, 1) == 1
    angles = [angles; zeros(2, size(angles, 2))];
end
assert(size(angles, 1) == 3, 'TIGRE:animateGeometry:angles', ...
    'angles must be 1xN or 3xN');
n = size(angles, 2);
pos = round(pos);
if pos < 1 || pos > n, pos = 1; end

geo = checkGeo(geo, angles);            % expands every field to per-view
ln = min(geo.sVoxel) / 2;               % coordinate-arrow length

%% Per-view rotations (ZYZ Euler, as the projectors interpret angles)
Rs = zeros(3, 3, n);
for i = 1:n
    Rs(:, :, i) = tiltedAxisRotation('z', angles(1, i)) * ...
                  tiltedAxisRotation('y', angles(2, i)) * ...
                  tiltedAxisRotation('z', angles(3, i));
end

%% Source / detector trajectories (lab x = beam, y = u, z = v)
scent = [geo.DSO; zeros(2, n)];                                  % 3xN
dcent = [-geo.DSD + geo.DSO; geo.offDetector(1, :); geo.offDetector(2, :)];
stj = zeros(3, n); dtj = zeros(3, n);
if strcmp(rotation, 'SD')
    for i = 1:n
        stj(:, i) = Rs(:, :, i) * scent(:, i);
        dtj(:, i) = Rs(:, :, i) * dcent(:, i);
    end
else
    stj(:, pos) = Rs(:, :, pos) * scent(:, pos);
    dtj(:, pos) = Rs(:, :, pos) * dcent(:, pos);
end
stj(2, :) = stj(2, :) + geo.COR;        % COR shifts source AND detector in y
dtj(2, :) = dtj(2, :) + geo.COR;

%% Detector orientation and cube
ddp = 30;                               % drawn detector depth (mm)
Rdet = zeros(3, 3, n);
for i = 1:n
    r = geo.rotDetector(:, i);
    Rdet(:, :, i) = Rs(:, :, i) * tiltedAxisRotation('x', r(1)) * ...
                    tiltedAxisRotation('y', r(2)) * tiltedAxisRotation('z', r(3));
end
dsz = [ddp; geo.sDetector(1); geo.sDetector(2)];     % [beam; u; v]
dverts = cubeVertices(dtj, dsz, Rdet, [-ddp/2; 0; 0]);

%% Object cube (offOrigin is [x;y;z] in MATLAB)
otj = geo.offOrigin;
otj(2, :) = otj(2, :) + geo.COR;
if strcmp(rotation, 'obj')
    Robj = permute(Rs, [2, 1, 3]);      % object rotates the other way
else
    Robj = repmat(eye(3), [1, 1, n]);
end
overts = cubeVertices(otj, geo.sVoxel, Robj, [0; 0; 0]);
FACES = [2 3 7 6; 1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 1 4 8 5];

%% Figure
h = figure('Name', 'Cone Beam Computed Tomography geometry', 'Color', 'w', ...
           'Position', [100, 100, 900, 720]);
ax = axes('Parent', h); hold(ax, 'on'); grid(ax, 'on');
xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
view(ax, 52, 26);

plot3(ax, stj(1, :), stj(2, :), stj(3, :), '.', 'Color', [0.87 0.63 0.87], 'MarkerSize', 6);
plot3(ax, dtj(1, :), dtj(2, :), dtj(3, :), '.', 'Color', [0.53 0.81 0.92], 'MarkerSize', 6);
plot3(ax, otj(1, :), otj(2, :), otj(3, :), '.', 'Color', [0.5 0.5 0.5], 'MarkerSize', 6);

hS = plot3(ax, stj(1, pos), stj(2, pos), stj(3, pos), 'o', 'MarkerSize', 6, ...
           'MarkerFaceColor', [0.87 0.63 0.87], 'Color', [0.87 0.63 0.87]);
hD = plot3(ax, dtj(1, pos), dtj(2, pos), dtj(3, pos), 'o', 'MarkerSize', 6, ...
           'MarkerFaceColor', [0.53 0.81 0.92], 'Color', [0.53 0.81 0.92]);
tS = text(ax, stj(1, pos), stj(2, pos), stj(3, pos) + 30, 'S');
tD = text(ax, dtj(1, pos), dtj(2, pos), dtj(3, pos) + 30, 'D');

hDet = patch(ax, 'Vertices', dverts(:, :, pos), 'Faces', FACES, ...
             'FaceColor', [0.65 0.16 0.16], 'FaceAlpha', 0.2, 'EdgeColor', 'k');
hObj = patch(ax, 'Vertices', overts(:, :, pos), 'Faces', FACES, ...
             'FaceColor', 'c', 'FaceAlpha', 0.2, 'EdgeColor', 'k');

quiver3(ax, 0, 0, 0, ln, 0, 0, 0, 'r', 'LineWidth', 1.5);
quiver3(ax, 0, 0, 0, 0, ln, 0, 0, 'b', 'LineWidth', 1.5);
quiver3(ax, 0, 0, 0, 0, 0, ln, 0, 'g', 'LineWidth', 1.5);
text(ax, -10, -10, -10, 'O');

% beam: central ray + the four rays to the source-facing detector corners
% (cube face 1 = the -beam face, vertices 2 3 7 6)
front = FACES(1, :);
hBeam = gobjects(4, 1);
for k = 1:4
    c = dverts(front(k), :, pos);
    hBeam(k) = plot3(ax, [stj(1, pos), c(1)], [stj(2, pos), c(2)], [stj(3, pos), c(3)], 'y');
end
hC = plot3(ax, [stj(1, pos), dtj(1, pos)], [stj(2, pos), dtj(2, pos)], ...
           [stj(3, pos), dtj(3, pos)], 'Color', [1 0.6 0.8]);

axis(ax, 'equal');
hT = title(ax, frameTitle(pos));

%% Animate
if ~animate
    return
end
frames = 1:n;
gif = '';
if ~isempty(fname)
    gif = [fname, '_geometry.gif'];
end
for k = frames
    set(hObj, 'Vertices', overts(:, :, k));
    if strcmp(rotation, 'SD')
        set(hS, 'XData', stj(1, k), 'YData', stj(2, k), 'ZData', stj(3, k));
        set(hD, 'XData', dtj(1, k), 'YData', dtj(2, k), 'ZData', dtj(3, k));
        set(tS, 'Position', [stj(1, k), stj(2, k), stj(3, k) + 30]);
        set(tD, 'Position', [dtj(1, k), dtj(2, k), dtj(3, k) + 30]);
        set(hDet, 'Vertices', dverts(:, :, k));
        set(hC, 'XData', [stj(1, k), dtj(1, k)], 'YData', [stj(2, k), dtj(2, k)], ...
                'ZData', [stj(3, k), dtj(3, k)]);
        for j = 1:4
            c = dverts(front(j), :, k);
            set(hBeam(j), 'XData', [stj(1, k), c(1)], 'YData', [stj(2, k), c(2)], ...
                          'ZData', [stj(3, k), c(3)]);
        end
    end
    set(hT, 'String', frameTitle(k));
    drawnow;
    if ~isempty(gif)
        fr = getframe(h);
        [im, map] = rgb2ind(frame2im(fr), 256);
        if k == frames(1)
            imwrite(im, map, gif, 'gif', 'LoopCount', inf, 'DelayTime', 1/30);
        else
            imwrite(im, map, gif, 'gif', 'WriteMode', 'append', 'DelayTime', 1/30);
        end
    end
end

%% ---------------------------------------------------------------------
function s = frameTitle(k)
    rd = geo.rotDetector(:, k) * 180 / pi;
    if strcmp(rotation, 'SD')
        ang = angles(:, k) * 180 / pi;
        who = 'Source';
    else
        ang = -angles(:, k) * 180 / pi;
        who = 'Object';
    end
    s = sprintf(['CBCT geometry in scale - %s at angle [%.1f, %.1f, %.1f]%c\n' ...
                 'Detector rotation [%.1f, %.1f, %.1f]%c   ' ...
                 'Detector offset [u %.1f, v %.1f] mm'], ...
                who, ang(1), ang(2), ang(3), char(176), ...
                rd(1), rd(2), rd(3), char(176), ...
                geo.offDetector(1, k), geo.offDetector(2, k));
end
end

%% ------------------------------------------------------------------------
function V = cubeVertices(centre, sz, R, offcent)
%CUBEVERTICES 8x3xN corner coordinates of cuboids centred at CENTRE (3xN),
%   of size SZ (3x1), rotated by R (3x3xN), shifted by OFFCENT (3x1) before
%   rotation. Corner order matches the FACES table in the caller.
CORNERS = [-1 -1 -1; 1 -1 -1; 1 1 -1; -1 1 -1; -1 -1 1; 1 -1 1; 1 1 1; -1 1 1];
n = size(centre, 2);
V = zeros(8, 3, n);
base = CORNERS .* (sz(:)' / 2) + offcent(:)';
for i = 1:n
    V(:, :, i) = base * R(:, :, i)' + centre(:, i)';
end
end
