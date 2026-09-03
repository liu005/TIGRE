function test_tiltedAxisGeo()
%TEST_TILTEDAXISGEO Pure-math tests for tiltedAxisGeo / projectPointsTilted.
%
%   No GPU needed. Four checks:
%     1. zero tilt returns the input geometry EXACTLY (COR, offDetector,
%        rotDetector preserved; phase 0);
%     2. parity with the Python implementation (tigre.utilities.
%        common_geometry.tilted_axis_geo) on data/tilted_axis_reference.mat,
%        with the MATLAB<->Python field-order mapping (offOrigin xyz<->zyx,
%        offDetector uv<->vu);
%     3. parity of projectPointsTilted with project_points_tilted;
%     4. internal consistency: an analytic pinhole model of TIGRE's own
%        placement (source at (DSO,COR,0), detector centre at
%        (-(DSD-DSO), COR+offU, offV), axes R*y/R*z, volume shifted by
%        offOrigin, gantry rotation Rz(angle)) applied to the BUILT geometry
%        reproduces the PHYSICAL tilted-axis projection - i.e. the emitted
%        per-view arrays mean what TIGRE will read them as.
%
%   Run:  cd MATLAB/Tests; test_tiltedAxisGeo
%   (or   matlab -batch "run('MATLAB/Tests/test_tiltedAxisGeo.m')" )
%--------------------------------------------------------------------------
% This file is part of the TIGRE Toolbox. License: BSD, see
% https://github.com/CERN/TIGRE/blob/master/LICENSE
% Coded by:           Yi Liu
%--------------------------------------------------------------------------
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'Utilities'));
ref = load(fullfile(here, 'data', 'tilted_axis_reference.mat'));

% --- geometry in MATLAB conventions (the reference stores Python orders) ---
geo.DSD = ref.DSD;  geo.DSO = ref.DSO;
geo.nDetector = [ref.nDetector_vu(2); ref.nDetector_vu(1)];     % [u; v]
geo.dDetector = [ref.dDetector_vu(2); ref.dDetector_vu(1)];
geo.sDetector = geo.nDetector .* geo.dDetector;
geo.nVoxel = [128; 96; 64];  geo.sVoxel = [128; 96; 64];         % [x; y; z]
geo.dVoxel = geo.sVoxel ./ geo.nVoxel;
geo.offOrigin = [0; 0; 0];
geo.COR = ref.COR;
geo.offDetector = [ref.offDetector_vu(2); ref.offDetector_vu(1)];  % [u; v]
geo.rotDetector = ref.rotDetector(:);
geo.mode = 'cone';  geo.accuracy = 0.5;
angles = ref.angles(:)';
points = ref.points;                                                % Px3 (x,y,z)

% --- 1. zero tilt is the identity -----------------------------------------
[g0, a0] = tiltedAxisGeo(geo, angles, 0, 0);
assertClose(a0, angles, 1e-12, 'zero tilt: angles unchanged');
assertClose(g0.DSO, repmat(geo.DSO, size(angles)), 1e-12, 'zero tilt: DSO');
assertClose(g0.DSD, repmat(geo.DSD, size(angles)), 1e-12, 'zero tilt: DSD');
assertClose(g0.COR, repmat(geo.COR, size(angles)), 1e-12, 'zero tilt: COR');
assertClose(g0.offDetector, repmat(geo.offDetector, size(angles)), 1e-12, ...
            'zero tilt: offDetector');
assertClose(g0.rotDetector, repmat(geo.rotDetector, size(angles)), 1e-9, ...
            'zero tilt: rotDetector');
assertClose(g0.offOrigin, zeros(3, numel(angles)), 1e-12, 'zero tilt: offOrigin');
fprintf('PASS 1: zero tilt returns the input geometry exactly\n');

% --- 2. parity with the Python implementation -----------------------------
[g, a] = tiltedAxisGeo(geo, angles, ref.tilt_x, ref.tilt_y);
assertClose(g.DSO, ref.out_DSO(:)', 1e-9, 'DSO vs Python');
assertClose(g.DSD, ref.out_DSD(:)', 1e-9, 'DSD vs Python');
assertClose(g.COR, ref.out_COR(:)', 1e-9, 'COR vs Python');
assertClose(g.offOrigin(3, :), ref.out_offOrigin_zyx(:, 1)', 1e-9, 'offOrigin z vs Python');
assertClose(g.offOrigin(1:2, :), zeros(2, numel(angles)), 1e-12, 'offOrigin x,y are zero');
assertClose(ref.out_offOrigin_zyx(:, 2:3), zeros(numel(angles), 2), 1e-12, 'Python offOrigin y,x zero');
assertClose(g.offDetector(1, :), ref.out_offDetector_vu(:, 2)', 1e-9, 'offDetector u vs Python');
assertClose(g.offDetector(2, :), ref.out_offDetector_vu(:, 1)', 1e-9, 'offDetector v vs Python');
assertClose(g.rotDetector, ref.out_rotDetector', 1e-9, 'rotDetector vs Python');
assertClose(a, ref.out_angles(:)', 2e-6, 'angles vs Python (float32 there)');
fprintf('PASS 2: parity with Python tilted_axis_geo (DSO %.6f DSD %.6f COR %.6f)\n', ...
        g.DSO(1), g.DSD(1), g.COR(1));

% --- 3. projectPointsTilted parity ----------------------------------------
proj = projectPointsTilted(points, geo, angles, ref.tilt_x, ref.tilt_y);
assertClose(proj, ref.proj_uv, 1e-8, 'projectPointsTilted vs Python');
fprintf('PASS 3: projectPointsTilted matches Python (max |d| = %.2e px)\n', ...
        max(abs(proj(:) - ref.proj_uv(:))));

% --- 4. built geometry, read as TIGRE reads it, reproduces the physics ----
predicted = pinholeTigre(points, g, a);
d = abs(predicted(:) - proj(:));
assert(max(d) < 1e-7, 'TIGRE-placement model of the built geometry disagrees with the physical model: max %.3e px', max(d));
fprintf('PASS 4: built per-view arrays reproduce the physical projection (max |d| = %.2e px)\n', max(d));

fprintf('test_tiltedAxisGeo: ALL PASSED\n');
end

%% ------------------------------------------------------------------------
function out = pinholeTigre(points, geo, angles)
% Analytic pinhole through TIGRE's placement of a (per-view) geometry.
du = geo.dDetector(1);  dv = geo.dDetector(2);
nu = geo.nDetector(1);  nv = geo.nDetector(2);
n = numel(angles);  P = size(points, 1);
out = zeros(n, P, 2);
for i = 1:n
    R = tiltedAxisRotation('z', angles(i));
    rot = geo.rotDetector(:, i);
    Rd = tiltedAxisRotation('z', rot(3)) * tiltedAxisRotation('y', rot(2)) * ...
         tiltedAxisRotation('x', rot(1));
    S = R * [geo.DSO(i); geo.COR(i); 0];
    C = R * [-(geo.DSD(i) - geo.DSO(i)); geo.COR(i) + geo.offDetector(1, i); geo.offDetector(2, i)];
    u = R * Rd * [0; 1; 0];
    v = R * Rd * [0; 0; 1];
    nrm = cross(u, v);
    for j = 1:P
        p = points(j, :)' + geo.offOrigin(:, i);     % volume centre sits at offOrigin
        ray = p - S;
        t = dot(C - S, nrm) / dot(ray, nrm);
        hit = S + t * ray - C;
        out(i, j, 1) = dot(hit, u) / du + nu / 2 - 0.5;
        out(i, j, 2) = dot(hit, v) / dv + nv / 2 - 0.5;
    end
end
end

function assertClose(a, b, tol, what)
d = max(abs(double(a(:)) - double(b(:))));
assert(isequal(size(a), size(b)), '%s: shape %s vs %s', what, mat2str(size(a)), mat2str(size(b)));
assert(d < tol, '%s: max |diff| = %.3e (tol %.0e)', what, d, tol);
end
