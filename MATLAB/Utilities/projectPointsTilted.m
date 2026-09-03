function out = projectPointsTilted(points, geo, angles, tiltX, tiltY)
%PROJECTPOINTSTILTED Analytic pinhole projection under the PHYSICAL
%   tilted-axis model - ground truth for validating tiltedAxisGeo, no GPU.
%
%   OUT = PROJECTPOINTSTILTED(POINTS, GEO, ANGLES, TILTX, TILTY) takes POINTS
%   (Px3, in the axis-aligned frame) and the NOMINAL geometry GEO (before
%   tiltedAxisGeo) and returns OUT (N x P x 2): the (u, v) pixel indices,
%   0-based, at which each point lands for each of the N ANGLES when the
%   object rotates by ANGLES about the axis T*[0;0;1]. Honours GEO's COR,
%   offDetector and rotDetector the same way tiltedAxisGeo does.
%
%   Conventions (pinned against tigre.Ax at zero tilt): world rotation
%   Rz(+theta), +y -> +u, +z -> +v, pixel k centred at (k + 0.5 - N/2)*d.
%   MATLAB detector fields are [u;v]: nDetector(1)/dDetector(1) are u.
%--------------------------------------------------------------------------
% This file is part of the TIGRE Toolbox. License: BSD, see
% https://github.com/CERN/TIGRE/blob/master/LICENSE
% Coded by:           Yi Liu
%--------------------------------------------------------------------------
du = geo.dDetector(1);  dv = geo.dDetector(2);
nu = geo.nDetector(1);  nv = geo.nDetector(2);

T = tiltedAxisTiltMatrix(tiltX, tiltY);
[S_lab, C_lab, u_lab, v_lab] = tiltedAxisLabEntities(geo);
angles = double(angles(:))';
P = size(points, 1);
out = zeros(numel(angles), P, 2);
for i = 1:numel(angles)
    Rw = tiltedAxisRotation('z', angles(i));
    S = Rw * (T' * S_lab);
    C = Rw * (T' * C_lab);
    u = Rw * (T' * u_lab);
    v = Rw * (T' * v_lab);
    nrm = cross(u, v);
    for j = 1:P
        p = points(j, :)';
        ray = p - S;
        t = dot(C - S, nrm) / dot(ray, nrm);
        hit = S + t * ray - C;
        out(i, j, 1) = dot(hit, u) / du + nu / 2 - 0.5;
        out(i, j, 2) = dot(hit, v) / dv + nv / 2 - 0.5;
    end
end
end
