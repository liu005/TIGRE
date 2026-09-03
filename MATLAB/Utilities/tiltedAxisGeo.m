function [geo, angles] = tiltedAxisGeo(geo, angles, tiltX, tiltY)
%TILTEDAXISGEO Per-view TIGRE geometry for a TILTED ROTATION AXIS.
%
%   [GEO, ANGLES] = TILTEDAXISGEO(GEO, ANGLES, TILTX, TILTY) fills GEO with
%   the per-view arrays (DSO, DSD, COR, offOrigin, offDetector, rotDetector)
%   that describe a scan whose rotation axis is tilted away from the ideal
%   vertical by TILTX (about x, the beam axis) and TILTY (about y), in
%   radians, and returns the ANGLES to reconstruct with. Reconstruct with
%   BOTH outputs, e.g.
%
%       [geo, ang] = tiltedAxisGeo(geo, angles, tx, ty);
%       img = FDK(proj, geo, ang);
%
%   WHY. A tilted axis cannot be expressed by any rigid detector transform
%   (offsets or rotDetector alone). TIGRE can describe an arbitrary rig
%   rotation with 3xN ZYZ Euler ANGLES (see d18_ArbitraryAxisOfRotation),
%   but FDK does not accept that form. This function instead reconstructs
%   in the AXIS-ALIGNED frame (the rotation axis IS z there, so the volume
%   has a clean vertical axis), where the tilt becomes a CONSTANT
%   re-expression of the rig plus a phase on ordinary 1xN angles - and FDK
%   works unchanged. The arrays are emitted per view because that is
%   TIGRE's interface, but they are the same for every view.
%
%   The model: the stage rotates the OBJECT by theta about the unit axis
%   a = T*[0;0;1], T = Rx(tiltX)*Ry(tiltY). Seen from the axis-aligned
%   frame the lab-fixed source/detector orbit the axis, X_i = Rz(theta_i)*
%   T'*X_lab for X in {S, C, u, v}. De-rotating by the azimuth of the
%   undisplaced source direction lets TIGRE's own placement be read off:
%   source at (DSO, COR, 0), detector centre at (-(DSD-DSO), COR+offU, offV),
%   detector axes R*y, R*z with R = Rz(rot3)*Ry(rot2)*Rx(rot1).
%
%   GEO may carry COR, offDetector and rotDetector (scalars / one column, or
%   per-view arrays that do not vary across views); all are composed as
%   lab-fixed entities. At ZERO tilt the input geometry is returned EXACTLY
%   (COR, offDetector, rotDetector included) - this matters because TIGRE's
%   offset-detector (Wang) weighting is gated on offDetector(1)==0, so a
%   builder that folded COR into offDetector would switch it on for a
%   centred detector and halve the reconstructed intensity. Any existing
%   geo.offOrigin is replaced (the volume is re-centred on the axis).
%
%   Cone mode only. MATLAB field conventions: offOrigin = [x;y;z],
%   offDetector = [u;v], rotDetector = [rot1;rot2;rot3] (the same order the
%   mex files read: dYaw <- rot1 is the in-plane roll about the beam).
%
%   This is the MATLAB port of tigre.utilities.common_geometry.tilted_axis_geo;
%   both are validated against the same analytic projection model
%   (see projectPointsTilted and Tests/test_tiltedAxisGeo.m).
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
% Coded by:           Yi Liu
%--------------------------------------------------------------------------

angles = double(angles(:))';           % 1xN
n = numel(angles);
if isfield(geo, 'mode') && ~strcmp(geo.mode, 'cone')
    error('TIGRE:tiltedAxisGeo:mode', 'tiltedAxisGeo: cone mode only');
end

% The tilt composes ONE rig: refuse per-view arrays that vary across views.
chk = {'COR', 'DSD', 'DSO', 'offDetector', 'rotDetector'};
for k = 1:numel(chk)
    f = chk{k};
    if isfield(geo, f) && size(geo.(f), 2) > 1
        if any(max(geo.(f), [], 2) - min(geo.(f), [], 2))
            error('TIGRE:tiltedAxisGeo:varying', ...
                ['tiltedAxisGeo: per-view %s arrays that vary across views ' ...
                 'are not supported (the tilt is composed with ONE rig)'], f);
        end
    end
end

[S_lab, C_lab, u_lab, v_lab] = tiltedAxisLabEntities(geo);
Tt = tiltedAxisTiltMatrix(tiltX, tiltY)';

% De-rotate by the azimuth of the UNDISPLACED source direction so the
% residual lateral displacement reads off as TIGRE's COR.
q = Tt * [1; 0; 0];
phase = atan2(q(2), q(1));
Rd = tiltedAxisRotation('z', -phase);                     % constant de-rotation
S = Rd * (Tt * S_lab);
C = Rd * (Tt * C_lab);
u = Rd * (Tt * u_lab);
v = Rd * (Tt * v_lab);

dso = S(1);  cor = S(2);  zsrc = S(3);
dsd = S(1) - C(1);

geo.DSO = repmat(dso, 1, n);
geo.DSD = repmat(dsd, 1, n);
geo.COR = repmat(cor, 1, n);

% offOrigin is [x;y;z] in MATLAB: the volume centre sits at -zsrc along the
% axis in the frame where the source is at z = 0.
offOrigin = zeros(3, n);
offOrigin(3, :) = -zsrc;

% Detector orientation: solve R with R*y = u, R*z = v (the observable axes),
% column form M = [u x v, u, v]; TIGRE stores [rot1;rot2;rot3] such that
% R = Rz(rot3)*Ry(rot2)*Rx(rot1).
M = [cross(u, v), u, v];
e = eulerZYX(M);                       % [a_z, b_y, c_x]
rot = [e(3); e(2); e(1)];

% Detector-centre residuals against TIGRE's placement at
% (-(DSD-DSO), COR+offU, offV) in the source-z=0 frame: [u;v] in MATLAB.
offDet = zeros(2, n);
offDet(1, :) = C(2) - cor;             % U on y
offDet(2, :) = C(3) - zsrc;            % V on z, frame shifted by zsrc

geo.offOrigin   = offOrigin;
geo.offDetector = offDet;
geo.rotDetector = repmat(rot, 1, n);
angles = angles + phase;
end

%% ------------------------------------------------------------------------
function e = eulerZYX(M)
% Euler angles [a b c] with M = Rz(a)*Ry(b)*Rx(c) (same as scipy 'ZYX').
a = atan2(M(2, 1), M(1, 1));
b = atan2(-M(3, 1), sqrt(M(1, 1)^2 + M(2, 1)^2));
c = atan2(M(3, 2), M(3, 3));
e = [a, b, c];
end
