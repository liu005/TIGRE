function [S_lab, C_lab, u_lab, v_lab] = tiltedAxisLabEntities(geo)
%TILTEDAXISLABENTITIES Lab-frame source, detector centre and detector axes.
%
%   Includes COR, offDetector and rotDetector under TIGRE's conventions:
%   source at (DSO, COR, 0), detector centre at (-(DSD-DSO), COR+offU, offV),
%   detector axes u = R*[0;1;0], v = R*[0;0;1] with
%   R = Rz(rot3)*Ry(rot2)*Rx(rot1). Shared by tiltedAxisGeo and
%   projectPointsTilted so both use ONE definition of the rig.
%
%   MATLAB field conventions: offDetector = [u;v]; rotDetector = [3x1].
%--------------------------------------------------------------------------
% This file is part of the TIGRE Toolbox. License: BSD, see
% https://github.com/CERN/TIGRE/blob/master/LICENSE
% Coded by:           Yi Liu
%--------------------------------------------------------------------------
DSD = geo.DSD(1);
DSO = geo.DSO(1);
cor = 0;
if isfield(geo, 'COR') && ~isempty(geo.COR)
    cor = geo.COR(1);
end
offU = 0; offV = 0;
if isfield(geo, 'offDetector') && ~isempty(geo.offDetector)
    offU = geo.offDetector(1, 1);
    offV = geo.offDetector(2, 1);
end
rot = [0; 0; 0];
if isfield(geo, 'rotDetector') && ~isempty(geo.rotDetector)
    rot = geo.rotDetector(:, 1);
end
R = tiltedAxisRotation('z', rot(3)) * tiltedAxisRotation('y', rot(2)) * ...
    tiltedAxisRotation('x', rot(1));
S_lab = [DSO; cor; 0];
C_lab = [-(DSD - DSO); cor + offU; offV];
u_lab = R * [0; 1; 0];
v_lab = R * [0; 0; 1];
end
