function R = tiltedAxisRotation(axis, a)
%TILTEDAXISROTATION 3x3 right-handed rotation by A radians about 'x','y' or 'z'.
%   Namespaced (rather than rotx/roty/rotz) so it cannot shadow the
%   functions of the same name in the Phased Array / Robotics toolboxes.
%--------------------------------------------------------------------------
% This file is part of the TIGRE Toolbox. License: BSD, see
% https://github.com/CERN/TIGRE/blob/master/LICENSE
% Coded by:           Yi Liu
%--------------------------------------------------------------------------
c = cos(a); s = sin(a);
switch lower(axis)
    case 'x'
        R = [1, 0, 0; 0, c, -s; 0, s, c];
    case 'y'
        R = [c, 0, s; 0, 1, 0; -s, 0, c];
    case 'z'
        R = [c, -s, 0; s, c, 0; 0, 0, 1];
    otherwise
        error('TIGRE:tiltedAxisRotation:axis', 'axis must be ''x'', ''y'' or ''z''');
end
end
