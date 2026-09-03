function T = tiltedAxisTiltMatrix(tiltX, tiltY)
%TILTEDAXISTILTMATRIX T mapping the ideal vertical to the physical rotation
%   axis: T = Rx(tiltX)*Ry(tiltY), so the axis is a = T*[0;0;1].
%--------------------------------------------------------------------------
% This file is part of the TIGRE Toolbox. License: BSD, see
% https://github.com/CERN/TIGRE/blob/master/LICENSE
% Coded by:           Yi Liu
%--------------------------------------------------------------------------
T = tiltedAxisRotation('x', tiltX) * tiltedAxisRotation('y', tiltY);
end
