function [V, F] = animateGeometryCube(centre, sz, R, offcent)
%ANIMATEGEOMETRYCUBE Corner coordinates of cuboids for animateGeometry.
%
%   [V, F] = ANIMATEGEOMETRYCUBE(CENTRE, SZ, R, OFFCENT) returns V (8x3xN),
%   the corners of N cuboids of size SZ (3x1) centred at CENTRE (3xN),
%   each rotated by R (3x3xN, or a single 3x3 applied to all) after being
%   shifted by OFFCENT (3x1) in the cuboid's own frame, and F (6x4), the
%   face index table (first face = the -x face, used by animateGeometry to
%   draw the rays to the source-facing detector corners). Draw with
%   patch('Vertices', V(:,:,i), 'Faces', F).
%
%   MATLAB counterpart of animate_geometry.calCube.
%--------------------------------------------------------------------------
% This file is part of the TIGRE Toolbox. License: BSD, see
% https://github.com/CERN/TIGRE/blob/master/LICENSE
% Coded by:           Yi Liu
%--------------------------------------------------------------------------
CORNERS = [-1 -1 -1; 1 -1 -1; 1 1 -1; -1 1 -1; -1 -1 1; 1 -1 1; 1 1 1; -1 1 1];
F = [2 3 7 6; 1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 1 4 8 5];
if nargin < 3 || isempty(R), R = eye(3); end
if nargin < 4 || isempty(offcent), offcent = [0; 0; 0]; end
centre = double(centre);
if size(centre, 1) ~= 3 && size(centre, 2) == 3   % accept Nx3 too
    centre = centre';
end
n = size(centre, 2);
if size(R, 3) == 1
    R = repmat(R, [1, 1, n]);
end
base = CORNERS .* (sz(:)' / 2) + offcent(:)';
V = zeros(8, 3, n);
for i = 1:n
    V(:, :, i) = base * R(:, :, i)' + centre(:, i)';
end
end
