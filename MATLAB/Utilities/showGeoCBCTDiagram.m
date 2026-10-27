function h=showGeoCBCTDiagram()
%SHOWGEODIAGRAM Shows an image describing the Geometry of TIGRE
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
% Coded by:           Ander Biguri
%--------------------------------------------------------------------------
% Local schematic: no network connection or image toolbox is needed.
h=figure('Name','Diagram of TIGRE Geometry','Color','white','Renderer','painters');
ax=axes('Parent',h); hold(ax,'on');
[V,F]=animateGeometryCube([0;0;0],[256;256;256]);
patch(ax,'Vertices',V,'Faces',F,'FaceColor','none','EdgeColor',[.2 .5 .5]);
corners=[-536 -204.8 -204.8;-536 204.8 -204.8;-536 204.8 204.8;-536 -204.8 204.8];
patch(ax,'Vertices',corners,'Faces',[1 2 3 4],'FaceColor',[.8 .85 1]);
for k=1:4
    plot3(ax,[1000 corners(k,1)],[0 corners(k,2)],[0 corners(k,3)],':','Color',[.7 .7 .7]);
end
plot3(ax,[-536 1000],[0 0],[0 0],'k--');
scatter3(ax,1000,0,0,70,'r','filled'); text(ax,1000,0,70,'Source S');
scatter3(ax,0,0,0,35,'k','filled'); text(ax,0,0,180,'Object / rotation origin O');
text(ax,-536,0,260,'Detector D (u = y, v = z)');
plot3(ax,[0 1000],[0 0],[-260 -260],'r','LineWidth',1.5);
text(ax,500,0,-290,'DSO: source to origin','HorizontalAlignment','center');
plot3(ax,[-536 1000],[0 0],[-360 -360],'b','LineWidth',1.5);
text(ax,232,0,-400,'DSD: source to detector','HorizontalAlignment','center');
xlabel(ax,'x (mm)'); ylabel(ax,'y (mm)'); zlabel(ax,'z (mm)');
title(ax,{'TIGRE cone-beam geometry','Illustrative centred flat detector; TIGRE supports more general layouts'});
axis(ax,'equal'); grid(ax,'on'); view(ax,52,26);
end
