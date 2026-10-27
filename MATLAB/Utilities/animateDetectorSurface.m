function h=animateDetectorSurface(surface,geo,pos,animate,fname)
%ANIMATEDETECTORSURFACE Render explicit curved/modular detector coordinates.
if nargin<2, geo=[]; end
if nargin<3 || isempty(pos), pos=1; end
if nargin<4, animate=true; end
if nargin<5, fname=''; end
n=size(surface.source_positions_mm,2); if pos<1 || pos>n, pos=1; end
h=figure('Name','TIGRE detector surface'); ax=axes('Parent',h);
scene=renderDetectorSurface(ax,surface,geo); scene.update(pos);
setappdata(h,'SurfaceScene',scene);
if ~animate, return; end
for k=pos:n
    if ~isgraphics(h), break; end
    scene.update(k); drawnow;
    if ~isempty(fname)
        frame=getframe(h); [image,map]=rgb2ind(frame2im(frame),256);
        file=[char(fname) '_geometry.gif'];
        if k==pos, imwrite(image,map,file,'gif','LoopCount',Inf,'DelayTime',.1);
        else, imwrite(image,map,file,'gif','WriteMode','append','DelayTime',.1); end
    else
        pause(.1);
    end
end
end
