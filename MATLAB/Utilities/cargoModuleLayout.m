function [centres,angles,nVertical,nHorizontal]=cargoModuleLayout(clearance,dso,horizontalLift)
%CARGOMODULELAYOUT Source-facing 36x4, 5 mm DR panels with aligned ray edges.
if nargin<1, clearance=20; end
if nargin<2, dso=4230; end
if nargin<3, horizontalLift=300; end
validateattributes(clearance,{'numeric'},{'scalar','finite','nonnegative'});
validateattributes(dso,{'numeric'},{'scalar','finite','positive'});
validateattributes(horizontalLift,{'numeric'},{'scalar','finite','nonnegative'});
source=[4700*cosd(13);0;-4700*sind(13)]; x=-1800*cosd(39); half=90;
shift=source*(dso/4700-1);
corners=[-1220 0 -1195;-1220 0 1395;1220 0 -1195;1220 0 1395]-shift';
limits=zeros(4,1); for j=1:4, limits(j)=fanAngle(corners(j,:)'); end
centre=[x;0;source(3)+(source(1)-x)*tan(min(limits))];
[row,angle]=pose(centre); centres=centre'; angles=angle; edge=centre+half*row;
nVertical=round(614/36)+1;
for j=2:nVertical
    [centre,angle,edge]=nextModule(edge,3,x);
    centres(end+1,:)=centre'; angles(end+1,1)=angle;
end
while edge(3)+shift(3)<1395
    assert(nVertical<128,'TIGRE:cargo:size','Vertical module limit exceeded');
    [centre,angle,edge]=nextModule(edge,3,x);
    centres(end+1,:)=centre'; angles(end+1,1)=angle; nVertical=nVertical+1;
end
joint=edge; height=joint(3)+clearance;
edge=source+(joint-source)*((height-source(3))/(joint(3)-source(3)));
ray=edge-source; distance=norm(ray); theta=atan2(ray(3),ray(1))-asin(half/distance);
radius=sqrt(distance^2-half^2); centre=source+radius*[cos(theta);0;sin(theta)];
[row,angle]=pose(centre); height=centre(3)+horizontalLift;
if horizontalLift>0, [centre,angle,edge]=nextModule(joint,1,height);
else, edge=centre+half*row; end
centres(end+1,:)=centre'; angles(end+1,1)=angle; nHorizontal=1;
while edge(1)<x+2050
    assert(nHorizontal<128,'TIGRE:cargo:size','Cargo module limit exceeded');
    [centre,angle,edge]=nextModule(edge,1,height);
    centres(end+1,:)=centre'; angles(end+1,1)=angle; nHorizontal=nHorizontal+1;
end
while fanAngle(edge)<max(limits)
    [centre,angle,edge]=nextModule(edge,1,height);
    centres(end+1,:)=centre'; angles(end+1,1)=angle; nHorizontal=nHorizontal+1;
    assert(nHorizontal<127,'TIGRE:cargo:size','Horizontal module limit exceeded');
end
% A complete additional module is beyond the container silhouette.
[centre,angle,~]=nextModule(edge,1,height);
centres(end+1,:)=centre'; angles(end+1,1)=angle; nHorizontal=nHorizontal+1;
centres=centres+shift';
    function angle=fanAngle(point)
        ray=point-source; angle=atan2(ray(3),-ray(1));
    end
    function [row,angle]=pose(centre)
        normal=source-centre; normal=normal/norm(normal);
        row=cross(normal,[0;1;0]); angle=atan2d(-normal(3),normal(1));
    end
    function [centre,angle,upper]=nextModule(edge,axis,fixed)
        ray=edge-source; lo=-10000;
        if axis==3, hi=10000; else, hi=source(1)-1; end
        first=evaluate(lo); assert(first*evaluate(hi)<0,'TIGRE:cargo:align','Cannot bracket module edge');
        for q=1:70
            middle=(lo+hi)/2;
            if evaluate(middle)*first>0, lo=middle; else, hi=middle; end
        end
        [~,centre,row,angle]=evaluate((lo+hi)/2); upper=centre+half*row;
        function [residual,centre,row,angle]=evaluate(value)
            if axis==3, centre=[fixed;0;value]; else, centre=[value;0;fixed]; end
            [row,angle]=pose(centre); lower=centre-half*row-source;
            residual=lower(1)*ray(3)-lower(3)*ray(1);
        end
    end
end
