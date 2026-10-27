function proj_out=proj_interp(proj_in,angles_in,target_angle,model,tolerance)
%PROJ_INTERP Match low-energy views on a continuous angle axis (radians).
% nearest selects the first closest input; skip returns scalar NaN if its
% absolute angular distance exceeds tolerance. linear sorts unique angles
% and clamps targets outside their endpoints. No implicit wrapping is used.
% Original dual-energy workflow: Andrew Keeler, February 28, 2024.
angles_in=angles_in(:);
validateattributes(angles_in,{'numeric'},{'real','finite','nonempty'});
validateattributes(target_angle,{'numeric'},{'scalar','real','finite'});
validateattributes(tolerance,{'numeric'},{'scalar','real','finite','nonnegative'});
assert(size(proj_in,3)==numel(angles_in),'TIGRE:DE:angles','Angles must match projection views.');
[distance,index]=min(abs(angles_in-target_angle));
switch model
    case 'nearest'
        proj_out=proj_in(:,:,index);
    case 'skip'
        if distance>tolerance, proj_out=NaN; else, proj_out=proj_in(:,:,index); end
    case 'linear'
        [values,order]=sort(angles_in);
        assert(numel(values)>=2 && all(diff(values)>0),'TIGRE:DE:angles', ...
            'Linear interpolation requires at least two distinct, unique angles.');
        upper=find(values>=target_angle,1);
        if isempty(upper), upper=numel(values); end
        upper=max(2,upper); lower=upper-1;
        weight=min(1,max(0,(target_angle-values(lower))/(values(upper)-values(lower))));
        proj_out=(1-weight)*proj_in(:,:,order(lower))+weight*proj_in(:,:,order(upper));
    otherwise
        error('TIGRE:DE:interpolation','Use nearest, linear or skip.');
end
end
