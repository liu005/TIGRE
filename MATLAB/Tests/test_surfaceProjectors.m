function test_surfaceProjectors()
% Native GPU integration: asymmetric xyz volume and non-square masked panels.
UseBlackwellBuild(); rng(7);
geo=defaultGeometry('nVoxel',[20;24;16]);
geo.dVoxel=[1;1.2;1.4]; geo.sVoxel=geo.nVoxel.*geo.dVoxel;
geo.offOrigin=[1;-2;0.5];
examples=detectorGeometryExamples();
settings=examples(strcmp({examples.name},'Circular CT: curved modular detector')).settings;
settings.n_views=40; settings.source_radius_mm=100;
settings.detector_radius_mm=70; settings.module_shape=[19 12];
settings.pixel_spacing_mm=[2 2]; settings.n_modules=3;
settings.angular_span_deg=35; settings.column_gap=2;
settings.cor_mm=3;
settings.panel_x_angles_deg=[-2 0 3]; settings.panel_y_angles_deg=[1 -3 2];
surface=buildModularSurface(settings);
x=rand(double(geo.nVoxel(:))','single');
q=rand(size(surface.valid_mask),'single');
ax=Ax_surface(x,geo,surface); atq=Atb_surface(q,geo,surface);
lhs=sum(double(ax(:)).*double(q(:))); rhs=sum(double(x(:)).*double(atq(:)));
err=abs(lhs-rhs)/max(abs([lhs rhs])); assert(err<2e-5);
assert(all(ax(~surface.valid_mask)==0));
q(~surface.valid_mask)=1e6;
masked=Atb_surface(q,geo,surface); assert(max(abs(masked(:)-atq(:)))<1e-3);
% Analytic centred box ray checks xyz order, spacing, origin, one-view support.
one=struct('source_positions_mm',single([100;-2;0.5]), ...
    'pixel_centers_mm',reshape(single([-100 -2 0.5]),1,1,3),'valid_mask',true);
line=Ax_surface(ones(size(x),'single'),geo,one);
assert(abs(double(line)-geo.sVoxel(1))<1e-3);
for axisIndex=2:3
    source=geo.offOrigin(:); pixel=source;
    source(axisIndex)=100; pixel(axisIndex)=-100;
    one.source_positions_mm=single(source);
    one.pixel_centers_mm=reshape(single(pixel),1,1,3);
    line=Ax_surface(ones(size(x),'single'),geo,one);
    assert(abs(double(line)-geo.sVoxel(axisIndex))<1e-3);
end
% Consistent data should reconstruct without flattening the tilted modules.
[xx,yy,zz]=ndgrid(((0:19)-9.5),((0:23)-11.5)*1.2,((0:15)-7.5)*1.4);
phantom=single(xx.^2+yy.^2+zz.^2<6^2);
b=Ax_surface(phantom,geo,surface);
[image,history]=CGLS_surface(b,geo,surface,35);
relative=norm(double(image(:)-phantom(:)))/norm(double(phantom(:)));
assert(relative<0.35 && history(end)<0.03*norm(double(b(:))));
[sirt,sh]=SIRT_surface(b,geo,surface,15);
assert(all(isfinite(sirt(:))) && min(sirt(:))>=0 && sh(end)<sh(1));
% A fixed panel and per-view masks exercise singleton broadcasting.
fixed=surface; fixed.pixel_centers_mm=fixed.pixel_centers_mm(:,:,:,1);
fixed.valid_mask=fixed.valid_mask(:,:,1);
fixed.valid_mask=repmat(fixed.valid_mask,1,1,settings.n_views);
fixed.valid_mask(5:8,5:9,2)=false;
fp=Ax_surface(x,geo,fixed); fq=Atb_surface(q,geo,fixed);
fl=sum(double(fp(:)).*double(q(:))); fr=sum(double(x(:)).*double(fq(:)));
assert(abs(fl-fr)/max(abs([fl fr]))<2e-5);
% Invalid shapes fail in MATLAB rather than reading beyond an array in MEX.
failed=false;
try, Ax_surface(x(:,:,1),geo,surface); catch e, failed=strcmp(e.identifier,'TIGRE:surface:shape'); end
assert(failed);
[zero,h]=CGLS_surface(zeros(size(b),'single'),geo,surface,5);
assert(~any(zero(:)) && isempty(h));
fprintf('PASS native surface CUDA: adjoint %.3g, CGLS relative error %.4f, %d iterations\n',err,relative,numel(history));
end
