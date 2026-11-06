function test_operatorPocs()
% Cross-language geometry, masks, weighted directions and conventional APIs.
UseBlackwellBuild();
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
reference=[tempname '.mat']; oldPath=getenv('PYTHONPATH');
cleanup=onCleanup(@() restore(reference,oldPath));
setenv('PYTHONPATH',[fullfile(root,'Python') pathsep oldPath]);
script=fullfile(root,'Python','tests','matlab_pocs_reference.py');
[status,message]=system(sprintf('conda run -n recon python "%s" "%s"',script,reference));
assert(status==0,'%s',message); ref=load(reference);
g=struct('nVoxel',double(ref.nVoxel(:)),'dVoxel',ref.dVoxel(:),'sVoxel',double(ref.nVoxel(:)).*ref.dVoxel(:),'offOrigin',ref.offOrigin(:));
s=struct('source_positions_mm',ref.source,'pixel_centers_mm',ref.pixels,'valid_mask',logical(ref.mask));
op=projectionOperator(g,s,'Device',0);
sub=op.subset([4 2 4]); nested=sub.subset([2 1]);
full=op.forward(ref.image); expected=full(:,:,[2 4]); actual=nested.forward(ref.image);
assert(max(abs(actual(:)-expected(:)))<1e-4);
names={'ASD_POCS','OS_ASD_POCS','AwASD_POCS','OS_AwASD_POCS','PCSD'};
for k=1:numel(names)
    name=names{k}; options={'MaxL2err',0,'TViter',2};
    if startsWith(name,'OS_'), options=[options {'BlockSize',5}]; end
    if contains(name,'Aw'), options=[options {'Delta',[-.03 -.02 -.01]}]; end
    [result,h]=feval(name,ref.data,[],[],3,'Operator',op,options{:});
    [wrapped,hw]=feval([name '_surface'],ref.data,g,s,3,options{:});
    direct=feval([name '_operator'],ref.data,op,3,options{:});
    assert(max(abs(result(:)-direct(:)))<2e-4);
    assert(max(abs(result(:)-wrapped(:)))<2e-4);
    assert(max(abs(h.residual-hw.residual))<1e-3);
    key=lower(name); expected=ref.(key);
    relative=norm(double(result(:)-expected(:)))/max(norm(double(expected(:))),1e-8);
    assert(relative<2e-4,'MATLAB/Python mismatch for %s: %.6g',name,relative);
    expectedHistory=ref.([key '_residual']);
    assert(max(abs(h.residual(:)-expectedHistory(:)))<1e-3);
    assert(all(isfinite(result(:))) && min(result(:))>=0);
    fprintf('PASS %s operator/surface/Python parity: relative %.4g\n',name,relative);
    [zero,h0]=feval([name '_surface'],ref.data,g,s,0,options{:});
    assert(~any(zero(:))&&isempty(h0.beta));
end
% Test weighted scalar/equal-vector consistency at the native gateway.
ids=GpuIds(); ids.devices=int32(0);
a=minimizeAwTV(ref.image,.01,2,-.005,'gpuids',ids);
b=minimizeAwTV(ref.image,.01,2,[-.005 -.005 -.005],'gpuids',ids);
assert(max(abs(a(:)-b(:)))<1e-5);
% Conventional calls retain legacy output types and options.
geo=defaultGeometry('nVoxel',[8;8;8]); geo.sVoxel=[8;8;8]; geo.dVoxel=[1;1;1];
geo.nDetector=[14;12]; geo.sDetector=geo.nDetector.*geo.dDetector;
angles=(0:11)*2*pi/12; data=Ax(ref.image,geo,angles);
standard=projectionOperator(geo,angles); direct=standard.forward(ref.image);
assert(max(abs(direct(:)-data(:)))<1e-4);
subset=standard.subset([3 1]); selected=subset.forward(ref.image); expected=data(:,:,[3 1]);
assert(max(abs(selected(:)-expected(:)))<1e-4);
back=standard.adjoint(data); assert(all(isfinite(back(:))));
varied=geo; varied.DSO=geo.DSO+(0:11); varied.DSD=geo.DSD+2*(0:11);
varied.offOrigin=[zeros(1,12);(0:11)*.01;zeros(1,12)];
varied.offDetector=[(0:11)*.01;zeros(1,12)]; varied.COR=(0:11)*.01;
varied.rotDetector=[zeros(2,12);(0:11)*.001];
moving=projectionOperator(varied,angles); allViews=moving.forward(ref.image);
sub=moving.subset([8 3 8]); sub=sub.subset([3 2]); actual=sub.forward(ref.image);
expected=allViews(:,:,[8 3]); assert(max(abs(actual(:)-expected(:)))<1e-4);
for k=1:numel(names)
    args={'MaxL2err',0,'TViter',2,'Verbose',0,'redundancy_weighting',false};
    if startsWith(names{k},'OS_'), args=[args {'BlockSize',5}]; end
    image=feval(names{k},data,geo,angles,2,args{:});
    assert(all(isfinite(image(:))));
end
mustFail(@() ASD_POCS_surface(ref.data,g,s,1));
mustFail(@() OS_AwASD_POCS_surface(ref.data,g,s,1,'MaxL2err',0,'Delta',[1 0 1]));
mustFail(@() OS_ASD_POCS_surface(ref.data,g,s,1,'MaxL2err',0,'BlockSize',0));
mustFail(@() ASD_POCS_surface(ref.data,g,s,1,'MaxL2err',0,'Init','FDK'));
testDense();
testPCSDGate();
for method={'CGLS','SIRT','OS_SART','fast_OS_SART'}
    name=method{1}; options={};
    if contains(name,'OS_SART'), options={'BlockSize',5}; end
    [result,~]=feval([name '_operator'],ref.data,op,3,options{:});
    if strcmp(name,'SIRT'), options={'Tolerance',0,'Nonnegative',false}; end
    wrapped=feval([name '_surface'],ref.data,g,s,3,options{:});
    assert(max(abs(result(:)-wrapped(:)))<1e-4);
    expected=ref.(lower(name)); relative=norm(double(result(:)-expected(:)))/max(norm(double(expected(:))),1e-8);
    assert(relative<2e-4,'Cross-language %s mismatch %.6g',name,relative);
    fprintf('PASS %s operator/surface/Python parity: relative %.4g\n',name,relative);
end
fprintf('PASS all four MATLAB ASD-POCS variants; conventional path and dense oracle\n');
end
function mustFail(action)
failed=false; try, action(); catch, failed=true; end
assert(failed,'Expected invalid operator options to fail');
end
function testPCSDGate()
op=struct('volume_shape',[1 1 1],'projection_shape',[1 1 1],'valid_mask',[]);
op.forward=@(x) x; op.adjoint=@(x) x; sub=op; op.subset=@(idx) sub;
b=single(.8);
result=PCSD_operator(b,op,1,'MaxL2err',.7,'TViter',0);
assert(result==b); % Absolute gate, not residual squared versus tolerance.
[result,h]=PCSD_operator(b,op,3,'MaxL2err',0,'TViter',2,'Init',b);
assert(result==b&&numel(h.beta)==3&&all(h.tv_step==0));
[result,h]=PCSD_operator(b,op,0,'MaxL2err',0,'Init',b);
assert(result==b&&isempty(h.beta));
mustFail(@() PCSD_operator(b,op,1));
mustFail(@() PCSD_operator(b,op,1,'MaxL2err',0,'alpha',.1));
end
function restore(file,oldPath)
setenv('PYTHONPATH',oldPath); if exist(file,'file'), delete(file); end
end
function testDense()
A=single([1 2 0 0;0 1 3 0;0 0 0 0;1 1 1 0;0 2 1 0;3 1 0 0]);
mask=true(2,1,3); mask(1,1,2)=false; op=dense(A,mask);
data=reshape(single(1:6),2,1,3); data(~mask)=NaN;
for block=[1 2 20]
    [actual,~]=ASD_POCS_operator(data,op,2,'MaxL2err',0,'TViter',0,'BlockSize',block,'lambda',.8,'lambda_red',.7);
    x=zeros(4,1); clean=double(data); clean(~mask)=0; nb=ceil(3/block); beta=.8;
    for iter=1:2
        for k=1:nb
            views=k:nb:3; rows=reshape(1:6,2,3); rows=rows(:,views); rows=rows(:);
            a=double(A(rows,:)); row=sum(a,2); col=sum(a,1)';
            w=zeros(size(row)); w(row>1e-6)=1./row(row>1e-6);
            v=zeros(size(col)); v(col>1e-6)=1./col(col>1e-6);
            x=max(x+beta*v.*(a'*(w.*(clean(rows)-a*x))),0);
        end
        beta=beta*.7;
    end
    assert(max(abs(double(actual(:))-x))<2e-6);
end
end
function op=dense(A,mask)
n=size(A,1)/2;
op=struct('volume_shape',[2 2 1],'projection_shape',[2 1 n],'valid_mask',mask);
op.forward=@(x) reshape(A*x(:),2,1,n);
op.adjoint=@(b) reshape(A'*b(:),2,2,1);
op.subset=@(idx) denseSubset(A,mask,idx);
end
function out=denseSubset(A,mask,idx)
rows=reshape(1:size(A,1),2,[]); rows=rows(:,idx);
out=dense(A(rows(:),:),mask(:,:,idx));
end
