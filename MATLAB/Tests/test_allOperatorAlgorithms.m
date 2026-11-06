function test_allOperatorAlgorithms()
% Every public iterative family on exact shared Python/MATLAB detector rays.
UseBlackwellBuild(); root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
path=[tempname '.mat']; previous=getenv('PYTHONPATH');
cleanup=onCleanup(@() restore(path,previous));
setenv('PYTHONPATH',[fullfile(root,'Python') pathsep previous]);
[status,message]=system(sprintf('conda run -n recon python "%s" "%s"', ...
    fullfile(root,'Python','tests','matlab_pocs_reference.py'),path));
assert(status==0,'%s',message); ref=load(path);
g=struct('nVoxel',double(ref.nVoxel(:)),'dVoxel',ref.dVoxel(:), ...
    'sVoxel',double(ref.nVoxel(:)).*ref.dVoxel(:),'offOrigin',ref.offOrigin(:));
s=struct('source_positions_mm',ref.source,'pixel_centers_mm',ref.pixels,'valid_mask',logical(ref.mask));
op=projectionOperator(g,s);
names={'SART','SIRT','OS_SART','fast_OS_SART','SART_TV','OS_SART_TV', ...
    'iterativereconalg','CGLS','LSQR','hybrid_LSQR','LSMR','IRN_TV_CGLS', ...
    'hybrid_fLSQR_TV','AB_GMRES','BA_GMRES','FISTA','ISTA','MLEM', ...
    'ASD_POCS','OS_ASD_POCS','AwASD_POCS','OS_AwASD_POCS','PCSD','AwPCSD','OS_PCSD','OS_AwPCSD'};
failures={};
for k=1:numel(names)
    name=names{k}; key=lower(name); args={};
    if contains(key,'pocs') || contains(key,'pcsd'), args={'MaxL2err',0,'TViter',2}; end
    if startsWith(key,'os_') || strcmp(key,'fast_os_sart'), args=[args {'BlockSize',5}]; end
    if contains(key,'aw'), args=[args {'Delta',[-.03 -.02 -.01]}]; end
    if ismember(key,{'sart_tv','os_sart_tv'}), args=[args {'TViter',2,'TVLambda',50}]; end
    if ismember(key,{'fista','ista'}), args=[args {'TViter',2,'lambda',.1,'Hyper',2000}]; end
    if strcmp(key,'irn_tv_cgls'), args=[args {'niter_outer',1}]; end
    switch key
        case 'awpcsd', key='aw_pcsd';
        case 'os_awpcsd', key='os_aw_pcsd';
    end
    try
        result=feval([name '_operator'],ref.data,op,3,args{:});
        wrapargs=args;
        if strcmpi(name,'SIRT'), wrapargs=[wrapargs {'Tolerance',0,'Nonnegative',false}]; end
        wrapped=feval([name '_surface'],ref.data,g,s,3,wrapargs{:});
        assert(all(isfinite(result(:))),'Nonfinite operator result');
        assert(isa(result,'single'),'Operator result must be single precision');
        wrapperDifference=norm(double(result(:)-wrapped(:)));
        assert(wrapperDifference<2e-3,'Operator/surface absolute difference %.6g exceeds 0.002',wrapperDifference);
        expected=ref.(key); relative=norm(double(result(:)-expected(:)))/max(norm(double(expected(:))),1e-9);
        fprintf('%s cross-language relative %.6g\n',name,relative);
        assert(relative<2e-3,'Cross-language relative difference %.6g exceeds 0.002',relative);
    catch error
        failures{end+1}=[name ': ' error.message]; %#ok<AGROW>
        fprintf('FAIL %s\n',failures{end});
    end
end
assert(isempty(failures),'TIGRE:operator:all','%s',strjoin(failures,newline));
fprintf('PASS all 26 operator algorithms, surface wrappers and cross-language references\n');
end
function restore(path,previous)
setenv('PYTHONPATH',previous); if exist(path,'file'), delete(path); end
end
