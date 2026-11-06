function test_operatorAuditContract()
% Shortcut/metadata audit cases independent of a reconstruction phantom.
UseBlackwellBuild();
op=struct('volume_shape',[3 3 3],'projection_shape',[3 3 3],'valid_mask',[]);
op.forward=@(x) x; op.adjoint=@(x) x;
op.subset=@(idx) error('Exact warm starts must not need subsets');
init=reshape(single(1:27),3,3,3);
for name={'LSQR','LSMR','hybrid_LSQR','AB_GMRES','BA_GMRES','hybrid_fLSQR_TV'}
    result=feval([name{1} '_operator'],init,op,3,'Init',init);
    assert(isequal(result,init),'Exact initializer changed for %s',name{1});
end
data=zeros(3,3,3,'single'); data(1)=1;
for name={'LSQR','LSMR','hybrid_LSQR','AB_GMRES','BA_GMRES','IRN_TV_CGLS'}
    options={}; if ismember(name{1},{'LSMR','hybrid_LSQR','IRN_TV_CGLS'}), options={'lambda',0}; end
    if strcmp(name{1},'IRN_TV_CGLS'), options=[options {'niter_outer',1}]; end
    result=feval([name{1} '_operator'],data,op,3,options{:});
    assert(max(abs(result(:)-data(:)))<1e-6,'Exact breakdown failed for %s',name{1});
end
for steps=[0 2]
    data=ones(3,3,3,'single'); if steps>0, data(:)=0; end
    mustFail(@() LSQR_operator(data,op,steps,'UnknownOption',1));
    mustFail(@() LSQR_operator(data,op,steps,'Init',nan(3,3,3)));
    mustFail(@() FISTA_operator(data,op,steps,'Hyper',-1));
    mustFail(@() MLEM_operator(-ones(3,3,3,'single'),op,steps));
end
bad=op; bad.projection_shape=[3 3 2];
mustFail(@() LSQR_operator(init,bad,0));
bad=op; bad.valid_mask=ones(3,3,3);
mustFail(@() CGLS_operator(init,bad,0));
fprintf('PASS MATLAB operator audit: exact warm starts, validation on shortcuts, shapes and masks\n');
end
function mustFail(action)
failed=false; try, action(); catch, failed=true; end
assert(failed,'Invalid operator inputs must fail before shortcuts');
end
