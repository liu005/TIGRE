function test_qualityMeasures()
% Pearson correlation must not require the Statistics Toolbox.
x=reshape(1:24,2,3,4); y=sin(x/3)+x/10;
a=x(:)-mean(x(:)); b=y(:)-mean(y(:));
reference=sum(a.*b)/sqrt(sum(a.^2)*sum(b.^2));
assert(abs(CC(x,y)-reference)<1e-12);
assert(abs(CC(single(x),single(2*x+5))-1)<1e-12);
assert(abs(CC(x,-x)+1)<1e-12);
assert(isnan(CC(ones(size(x)),x)));
bad=y; bad(1)=NaN; assert(isnan(CC(x,bad)));
quality=Measure_Quality(x,y,{'RMSE','CC','MSSIM','UQI'});
assert(numel(quality)==4 && all(isfinite(quality)));
assert(abs(quality(2)-reference)<1e-12);
fprintf('PASS base-MATLAB quality metrics and Pearson correlation edge cases\n');
end
