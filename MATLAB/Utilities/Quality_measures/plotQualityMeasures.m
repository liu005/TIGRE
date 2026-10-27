function axesHandles=plotQualityMeasures(histories,algorithmNames,metricNames)
%PLOTQUALITYMEASURES Plot unequal-length iteration histories without padding.
% Histories are metric-by-iteration matrices, one cell per algorithm.
assert(iscell(histories) && numel(histories)==numel(algorithmNames), ...
    'TIGRE:quality:histories','Supply one history per algorithm name');
for k=1:numel(histories)
    assert(ismatrix(histories{k}) && size(histories{k},1)==numel(metricNames), ...
        'TIGRE:quality:metrics','Each history needs one row per metric');
end
axesHandles=gobjects(1,numel(metricNames));
for metric=1:numel(metricNames)
    figure('Name',['Quality: ' metricNames{metric}]);
    ax=axes(); axesHandles(metric)=ax; hold(ax,'on');
    for algorithm=1:numel(histories)
        values=histories{algorithm}(metric,:);
        plot(ax,1:numel(values),values,'DisplayName',algorithmNames{algorithm});
    end
    title(ax,['Evolution of ' metricNames{metric} ' per iteration']);
    xlabel(ax,'Completed iteration'); ylabel(ax,metricNames{metric});
    legend(ax,'show'); grid(ax,'on');
end
end
