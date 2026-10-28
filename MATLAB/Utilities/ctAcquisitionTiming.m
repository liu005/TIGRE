function [a,times]=ctAcquisitionTiming(n,a)
%CTACQUISITIONTIMING One synchronized detector frame per source pulse, t=0 first.
validateattributes(n,{'numeric'},{'scalar','integer','positive','<=',6000});
names={'readout_period_ms','firing_rate_hz','rotation_period_s','table_speed_mm_s','playback_rate'};
assert(isstruct(a) && all(isfield(a,names)) && numel(fieldnames(a))==numel(names), ...
    'TIGRE:timing:fields','Supply all five acquisition timing fields');
for i=1:numel(names)
    a.(names{i})=double(a.(names{i}));
    validateattributes(a.(names{i}),{'numeric'},{'scalar','finite'});
    if ~strcmp(names{i},'table_speed_mm_s')
        assert(a.(names{i})>0,'TIGRE:timing:positive','Timing periods and rates must be positive');
    end
end
assert(a.readout_period_ms*a.firing_rate_hz<=1000*(1+1e-12), ...
    'TIGRE:timing:capacity','Source firing rate exceeds detector readout capacity');
times=(0:n-1)/a.firing_rate_hz;
end
