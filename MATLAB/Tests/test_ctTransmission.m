function test_ctTransmission()
geo=struct('sVoxel',[10;10;10],'offOrigin',zeros(3,1));
s=struct('source_positions_mm',[20;0;0],'pixel_centers_mm',reshape([-20;0;0],1,1,3), ...
    'valid_mask',true(1,1));
report=validateCTTransmission(s,geo); assert(report.crossing_samples==1);
s.pixel_centers_mm=reshape([30;0;0],1,1,3);
try, validateCTTransmission(s,geo); error('test:missed','Expected same-side rejection');
catch ex, assert(strcmp(ex.identifier,'TIGRE:physics:sameSide')); end
s.source_positions_mm=[6;0;0]; s.pixel_centers_mm=reshape([4;6;0],1,1,3);
report=validateCTTransmission(s,geo); assert(report.crossing_samples==1);
s.source_positions_mm=repmat([20;0;0],1,2);
s.pixel_centers_mm=repmat(reshape([-20;0;0],1,1,3),1,1,1,2); s.valid_mask=true(1,1,2);
geo.offOrigin=[0 0;0 0;0 20];
try, validateCTTransmission(s,geo); error('test:missed','Expected moving-volume rejection');
catch ex, assert(strcmp(ex.identifier,'TIGRE:physics:noRays')); end
fprintf('PASS physical transmission: opposite/same side, off-center corner rays, moving volume\n');
end
