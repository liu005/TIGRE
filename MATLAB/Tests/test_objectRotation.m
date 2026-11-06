function test_objectRotation()
% Arbitrary object poses, GUI preset, inverse geometry and shared YAML.
examples=detectorGeometryExamples(); s=examples(strcmp({examples.name},'TIGRE d18: arbitrary object rotation (ZYZ)')).settings;
s.n_views=3; s.module_shape=[3 4]; surface=buildModularSurface(s);
assert(isequal(surface.source_positions_mm,repmat(surface.source_positions_mm(:,1),1,3)));
g=struct('nVoxel',[6;4;2],'sVoxel',[60;40;20],'offOrigin',[3;2;1], ...
    'object_angles_deg',[0 0 0;30 90 60;-20 40 90]);
[fixed,gfixed]=fixedObjectGeometry(surface,g); rotations=objectRotationMatrices(g,3);
for k=1:3
    reconstructed=rotations(:,:,k)*double(fixed.source_positions_mm(:,k))+g.offOrigin;
    assert(max(abs(reconstructed-double(surface.source_positions_mm(:,k))))<.001);
end
assert(isequal(gfixed.offOrigin,zeros(3,1)) && ~isfield(gfixed,'object_angles_deg'));
report=validateCTTransmission(surface,g); assert(report.crossing_samples>0);
fprintf('PASS object rotation core geometry\n');
end
