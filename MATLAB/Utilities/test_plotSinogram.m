% test_plotSinogram.m
% Functional test for plotSinogram.m (TIGRE issue #278).
% Run with: octave --no-gui test_plotSinogram.m
% (or in real MATLAB: run this script directly.)
%
% Uses only top-level statements and anonymous function handles (no
% named local functions), so it behaves identically under MATLAB and
% GNU Octave and avoids their differing rules for local functions
% defined inside a script file.

set(0, 'DefaultFigureVisible', 'off');  % headless: don't try to pop a window
try
  graphics_toolkit('gnuplot');
catch
  % not Octave / toolkit unavailable — fine, figures are invisible anyway
end

% Synthetic projection data: [V (detector rows), U (detector cols), nAngles]
nV = 8; nU = 10; nAngles = 12;
proj = rand(nV, nU, nAngles);
alpha = linspace(0, 2*pi, nAngles);

names = {
  'default call (no options)'
  'Step option'
  'Slice option (single row)'
  'Colormap option (builtin name)'
  'Colormap option (custom Nx3 matrix)'
  'Clims option'
  'Combination: Step+Colormap+Clims'
};
calls = {
  @() plotSinogram(proj, alpha)
  @() plotSinogram(proj, alpha, 'Step', 2)
  @() plotSinogram(proj, alpha, 'Slice', 3)
  @() plotSinogram(proj, alpha, 'Colormap', 'hot')
  @() plotSinogram(proj, alpha, 'Colormap', gray(64))
  @() plotSinogram(proj, alpha, 'Clims', [0 1])
  @() plotSinogram(proj, alpha, 'Step', 3, 'Colormap', 'jet', 'Clims', [0 1])
};
% These should error and are checked separately below.
badNames = {
  'Invalid option name errors as expected'
  'Odd number of option args errors as expected'
};
badCalls = {
  @() plotSinogram(proj, alpha, 'NotARealOption', 1)
  @() plotSinogram(proj, alpha, 'Step')
};

nRun = 0;
nFailed = 0;

for i = 1:numel(calls)
  nRun = nRun + 1;
  try
    calls{i}();
    printf('  PASS: %s\n', names{i});
  catch err
    nFailed = nFailed + 1;
    printf('  FAIL: %s\n    %s\n', names{i}, err.message);
  end
  close all;
end

for i = 1:numel(badCalls)
  nRun = nRun + 1;
  threw = false;
  try
    badCalls{i}();
  catch
    threw = true;
  end
  close all;
  if threw
    printf('  PASS: %s\n', badNames{i});
  else
    nFailed = nFailed + 1;
    printf('  FAIL: %s\n    expected an error but none was thrown\n', badNames{i});
  end
end

% Correctness check: the plotted image for a given detector row must
% equal proj(row,:,:) exactly (i.e. the intended transposed slice),
% not e.g. an accidental copy of plotProj's angle-slicing behavior.
nRun = nRun + 1;
try
  rowToCheck = 5;
  plotSinogram(proj, alpha, 'Slice', rowToCheck);
  im = findobj(gca, 'Type', 'image');
  if isempty(im)
    error('no image object found on the axes');
  end
  plotted = get(im, 'CData');
  expected = squeeze(proj(rowToCheck, :, :));
  if ~isequal(size(plotted), size(expected))
    error('plotted image size %s does not match expected U-by-angle slice size %s', ...
      mat2str(size(plotted)), mat2str(size(expected)));
  end
  if max(abs(plotted(:) - expected(:))) > 1e-10
    error('plotted data does not match proj(row,:,:) - wrong axis sliced');
  end
  printf('  PASS: %s\n', 'plotted data matches the correct (transposed) slice');
catch err
  nFailed = nFailed + 1;
  printf('  FAIL: %s\n    %s\n', 'plotted data matches the correct (transposed) slice', err.message);
end
close all;

printf('\n%d/%d tests passed\n', nRun - nFailed, nRun);
if nFailed > 0
  error('%d test(s) failed', nFailed);
end
