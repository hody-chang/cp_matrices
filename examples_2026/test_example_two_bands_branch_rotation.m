function [pass, str] = test_example_two_bands_branch_rotation()
  str = 'two-band branch rotation example diagnostics';

  makePlots = false;
  scriptname = which('example_two_bands_branch_rotation.m');
  if (isempty(scriptname))
    scriptname = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                          'example_two_bands_branch_rotation.m');
  end
  run(scriptname);

  passlist = [];
  names = {};
  c = 0;

  c = c + 1;
  passlist(c) = (errorA < 1e-2) && (errorB < 1e-2);
  names{c} = 'heat errors';

  c = c + 1;
  passlist(c) = ~isempty(crossRowsA) && ~isempty(crossRowsB);
  names{c} = 'cross rows nonempty';

  c = c + 1;
  rowSums = full(sum(Eblock,2));
  passlist(c) = max(abs(rowSums - 1)) < 1e-10;
  names{c} = 'extension row sums';

  c = c + 1;
  passlist(c) = max(abs([thetaAtoB(:); thetaBtoA(:)])) < 100*eps;
  names{c} = 'zero cut-circle rotations';

  c = c + 1;
  passlist(c) = exist('vertexDiffFinal','var') && ...
      (length(vertexDiffFinal) == 2) && ...
      (max(vertexDiffFinal) < 1e-12);
  names{c} = 'shared vertex averaging';

  endpointRowsA = unique([ ...
      find(hypot(cpxgoutA - endpointA1(1), cpygoutA - endpointA1(2)) <= endpointTol); ...
      find(hypot(cpxgoutA - endpointA2(1), cpygoutA - endpointA2(2)) <= endpointTol)]);
  endpointRowsB = unique([ ...
      find(hypot(cpxgoutB - endpointB1(1), cpygoutB - endpointB1(2)) <= endpointTol); ...
      find(hypot(cpxgoutB - endpointB2(1), cpygoutB - endpointB2(2)) <= endpointTol)]);

  c = c + 1;
  passlist(c) = ~isempty(endpointRowsA) && all(ismember(endpointRowsA, crossRowsA));
  names{c} = 'branch A endpoint routing';

  c = c + 1;
  passlist(c) = ~isempty(endpointRowsB) && all(ismember(endpointRowsB, crossRowsB));
  names{c} = 'branch B endpoint routing';

  c = c + 1;
  th = pi/6;
  x0 = 0;  y0 = 0;
  dx0 = 1;  dy0 = 0;
  xr = x0 + cos(th).*dx0 - sin(th).*dy0;
  yr = y0 + sin(th).*dx0 + cos(th).*dy0;
  passlist(c) = norm([xr yr] - [cos(th) sin(th)], inf) < 100*eps;
  names{c} = 'nonzero rotation formula';

  pass = all(passlist);
  if (~pass)
    failed = find(~passlist);
    str = [str ': failed ' names{failed(1)}];
    for k = 2:length(failed)
      str = [str ', ' names{failed(k)}];
    end
  end
end
