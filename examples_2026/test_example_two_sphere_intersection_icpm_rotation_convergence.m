function [pass, str] = test_example_two_sphere_intersection_icpm_rotation_convergence()
  str = 'two-sphere intersection ICPM rotation convergence diagnostics';

  opts = struct('hvals', [0.2 0.1 0.05], ...
                'makePlots', false, ...
                'showDiagnostics', false);
  results = example_two_sphere_intersection_icpm_rotation_convergence(opts);

  passlist = [];
  names = {};
  c = 0;

  c = c + 1;
  passlist(c) = all(isfinite(results.errorsLinf(:))) && ...
                all(isfinite(results.errorsL2(:))) && ...
                all(results.errorsLinf(:) > 0) && ...
                all(results.errorsL2(:) > 0);
  names{c} = 'finite positive surface errors';

  c = c + 1;
  passlist(c) = all(results.crossRowsA(:) > 0) && ...
                all(results.crossRowsB(:) > 0);
  names{c} = 'cross rows nonempty';

  c = c + 1;
  passlist(c) = all(isfinite(results.maxExtensionRowSumError(:))) && ...
                max(results.maxExtensionRowSumError(:)) < 1e-4;
  names{c} = 'extension row sums';

  c = c + 1;
  singularDiff = results.maxSingularBranchDifference(:);
  passlist(c) = all(isfinite(singularDiff)) && ...
                all(diff(singularDiff) < 0) && ...
                singularDiff(end) < 1e-4;
  names{c} = 'singular branch difference';

  c = c + 1;
  passlist(c) = length(results.ratesLinf) >= 2 && ...
                length(results.ratesL2) >= 2 && ...
                isfinite(results.ratesLinf(end)) && ...
                isfinite(results.ratesL2(end)) && ...
                results.ratesLinf(end) > 1.7 && ...
                results.ratesL2(end) > 1.7;
  names{c} = 'refined surface rates';

  pass = passlist;
  if (~all(pass))
    failed = find(~pass);
    str = [str ': failed ' names{failed(1)}];
    for k = 2:length(failed)
      str = [str ', ' names{failed(k)}];
    end
  end
end
