function [pass, str] = test_param_output_aligned()
  str = ['call all param fcns, make sure curves/filaments return column vectors'];

  pass = [];
  c = 0;
  skipped = {};

  % TODO: will this path fail on windows?
  files = dir('../');
  % TODO: yuck:
  files(end+1).name = 'paramPolygon.m';

  for i=1:length(files)
    file = files(i).name;
    % TODO: check if it starts with param
    if strmatch('param', file)
      f = str2func(file(1:end-2));
      try
        [x,y] = f(10);
      catch err
        % some of these need functions we might not have (e.g., the
        % spline curves call cscvn from the Curve Fitting Toolbox).  A
        % missing toolbox is not a failure of this test, so skip it.
        if strcmp(err.identifier, 'MATLAB:UndefinedFunction')
          skipped{end+1} = file;
          continue
        end
        rethrow(err)
      end
      c = c + 1;
      %pass(c) = (size(x,1) >= 2);
      if (size(x,1) >= 2)
        pass(c) = 1;
      else
        disp('  Not a column vector:');
        f
        file
        pass(c) = 0;
      end
    end
  end

  if ~isempty(skipped)
    disp(['  (skipped, missing dependency: ' strjoin(skipped, ' ') ')']);
  end
