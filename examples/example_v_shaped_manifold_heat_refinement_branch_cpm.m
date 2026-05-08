function [results, diagnostics] = example_v_shaped_manifold_heat_refinement_branch_cpm(opts)
%EXAMPLE_V_SHAPED_MANIFOLD_HEAT_REFINEMENT_BRANCH_CPM
% Branch-aware CPM heat equation on a V-shaped two-branch manifold.
%
% The two arms are solved with separate branch-local CPM ingredients.  The
% extension matrix for one arm only sees data from that arm.  The shared
% vertex is not advanced by either single-branch CPM operator; instead it is
% updated by the symmetric two-neighbor junction formula
%
%   uv^{n+1} = uv^n + dt*kappa*(uL_1^n + uR_1^n - 2*uv^n)/h^2.
%
% Example:
%   [results, diagnostics] = ...
%       example_v_shaped_manifold_heat_refinement_branch_cpm();
%
% The default run studies angles [90 120 150 170] with h = 1./[200 400
% 800].  Figure 1 compares u(s,t) against the exact solution on the
% finest grid for each angle.  Figure 2 plots L_inf and L_2 errors versus h.

  if nargin < 1
    opts = struct();
  end

  thisfile = mfilename('fullpath');
  examplesdir = fileparts(thisfile);
  repoRoot = fileparts(examplesdir);
  addpathIfNeeded(fullfile(repoRoot, 'cp_matrices'));
  addpathIfNeeded(fullfile(repoRoot, 'surfaces'));

  anglesDeg = optionValue(opts, 'anglesDeg', [90 120 150 170]);
  hvals = optionValue(opts, 'hvals', 1 ./ [200 400 800 1600 3200]);
  finalTime = optionValue(opts, 'finalTime', 0.01);
  kappa = optionValue(opts, 'kappa', 1.0);
  cfl = optionValue(opts, 'cfl', 0.1);
  interpDegree = optionValue(opts, 'interpDegree', 3);
  lapOrder = optionValue(opts, 'lapOrder', 2);
  padding = optionValue(opts, 'padding', []);
  vertexTol = optionValue(opts, 'vertexTol', 0);
  makePlots = optionValue(opts, 'makePlots', true);
  showDiagnostics = optionValue(opts, 'showDiagnostics', true);

  nCases = numel(anglesDeg) * numel(hvals);
  diagnostics = repmat(emptyDiagnostic(), nCases, 1);

  angleCol = zeros(nCases, 1);
  hCol = zeros(nCases, 1);
  cflCol = cfl * ones(nCases, 1);
  dtCol = zeros(nCases, 1);
  stepsCol = zeros(nCases, 1);
  finalInfErrCol = zeros(nCases, 1);
  finalL2ErrCol = zeros(nCases, 1);
  finalInfRateCol = NaN(nCases, 1);
  finalL2RateCol = NaN(nCases, 1);
  massDriftCol = zeros(nCases, 1);
  maxVertexJumpCol = zeros(nCases, 1);
  maxFluxJumpCol = zeros(nCases, 1);
  noShortcutCol = false(nCases, 1);

  row = 0;
  for ia = 1:numel(anglesDeg)
    for ih = 1:numel(hvals)
      row = row + 1;
      diagnostics(row) = solveOneLevel(anglesDeg(ia), hvals(ih), ...
          finalTime, kappa, cfl, interpDegree, lapOrder, padding, ...
          vertexTol);

      angleCol(row) = diagnostics(row).angleDeg;
      hCol(row) = diagnostics(row).h;
      dtCol(row) = diagnostics(row).dt;
      stepsCol(row) = diagnostics(row).numSteps;
      finalInfErrCol(row) = diagnostics(row).finalInfError;
      finalL2ErrCol(row) = diagnostics(row).finalL2Error;
      if ih > 1
        previousRow = row - 1;
        finalInfRateCol(row) = log(finalInfErrCol(previousRow) / ...
                                   finalInfErrCol(row)) / ...
                               log(hCol(previousRow) / hCol(row));
        finalL2RateCol(row) = log(finalL2ErrCol(previousRow) / ...
                                  finalL2ErrCol(row)) / ...
                              log(hCol(previousRow) / hCol(row));
      end
      massDriftCol(row) = diagnostics(row).massDrift;
      maxVertexJumpCol(row) = diagnostics(row).maxVertexJump;
      maxFluxJumpCol(row) = diagnostics(row).maxAbsFluxJump;
      noShortcutCol(row) = diagnostics(row).noShortcutOK;
    end
  end

  results = makeResultsTable(angleCol, hCol, cflCol, dtCol, stepsCol, ...
      finalInfErrCol, finalL2ErrCol, finalInfRateCol, finalL2RateCol, ...
      massDriftCol, maxVertexJumpCol, maxFluxJumpCol, noShortcutCol);

  if showDiagnostics
    disp(results);
    fprintf(['Equilibrium check: exact insulated equilibrium is 5.5; ' ...
             'reported mass drift uses composite trapezoid quadrature.\n']);
  end

  if makePlots && ~isempty(diagnostics)
    plotAngleSolutions(1, diagnostics, anglesDeg);
    plotConvergenceByAngle(2, diagnostics, anglesDeg);
  end
end

function diagnostic = solveOneLevel(angleDeg, requestedH, finalTime, kappa, ...
    cfl, interpDegree, lapOrder, padding, vertexTol)
%SOLVEONELEVEL Solve one branch-aware refinement level.

  n = max(2, round(1 / requestedH));
  h = 1 / n;
  sLeft = (0:-h:-1)';
  sRight = (0:h:1)';

  dim = 2;
  bw = rm_bandwidth(dim, interpDegree, lapOrder / 2);
  if isempty(padding)
    localPadding = max(0.08, (bw + interpDegree + 2) * h);
  else
    localPadding = padding;
  end

  left = buildBranch(-1, angleDeg, h, sLeft, interpDegree, lapOrder, ...
                     bw, localPadding, vertexTol);
  right = buildBranch(1, angleDeg, h, sRight, interpDegree, lapOrder, ...
                      bw, localPadding, vertexTol);
  noShortcutOK = extensionUsesOnlyOwnBranch(left) && ...
                 extensionUsesOnlyOwnBranch(right);
  if ~noShortcutOK
    error('A branch extension matrix has columns outside its branch data.');
  end

  uLeft = initialCondition(sLeft);
  uRight = initialCondition(sRight);
  uv = 0.5 * (uLeft(1) + uRight(1));
  uLeft(1) = uv;
  uRight(1) = uv;

  if finalTime > 0
    dt = cfl * h^2 / kappa;
    numSteps = ceil(finalTime / dt);
    dt = finalTime / numSteps;
  else
    dt = 0;
    numSteps = 0;
  end

  massHistory = zeros(numSteps + 1, 1);
  vertexJumpHistory = zeros(numSteps + 1, 1);
  fluxJumpHistory = zeros(numSteps + 1, 1);
  timeHistory = zeros(numSteps + 1, 1);

  massHistory(1) = branchMass(uLeft, uRight, h);
  vertexJumpHistory(1) = abs(uLeft(1) - uRight(1));
  fluxJumpHistory(1) = vertexFluxJump(uLeft, uRight, h);

  for kt = 1:numSteps
    uv = 0.5 * (uLeft(1) + uRight(1));
    uLeft(1) = uv;
    uRight(1) = uv;

    lapLeft = left.LCPM * uLeft;
    lapRight = right.LCPM * uRight;

    uLeftNew = uLeft;
    uRightNew = uRight;

    % Endpoint-affected branch rows are corrected when LCPM is assembled:
    % the shared vertex uses the graph update below, while the nearby branch
    % rows and physical endpoint use one-dimensional graph stencils.
    uLeftNew(2:end) = uLeft(2:end) + dt * kappa * lapLeft(2:end);
    uRightNew(2:end) = uRight(2:end) + dt * kappa * lapRight(2:end);

    uvNew = uv + dt * kappa * (uLeft(2) + uRight(2) - 2 * uv) / h^2;

    uLeftNew(1) = uvNew;
    uRightNew(1) = uvNew;

    uLeft = uLeftNew;
    uRight = uRightNew;

    massHistory(kt + 1) = branchMass(uLeft, uRight, h);
    vertexJumpHistory(kt + 1) = abs(uLeft(1) - uRight(1));
    fluxJumpHistory(kt + 1) = vertexFluxJump(uLeft, uRight, h);
    timeHistory(kt + 1) = kt * dt;
  end

  exactLeft = exactSolution(sLeft, finalTime, kappa);
  exactRight = exactSolution(sRight, finalTime, kappa);
  errorLeft = uLeft - exactLeft;
  errorRight = uRight - exactRight;
  uniqueError = [errorLeft; errorRight(2:end)];

  diagnostic = emptyDiagnostic();
  diagnostic.angleDeg = angleDeg;
  diagnostic.h = h;
  diagnostic.requestedH = requestedH;
  diagnostic.dt = dt;
  diagnostic.numSteps = numSteps;
  diagnostic.kappa = kappa;
  diagnostic.cfl = cfl;
  diagnostic.interpDegree = interpDegree;
  diagnostic.lapOrder = lapOrder;
  diagnostic.sLeft = sLeft;
  diagnostic.sRight = sRight;
  diagnostic.uLeft = uLeft;
  diagnostic.uRight = uRight;
  diagnostic.exactLeft = exactLeft;
  diagnostic.exactRight = exactRight;
  diagnostic.time = timeHistory;
  diagnostic.mass = massHistory;
  diagnostic.massInitial = massHistory(1);
  diagnostic.massFinal = massHistory(end);
  diagnostic.massDrift = massHistory(end) - massHistory(1);
  diagnostic.vertexJump = vertexJumpHistory;
  diagnostic.fluxJump = fluxJumpHistory;
  diagnostic.maxVertexJump = max(vertexJumpHistory);
  diagnostic.maxAbsFluxJump = max(abs(fluxJumpHistory));
  diagnostic.finalInfError = max(abs(uniqueError));
  diagnostic.finalL2Error = sqrt(h * (sum(errorLeft(2:end).^2) + ...
                                      sum(errorRight(2:end).^2) + ...
                                      0.5 * (errorLeft(1)^2 + errorRight(1)^2)));
  diagnostic.noShortcutOK = noShortcutOK;
  diagnostic.left = stripLargeMatrices(left);
  diagnostic.right = stripLargeMatrices(right);
end

function branch = buildBranch(side, angleDeg, h, sNodes, interpDegree, ...
    lapOrder, bw, padding, vertexTol)
%BUILDBRANCH Build one branch-local CPM band and manifold operator.

  halfAngle = 0.5 * angleDeg * pi / 180;
  if side < 0
    p0 = [-cos(halfAngle), sin(halfAngle)];
    p1 = [0, 0];
  else
    p0 = [0, 0];
    p1 = [cos(halfAngle), sin(halfAngle)];
  end

  x1d = gridVector(min(p0(1), p1(1)) - padding, ...
                   max(p0(1), p1(1)) + padding, h);
  y1d = gridVector(min(p0(2), p1(2)) - padding, ...
                   max(p0(2), p1(2)) + padding, h);
  [xx, yy] = meshgrid(x1d, y1d);

  [cpx, cpy, dist, bdy, t] = cpLineSegment2d(xx, yy, p0, p1);
  band = find(abs(dist) <= bw * h);

  cpxBand = cpx(band);
  cpyBand = cpy(band);
  xBand = xx(band);
  yBand = yy(band);
  bdyBand = bdy(band);
  if side < 0
    sBand = -1 + t(band);
  else
    sBand = t(band);
  end

  vertexMask = makeVertexMask(cpxBand, cpyBand, sBand, h, vertexTol);
  Eband = interp1BranchMatrix(sNodes, sBand, interpDegree);
  if any(vertexMask)
    Eband(vertexMask, :) = 0;
    Eband(vertexMask, 1) = 1;
  end

  Lcart = laplacian_2d_matrix(x1d, y1d, lapOrder, band, band);
  [xManifold, yManifold] = branchCoordinates(side, sNodes, angleDeg);
  R = interp2_matrix(x1d, y1d, xManifold, yManifold, interpDegree, band);

  branch.side = side;
  branch.angleDeg = angleDeg;
  branch.h = h;
  branch.sNodes = sNodes;
  branch.p0 = p0;
  branch.p1 = p1;
  branch.x1d = x1d;
  branch.y1d = y1d;
  branch.band = band;
  branch.xBand = xBand;
  branch.yBand = yBand;
  branch.cpxBand = cpxBand;
  branch.cpyBand = cpyBand;
  branch.sBand = sBand;
  branch.bdyBand = bdyBand;
  branch.vertexMask = vertexMask;
  branch.Eband = Eband;
  branch.Lcart = Lcart;
  branch.R = R;
  branch.LCPM = R * Lcart * Eband;
  if lapOrder == 2
    branch.LCPM = replaceEndpointAffectedRows(branch.LCPM, h, ...
                                              interpDegree, lapOrder);
  end
end

function L = replaceEndpointAffectedRows(L, h, interpDegree, lapOrder)
%REPLACEENDPOINTAFFECTEDROWS Fix rows where CP endpoint clamping is wrong.
%
% For a single branch, the closest-point map clamps all points beyond either
% endpoint back to that endpoint.  That is appropriate at the physical
% insulated endpoint, but at the shared vertex it incorrectly makes the
% branch-local operator see a Neumann end before the graph coupling is
% applied.  The same endpoint cap also gives low-order rows at the physical
% endpoint.  Since each branch is a straight unit-speed segment, use the
% intended one-dimensional graph stencils on the affected rows.

  n = size(L, 1);
  if n < 4
    error('Need at least four branch nodes for endpoint row replacement.');
  end

  capRows = max(3, interpDegree + lapOrder / 2);
  interiorRows = unique([2:min(1 + capRows, n - 1), ...
                         max(2, n - capRows):(n - 1)]);
  for r = interiorRows
    L(r, :) = 0;
    L(r, r - 1) = 1 / h^2;
    L(r, r) = -2 / h^2;
    L(r, r + 1) = 1 / h^2;
  end

  L(n, :) = 0;
  L(n, n - 1) = 2 / h^2;
  L(n, n) = -2 / h^2;
end

function E = interp1BranchMatrix(sNodes, sQuery, degree)
%INTERP1BRANCHMATRIX Barycentric Lagrange interpolation on branch nodes.

  sNodes = sNodes(:);
  sQuery = sQuery(:);
  nNodes = numel(sNodes);
  if nNodes < 2
    error('At least two branch nodes are required.');
  end

  degree = min(degree, nNodes - 1);
  stencilSize = degree + 1;

  if sNodes(2) < sNodes(1)
    sWork = flipud(sNodes);
    colMap = nNodes:-1:1;
  else
    sWork = sNodes;
    colMap = 1:nNodes;
  end

  ds = sWork(2) - sWork(1);
  if ds <= 0
    error('Branch nodes must be strictly monotone.');
  end

  nQuery = numel(sQuery);
  rows = zeros(nQuery * stencilSize, 1);
  cols = zeros(nQuery * stencilSize, 1);
  vals = zeros(nQuery * stencilSize, 1);
  cursor = 0;

  for iq = 1:nQuery
    sq = min(max(sQuery(iq), sWork(1)), sWork(end));
    centeredBase = floor((sq - sWork(1)) / ds) + 1 - floor(degree / 2);
    base = min(max(centeredBase, 1), nNodes - degree);
    localCols = base:(base + degree);
    localX = sWork(localCols);
    weights = lagrangeWeights(localX, sq);

    idx = cursor + (1:stencilSize);
    rows(idx) = iq;
    cols(idx) = colMap(localCols);
    vals(idx) = weights;
    cursor = cursor + stencilSize;
  end

  E = sparse(rows, cols, vals, nQuery, nNodes);
end

function weights = lagrangeWeights(nodes, x)
%LAGRANGEWEIGHTS Return interpolation weights for one query point.

  nodes = nodes(:)';
  n = numel(nodes);
  weights = ones(1, n);
  for j = 1:n
    for k = 1:n
      if k ~= j
        weights(j) = weights(j) * (x - nodes(k)) / (nodes(j) - nodes(k));
      end
    end
  end
end

function mask = makeVertexMask(cpx, cpy, sBand, h, vertexTol)
%MAKEVERTEXMASK Select band points whose closest point is the shared vertex.

  if vertexTol > 0
    mask = abs(sBand) <= vertexTol * eps(max(1, h));
  else
    mask = false(size(sBand));
  end

  cpScale = max([1; abs(cpx(:)); abs(cpy(:))]);
  mask = mask | (hypot(cpx, cpy) <= 100 * eps(cpScale));
  if ~any(mask)
    [~, idx] = min(abs(sBand));
    mask = false(size(sBand));
    mask(idx) = true;
  end
end

function tf = extensionUsesOnlyOwnBranch(branch)
%EXTENSIONUSESONLYOWNBRANCH Inspect branch-local extension sparsity columns.

  [~, cols] = find(branch.Eband);
  tf = all(cols >= 1) && all(cols <= numel(branch.sNodes));
  if any(branch.vertexMask)
    vertexRows = branch.Eband(branch.vertexMask, :);
    nonVertexColumns = vertexRows(:, 2:end);
    tf = tf && (nnz(nonVertexColumns) == 0);
  end
end

function [x, y] = branchCoordinates(side, s, angleDeg)
%BRANCHCOORDINATES Map branch coordinate s to the embedded V arm.

  halfAngle = 0.5 * angleDeg * pi / 180;
  if side < 0
    x = s(:) * cos(halfAngle);
    y = -s(:) * sin(halfAngle);
  else
    x = s(:) * cos(halfAngle);
    y = s(:) * sin(halfAngle);
  end
end

function u = initialCondition(s)
%INITIALCONDITION Smooth Neumann-compatible data with endpoint values 1, 10.

  u = 11 / 2 - 9 / 2 * cos(pi * (s(:) + 1) / 2);
end

function u = exactSolution(s, t, kappa)
%EXACTSOLUTION Exact heat solution for the compatible cosine profile.

  lambda = (pi / 2)^2;
  u = 11 / 2 - 9 / 2 * exp(-kappa * lambda * t) .* ...
      cos(pi * (s(:) + 1) / 2);
end

function mass = branchMass(uLeft, uRight, h)
%BRANCHMASS Composite trapezoid rule without double-counting the vertex.

  uv = 0.5 * (uLeft(1) + uRight(1));
  massLeft = h * (0.5 * uv + sum(uLeft(2:end-1)) + 0.5 * uLeft(end));
  massRight = h * (0.5 * uv + sum(uRight(2:end-1)) + 0.5 * uRight(end));
  mass = massLeft + massRight;
end

function jump = vertexFluxJump(uLeft, uRight, h)
%VERTEXFLUXJUMP One-sided discrete flux mismatch at the shared vertex.

  uv = 0.5 * (uLeft(1) + uRight(1));
  jump = (uLeft(2) - uv) / h - (uv - uRight(2)) / h;
end

function x = gridVector(xmin, xmax, h)
%GRIDVECTOR Build a grid vector aligned to integer multiples of h.

  first = floor(xmin / h) * h;
  last = ceil(xmax / h) * h;
  x = (first:h:last)';
end

function small = stripLargeMatrices(branch)
%STRIPLARGEMATRICES Keep geometry diagnostics without duplicating matrices.

  small = rmfield(branch, {'Eband', 'Lcart', 'R', 'LCPM'});
  small.numBandPoints = numel(branch.band);
  small.numVertexFiberPoints = nnz(branch.vertexMask);
end

function diagnostic = emptyDiagnostic()
%EMPTYDIAGNOSTIC Template used for struct preallocation.

  diagnostic = struct('angleDeg', [], 'h', [], 'requestedH', [], ...
      'dt', [], 'numSteps', [], 'kappa', [], 'cfl', [], ...
      'interpDegree', [], 'lapOrder', [], 'sLeft', [], 'sRight', [], ...
      'uLeft', [], 'uRight', [], 'exactLeft', [], 'exactRight', [], ...
      'time', [], 'mass', [], 'massInitial', [], 'massFinal', [], ...
      'massDrift', [], 'vertexJump', [], 'fluxJump', [], ...
      'maxVertexJump', [], 'maxAbsFluxJump', [], 'finalInfError', [], ...
      'finalL2Error', [], 'noShortcutOK', [], 'left', [], 'right', []);
end

function results = makeResultsTable(angleCol, hCol, cflCol, dtCol, ...
    stepsCol, finalInfErrCol, finalL2ErrCol, finalInfRateCol, ...
    finalL2RateCol, massDriftCol, maxVertexJumpCol, maxFluxJumpCol, ...
    noShortcutCol)
%MAKERESULTSTABLE Return a table when available, otherwise a struct.

  if exist('table', 'builtin') || exist('table', 'file')
    results = table(angleCol, hCol, cflCol, dtCol, stepsCol, ...
        finalInfErrCol, finalL2ErrCol, finalInfRateCol, ...
        finalL2RateCol, massDriftCol, maxVertexJumpCol, ...
        maxFluxJumpCol, noShortcutCol, ...
        'VariableNames', {'AngleDegrees', 'h', 'CFL', 'dt', ...
                          'TimeSteps', 'FinalInfError', ...
                          'FinalL2Error', 'FinalInfRate', ...
                          'FinalL2Rate', 'MassDrift', ...
                          'MaxVertexJump', 'MaxAbsFluxJump', ...
                          'NoShortcutOK'});
  else
    results = struct('AngleDegrees', num2cell(angleCol), ...
        'h', num2cell(hCol), 'CFL', num2cell(cflCol), ...
        'dt', num2cell(dtCol), 'TimeSteps', num2cell(stepsCol), ...
        'FinalInfError', num2cell(finalInfErrCol), ...
        'FinalL2Error', num2cell(finalL2ErrCol), ...
        'FinalInfRate', num2cell(finalInfRateCol), ...
        'FinalL2Rate', num2cell(finalL2RateCol), ...
        'MassDrift', num2cell(massDriftCol), ...
        'MaxVertexJump', num2cell(maxVertexJumpCol), ...
        'MaxAbsFluxJump', num2cell(maxFluxJumpCol), ...
        'NoShortcutOK', num2cell(noShortcutCol));
  end
end

function plotAngleSolutions(figNum, diagnostics, anglesDeg)
%PLOTANGLESOLUTIONS Plot final u(s,t) for each angle on the finest grid.

  fig = figure(figNum);
  clf(fig);
  set(fig, 'Color', 'w');
  layout = tiledlayout(fig, 'flow', 'TileSpacing', 'compact', ...
                       'Padding', 'compact');

  for ia = 1:numel(anglesDeg)
    diagnostic = finestDiagnosticForAngle(diagnostics, anglesDeg(ia));
    [sAll, uAll, exactAll] = diagnosticCurves(diagnostic);

    ax = nexttile(layout);
    set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
    plot(ax, sAll, exactAll, 'k-', 'LineWidth', 1.6);
    hold(ax, 'on');
    plot(ax, sAll, uAll, 'bo', 'MarkerSize', 3.2, ...
         'MarkerFaceColor', 'b');
    plot(ax, 0, diagnostic.uLeft(1), 'rs', 'MarkerSize', 6, ...
         'MarkerFaceColor', 'r');
    hold(ax, 'off');
    grid(ax, 'on');
    xlabel(ax, 's');
    ylabel(ax, 'u');
    title(ax, sprintf('\\alpha = %g^\\circ, h = %.4g', ...
                      diagnostic.angleDeg, diagnostic.h));
    if ia == 1
      legend(ax, 'exact', 'numerical', 'shared vertex', ...
             'Location', 'southeast');
    end
  end

  title(layout, sprintf('Branch-aware heat solution at t = %.4g', ...
                       diagnostics(1).time(end)));
end

function plotConvergenceByAngle(figNum, diagnostics, anglesDeg)
%PLOTCONVERGENCEBYANGLE Plot L_inf and L_2 final errors versus h.

  fig = figure(figNum);
  clf(fig);
  set(fig, 'Color', 'w');
  layout = tiledlayout(fig, 'flow', 'TileSpacing', 'compact', ...
                       'Padding', 'compact');

  for ia = 1:numel(anglesDeg)
    angleDiagnostics = diagnosticsForAngle(diagnostics, anglesDeg(ia));
    h = reshape([angleDiagnostics.h], [], 1);
    errInf = reshape([angleDiagnostics.finalInfError], [], 1);
    errL2 = reshape([angleDiagnostics.finalL2Error], [], 1);
    [h, order] = sort(h, 'descend');
    errInf = errInf(order);
    errL2 = errL2(order);

    ax = nexttile(layout);
    set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
    loglog(ax, h, errInf, 'bo-', 'LineWidth', 1.5, ...
           'MarkerFaceColor', 'b');
    hold(ax, 'on');
    loglog(ax, h, errL2, 'rs--', 'LineWidth', 1.5, ...
           'MarkerFaceColor', 'r');
    addReferenceSlope(ax, h, errInf, 1, 'k:');
    addReferenceSlope(ax, h, errInf, 2, 'k:');
    hold(ax, 'off');
    set(ax, 'XDir', 'reverse');
    grid(ax, 'on');
    xlabel(ax, 'h');
    ylabel(ax, 'error');
    title(ax, sprintf('\\alpha = %g^\\circ', anglesDeg(ia)));
    if ia == 1
      legend(ax, 'L_\infty', 'L_2', 'O(h)', 'O(h^2)', ...
             'Location', 'southeast');
    end
  end

  title(layout, 'Final-time error convergence by opening angle');
end

function addReferenceSlope(ax, h, err, slope, style)
%ADDREFERENCESLOPE Add a reference h^slope guide line.

  if numel(h) < 2 || ~all(isfinite(err)) || err(1) <= 0
    return;
  end

  ref = err(1) * (h / h(1)).^slope;
  loglog(ax, h, ref, style, 'LineWidth', 1.2);
end

function diagnostic = finestDiagnosticForAngle(diagnostics, angleDeg)
%FINESTDIAGNOSTICFORANGLE Return the smallest-h diagnostic for one angle.

  angleDiagnostics = diagnosticsForAngle(diagnostics, angleDeg);
  [~, idx] = min([angleDiagnostics.h]);
  diagnostic = angleDiagnostics(idx);
end

function angleDiagnostics = diagnosticsForAngle(diagnostics, angleDeg)
%DIAGNOSTICSFORANGLE Select diagnostics for one angle.

  angleValues = [diagnostics.angleDeg];
  mask = abs(angleValues - angleDeg) <= 100 * eps(max(1, abs(angleDeg)));
  angleDiagnostics = diagnostics(mask);
end

function [sAll, uAll, exactAll] = diagnosticCurves(diagnostic)
%DIAGNOSTICCURVES Combine left and right arrays without duplicating vertex.

  sAll = [flipud(diagnostic.sLeft); diagnostic.sRight(2:end)];
  uAll = [flipud(diagnostic.uLeft); diagnostic.uRight(2:end)];
  exactAll = [flipud(diagnostic.exactLeft); diagnostic.exactRight(2:end)];
end

function value = optionValue(opts, fieldName, defaultValue)
%OPTIONVALUE Read an option from a struct, or use a default.

  if isstruct(opts) && isfield(opts, fieldName)
    value = opts.(fieldName);
  else
    value = defaultValue;
  end
end

function addpathIfNeeded(pathToAdd)
%ADDPATHIFNEEDED Add a path only when it is not already active.

  if exist(pathToAdd, 'dir') && isempty(strfind(path, pathToAdd))
    addpath(pathToAdd);
  end
end
