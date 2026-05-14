function [results, diagnostics] = example_v_shaped_manifold_poisson_branch_cpm(opts)
%EXAMPLE_V_SHAPED_MANIFOLD_POISSON_BRANCH_CPM
% Branch-aware CPM Poisson equation on a two-branch manifold.
%
% This solves the well-posed manufactured problem
%
%   -kappa*u_ss = f(s),       -1 < s < 1,
%   u(-1) = 0,  u(1) = 0,
%   u_L(0) = u_R(0),  u_L'(0) = u_R'(0),
%
% on a two-branch embedding.  By default the left branch is a straight unit
% segment and the right branch is the upper semicircle from (0,0) to (1,0).
% The default exact solution is the mixed mode
%
%   u(s) = cos(pi*s/2) + 0.35*sin(pi*(s+1)) + 0.2*cos(3*pi*s/2),
%
% which has zero values at s = +/-1 and a nonzero derivative through the
% shared vertex.  Each arm is assembled with branch-local CPM ingredients;
% the shared vertex row is assembled as the centered graph Poisson equation.
%
% Example:
%   [results, diagnostics] = ...
%       example_v_shaped_manifold_poisson_branch_cpm();

  if nargin < 1
    opts = struct();
  end

  thisfile = mfilename('fullpath');
  examplesdir = fileparts(thisfile);
  repoRoot = fileparts(examplesdir);
  addpathIfNeeded(fullfile(repoRoot, 'cp_matrices'));
  addpathIfNeeded(fullfile(repoRoot, 'surfaces'));

  anglesDeg = optionValue(opts, 'anglesDeg', [90 120 150 170]);
  hvals = optionValue(opts, 'hvals', 1 ./ [400 800 1200 1600 3200 6400]);
  kappa = optionValue(opts, 'kappa', 1.0);
  exactMode = optionValue(opts, 'exactMode', 'hard');
  rightBranchShape = optionValue(opts, 'rightBranchShape', 'line');
  rightArcRadius = optionValue(opts, 'rightArcRadius', 0.5);
  interpDegree = optionValue(opts, 'interpDegree', 3);
  lapOrder = optionValue(opts, 'lapOrder', 2);
  padding = optionValue(opts, 'padding', []);
  vertexTol = optionValue(opts, 'vertexTol', 0);
  makePlots = optionValue(opts, 'makePlots', true);
  showDiagnostics = optionValue(opts, 'showDiagnostics', true);

  if lapOrder ~= 2
    error('This Poisson example currently implements the second-order graph closure only.');
  end

  nCases = numel(anglesDeg) * numel(hvals);
  diagnostics = repmat(emptyDiagnostic(), nCases, 1);

  angleCol = zeros(nCases, 1);
  hCol = zeros(nCases, 1);
  finalInfErrCol = zeros(nCases, 1);
  finalL2ErrCol = zeros(nCases, 1);
  finalInfRateCol = NaN(nCases, 1);
  finalL2RateCol = NaN(nCases, 1);
  vertexJumpCol = zeros(nCases, 1);
  fluxJumpCol = zeros(nCases, 1);
  residualCol = zeros(nCases, 1);

  row = 0;
  for ia = 1:numel(anglesDeg)
    for ih = 1:numel(hvals)
      row = row + 1;
      diagnostics(row) = solveOneLevel(anglesDeg(ia), hvals(ih), kappa, ...
          exactMode, rightBranchShape, rightArcRadius, interpDegree, ...
          lapOrder, padding, vertexTol);

      angleCol(row) = diagnostics(row).angleDeg;
      hCol(row) = diagnostics(row).h;
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
      vertexJumpCol(row) = diagnostics(row).vertexJump;
      fluxJumpCol(row) = diagnostics(row).fluxJump;
      residualCol(row) = diagnostics(row).linearResidualInf;
    end
  end

  results = makeResultsTable(angleCol, hCol, finalInfErrCol, ...
      finalL2ErrCol, finalInfRateCol, finalL2RateCol, vertexJumpCol, ...
      fluxJumpCol, residualCol);

  if showDiagnostics
    disp(results);
  end

  if makePlots && ~isempty(diagnostics)
    plotAngleSolutions(11, diagnostics, anglesDeg);
    plotConvergenceByAngle(12, diagnostics, anglesDeg);
  end
end

function diagnostic = solveOneLevel(angleDeg, requestedH, kappa, exactMode, ...
    rightBranchShape, rightArcRadius, interpDegree, lapOrder, padding, ...
    vertexTol)
%SOLVEONELEVEL Assemble and solve one Poisson refinement level.

  leftGeom = branchGeometry(-1, angleDeg, 'line', rightArcRadius);
  rightGeom = branchGeometry(1, angleDeg, rightBranchShape, rightArcRadius);

  nLeft = max(2, round(leftGeom.length / requestedH));
  nRight = max(2, round(rightGeom.length / requestedH));
  hLeft = leftGeom.length / nLeft;
  hRight = rightGeom.length / nRight;
  h = max(hLeft, hRight);
  sLeft = (0:-hLeft:-leftGeom.length)';
  sRight = (0:hRight:rightGeom.length)';

  dim = 2;
  bw = rm_bandwidth(dim, interpDegree, lapOrder / 2);
  if isempty(padding)
    localPadding = max(0.08, (bw + interpDegree + 2) * h);
  else
    localPadding = padding;
  end

  left = buildBranch(leftGeom, hLeft, sLeft, interpDegree, lapOrder, ...
                     bw, localPadding, vertexTol);
  right = buildBranch(rightGeom, hRight, sRight, interpDegree, lapOrder, ...
                      bw, localPadding, vertexTol);

  [A, rhs, leftMap, rightMap] = assemblePoissonSystem(left, right, h, ...
                                                      kappa, exactMode);
  q = A \ rhs;

  uLeft = leftMap * q;
  uRight = rightMap * q;
  exactLeft = exactSolution(sLeft, exactMode, leftGeom.length, ...
                            rightGeom.length);
  exactRight = exactSolution(sRight, exactMode, leftGeom.length, ...
                             rightGeom.length);
  errorLeft = uLeft - exactLeft;
  errorRight = uRight - exactRight;
  uniqueError = [errorLeft; errorRight(2:end)];

  linearResidual = A*q - rhs;

  diagnostic = emptyDiagnostic();
  diagnostic.angleDeg = angleDeg;
  diagnostic.h = h;
  diagnostic.requestedH = requestedH;
  diagnostic.kappa = kappa;
  diagnostic.exactMode = exactMode;
  diagnostic.interpDegree = interpDegree;
  diagnostic.lapOrder = lapOrder;
  diagnostic.sLeft = sLeft;
  diagnostic.sRight = sRight;
  diagnostic.uLeft = uLeft;
  diagnostic.uRight = uRight;
  diagnostic.exactLeft = exactLeft;
  diagnostic.exactRight = exactRight;
  diagnostic.finalInfError = max(abs(uniqueError));
  diagnostic.finalL2Error = branchL2Error(errorLeft, errorRight, hLeft, hRight);
  diagnostic.vertexJump = abs(uLeft(1) - uRight(1));
  diagnostic.fluxJump = vertexFluxJump(uLeft, uRight, hLeft, hRight);
  diagnostic.linearResidualInf = norm(linearResidual, inf);
  diagnostic.left = stripLargeMatrices(left);
  diagnostic.right = stripLargeMatrices(right);
end

function [A, rhs, leftMap, rightMap] = assemblePoissonSystem(left, right, ...
    h, kappa, exactMode)
%ASSEMBLEPOISSONSYSTEM Build the graph-coupled Dirichlet Poisson system.

  nLeft = numel(left.sNodes) - 1;
  nRight = numel(right.sNodes) - 1;
  hLeft = left.hStep;
  hRight = right.hStep;

  nUnknowns = nLeft + nRight + 1;
  leftMap = sparse([], [], [], nLeft + 1, nUnknowns, nLeft + 1);
  rightMap = sparse([], [], [], nRight + 1, nUnknowns, nRight + 1);

  leftMap(1, 1) = 1;
  leftMap(2:end, 2:(nLeft + 1)) = speye(nLeft);

  rightMap(1, 1) = 1;
  rightMap(2:end, (nLeft + 2):(nLeft + nRight + 1)) = speye(nRight);

  A = sparse([], [], [], nUnknowns, nUnknowns, 5*max(nLeft, nRight));
  rhs = zeros(nUnknowns, 1);
  row = 1;

  A(row, 1) = 2*kappa / (hLeft*hRight);
  A(row, 2) = -2*kappa / (hLeft*(hLeft + hRight));
  A(row, nLeft + 2) = -2*kappa / (hRight*(hLeft + hRight));
  rhs(row) = forcing(0, kappa, exactMode, left.length, right.length);
  row = row + 1;

  leftInterior = 2:nLeft;
  rightInterior = 2:nRight;
  nLeftInterior = numel(leftInterior);
  nRightInterior = numel(rightInterior);

  A(row:(row + nLeftInterior - 1), :) = ...
      -kappa * left.LCPM(leftInterior, :) * leftMap;
  rhs(row:(row + nLeftInterior - 1)) = forcing(left.sNodes(leftInterior), ...
      kappa, exactMode, left.length, right.length);
  row = row + nLeftInterior;

  A(row:(row + nRightInterior - 1), :) = ...
      -kappa * right.LCPM(rightInterior, :) * rightMap;
  rhs(row:(row + nRightInterior - 1)) = forcing(right.sNodes(rightInterior), ...
      kappa, exactMode, left.length, right.length);
  row = row + nRightInterior;

  A(row, nLeft + 1) = 1;
  rhs(row) = exactSolution(-left.length, exactMode, left.length, right.length);
  row = row + 1;

  A(row, nLeft + nRight + 1) = 1;
  rhs(row) = exactSolution(right.length, exactMode, left.length, right.length);
end

function branch = buildBranch(geom, h, sNodes, interpDegree, ...
    lapOrder, bw, padding, vertexTol)
%BUILDBRANCH Build one branch-local CPM band and manifold operator.

  bb = branchBoundingBox(geom, padding);
  x1d = gridVector(bb(1), bb(3), h);
  y1d = gridVector(bb(2), bb(4), h);
  [xx, yy] = meshgrid(x1d, y1d);

  [cpx, cpy, dist, bdy, sGrid] = branchClosestPoint(xx, yy, geom);
  band = find(abs(dist) <= bw * h);

  cpxBand = cpx(band);
  cpyBand = cpy(band);
  xBand = xx(band);
  yBand = yy(band);
  bdyBand = bdy(band);
  sBand = sGrid(band);

  vertexMask = makeVertexMask(cpxBand, cpyBand, sBand, h, vertexTol);
  Eband = interp1BranchMatrix(sNodes, sBand, interpDegree);
  if any(vertexMask)
    Eband(vertexMask, :) = 0;
    Eband(vertexMask, 1) = 1;
  end

  Lcart = laplacian_2d_matrix(x1d, y1d, lapOrder, band, band);
  [xManifold, yManifold] = branchCoordinates(geom, sNodes);
  R = interp2_matrix(x1d, y1d, xManifold, yManifold, interpDegree, band);

  branch.side = geom.side;
  branch.angleDeg = geom.angleDeg;
  branch.geometryType = geom.type;
  branch.length = geom.length;
  branch.h = h;
  branch.hStep = h;
  branch.sNodes = sNodes;
  branch.p0 = geom.p0;
  branch.p1 = geom.p1;
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
  branch.LCPM = replaceEndpointAffectedRows(branch.LCPM, h, ...
                                            interpDegree, lapOrder);
end

function L = replaceEndpointAffectedRows(L, h, interpDegree, lapOrder)
%REPLACEENDPOINTAFFECTEDROWS Fix rows where CP endpoint clamping is wrong.

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

function geom = branchGeometry(side, angleDeg, rightBranchShape, rightArcRadius)
%BRANCHGEOMETRY Return geometry data for one branch.

  geom.side = side;
  geom.angleDeg = angleDeg;
  halfAngle = 0.5 * angleDeg * pi / 180;

  if side < 0
    geom.type = 'line';
    geom.p0 = [-cos(halfAngle), sin(halfAngle)];
    geom.p1 = [0, 0];
    geom.length = norm(geom.p1 - geom.p0);
    return;
  end

  shape = lower(char(rightBranchShape));
  if strcmp(shape, 'semicircle') || strcmp(shape, 'arc') || ...
     strcmp(shape, 'halfcircle')
    geom.type = 'semicircle';
    geom.radius = rightArcRadius;
    geom.center = [rightArcRadius, 0];
    geom.angle1 = 0;
    geom.angle2 = pi;
    geom.p0 = [0, 0];
    geom.p1 = [2*rightArcRadius, 0];
    geom.length = pi * rightArcRadius;
  elseif strcmp(shape, 'line') || strcmp(shape, 'straight')
    geom.type = 'line';
    geom.p0 = [0, 0];
    geom.p1 = [cos(halfAngle), sin(halfAngle)];
    geom.length = norm(geom.p1 - geom.p0);
  else
    error('Unknown rightBranchShape: %s', char(rightBranchShape));
  end
end

function bb = branchBoundingBox(geom, padding)
%BRANCHBOUNDINGBOX Return [xmin ymin xmax ymax] for one branch.

  if strcmp(geom.type, 'line')
    bb = [min(geom.p0(1), geom.p1(1)) - padding, ...
          min(geom.p0(2), geom.p1(2)) - padding, ...
          max(geom.p0(1), geom.p1(1)) + padding, ...
          max(geom.p0(2), geom.p1(2)) + padding];
  elseif strcmp(geom.type, 'semicircle')
    R = geom.radius;
    cen = geom.center;
    bb = [cen(1) - R - padding, cen(2) - padding, ...
          cen(1) + R + padding, cen(2) + R + padding];
  else
    error('Unknown geometry type: %s', geom.type);
  end
end

function [cpx, cpy, dist, bdy, sGrid] = branchClosestPoint(x, y, geom)
%BRANCHCLOSESTPOINT Closest point and arclength coordinate for one branch.

  if strcmp(geom.type, 'line')
    [cpx, cpy, dist, bdy, t] = cpLineSegment2d(x, y, geom.p0, geom.p1);
    if geom.side < 0
      sGrid = -geom.length + geom.length * t;
    else
      sGrid = geom.length * t;
    end
  elseif strcmp(geom.type, 'semicircle')
    [cpx, cpy, dist, bdy] = cpArc(x, y, geom.radius, geom.center, ...
                                  geom.angle1, geom.angle2);
    theta = atan2(cpy - geom.center(2), cpx - geom.center(1));
    theta(theta < 0) = theta(theta < 0) + 2*pi;
    theta = min(max(theta, 0), pi);
    sGrid = geom.radius * (pi - theta);
  else
    error('Unknown geometry type: %s', geom.type);
  end
end

function [x, y] = branchCoordinates(geom, s)
%BRANCHCOORDINATES Map branch arclength coordinate to the embedding.

  s = s(:);
  if strcmp(geom.type, 'line')
    if geom.side < 0
      tau = (s + geom.length) / geom.length;
    else
      tau = s / geom.length;
    end
    x = geom.p0(1) + tau * (geom.p1(1) - geom.p0(1));
    y = geom.p0(2) + tau * (geom.p1(2) - geom.p0(2));
  elseif strcmp(geom.type, 'semicircle')
    theta = pi - s / geom.radius;
    x = geom.center(1) + geom.radius * cos(theta);
    y = geom.center(2) + geom.radius * sin(theta);
  else
    error('Unknown geometry type: %s', geom.type);
  end
end

function u = exactSolution(s, exactMode, leftLength, rightLength)
%EXACTSOLUTION Manufactured Dirichlet solution.

  mode = lower(char(exactMode));
  s = s(:);
  totalLength = leftLength + rightLength;
  sigma = s + leftLength;
  if strcmp(mode, 'hard') || strcmp(mode, 'multimode') || strcmp(mode, 'mixed')
    u = sin(pi * sigma / totalLength) + ...
        0.35 * sin(2*pi * sigma / totalLength) - ...
        0.2 * sin(3*pi * sigma / totalLength);
  elseif strcmp(mode, 'simple') || strcmp(mode, 'cosine') || ...
         strcmp(mode, 'singlecosine')
    u = sin(pi * sigma / totalLength);
  else
    error('Unknown exactMode: %s', char(exactMode));
  end
end

function f = forcing(s, kappa, exactMode, leftLength, rightLength)
%FORCING Right-hand side for -kappa*u_ss = f.

  mode = lower(char(exactMode));
  s = s(:);
  totalLength = leftLength + rightLength;
  sigma = s + leftLength;
  if strcmp(mode, 'hard') || strcmp(mode, 'multimode') || strcmp(mode, 'mixed')
    f = kappa * ((pi / totalLength)^2 * sin(pi * sigma / totalLength) + ...
                 0.35 * (2*pi / totalLength)^2 * ...
                 sin(2*pi * sigma / totalLength) - ...
                 0.2 * (3*pi / totalLength)^2 * ...
                 sin(3*pi * sigma / totalLength));
  elseif strcmp(mode, 'simple') || strcmp(mode, 'cosine') || ...
         strcmp(mode, 'singlecosine')
    f = kappa * (pi / totalLength)^2 * ...
        exactSolution(s, exactMode, leftLength, rightLength);
  else
    error('Unknown exactMode: %s', char(exactMode));
  end
end

function jump = vertexFluxJump(uLeft, uRight, hLeft, hRight)
%VERTEXFLUXJUMP One-sided discrete flux mismatch at the shared vertex.

  uv = 0.5 * (uLeft(1) + uRight(1));
  jump = (uv - uLeft(2)) / hLeft - (uRight(2) - uv) / hRight;
end

function err = branchL2Error(errorLeft, errorRight, hLeft, hRight)
%BRANCHL2ERROR Composite trapezoid L2 error on the two graph branches.

  leftPart = hLeft * (0.5 * errorLeft(1)^2 + ...
                      sum(errorLeft(2:end-1).^2) + ...
                      0.5 * errorLeft(end)^2);
  rightPart = hRight * (0.5 * errorRight(1)^2 + ...
                        sum(errorRight(2:end-1).^2) + ...
                        0.5 * errorRight(end)^2);
  err = sqrt(leftPart + rightPart);
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
      'kappa', [], 'exactMode', [], 'interpDegree', [], 'lapOrder', [], ...
      'sLeft', [], 'sRight', [], 'uLeft', [], 'uRight', [], ...
      'exactLeft', [], 'exactRight', [], 'finalInfError', [], ...
      'finalL2Error', [], 'vertexJump', [], 'fluxJump', [], ...
      'linearResidualInf', [], 'left', [], 'right', []);
end

function results = makeResultsTable(angleCol, hCol, finalInfErrCol, ...
    finalL2ErrCol, finalInfRateCol, finalL2RateCol, vertexJumpCol, ...
    fluxJumpCol, residualCol)
%MAKERESULTSTABLE Return a table when available, otherwise a struct.

  if exist('table', 'builtin') || exist('table', 'file')
    results = table(angleCol, hCol, finalInfErrCol, finalL2ErrCol, ...
        finalInfRateCol, finalL2RateCol, vertexJumpCol, fluxJumpCol, ...
        residualCol, ...
        'VariableNames', {'AngleDegrees', 'h', 'FinalInfError', ...
                          'FinalL2Error', 'FinalInfRate', ...
                          'FinalL2Rate', 'VertexJump', 'FluxJump', ...
                          'LinearResidualInf'});
  else
    results = struct('AngleDegrees', num2cell(angleCol), ...
        'h', num2cell(hCol), ...
        'FinalInfError', num2cell(finalInfErrCol), ...
        'FinalL2Error', num2cell(finalL2ErrCol), ...
        'FinalInfRate', num2cell(finalInfRateCol), ...
        'FinalL2Rate', num2cell(finalL2RateCol), ...
        'VertexJump', num2cell(vertexJumpCol), ...
        'FluxJump', num2cell(fluxJumpCol), ...
        'LinearResidualInf', num2cell(residualCol));
  end
end

function plotAngleSolutions(figNum, diagnostics, anglesDeg)
%PLOTANGLESOLUTIONS Plot final u(s) for each angle on the finest grid.

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
    plot(ax, sAll, exactAll, 'w-', 'LineWidth', 2);
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

  title(layout, sprintf('Branch-aware CPM Poisson solution, exactMode = %s', ...
                        char(diagnostics(1).exactMode)));
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
    addReferenceSlope(ax, h, errInf, 2, 'w:');
    hold(ax, 'off');
    set(ax, 'XDir', 'reverse');
    grid(ax, 'on');
    xlabel(ax, 'h');
    ylabel(ax, 'error');
    title(ax, sprintf('\\alpha = %g^\\circ', anglesDeg(ia)));
    if ia == 1
      legend(ax, 'L_\infty', 'L_2', 'O(h^2)', ...
             'Location', 'southeast');
    end
  end

  title(layout, sprintf('Poisson error convergence, exactMode = %s', ...
                        char(diagnostics(1).exactMode)));
end

function addReferenceSlope(ax, h, err, slope, style)
%ADDREFERENCESLOPE Add a reference h^slope guide line.

  if numel(h) < 2 || ~all(isfinite(err)) || err(1) <= 0
    return;
  end

  ref = err(1) * (h / h(1)).^slope;
  loglog(ax, h, ref, style, 'LineWidth', 2);
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
