function results = example_v_shaped_manifold_heat_refinement(opts)
%EXAMPLE_V_SHAPED_MANIFOLD_HEAT_REFINEMENT
% Vertex-coupled branch-aware CPM heat-equation refinement test on V-shaped curves.
%
% This diagnostic solves the two V arms as labeled line-segment CPM
% problems.  Each branch keeps the usual cubic interpolation support, while
% communication between labels is allowed only through grid-band values
% whose closest point is the shared vertex.

  if (nargin < 1)
    opts = struct();
  end

  thisfile = mfilename('fullpath');
  examplesdir = fileparts(thisfile);
  repoRoot = fileparts(examplesdir);
  addpathIfNeeded(fullfile(repoRoot, 'cp_matrices'));
  addpathIfNeeded(fullfile(repoRoot, 'surfaces'));

  anglesDeg = optionValue(opts, 'anglesDeg', [90 120 150 170]);
  hvals = optionValue(opts, 'hvals', 1 ./ [400 800 1200]);
  finalTime = optionValue(opts, 'finalTime', 0.01);
  cfl = optionValue(opts, 'cfl', 0.2);
  interpDegree = optionValue(opts, 'interpDegree', 3);
  lapOrder = optionValue(opts, 'lapOrder', 2);
  couplingMode = optionValue(opts, 'couplingMode', 'reflectedGhost');
  exactMode = optionValue(opts, 'exactMode', 'rightThree');
  vertexTol = optionValue(opts, 'vertexTol', 0);
  makePlots = optionValue(opts, 'makePlots', true);
  xyFigNum = optionValue(opts, 'xyFigNum', 1);
  convFigNum = optionValue(opts, 'convFigNum', 2);
  ownershipFigNum = optionValue(opts, 'ownershipFigNum', 3);
  showBranchOwnership = optionValue(opts, 'showBranchOwnership', true);
  ownershipAngleDeg = optionValue(opts, 'ownershipAngleDeg', 150);
  ownershipH = optionValue(opts, 'ownershipH', 1/120);

  dim = 2;
  padding = 0.18;
  sSample = linspace(-1, 1, 8001)';

  absGlobalInf = zeros(length(anglesDeg), length(hvals));
  absGlobalL2 = zeros(size(absGlobalInf));
  errGlobalInf = zeros(length(anglesDeg), length(hvals));
  errGlobalL2 = zeros(size(errGlobalInf));
  rateGlobalInf = NaN(size(errGlobalInf));
  timeSteps = zeros(size(errGlobalInf));

  xyPlots = struct('alphaDeg', {}, 'h', {}, 'xBand', {}, 'yBand', {}, ...
                   'errBand', {}, 'xCurve', {}, 'yCurve', {});

  bw = 1.0001*sqrt((dim-1)*((interpDegree+1)/2)^2 + ...
                   ((lapOrder/2+(interpDegree+1)/2)^2));

  for ia = 1:length(anglesDeg)
    alphaDeg = anglesDeg(ia);
    m = tan(alphaDeg*pi/360);
    exactSample = exactSolution(sSample, finalTime, m, exactMode);

    for ih = 1:length(hvals)
      h = hvals(ih);
      [uSample, branchPlot, nsteps] = solveBranchAwareLevel( ...
          m, h, sSample, finalTime, cfl, interpDegree, lapOrder, ...
          bw, padding, couplingMode, exactMode, vertexTol);

      timeSteps(ia, ih) = nsteps;
      sampleError = abs(uSample - exactSample);
      exactGlobalInf = max(abs(exactSample));
      exactGlobalL2 = sqrt(mean(exactSample.^2));

      absGlobalInf(ia, ih) = max(sampleError);
      absGlobalL2(ia, ih) = sqrt(mean(sampleError.^2));
      errGlobalInf(ia, ih) = absGlobalInf(ia, ih) / exactGlobalInf;
      errGlobalL2(ia, ih) = absGlobalL2(ia, ih) / exactGlobalL2;

      if ih > 1
        rateGlobalInf(ia, ih) = log(errGlobalInf(ia, ih-1) / ...
                                    errGlobalInf(ia, ih)) / ...
                                log(hvals(ih-1) / hvals(ih));
      end

      if ih == length(hvals)
        xyPlots(ia).alphaDeg = alphaDeg;
        xyPlots(ia).h = h;
        xyPlots(ia).xBand = branchPlot.xBand;
        xyPlots(ia).yBand = branchPlot.yBand;
        xyPlots(ia).errBand = branchPlot.errBand;
        xyPlots(ia).xCurve = sSample;
        xyPlots(ia).yCurve = m*abs(sSample);
      end
    end
  end

  results = makeResultsTable(anglesDeg, hvals, timeSteps, cfl, ...
                             interpDegree, errGlobalInf, errGlobalL2, ...
                             rateGlobalInf);
  disp(results);

  if makePlots
    plotXYPlanes(xyFigNum, xyPlots, interpDegree, couplingMode);
    plotConvergence(convFigNum, anglesDeg, hvals, absGlobalInf, ...
                    absGlobalL2, interpDegree, couplingMode, exactMode);
    if showBranchOwnership
      plotBranchOwnership(ownershipFigNum, ownershipAngleDeg, ownershipH, ...
                          interpDegree, lapOrder, padding);
    end
  end
end

function [uSample, branchPlot, nsteps] = solveBranchAwareLevel( ...
    m, h, sSample, finalTime, cfl, interpDegree, lapOrder, bw, padding, ...
    couplingMode, exactMode, vertexTol)
%SOLVEBRANCHAWARELEVEL Solve one refinement level on separately labeled arms.

  if isVertexCoupled(couplingMode)
    [uSample, branchPlot, nsteps] = solveVertexCoupledLevel( ...
        m, h, sSample, finalTime, cfl, interpDegree, lapOrder, bw, ...
        padding, exactMode, vertexTol);
    return;
  end

  if isReflectedGhostCoupled(couplingMode)
    [uSample, branchPlot, nsteps] = solveReflectedGhostLevel( ...
        m, h, sSample, finalTime, cfl, interpDegree, lapOrder, bw, ...
        padding, exactMode, vertexTol);
    return;
  end

  leftMask = sSample <= 0;
  rightMask = sSample >= 0;
  uSample = zeros(size(sSample));

  [uLeft, leftPlot, nsteps] = solveOneBranch( ...
      -1, m, h, sSample(leftMask), finalTime, cfl, interpDegree, ...
      lapOrder, bw, padding, exactMode, vertexTol);
  [uRight, rightPlot, nstepsRight] = solveOneBranch( ...
      1, m, h, sSample(rightMask), finalTime, cfl, interpDegree, ...
      lapOrder, bw, padding, exactMode, vertexTol);

  if nstepsRight ~= nsteps
    error('left and right branch time-step counts differ');
  end

  uSample(leftMask) = uLeft;
  uSample(rightMask) = uRight;

  branchPlot.xBand = [leftPlot.xBand; rightPlot.xBand];
  branchPlot.yBand = [leftPlot.yBand; rightPlot.yBand];
  branchPlot.errBand = [leftPlot.errBand; rightPlot.errBand];
end

function [uSample, branchPlot, nsteps] = solveVertexCoupledLevel( ...
    m, h, sSample, finalTime, cfl, interpDegree, lapOrder, bw, padding, ...
    exactMode, vertexTol)
%SOLVEVERTEXCOUPLEDLEVEL Couple the two labeled arms only at the vertex.

  leftMask = sSample <= 0;
  rightMask = sSample >= 0;
  uSample = zeros(size(sSample));

  left = buildOneBranch(-1, m, h, sSample(leftMask), interpDegree, ...
                        lapOrder, bw, padding, vertexTol);
  right = buildOneBranch(1, m, h, sSample(rightMask), interpDegree, ...
                         lapOrder, bw, padding, vertexTol);

  dt = cfl*h^2;
  nsteps = ceil(finalTime/dt);
  dt = finalTime/nsteps;

  uLeft = exactSolution(left.sBand, 0, m, exactMode);
  uRight = exactSolution(right.sBand, 0, m, exactMode);
  [uLeft, uRight] = coupleVertexValues(uLeft, uRight, ...
                                       left.vertexMask, right.vertexMask);

  for kt = 1:nsteps
    uLeft = uLeft + dt*(left.L*uLeft);
    uRight = uRight + dt*(right.L*uRight);

    uLeft = left.E*uLeft;
    uRight = right.E*uRight;
    [uLeft, uRight] = coupleVertexValues(uLeft, uRight, ...
                                         left.vertexMask, right.vertexMask);
  end

  uLeftPlot = left.Eplot*uLeft;
  uRightPlot = right.Eplot*uRight;
  [uLeftPlot, uRightPlot] = coupleSampleVertexValues( ...
      uLeftPlot, uRightPlot, sSample(leftMask), sSample(rightMask));

  uSample(leftMask) = uLeftPlot;
  uSample(rightMask) = uRightPlot;

  errLeftBand = abs(uLeft - exactSolution(left.sBand, finalTime, ...
                                          m, exactMode));
  errRightBand = abs(uRight - exactSolution(right.sBand, finalTime, ...
                                            m, exactMode));

  branchPlot.xBand = [left.xBand; right.xBand];
  branchPlot.yBand = [left.yBand; right.yBand];
  branchPlot.errBand = [errLeftBand; errRightBand];
end

function [uSample, branchPlot, nsteps] = solveReflectedGhostLevel( ...
    m, h, sSample, finalTime, cfl, interpDegree, lapOrder, bw, padding, ...
    exactMode, vertexTol)
%SOLVEREFLECTEDGHOSTLEVEL Use reflected edge ghost values from the other arm.

  leftMask = sSample <= 0;
  rightMask = sSample >= 0;
  uSample = zeros(size(sSample));

  left = buildOneBranch(-1, m, h, sSample(leftMask), interpDegree, ...
                        lapOrder, bw, padding, vertexTol);
  right = buildOneBranch(1, m, h, sSample(rightMask), interpDegree, ...
                         lapOrder, bw, padding, vertexTol);
  [left, right] = addReflectedGhostMaps(left, right, interpDegree);

  dt = cfl*h^2;
  nsteps = ceil(finalTime/dt);
  dt = finalTime/nsteps;

  uLeft = exactSolution(left.sBand, 0, m, exactMode);
  uRight = exactSolution(right.sBand, 0, m, exactMode);
  [uLeft, uRight] = coupleVertexValues(uLeft, uRight, ...
                                       left.vertexMask, right.vertexMask);
  [uLeft, uRight] = applyReflectedGhostValues(uLeft, uRight, left, right);

  for kt = 1:nsteps
    uLeft = uLeft + dt*(left.L*uLeft);
    uRight = uRight + dt*(right.L*uRight);

    uLeft = left.E*uLeft;
    uRight = right.E*uRight;
    [uLeft, uRight] = coupleVertexValues(uLeft, uRight, ...
                                         left.vertexMask, right.vertexMask);
    [uLeft, uRight] = applyReflectedGhostValues(uLeft, uRight, left, right);
  end

  uLeftPlot = left.Eplot*uLeft;
  uRightPlot = right.Eplot*uRight;
  [uLeftPlot, uRightPlot] = coupleSampleVertexValues( ...
      uLeftPlot, uRightPlot, sSample(leftMask), sSample(rightMask));

  uSample(leftMask) = uLeftPlot;
  uSample(rightMask) = uRightPlot;

  errLeftBand = abs(uLeft - exactSolution(left.sBand, finalTime, ...
                                          m, exactMode));
  errRightBand = abs(uRight - exactSolution(right.sBand, finalTime, ...
                                            m, exactMode));

  branchPlot.xBand = [left.xBand; right.xBand];
  branchPlot.yBand = [left.yBand; right.yBand];
  branchPlot.errBand = [errLeftBand; errRightBand];
end

function [uPlot, branchPlot, nsteps] = solveOneBranch( ...
    side, m, h, sPlot, finalTime, cfl, interpDegree, lapOrder, bw, ...
    padding, exactMode, vertexTol)
%SOLVEONEBRANCH Run CPM heat evolution on one labeled V arm.

  branch = buildOneBranch(side, m, h, sPlot, interpDegree, lapOrder, ...
                          bw, padding, vertexTol);

  dt = cfl*h^2;
  nsteps = ceil(finalTime/dt);
  dt = finalTime/nsteps;

  u = exactSolution(branch.sBand, 0, m, exactMode);
  for kt = 1:nsteps
    u = u + dt*(branch.L*u);
    u = branch.E*u;
  end

  uPlot = branch.Eplot*u;

  branchPlot.xBand = branch.xBand;
  branchPlot.yBand = branch.yBand;
  branchPlot.errBand = abs(u - exactSolution(branch.sBand, finalTime, ...
                                             m, exactMode));
end

function branch = buildOneBranch( ...
    side, m, h, sPlot, interpDegree, lapOrder, bw, padding, vertexTol)
%BUILDONEBRANCH Build branch-local CPM matrices and plotting data.

  if side < 0
    p0 = [-1, m];
    p1 = [0, 0];
    x1d = (-1-padding:h:padding)';
  else
    p0 = [0, 0];
    p1 = [1, m];
    x1d = (-padding:h:1+padding)';
  end

  y1d = (-padding:h:m+padding)';
  [xx, yy] = meshgrid(x1d, y1d);
  [cpx, cpy, dist, bdy, t] = cpLineSegment2d(xx, yy, p0, p1);
  band = find(abs(dist) <= bw*h);

  cpxBand = cpx(band);
  cpyBand = cpy(band);
  bdyBand = bdy(band);
  sBand = branchParameter(side, t(band));
  xBand = xx(band);
  yBand = yy(band);

  xPlot = sPlot;
  yPlot = m*abs(sPlot);
  Eplot = interp2_matrix(x1d, y1d, xPlot, yPlot, interpDegree, band);

  branch.E = interp2_matrix(x1d, y1d, cpxBand, cpyBand, ...
                            interpDegree, band);
  branch.L = laplacian_2d_matrix(x1d, y1d, lapOrder, band, band);
  branch.Eplot = Eplot;
  branch.side = side;
  branch.p0 = p0;
  branch.p1 = p1;
  branch.x1d = x1d;
  branch.y1d = y1d;
  branch.band = band;
  branch.cpxBand = cpxBand;
  branch.cpyBand = cpyBand;
  branch.bdyBand = bdyBand;
  branch.sBand = sBand;
  branch.xBand = xBand;
  branch.yBand = yBand;
  branch.vertexMask = makeVertexMask(cpxBand, cpyBand, sBand, h, vertexTol);
  branch.sharedGhostMask = makeSharedGhostMask(side, bdyBand);
end

function mask = makeSharedGhostMask(side, bdyBand)
%MAKESHAREDGHOSTMASK Select branch-edge ghosts at the shared vertex.

  if side < 0
    mask = bdyBand == 2;
  else
    mask = bdyBand == 1;
  end
end

function [left, right] = addReflectedGhostMaps(left, right, interpDegree)
%ADDREFLECTEDGHOSTMAPS Build direct-or-reflected cross-branch ghost maps.

  left = addReflectedGhostMap(left, right, interpDegree);
  right = addReflectedGhostMap(right, left, interpDegree);
end

function branch = addReflectedGhostMap(branch, other, interpDegree)
%ADDREFLECTEDGHOSTMAP Map ghost points to values on the other branch.

  ghostMask = branch.sharedGhostMask;
  if any(ghostMask)
    ghostX = branch.xBand(ghostMask);
    ghostY = branch.yBand(ghostMask);
    [targetX, targetY, tilde, targetBdy] = ...
        cpLineSegment2d(ghostX, ghostY, other.p0, other.p1); %#ok<ASGLU>

    useReflected = targetBdy ~= 0;
    if any(useReflected)
      ownCpX = branch.cpxBand(ghostMask);
      ownCpY = branch.cpyBand(ghostMask);
      tempX = ownCpX(useReflected) + ...
              (ownCpX(useReflected) - ghostX(useReflected));
      tempY = ownCpY(useReflected) + ...
              (ownCpY(useReflected) - ghostY(useReflected));
      [targetX(useReflected), targetY(useReflected)] = ...
          cpLineSegment2d(tempX, tempY, other.p0, other.p1);
    end

    branch.ghostFromOther = interp2_matrix(other.x1d, other.y1d, ...
                                           targetX, targetY, interpDegree, ...
                                           other.band);
    branch.ghostUsesDirectCp = ~useReflected;
    branch.ghostUsesReflectedCp = useReflected;
  else
    branch.ghostFromOther = sparse(0, length(other.band));
    branch.ghostUsesDirectCp = false(0, 1);
    branch.ghostUsesReflectedCp = false(0, 1);
  end
end

function [uLeft, uRight] = applyReflectedGhostValues( ...
    uLeft, uRight, left, right)
%APPLYREFLECTEDGHOSTVALUES Fill edge ghosts from opposite-branch cp values.

  oldLeft = uLeft;
  oldRight = uRight;

  if any(left.sharedGhostMask)
    uLeft(left.sharedGhostMask) = left.ghostFromOther * oldRight;
  end
  if any(right.sharedGhostMask)
    uRight(right.sharedGhostMask) = right.ghostFromOther * oldLeft;
  end
end

function mask = makeVertexMask(cpxBand, cpyBand, sBand, h, vertexTol)
%MAKEVERTEXMASK Select rows whose closest point is the shared vertex.

  if vertexTol > 0
    mask = abs(sBand) <= vertexTol*h;
  else
    cpScale = max([1; abs(cpxBand(:)); abs(cpyBand(:))]);
    mask = hypot(cpxBand, cpyBand) <= 100*eps(cpScale);
  end

  if ~any(mask)
    [~, idx] = min(abs(sBand));
    mask = false(size(sBand));
    mask(idx) = true;
  end
end

function [uLeft, uRight] = coupleVertexValues( ...
    uLeft, uRight, leftVertexMask, rightVertexMask)
%COUPLEVERTEXVALUES Share only the two vertex closest-point values.

  hasLeft = any(leftVertexMask);
  hasRight = any(rightVertexMask);

  if hasLeft && hasRight
    sharedValue = 0.5*(mean(uLeft(leftVertexMask)) + ...
                       mean(uRight(rightVertexMask)));
    uLeft(leftVertexMask) = sharedValue;
    uRight(rightVertexMask) = sharedValue;
  end
end

function [uLeftPlot, uRightPlot] = coupleSampleVertexValues( ...
    uLeftPlot, uRightPlot, sLeft, sRight)
%COUPLESAMPLEVERTEXVALUES Use one reported value at s = 0.

  leftVertex = abs(sLeft) <= 100*eps(1);
  rightVertex = abs(sRight) <= 100*eps(1);

  if any(leftVertex) && any(rightVertex)
    sharedValue = 0.5*(mean(uLeftPlot(leftVertex)) + ...
                       mean(uRightPlot(rightVertex)));
    uLeftPlot(leftVertex) = sharedValue;
    uRightPlot(rightVertex) = sharedValue;
  end
end

function tf = isVertexCoupled(couplingMode)
%ISVERTEXCOUPLED True for the vertex-coupled branch-aware method.

  mode = lower(couplingMode);
  tf = strcmp(mode, 'vertex') || strcmp(mode, 'vertexcoupled') || ...
       strcmp(mode, 'coupled');
end

function tf = isReflectedGhostCoupled(couplingMode)
%ISREFLECTEDGHOSTCOUPLED True for reflected cross-branch ghost values.

  mode = lower(couplingMode);
  tf = strcmp(mode, 'reflectedghost') || strcmp(mode, 'edgeghost') || ...
       strcmp(mode, 'crossghost') || strcmp(mode, 'ghost');
end

function s = branchParameter(side, t)
%BRANCHPARAMETER Convert cpLineSegment2d parameter to the global V parameter.

  if side < 0
    s = -1 + t;
  else
    s = t;
  end
end

function u = exactSolution(s, t, m, exactMode)
%EXACTSOLUTION Manufactured heat-equation solution on Gamma_alpha.

  mode = lower(exactMode);
  if strcmp(mode, 'rightthree') || strcmp(mode, 'endpointthree')
    lambda = (pi/2)^2/(1 + m^2);
    u = 2 + exp(-lambda*t).*sin(0.5*pi*s);
  elseif strcmp(mode, 'hotcold') || strcmp(mode, 'connectedfirst') || ...
     strcmp(mode, 'connected')
    lambda = (pi/2)^2/(1 + m^2);
    u = 0.5 - 0.5*exp(-lambda*t).*sin(0.5*pi*s);
  elseif strcmp(mode, 'cospi') || strcmp(mode, 'symmetric')
    lambda = pi^2/(1 + m^2);
    u = exp(-lambda*t).*cos(pi*s);
  else
    error('unknown exactMode: %s', exactMode);
  end
end

function results = makeResultsTable(anglesDeg, hvals, timeSteps, cfl, ...
                                    interpDegree, errGlobalInf, ...
                                    errGlobalL2, rateGlobalInf)
%MAKERESULTSTABLE Return a table when available, otherwise a struct array.

  nrows = numel(anglesDeg)*numel(hvals);
  angle = zeros(nrows, 1);
  h = zeros(nrows, 1);
  cflCol = cfl*ones(nrows, 1);
  degree = interpDegree*ones(nrows, 1);
  steps = zeros(nrows, 1);
  globalInf = zeros(nrows, 1);
  globalL2 = zeros(nrows, 1);
  globalRate = NaN(nrows, 1);

  row = 0;
  for ia = 1:numel(anglesDeg)
    for ih = 1:numel(hvals)
      row = row + 1;
      angle(row) = anglesDeg(ia);
      h(row) = hvals(ih);
      steps(row) = timeSteps(ia, ih);
      globalInf(row) = errGlobalInf(ia, ih);
      globalL2(row) = errGlobalL2(ia, ih);
      globalRate(row) = rateGlobalInf(ia, ih);
    end
  end

  if exist('table', 'builtin') || exist('table', 'file')
    results = table(angle, h, cflCol, degree, steps, globalInf, ...
                    globalL2, globalRate, ...
                    'VariableNames', {'AngleDegrees', 'h', 'CFL', ...
                                      'InterpDegree', 'TimeSteps', ...
                                      'RelGlobalInf', 'RelGlobalL2', ...
                                      'RelGlobalInfRate'});
  else
    results = struct('AngleDegrees', num2cell(angle), ...
                     'h', num2cell(h), ...
                     'CFL', num2cell(cflCol), ...
                     'InterpDegree', num2cell(degree), ...
                     'TimeSteps', num2cell(steps), ...
                     'RelGlobalInf', num2cell(globalInf), ...
                     'RelGlobalL2', num2cell(globalL2), ...
                     'RelGlobalInfRate', num2cell(globalRate));
  end
end

function plotXYPlanes(figNum, xyPlots, interpDegree, couplingMode)
%PLOTXYPLANES Show finest-grid x-y pointwise errors for all angles.

  figure(figNum);
  clf;
  set(gcf, 'color', 'w');
  set(gcf, 'Position', [100, 100, 1250, 720]);

  xlimAll = [inf, -inf];
  climAll = [inf, -inf];
  for ia = 1:length(xyPlots)
    xlimAll(1) = min(xlimAll(1), min(xyPlots(ia).xBand));
    xlimAll(2) = max(xlimAll(2), max(xyPlots(ia).xBand));
    climAll(1) = min(climAll(1), min(xyPlots(ia).errBand));
    climAll(2) = max(climAll(2), max(xyPlots(ia).errBand));
  end

  xPad = 0.02 * diff(xlimAll);
  xlimAll = xlimAll + [-xPad, xPad];
  if climAll(2) <= 0 || ~isfinite(climAll(2))
    climAll = [0, 1];
  else
    climAll = [0, 1.03*climAll(2)];
  end

  positions = panelPositions(length(xyPlots));
  ax = cell(length(xyPlots), 1);

  for ia = 1:length(xyPlots)
    ax{ia} = axes('Position', positions(ia, :)); %#ok<LAXES>
    scatter(xyPlots(ia).xBand, xyPlots(ia).yBand, 7, ...
            xyPlots(ia).errBand, 'filled', 'MarkerEdgeColor', 'none');
    hold on;
    plot(xyPlots(ia).xCurve, xyPlots(ia).yCurve, 'k-', ...
         'LineWidth', 1.25);
    plot(0, 0, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'w');
    hold off;

    ylimLocal = [min(xyPlots(ia).yBand), max(xyPlots(ia).yBand)];
    yPad = 0.04 * diff(ylimLocal);
    if yPad == 0
      yPad = 1;
    end
    xlim(xlimAll);
    ylim(ylimLocal + [-yPad, yPad]);
    caxis(climAll);
    grid on;
    box on;
    set(gca, 'FontSize', 9, 'LineWidth', 0.8, 'Layer', 'top');
    xlabel('x');
    ylabel('y');
    title(sprintf('\\alpha = %d^\\circ, h = %.4g, interpDegree = %d', ...
                  xyPlots(ia).alphaDeg, xyPlots(ia).h, interpDegree));
  end

  colormap(parula);
  axes(ax{end}); %#ok<LAXES>
  cb = colorbar('Position', [0.91, 0.18, 0.018, 0.64]);
  ylabel(cb, '|u_h - u|');
  for ia = 1:length(ax)
    set(ax{ia}, 'Position', positions(ia, :));
  end

  if exist('sgtitle', 'file') || exist('sgtitle', 'builtin')
    sgtitle(sprintf('Finest-grid pointwise error, coupling = %s', ...
                    couplingMode));
  end
end

function plotBranchOwnership(figNum, alphaDeg, h, interpDegree, lapOrder, padding)
%PLOTBRANCHOWNERSHIP Show which branch-local band owns each grid point.

  dim = 2;
  m = tan(alphaDeg*pi/360);
  bw = 1.0001*sqrt((dim-1)*((interpDegree+1)/2)^2 + ...
                   ((lapOrder/2+(interpDegree+1)/2)^2));

  left = branchOwnershipBand(-1, m, h, bw, padding);
  right = branchOwnershipBand(1, m, h, bw, padding);
  [leftOnly, rightOnly, both] = splitBranchOwnership(left, right, h);

  figure(figNum);
  clf;
  set(gcf, 'color', 'w');
  set(gcf, 'Position', [130, 130, 1120, 760]);

  hLeft = scatter(leftOnly.x, leftOnly.y, 20, [0.10 0.35 0.85], ...
                  'filled', 'MarkerFaceAlpha', 0.70, ...
                  'MarkerEdgeColor', 'none');
  hold on;
  hRight = scatter(rightOnly.x, rightOnly.y, 20, [0.90 0.30 0.12], ...
                   'filled', 'MarkerFaceAlpha', 0.70, ...
                   'MarkerEdgeColor', 'none');
  hBoth = scatter(both.x, both.y, 28, [0.45 0.18 0.65], ...
                  'filled', 'MarkerFaceAlpha', 0.85, ...
                  'MarkerEdgeColor', 'none');
  hCurve = plot([-1 0 1], [m 0 m], 'k-', 'LineWidth', 2.1);
  hVertex = plot(0, 0, 'ko', 'MarkerSize', 7, ...
                 'MarkerFaceColor', 'w', 'LineWidth', 1.2);
  hold off;

  allX = [left.x; right.x];
  allY = [left.y; right.y];
  axis equal;
  grid on;
  box on;
  xlim([min(allX)-0.05, max(allX)+0.05]);
  ylim([min(allY)-0.05, max(allY)+0.08]);
  xlabel('x');
  ylabel('y');
  title(sprintf('Branch-local band ownership, \\alpha = %d^\\circ, h = %.4g', ...
                alphaDeg, h));
  legend([hLeft, hRight, hBoth, hCurve, hVertex], ...
         {'left branch only', 'right branch only', 'both branch bands', ...
          'V manifold', 'shared vertex'}, ...
         'Location', 'northoutside', 'Orientation', 'horizontal');
  set(gca, 'FontSize', 12, 'LineWidth', 1, 'Layer', 'top');
end

function data = branchOwnershipBand(side, m, h, bw, padding)
%BRANCHOWNERSHIPBAND Build one branch-local band for diagnostics.

  if side < 0
    p0 = [-1, m];
    p1 = [0, 0];
    x1d = (-1-padding:h:padding)';
  else
    p0 = [0, 0];
    p1 = [1, m];
    x1d = (-padding:h:1+padding)';
  end

  y1d = (-padding:h:m+padding)';
  [xx, yy] = meshgrid(x1d, y1d);
  [tilde, tilde, dist] = cpLineSegment2d(xx, yy, p0, p1); %#ok<ASGLU>
  band = find(abs(dist) <= bw*h);

  data.x = xx(band);
  data.y = yy(band);
end

function [leftOnly, rightOnly, both] = splitBranchOwnership(left, right, h)
%SPLITBRANCHOWNERSHIP Separate unique and overlapping branch-band points.

  leftKeys = branchOwnershipKeys(left.x, left.y, h);
  rightKeys = branchOwnershipKeys(right.x, right.y, h);
  [tilde, leftBothIdx, rightBothIdx] = intersect(leftKeys, rightKeys, 'rows'); %#ok<ASGLU>

  leftBothMask = false(size(left.x));
  rightBothMask = false(size(right.x));
  leftBothMask(leftBothIdx) = true;
  rightBothMask(rightBothIdx) = true;

  leftOnly.x = left.x(~leftBothMask);
  leftOnly.y = left.y(~leftBothMask);
  rightOnly.x = right.x(~rightBothMask);
  rightOnly.y = right.y(~rightBothMask);
  both.x = left.x(leftBothMask);
  both.y = left.y(leftBothMask);
end

function keys = branchOwnershipKeys(x, y, h)
%BRANCHOWNERSHIPKEYS Make stable integer keys for Cartesian grid points.

  keys = round([x(:), y(:)] / h);
end

function positions = panelPositions(nPanels)
%PANELPOSITIONS Return uniform axes positions, centering a short last row.

  nCols = min(3, nPanels);
  nRows = ceil(nPanels/nCols);
  left = 0.065;
  right = 0.14;
  bottom = 0.085;
  top = 0.09;
  hGap = 0.065;
  vGap = 0.10;
  width = (1 - left - right - (nCols - 1)*hGap)/nCols;
  height = (1 - bottom - top - (nRows - 1)*vGap)/nRows;
  positions = zeros(nPanels, 4);

  for ia = 1:nPanels
    row = floor((ia - 1)/nCols) + 1;
    col = mod(ia - 1, nCols) + 1;
    rowCount = min(nCols, nPanels - (row - 1)*nCols);
    rowOffset = (nCols - rowCount) * (width + hGap) / 2;
    x = left + rowOffset + (col - 1)*(width + hGap);
    y = 1 - top - row*height - (row - 1)*vGap;
    positions(ia, :) = [x, y, width, height];
  end
end

function plotConvergence(figNum, anglesDeg, hvals, absGlobalInf, ...
                         absGlobalL2, interpDegree, couplingMode, exactMode)
%PLOTCONVERGENCE Show absolute-error convergence curves.

  figure(figNum);
  clf;
  set(gcf, 'color', 'w');
  set(gcf, 'Position', [120, 120, 1180, 640]);
  colors = lines(length(anglesDeg));
  positions = panelPositions(length(anglesDeg));

  for ia = 1:length(anglesDeg)
    axes('Position', positions(ia, :)); %#ok<LAXES>
    loglog(hvals, absGlobalInf(ia, :), 'o-', ...
           'LineWidth', 1.5, 'MarkerSize', 7, 'Color', colors(ia, :));
    hold on;
    loglog(hvals, absGlobalL2(ia, :), 's--', ...
           'LineWidth', 1.4, 'MarkerSize', 7, 'Color', colors(ia, :));
    ref = absGlobalInf(ia, 1) * hvals ./ hvals(1);
    loglog(hvals, ref, 'w:', 'LineWidth', 1.8);
    hold off;

    set(gca, 'XDir', 'reverse', 'FontSize', 9, ...
             'LineWidth', 0.8, 'Layer', 'top');
    grid on;
    box on;
    xlabel('h');
    ylabel('absolute error');
    title(sprintf('\\alpha = %d^\\circ', anglesDeg(ia)));
    legend({'E_{\infty}', 'E_2', 'O(h)'}, 'Location', 'southwest');
  end

  if exist('sgtitle', 'file') || exist('sgtitle', 'builtin')
    sgtitle(sprintf(['Heat convergence, coupling = %s, exact = %s, ' ...
                    'interpDegree = %d'], ...
                    couplingMode, exactMode, interpDegree));
  end
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
