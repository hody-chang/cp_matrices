function [results, diagnostics] = example_v_shaped_manifold_heat_refinement_angle_rotation(opts)
%EXAMPLE_V_SHAPED_MANIFOLD_HEAT_REFINEMENT_ANGLE_ROTATION
% Angle-rotation CPM heat equation on a V-shaped two-branch manifold.
%
% The two arms are solved on separate branch-local grids.  When a Cartesian
% band point on one branch has the shared vertex as its closest point, that
% point is rotated so the current branch's outward tangent matches the
% opposite branch's inward tangent.  The extension value is then interpolated
% from the opposite branch closest point.  The shared vertex itself is
% advanced by the symmetric two-neighbor junction formula
%
%   uv^{n+1} = uv^n + dt*kappa*(uL_2^n + uR_2^n - 2*uv^n)/h^2.
%
% The default time stepper stores the evolving state on the branch nodes and
% uses the CPM matrices as the sampled spatial operator R*L*E.  This avoids
% applying the interpolation projection R*E to the old solution at every
% tiny explicit step.  The optional 'explicitBand' stepper runs the literal
% Cartesian band update and re-extension loop for comparison.  The two
% physical endpoints use reflected same-branch ghost values, giving the
% standard second-order Neumann endpoint closure.
%
% Example:
%   [results, diagnostics] = ...
%       example_v_shaped_manifold_heat_refinement_angle_rotation();
%
% The default run studies angles [90 120 150 170] with h =  
% 1./[200 400 800 1600 3200].  Figure 1 compares u(s,t) against the exact
% solution on the finest grid for each angle.  Figure 2 plots L_inf and L_2
% errors versus h.  Figure 3 plots final-time |u_h - u| versus s.

  if nargin < 1
    opts = struct();
  end

  thisfile = mfilename('fullpath');
  examplesdir = fileparts(thisfile);
  repoRoot = fileparts(examplesdir);
  addpathIfNeeded(fullfile(repoRoot, 'cp_matrices'));
  addpathIfNeeded(fullfile(repoRoot, 'surfaces'));

  anglesDeg = optionValue(opts, 'anglesDeg', [90 120 150 170]);
  hvals = optionValue(opts, 'hvals', 1 ./ [200 400 800]);
  finalTime = optionValue(opts, 'finalTime', 0.01);
  kappa = optionValue(opts, 'kappa', 1.0);
  cfl = optionValue(opts, 'cfl', 0.1);
  interpDegree = optionValue(opts, 'interpDegree', 3);
  lapOrder = optionValue(opts, 'lapOrder', 2);
  padding = optionValue(opts, 'padding', []);
  vertexTol = optionValue(opts, 'vertexTol', 0);
  makePlots = optionValue(opts, 'makePlots', true);
  showDiagnostics = optionValue(opts, 'showDiagnostics', true);
  showProgress = optionValue(opts, 'showProgress', true);
  errorFigNum = optionValue(opts, 'errorFigNum', 3);
  timeStepper = optionValue(opts, 'timeStepper', 'explicitManifold');

  if showProgress
    fprintf(['Angle-rotation CPM run: %d angle(s), %d h value(s), ' ...
             'finalTime = %.4g, timeStepper = %s\n'], ...
            numel(anglesDeg), numel(hvals), finalTime, timeStepper);
  end

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
  thetaLeftToRightCol = zeros(nCases, 1);
  thetaRightToLeftCol = zeros(nCases, 1);
  angleRotationCol = false(nCases, 1);

  row = 0;
  for ia = 1:numel(anglesDeg)
    if showProgress
      fprintf('Angle %d/%d: alpha = %.6g degrees\n', ...
              ia, numel(anglesDeg), anglesDeg(ia));
    end
    for ih = 1:numel(hvals)
      row = row + 1;
      if showProgress
        fprintf('  Grid %d/%d: requested h = %.6g\n', ...
                ih, numel(hvals), hvals(ih));
      end
      diagnostics(row) = solveOneLevel(anglesDeg(ia), hvals(ih), ...
          finalTime, kappa, cfl, interpDegree, lapOrder, padding, ...
          vertexTol, showProgress, timeStepper);

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
      thetaLeftToRightCol(row) = diagnostics(row).thetaLeftToRightDeg;
      thetaRightToLeftCol(row) = diagnostics(row).thetaRightToLeftDeg;
      angleRotationCol(row) = diagnostics(row).angleRotationOK;
      if showProgress
        fprintf(['  Finished alpha = %.6g, h = %.6g: steps = %d, ' ...
                 'FinalInfError = %.4e\n'], ...
                diagnostics(row).angleDeg, diagnostics(row).h, ...
                diagnostics(row).numSteps, diagnostics(row).finalInfError);
      end
    end
  end

  results = makeResultsTable(angleCol, hCol, cflCol, dtCol, stepsCol, ...
      finalInfErrCol, finalL2ErrCol, finalInfRateCol, finalL2RateCol, ...
      massDriftCol, maxVertexJumpCol, maxFluxJumpCol, ...
      thetaLeftToRightCol, thetaRightToLeftCol, angleRotationCol);

  if showDiagnostics
    disp(results);
    fprintf(['Equilibrium check: exact insulated equilibrium is 5.5; ' ...
             'reported mass drift uses composite trapezoid quadrature.\n']);
  end

  if makePlots && ~isempty(diagnostics)
    plotAngleSolutions(1, diagnostics, anglesDeg);
    plotConvergenceByAngle(2, diagnostics, anglesDeg);
    plotErrorByAngle(errorFigNum, diagnostics, anglesDeg);
  end
end

function diagnostic = solveOneLevel(angleDeg, requestedH, finalTime, kappa, ...
    cfl, interpDegree, lapOrder, padding, vertexTol, showProgress, ...
    timeStepper)
%SOLVEONELEVEL Solve one angle-rotation CPM refinement level.

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

  if showProgress
    fprintf('    Building branch grids and CPM matrices...\n');
  end

  left = buildBranch(-1, angleDeg, h, sLeft, interpDegree, lapOrder, ...
                     bw, localPadding, vertexTol);
  right = buildBranch(1, angleDeg, h, sRight, interpDegree, lapOrder, ...
                      bw, localPadding, vertexTol);
  [left, right, rotationInfo] = addAngleRotationMaps(left, right, ...
                                                     interpDegree);
  angleRotationOK = angleRotationUsesOnlyVertexFibers(left) && ...
                    angleRotationUsesOnlyVertexFibers(right);
  if ~angleRotationOK
    error('Angle-rotation extension rows are not limited to vertex fibers.');
  end

  if showProgress
    fprintf('    Rotation angles: left-to-right = %.6g deg, right-to-left = %.6g deg\n', ...
            rotationInfo.thetaLeftToRightDeg, rotationInfo.thetaRightToLeftDeg);
  end

  if showProgress
    fprintf('    Initializing branch-node values...\n');
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

  if showProgress
    fprintf('    Time integration: %d step(s), dt = %.6g\n', numSteps, dt);
  end

  switch lower(timeStepper)
    case {'explicitmanifold', 'manifold'}
      if showProgress
        fprintf('    Using explicit manifold CPM operator R*L*E.\n');
      end
      [uLeft, uRight, massHistory, timeHistory, vertexJumpHistory, ...
          fluxJumpHistory] = integrateExplicitManifold(uLeft, uRight, ...
          left, right, h, dt, numSteps, kappa, showProgress);

    case {'explicitband', 'band'}
      if showProgress
        fprintf('    Using explicit Cartesian band CPM re-extension.\n');
      end
      [uLeft, uRight, massHistory, timeHistory, vertexJumpHistory, ...
          fluxJumpHistory] = integrateExplicitBand(uLeft, uRight, ...
          left, right, h, dt, numSteps, kappa, showProgress);

    otherwise
      error('Unknown timeStepper "%s". Use explicitManifold or explicitBand.', ...
            timeStepper);
  end

  massInitial = massHistory(1);
  massFinal = massHistory(end);

  exactLeft = exactSolution(sLeft, finalTime, kappa);
  exactRight = exactSolution(sRight, finalTime, kappa);
  errorLeft = uLeft - exactLeft;
  errorRight = uRight - exactRight;
  uniqueError = [errorLeft; errorRight(2:end)];
  [errorS, finalAbsError] = errorCurve(errorLeft, errorRight, sLeft, sRight);

  diagnostic = emptyDiagnostic();
  diagnostic.angleDeg = angleDeg;
  diagnostic.h = h;
  diagnostic.requestedH = requestedH;
  diagnostic.dt = dt;
  diagnostic.numSteps = numSteps;
  diagnostic.kappa = kappa;
  diagnostic.cfl = cfl;
  diagnostic.timeStepper = timeStepper;
  diagnostic.interpDegree = interpDegree;
  diagnostic.lapOrder = lapOrder;
  diagnostic.sLeft = sLeft;
  diagnostic.sRight = sRight;
  diagnostic.uLeft = uLeft;
  diagnostic.uRight = uRight;
  diagnostic.exactLeft = exactLeft;
  diagnostic.exactRight = exactRight;
  diagnostic.time = timeHistory;
  diagnostic.errorS = errorS;
  diagnostic.finalAbsError = finalAbsError;
  diagnostic.mass = massHistory;
  diagnostic.massInitial = massInitial;
  diagnostic.massFinal = massFinal;
  diagnostic.massDrift = massFinal - massInitial;
  diagnostic.vertexJump = vertexJumpHistory;
  diagnostic.fluxJump = fluxJumpHistory;
  diagnostic.maxVertexJump = max(vertexJumpHistory);
  diagnostic.maxAbsFluxJump = max(abs(fluxJumpHistory));
  diagnostic.finalInfError = max(abs(uniqueError));
  diagnostic.finalL2Error = sqrt(h * (sum(errorLeft(2:end).^2) + ...
                                      sum(errorRight(2:end).^2) + ...
                                      0.5 * (errorLeft(1)^2 + errorRight(1)^2)));
  diagnostic.thetaLeftToRightDeg = rotationInfo.thetaLeftToRightDeg;
  diagnostic.thetaRightToLeftDeg = rotationInfo.thetaRightToLeftDeg;
  diagnostic.angleRotationOK = angleRotationOK;
  diagnostic.maxBandExtensionRowSumError = max( ...
      maxExtensionRowSumError(left, 'Bown', 'Bother'), ...
      maxExtensionRowSumError(right, 'Bown', 'Bother'));
  diagnostic.maxNodeExtensionRowSumError = max( ...
      maxExtensionRowSumError(left, 'Nown', 'Nother'), ...
      maxExtensionRowSumError(right, 'Nown', 'Nother'));
  diagnostic.left = stripLargeMatrices(left);
  diagnostic.right = stripLargeMatrices(right);
end

function branch = buildBranch(side, angleDeg, h, sNodes, interpDegree, ...
    lapOrder, bw, padding, vertexTol)
%BUILDBRANCH Build one branch-local CPM band and geometry.

  halfAngle = 0.5 * angleDeg * pi / 180;
  if side < 0
    p0 = [-cos(halfAngle), sin(halfAngle)];
    p1 = [0, 0];
    physicalEndpoint = p0;
  else
    p0 = [0, 0];
    p1 = [cos(halfAngle), sin(halfAngle)];
    physicalEndpoint = p1;
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
  physicalEndpointMask = makePhysicalEndpointMask(side, bdyBand);
  Nown = interp1BranchMatrix(sNodes, sBand, interpDegree);
  Nown = applyPhysicalEndpointNeumannNodeRows(Nown, physicalEndpointMask, ...
      xBand, yBand, side, physicalEndpoint, p0, p1, sNodes, interpDegree);
  Bown = interp2_matrix(x1d, y1d, cpxBand, cpyBand, interpDegree, band);
  Bown = applyPhysicalEndpointNeumannBandRows(Bown, physicalEndpointMask, ...
      xBand, yBand, side, physicalEndpoint, p0, p1, x1d, y1d, band, ...
      interpDegree);
  if any(vertexMask)
    Nown(vertexMask, :) = 0;
    Bown(vertexMask, :) = 0;
  end

  Lcart = laplacian_2d_matrix(x1d, y1d, lapOrder, band, band);
  [xManifold, yManifold] = branchCoordinates(side, sNodes, angleDeg);
  R = interp2_matrix(x1d, y1d, xManifold, yManifold, interpDegree, band);
  vertexGridMask = makeVertexGridMask(xBand, yBand);

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
  branch.physicalEndpointMask = physicalEndpointMask;
  branch.vertexGridMask = vertexGridMask;
  branch.Nown = Nown;
  branch.Nother = sparse(numel(band), 0);
  branch.Bown = Bown;
  branch.Bother = sparse(numel(band), 0);
  branch.Lcart = Lcart;
  branch.R = R;
end

function [left, right, info] = addAngleRotationMaps(left, right, interpDegree)
%ADDANGLEROTATIONMAPS Add cross-branch extension maps at the shared vertex.

  leftTangents = branchTangents(left);
  rightTangents = branchTangents(right);
  thetaLeftToRight = signedAngle(leftTangents.outward, ...
                                 rightTangents.inward);
  thetaRightToLeft = signedAngle(rightTangents.outward, ...
                                 leftTangents.inward);

  left = addAngleRotationMap(left, right, thetaLeftToRight, interpDegree);
  right = addAngleRotationMap(right, left, thetaRightToLeft, interpDegree);

  left.rotationTheta = thetaLeftToRight;
  right.rotationTheta = thetaRightToLeft;
  info.thetaLeftToRight = thetaLeftToRight;
  info.thetaRightToLeft = thetaRightToLeft;
  info.thetaLeftToRightDeg = 180 * thetaLeftToRight / pi;
  info.thetaRightToLeftDeg = 180 * thetaRightToLeft / pi;
end

function branch = addAngleRotationMap(branch, other, theta, interpDegree)
%ADDANGLEROTATIONMAP Interpolate vertex-fiber rows from the other branch.

  nBand = numel(branch.band);
  nOtherNodes = numel(other.sNodes);
  nOtherBand = numel(other.band);
  vertexRows = find(branch.vertexMask);

  if isempty(vertexRows)
    branch.Nother = sparse(nBand, nOtherNodes);
    branch.Bother = sparse(nBand, nOtherBand);
  else
    [xRot, yRot] = rotatePoints(branch.xBand(vertexRows), ...
                                branch.yBand(vertexRows), theta);
    [targetX, targetY, tilde, tilde, targetT] = ...
        cpLineSegment2d(xRot, yRot, other.p0, other.p1); %#ok<ASGLU>
    targetS = branchParameter(other.side, targetT);
    Nvertex = interp1BranchMatrix(other.sNodes, targetS, interpDegree);
    Bvertex = interp2_matrix(other.x1d, other.y1d, targetX, targetY, ...
                             interpDegree, other.band);
    rowSelector = sparse(vertexRows, 1:numel(vertexRows), 1, ...
                         nBand, numel(vertexRows));

    branch.Nother = rowSelector * Nvertex;
    branch.Bother = rowSelector * Bvertex;
    branch.rotatedVertexX = xRot;
    branch.rotatedVertexY = yRot;
    branch.rotatedTargetX = targetX;
    branch.rotatedTargetY = targetY;
    branch.rotatedTargetS = targetS;
  end
end

function Nown = applyPhysicalEndpointNeumannNodeRows(Nown, mask, xBand, yBand, ...
    side, endpoint, p0, p1, sNodes, interpDegree)
%APPLYPHYSICALENDPOINTNEUMANNNODEROWS Reflect endpoint rows from nodes.

  if ~any(mask)
    return;
  end

  xReflect = 2 * endpoint(1) - xBand(mask);
  yReflect = 2 * endpoint(2) - yBand(mask);
  [tilde, tilde, tilde, tilde, targetT] = ...
      cpLineSegment2d(xReflect, yReflect, p0, p1); %#ok<ASGLU>
  targetS = branchParameter(side, targetT);
  Ereflect = interp1BranchMatrix(sNodes, targetS, interpDegree);
  Nown(mask, :) = Ereflect;
end

function Bown = applyPhysicalEndpointNeumannBandRows(Bown, mask, xBand, yBand, ...
    side, endpoint, p0, p1, x1d, y1d, band, interpDegree)
%APPLYPHYSICALENDPOINTNEUMANNBANDROWS Reflect endpoint rows from band data.

  if ~any(mask)
    return;
  end

  xReflect = 2 * endpoint(1) - xBand(mask);
  yReflect = 2 * endpoint(2) - yBand(mask);
  [targetX, targetY] = cpLineSegment2d(xReflect, yReflect, p0, p1);
  Ereflect = interp2_matrix(x1d, y1d, targetX, targetY, interpDegree, band);
  Bown(mask, :) = Ereflect;
end

function [uLeftNew, uRightNew] = extendFromBranchNodes( ...
    uLeft, uRight, left, right)
%EXTENDFROMBRANCHNODES Build Cartesian band extensions from branch nodes.

  oldLeft = uLeft;
  oldRight = uRight;
  uLeftNew = left.Nown * oldLeft + left.Nother * oldRight;
  uRightNew = right.Nown * oldRight + right.Nother * oldLeft;
end

function [uLeftNew, uRightNew] = extendFromBandValues( ...
    uLeft, uRight, left, right)
%EXTENDFROMBANDVALUES Re-extend Cartesian band values after a time step.

  oldLeft = uLeft;
  oldRight = uRight;
  uLeftNew = left.Bown * oldLeft + left.Bother * oldRight;
  uRightNew = right.Bown * oldRight + right.Bother * oldLeft;
end

function [uLeft, uRight, massHistory, timeHistory, vertexJumpHistory, ...
    fluxJumpHistory] = integrateExplicitManifold(uLeft, uRight, left, ...
    right, h, dt, numSteps, kappa, showProgress)
%INTEGRATEEXPLICITMANIFOLD Forward Euler for the sampled CPM operator.

  [ALL, ALR, ARL, ARR] = buildManifoldOperatorBlocks(left, right);
  massHistory = zeros(numSteps + 1, 1);
  vertexJumpHistory = zeros(numSteps + 1, 1);
  fluxJumpHistory = zeros(numSteps + 1, 1);
  timeHistory = zeros(numSteps + 1, 1);
  massHistory(1) = branchMass(uLeft, uRight, h);
  vertexJumpHistory(1) = abs(uLeft(1) - uRight(1));
  fluxJumpHistory(1) = vertexFluxJump(uLeft, uRight, h);
  progressSteps = makeProgressSteps(numSteps);
  progressCursor = 1;

  for kt = 1:numSteps
    uv = 0.5 * (uLeft(1) + uRight(1));
    uLeft(1) = uv;
    uRight(1) = uv;

    lapLeft = ALL * uLeft + ALR * uRight;
    lapRight = ARR * uRight + ARL * uLeft;

    uLeftNew = uLeft + dt * kappa * lapLeft;
    uRightNew = uRight + dt * kappa * lapRight;
    uvNew = uv + dt * kappa * ...
        (uLeft(2) + uRight(2) - 2 * uv) / h^2;
    uLeftNew(1) = uvNew;
    uRightNew(1) = uvNew;

    uLeft = uLeftNew;
    uRight = uRightNew;
    massHistory(kt + 1) = branchMass(uLeft, uRight, h);
    vertexJumpHistory(kt + 1) = abs(uLeft(1) - uRight(1));
    fluxJumpHistory(kt + 1) = vertexFluxJump(uLeft, uRight, h);
    timeHistory(kt + 1) = kt * dt;

    if showProgress && progressCursor <= numel(progressSteps) && ...
       kt == progressSteps(progressCursor)
      fprintf('      step %d/%d (t = %.4g)\n', kt, numSteps, kt * dt);
      progressCursor = progressCursor + 1;
    end
  end
end

function [uLeft, uRight, massHistory, timeHistory, vertexJumpHistory, ...
    fluxJumpHistory] = integrateExplicitBand(uLeft, uRight, left, right, ...
    h, dt, numSteps, kappa, showProgress)
%INTEGRATEEXPLICITBAND Forward Euler with CPM band re-extension each step.

  uv = 0.5 * (uLeft(1) + uRight(1));
  uLeft(1) = uv;
  uRight(1) = uv;
  [uLeftBand, uRightBand] = extendFromBranchNodes(uLeft, uRight, ...
                                                  left, right);
  uLeftBand = setVertexGridValue(uLeftBand, left, uv);
  uRightBand = setVertexGridValue(uRightBand, right, uv);
  [uLeft, uRight, uv] = sampleBranchNodes(uLeftBand, uRightBand, ...
                                          left, right, uv);

  massInitial = branchMass(uLeft, uRight, h);
  vertexJumpHistory = zeros(numSteps + 1, 1);
  fluxJumpHistory = zeros(numSteps + 1, 1);
  timeHistory = [0; numSteps * dt];

  vertexJumpHistory(1) = abs(uLeft(1) - uRight(1));
  fluxJumpHistory(1) = vertexFluxJump(uLeft, uRight, h);
  progressSteps = makeProgressSteps(numSteps);
  progressCursor = 1;

  for kt = 1:numSteps
    [uLeftHead, uRightHead, uv] = sampleBranchHeads( ...
        uLeftBand, uRightBand, left, right);
    uLeftBand = setVertexGridValue(uLeftBand, left, uv);
    uRightBand = setVertexGridValue(uRightBand, right, uv);

    uvNew = uv + dt * kappa * ...
        (uLeftHead(2) + uRightHead(2) - 2 * uv) / h^2;

    uLeftTemp = uLeftBand + dt * kappa * (left.Lcart * uLeftBand);
    uRightTemp = uRightBand + dt * kappa * (right.Lcart * uRightBand);
    uLeftTemp = setVertexGridValue(uLeftTemp, left, uvNew);
    uRightTemp = setVertexGridValue(uRightTemp, right, uvNew);

    [uLeftBand, uRightBand] = extendFromBandValues(uLeftTemp, ...
        uRightTemp, left, right);
    uLeftBand = setVertexGridValue(uLeftBand, left, uvNew);
    uRightBand = setVertexGridValue(uRightBand, right, uvNew);

    [uLeftHead, uRightHead] = sampleBranchHeads( ...
        uLeftBand, uRightBand, left, right, uvNew);
    vertexJumpHistory(kt + 1) = abs(uLeftHead(1) - uRightHead(1));
    fluxJumpHistory(kt + 1) = vertexFluxJump(uLeftHead, uRightHead, h);

    if showProgress && progressCursor <= numel(progressSteps) && ...
       kt == progressSteps(progressCursor)
      fprintf('      step %d/%d (t = %.4g)\n', kt, numSteps, kt * dt);
      progressCursor = progressCursor + 1;
    end
  end

  [uLeft, uRight] = sampleBranchNodes(uLeftBand, uRightBand, ...
                                      left, right);
  massFinal = branchMass(uLeft, uRight, h);
  massHistory = [massInitial; massFinal];
end

function [ALL, ALR, ARL, ARR] = buildManifoldOperatorBlocks(left, right)
%BUILDMANIFOLDOPERATORBLOCKS Assemble R*L*E blocks on branch nodes.

  ALL = left.R * (left.Lcart * left.Nown);
  ALR = left.R * (left.Lcart * left.Nother);
  ARR = right.R * (right.Lcart * right.Nown);
  ARL = right.R * (right.Lcart * right.Nother);
end

function u = setVertexGridValue(u, branch, uv)
%SETVERTEXGRIDVALUE Store the graph-update value at the actual vertex only.

  u(branch.vertexGridMask) = uv;
end

function [uLeft, uRight, uv] = sampleBranchNodes( ...
    uLeftBand, uRightBand, left, right, uvOverride)
%SAMPLEBRANCHNODES Interpolate Cartesian band states to branch nodes.

  uLeft = left.R * uLeftBand;
  uRight = right.R * uRightBand;
  if nargin >= 5 && ~isempty(uvOverride)
    uv = uvOverride;
  else
    uv = 0.5 * (uLeft(1) + uRight(1));
  end
  uLeft(1) = uv;
  uRight(1) = uv;
end

function [uLeftHead, uRightHead, uv] = sampleBranchHeads( ...
    uLeftBand, uRightBand, left, right, uvOverride)
%SAMPLEBRANCHHEADS Interpolate the vertex and first neighbor only.

  uLeftHead = left.R(1:2, :) * uLeftBand;
  uRightHead = right.R(1:2, :) * uRightBand;
  if nargin >= 5 && ~isempty(uvOverride)
    uv = uvOverride;
  else
    uv = 0.5 * (uLeftHead(1) + uRightHead(1));
  end
  uLeftHead(1) = uv;
  uRightHead(1) = uv;
end

function tangents = branchTangents(branch)
%BRANCHTANGENTS Return unit tangents at the shared vertex.

  vertex = [0, 0];
  if branch.side < 0
    endpoint = branch.p0;
  else
    endpoint = branch.p1;
  end

  tangents.outward = unitVector(endpoint - vertex);
  tangents.inward = -tangents.outward;
end

function v = unitVector(v)
%UNITVECTOR Normalize a row vector.

  nrm = sqrt(sum(v.^2));
  if nrm == 0
    error('Cannot normalize a zero tangent vector.');
  end
  v = v / nrm;
end

function theta = signedAngle(fromVector, toVector)
%SIGNEDANGLE Return the counterclockwise angle from one vector to another.

  crossValue = fromVector(1) * toVector(2) - fromVector(2) * toVector(1);
  dotValue = dot(fromVector, toVector);
  theta = atan2(crossValue, dotValue);
end

function [xRot, yRot] = rotatePoints(x, y, theta)
%ROTATEPOINTS Rotate points about the shared vertex.

  c = cos(theta);
  s = sin(theta);
  xRot = c * x - s * y;
  yRot = s * x + c * y;
end

function s = branchParameter(side, t)
%BRANCHPARAMETER Convert cpLineSegment2d parameter to branch coordinate.

  if side < 0
    s = -1 + t;
  else
    s = t;
  end
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
    mask = abs(sBand) <= vertexTol * h;
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

function mask = makePhysicalEndpointMask(side, bdyBand)
%MAKEPHYSICALENDPOINTMASK Select rows clamped at the non-vertex endpoint.

  if side < 0
    mask = bdyBand == 1;
  else
    mask = bdyBand == 2;
  end
end

function mask = makeVertexGridMask(xBand, yBand)
%MAKEVERTEXGRIDMASK Select the actual Cartesian grid point at the vertex.

  scale = max([1; abs(xBand(:)); abs(yBand(:))]);
  mask = hypot(xBand, yBand) <= 100 * eps(scale);
  if ~any(mask)
    [~, idx] = min(hypot(xBand, yBand));
    mask = false(size(xBand));
    mask(idx) = true;
  end
end

function tf = angleRotationUsesOnlyVertexFibers(branch)
%ANGLEROTATIONUSESONLYVERTEXFIBERS Check cross rows are vertex rows only.

  crossRowHasData = full(any(branch.Nother ~= 0, 2)) | ...
                    full(any(branch.Bother ~= 0, 2));
  ownVertexHasData = full(any(branch.Nown(branch.vertexMask, :) ~= 0, 2)) | ...
                     full(any(branch.Bown(branch.vertexMask, :) ~= 0, 2));
  tf = all(~crossRowHasData | branch.vertexMask) && ~any(ownVertexHasData);
end

function err = maxExtensionRowSumError(branch, ownField, otherField)
%MAXEXTENSIONROWSUMERROR Check whether an extension map preserves constants.

  rowSums = full(sum(branch.(ownField), 2));
  if size(branch.(otherField), 2) > 0
    rowSums = rowSums + full(sum(branch.(otherField), 2));
  else
    rowSums = rowSums + zeros(numel(branch.band), 1);
  end
  err = max(abs(rowSums - 1));
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

function [sAll, absError] = errorCurve(errorLeft, errorRight, sLeft, sRight)
%ERRORCURVE Return final-time |u_h - u| on the combined branch coordinate.

  sAll = [flipud(sLeft); sRight(2:end)]';
  errorAll = [flipud(errorLeft); errorRight(2:end)];
  absError = abs(errorAll(:))';
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

function steps = makeProgressSteps(numSteps)
%MAKEPROGRESSSTEPS Choose approximately 10 progress checkpoints.

  if numSteps <= 0
    steps = [];
  else
    steps = unique(round(linspace(0, numSteps, min(11, numSteps + 1))));
    steps = steps(steps > 0);
  end
end

function small = stripLargeMatrices(branch)
%STRIPLARGEMATRICES Keep geometry diagnostics without duplicating matrices.

  small = rmfield(branch, {'Nown', 'Nother', 'Bown', 'Bother', ...
                           'Lcart', 'R'});
  small.numBandPoints = numel(branch.band);
  small.numVertexFiberPoints = nnz(branch.vertexMask);
end

function diagnostic = emptyDiagnostic()
%EMPTYDIAGNOSTIC Template used for struct preallocation.

  diagnostic = struct('angleDeg', [], 'h', [], 'requestedH', [], ...
      'dt', [], 'numSteps', [], 'kappa', [], 'cfl', [], ...
      'timeStepper', [], 'interpDegree', [], 'lapOrder', [], ...
      'sLeft', [], 'sRight', [], ...
      'uLeft', [], 'uRight', [], 'exactLeft', [], 'exactRight', [], ...
      'time', [], 'errorS', [], 'finalAbsError', [], ...
      'mass', [], 'massInitial', [], 'massFinal', [], ...
      'massDrift', [], 'vertexJump', [], 'fluxJump', [], ...
      'maxVertexJump', [], 'maxAbsFluxJump', [], 'finalInfError', [], ...
      'finalL2Error', [], 'thetaLeftToRightDeg', [], ...
      'thetaRightToLeftDeg', [], 'angleRotationOK', [], ...
      'maxBandExtensionRowSumError', [], ...
      'maxNodeExtensionRowSumError', [], ...
      'left', [], 'right', []);
end

function results = makeResultsTable(angleCol, hCol, cflCol, dtCol, ...
    stepsCol, finalInfErrCol, finalL2ErrCol, finalInfRateCol, ...
    finalL2RateCol, massDriftCol, maxVertexJumpCol, maxFluxJumpCol, ...
    thetaLeftToRightCol, thetaRightToLeftCol, angleRotationCol)
%MAKERESULTSTABLE Return a table when available, otherwise a struct.

  if exist('table', 'builtin') || exist('table', 'file')
    results = table(angleCol, hCol, cflCol, dtCol, stepsCol, ...
        finalInfErrCol, finalL2ErrCol, finalInfRateCol, ...
        finalL2RateCol, massDriftCol, maxVertexJumpCol, ...
        maxFluxJumpCol, thetaLeftToRightCol, thetaRightToLeftCol, ...
        angleRotationCol, ...
        'VariableNames', {'AngleDegrees', 'h', 'CFL', 'dt', ...
                          'TimeSteps', 'FinalInfError', ...
                          'FinalL2Error', 'FinalInfRate', ...
                          'FinalL2Rate', 'MassDrift', ...
                          'MaxVertexJump', 'MaxAbsFluxJump', ...
                          'ThetaLeftToRightDeg', ...
                          'ThetaRightToLeftDeg', 'AngleRotationOK'});
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
        'ThetaLeftToRightDeg', num2cell(thetaLeftToRightCol), ...
        'ThetaRightToLeftDeg', num2cell(thetaRightToLeftCol), ...
        'AngleRotationOK', num2cell(angleRotationCol));
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

  title(layout, sprintf('Angle-rotation heat solution at t = %.4g', ...
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
      if numel(h) >= 2
        legend(ax, 'L_\infty', 'L_2', 'O(h)', 'O(h^2)', ...
               'Location', 'southeast');
      else
        legend(ax, 'L_\infty', 'L_2', 'Location', 'southeast');
      end
    end
  end

  title(layout, 'Angle-rotation final-time error convergence by opening angle');
end

function plotErrorByAngle(figNum, diagnostics, anglesDeg)
%PLOTERRORBYANGLE Plot final-time |u_h - u| versus s for each angle.

  fig = figure(figNum);
  clf(fig);
  set(fig, 'Color', 'w');
  layout = tiledlayout(fig, 'flow', 'TileSpacing', 'compact', ...
                       'Padding', 'compact');

  maxError = 0;
  for ia = 1:numel(anglesDeg)
    diagnostic = finestDiagnosticForAngle(diagnostics, anglesDeg(ia));
    if ~isempty(diagnostic.finalAbsError)
      maxError = max(maxError, max(diagnostic.finalAbsError(:)));
    end
  end

  for ia = 1:numel(anglesDeg)
    diagnostic = finestDiagnosticForAngle(diagnostics, anglesDeg(ia));
    ax = nexttile(layout);
    plot(ax, diagnostic.errorS, diagnostic.finalAbsError, 'b-', ...
         'LineWidth', 1.5);
    hold(ax, 'on');
    plot(ax, diagnostic.errorS, diagnostic.finalAbsError, 'bo', ...
         'MarkerSize', 2.5, 'MarkerFaceColor', 'k');
    vertexIdx = find(abs(diagnostic.errorS) <= 100 * eps(1), 1);
    if ~isempty(vertexIdx)
      plot(ax, 0, diagnostic.finalAbsError(vertexIdx), ...
           'rs', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    end
    hold(ax, 'off');
    grid(ax, 'on');
    if maxError > 0 && isfinite(maxError)
      ylim(ax, [0, 1.05 * maxError]);
    end
    xlabel(ax, 's');
    ylabel(ax, '|u_h - u|');
    title(ax, sprintf('\\alpha = %g^\\circ, h = %.4g', ...
                      diagnostic.angleDeg, diagnostic.h));
  end

  title(layout, 'Final-time |u_h - u| versus s, finest grid by angle');
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
