function results = example_two_sphere_intersection_icpm_rotation_convergence(opts)
%EXAMPLE_TWO_SPHERE_INTERSECTION_ICPM_ROTATION_CONVERGENCE
% Manufactured-solution convergence for two sphere caps glued along a circle.
%
% The surface is the boundary of the intersection of two unit spheres with
% centers [-a 0 0] and [a 0 0].  Branch A is the cap x >= 0 on the left
% sphere and branch B is the cap x <= 0 on the right sphere.  The branch
% extension rows whose closest points are on the shared singular circle are
% rotated into the opposite branch and placed in the off-diagonal block of
% the closest point extension matrix.

  if (nargin < 1)
    opts = struct();
  end

  thisdir = fileparts(mfilename('fullpath'));
  repodir = fileparts(thisdir);
  addpath(thisdir);
  addpath(fullfile(repodir, 'cp_matrices'));
  addpath(fullfile(repodir, 'surfaces'));

  R = getOption(opts, 'R', 1);
  a = getOption(opts, 'a', 0.5);
  hvals = getOption(opts, 'hvals', 1./[20 40]);
  p = getOption(opts, 'p', 3);
  order = getOption(opts, 'order', 2);
  makePlots = getOption(opts, 'makePlots', true);
  showDiagnostics = getOption(opts, 'showDiagnostics', true);
  solver = getOption(opts, 'solver', 'auto');
  directMaxUnknowns = getOption(opts, 'directMaxUnknowns', 2e5);
  iterTol = getOption(opts, 'iterTol', 1e-8);
  iterMaxit = getOption(opts, 'iterMaxit', 100);
  gmresRestart = getOption(opts, 'gmresRestart', 50);

  if (a <= 0 || a >= R)
    error('Expected 0 < a < R.');
  end

  hvals = hvals(:).';
  if (any(hvals <= 0))
    error('All h values must be positive.');
  end

  global ICPM2009BANDINGCHECKS
  oldBandingChecks = ICPM2009BANDINGCHECKS;
  ICPM2009BANDINGCHECKS = 1;
  cleanupICPM2009BANDINGCHECKS = onCleanup(@() reset_icpm2009bandingchecks(oldBandingChecks));

  levelResults = repmat(emptyLevelResult(), length(hvals), 1);

  for k = 1:length(hvals)
    levelResults(k) = solveOneLevel(hvals(k), R, a, p, order, ...
                                    makePlots, showDiagnostics, solver, ...
                                    directMaxUnknowns, iterTol, iterMaxit, ...
                                    gmresRestart);
  end

  results.h = hvals;
  results.N = 1 ./ hvals;
  results.errorsLinf = [levelResults.errorLinf];
  results.errorsL2 = [levelResults.errorL2];
  results.ratesLinf = convergenceRates(results.h, results.errorsLinf);
  results.ratesL2 = convergenceRates(results.h, results.errorsL2);
  results.branchErrorsA = [[levelResults.errorLinfA].' [levelResults.errorL2A].'];
  results.branchErrorsB = [[levelResults.errorLinfB].' [levelResults.errorL2B].'];
  results.bandErrorsLinf = [levelResults.bandErrorLinf];
  results.bandErrorsL2 = [levelResults.bandErrorL2];
  results.crossRowsA = [levelResults.crossRowsA];
  results.crossRowsB = [levelResults.crossRowsB];
  results.maxExtensionRowSumError = [levelResults.maxExtensionRowSumError];
  results.maxSingularBranchDifference = [levelResults.maxSingularBranchDifference];
  results.levels = levelResults;

  if (showDiagnostics)
    printConvergenceTable(results);
  end

  if (makePlots)
    plotConvergenceSummary(results);
  end


function level = solveOneLevel(h, R, a, p, order, makePlots, showDiagnostics, ...
                               solver, directMaxUnknowns, iterTol, iterMaxit, ...
                               gmresRestart)

  dim = 3;
  fd_stenrad = order/2;
  bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ...
                   ((fd_stenrad+(p+1)/2)^2));

  circleRadius = sqrt(R^2 - a^2);
  pad = (bw + fd_stenrad + (p+1)/2 + 2)*h;
  xmax = h*ceil((R - a + pad)/h);
  yzmax = h*ceil((circleRadius + pad)/h);
  x1d = (-xmax:h:xmax).';
  y1d = (-yzmax:h:yzmax).';
  z1d = y1d;

  [xx, yy, zz] = meshgrid(x1d, y1d, z1d);

  cenA = [-a 0 0];
  cenB = [ a 0 0];
  sideA = 1;     % branch A cap has x >= 0
  sideB = -1;    % branch B cap has x <= 0

  [cpxA, cpyA, cpzA, distA, bdyA] = cpSphereCap(xx, yy, zz, R, cenA, sideA);
  [cpxB, cpyB, cpzB, distB, bdyB] = cpSphereCap(xx, yy, zz, R, cenB, sideB);

  bandInitA = find(abs(distA) <= bw*h);
  bandInitB = find(abs(distB) <= bw*h);

  cpxInitA = cpxA(bandInitA);  cpyInitA = cpyA(bandInitA);
  cpzInitA = cpzA(bandInitA);
  xInitA = xx(bandInitA);  yInitA = yy(bandInitA);  zInitA = zz(bandInitA);
  bdyInitA = bdyA(bandInitA);

  cpxInitB = cpxB(bandInitB);  cpyInitB = cpyB(bandInitB);
  cpzInitB = cpzB(bandInitB);
  xInitB = xx(bandInitB);  yInitB = yy(bandInitB);  zInitB = zz(bandInitB);
  bdyInitB = bdyB(bandInitB);

  if (showDiagnostics)
    fprintf('\nh = %g, grid %d x %d x %d\n', h, length(x1d), length(y1d), length(z1d));
    fprintf('Initial bands A/B: %d / %d\n', length(bandInitA), length(bandInitB));
  end

  [LA, EAA, RA, ibandA, obandA, ibandfullA, obandfullA] = ...
      ops_and_bands3d(x1d, y1d, z1d, xInitA, yInitA, zInitA, ...
                      cpxInitA, cpyInitA, cpzInitA, bandInitA, p, order);
  [LB, EBB, RB, ibandB, obandB, ibandfullB, obandfullB] = ...
      ops_and_bands3d(x1d, y1d, z1d, xInitB, yInitB, zInitB, ...
                      cpxInitB, cpyInitB, cpzInitB, bandInitB, p, order);

  cpxOutA = cpxInitA(obandA);  cpyOutA = cpyInitA(obandA);
  cpzOutA = cpzInitA(obandA);
  xOutA = xInitA(obandA);  yOutA = yInitA(obandA);  zOutA = zInitA(obandA);
  bdyOutA = bdyInitA(obandA);

  cpxOutB = cpxInitB(obandB);  cpyOutB = cpyInitB(obandB);
  cpzOutB = cpzInitB(obandB);
  xOutB = xInitB(obandB);  yOutB = yInitB(obandB);  zOutB = zInitB(obandB);
  bdyOutB = bdyInitB(obandB);

  cpxInA = RA*cpxOutA;  cpyInA = RA*cpyOutA;  cpzInA = RA*cpzOutA;
  cpxInB = RB*cpxOutB;  cpyInB = RB*cpyOutB;  cpzInB = RB*cpzOutB;

  circleTol = 100*eps(max(1, R));
  crossRowsA = find(bdyOutA | (abs(cpxOutA) <= circleTol));
  crossRowsB = find(bdyOutB | (abs(cpxOutB) <= circleTol));

  EAB = sparse(size(EAA, 1), size(EBB, 2));
  EBA = sparse(size(EBB, 1), size(EAA, 2));

  thetaAtoB = [];
  thetaBtoA = [];

  if (~isempty(crossRowsA))
    sA = [cpxOutA(crossRowsA) cpyOutA(crossRowsA) cpzOutA(crossRowsA)];
    pA = [xOutA(crossRowsA) yOutA(crossRowsA) zOutA(crossRowsA)];
    [rotA, thetaAtoB] = rotateBranchPoints(pA, sA, R, cenA, cenB, sideA, sideB);
    [cpxAtoB, cpyAtoB, cpzAtoB] = cpSphereCap(rotA(:,1), rotA(:,2), rotA(:,3), ...
                                              R, cenB, sideB);
    EAB(crossRowsA,:) = interp3_matrix(x1d, y1d, z1d, ...
                                       cpxAtoB(:), cpyAtoB(:), cpzAtoB(:), ...
                                       p, ibandfullB);
    EAA(crossRowsA,:) = 0;
  end

  if (~isempty(crossRowsB))
    sB = [cpxOutB(crossRowsB) cpyOutB(crossRowsB) cpzOutB(crossRowsB)];
    pB = [xOutB(crossRowsB) yOutB(crossRowsB) zOutB(crossRowsB)];
    [rotB, thetaBtoA] = rotateBranchPoints(pB, sB, R, cenB, cenA, sideB, sideA);
    [cpxBtoA, cpyBtoA, cpzBtoA] = cpSphereCap(rotB(:,1), rotB(:,2), rotB(:,3), ...
                                              R, cenA, sideA);
    EBA(crossRowsB,:) = interp3_matrix(x1d, y1d, z1d, ...
                                       cpxBtoA(:), cpyBtoA(:), cpzBtoA(:), ...
                                       p, ibandfullA);
    EBB(crossRowsB,:) = 0;
  end

  rhsA = rhsMmsBranch(cpxInA, cpyInA, cpzInA, 'A', cenA, a, R);
  rhsB = rhsMmsBranch(cpxInB, cpyInB, cpzInB, 'B', cenB, a, R);
  rhs = [rhsA; rhsB];
  if (any(~isfinite(rhs)))
    error('Non-finite MMS right-hand side encountered at branch closest points.');
  end

  u = solveIcpLinearSystem(LA, EAA, EAB, RA, LB, EBA, EBB, RB, rhs, ...
                           solver, directMaxUnknowns, iterTol, iterMaxit, ...
                           gmresRestart, showDiagnostics);
  if (any(~isfinite(u)))
    error('Non-finite ICPM solution encountered after solving the elliptic system.');
  end

  uA = u(1:length(rhsA));
  uB = u(length(rhsA)+1:end);

  bandExactA = exactMmsBranch(cpxInA, cpyInA, cpzInA, 'A', cenA, a, R);
  bandExactB = exactMmsBranch(cpxInB, cpyInB, cpzInB, 'B', cenB, a, R);
  bandErrA = uA - bandExactA;
  bandErrB = uB - bandExactB;

  if (any(~isfinite([bandExactA; bandExactB; bandErrA; bandErrB])))
    error('Non-finite MMS exact values or errors encountered; check closest points and RHS.');
  end

  bandErrorLinfA = norm(bandErrA, inf);
  bandErrorLinfB = norm(bandErrB, inf);
  bandErrorL2A = sqrt(mean(bandErrA.^2));
  bandErrorL2B = sqrt(mean(bandErrB.^2));

  [errorLinfA, errorL2A, surfaceWeightA, surfaceEvalPointsA] = ...
      surfaceCapError(x1d, y1d, z1d, ibandfullA, uA, p, 'A', cenA, a, R, h);
  [errorLinfB, errorL2B, surfaceWeightB, surfaceEvalPointsB] = ...
      surfaceCapError(x1d, y1d, z1d, ibandfullB, uB, p, 'B', cenB, a, R, h);
  surfaceWeight = surfaceWeightA + surfaceWeightB;
  errorLinf = max(errorLinfA, errorLinfB);
  errorL2 = sqrt((surfaceWeightA*errorL2A^2 + surfaceWeightB*errorL2B^2) / surfaceWeight);

  if (any(~isfinite([errorLinfA; errorLinfB; errorLinf; errorL2A; errorL2B; errorL2])))
    error('Non-finite surface error diagnostic encountered.');
  end

  rowSums = full([sum(EAA, 2) + sum(EAB, 2); ...
                  sum(EBA, 2) + sum(EBB, 2)]);
  maxRowSumError = max(abs(rowSums - 1));

  [maxSingularDiff, singularExactError] = singularCircleDiagnostic( ...
      x1d, y1d, z1d, ibandfullA, ibandfullB, uA, uB, p, circleRadius, h, ...
      cenA, a, R);

  if (showDiagnostics)
    fprintf('Inner bands A/B: %d / %d\n', length(ibandfullA), length(ibandfullB));
    fprintf('Outer bands A/B: %d / %d\n', length(obandfullA), length(obandfullB));
    fprintf('Cross rows A->B / B->A: %d / %d\n', length(crossRowsA), length(crossRowsB));
    fprintf('Rotation angles A->B range: [%g %g]\n', minValue(thetaAtoB), maxValue(thetaAtoB));
    fprintf('Rotation angles B->A range: [%g %g]\n', minValue(thetaBtoA), maxValue(thetaBtoA));
    fprintf('E row-sum max error: %g\n', maxRowSumError);
    fprintf('Singular-circle branch max difference: %g\n', maxSingularDiff);
    fprintf('Singular-circle exact max error: %g\n', singularExactError);
    fprintf('Surface eval points A/B: %d / %d\n', surfaceEvalPointsA, surfaceEvalPointsB);
    fprintf('Surface errors Linf A/B/combined: %g / %g / %g\n', ...
            errorLinfA, errorLinfB, errorLinf);
    fprintf('Surface errors L2   A/B/combined: %g / %g / %g\n', ...
            errorL2A, errorL2B, errorL2);
    fprintf('Band errors Linf A/B/combined: %g / %g / %g\n', ...
            bandErrorLinfA, bandErrorLinfB, max(bandErrorLinfA, bandErrorLinfB));
    fprintf('Band errors RMS  A/B/combined: %g / %g / %g\n', ...
            bandErrorL2A, bandErrorL2B, sqrt(mean([bandErrA; bandErrB].^2)));
  end

  if (makePlots)
    plotLensSolution(x1d, y1d, z1d, ibandfullA, uA, ibandfullB, uB, ...
                     p, cenA, cenB, a, R, h);
  end

  level = emptyLevelResult();
  level.h = h;
  level.errorLinfA = errorLinfA;
  level.errorLinfB = errorLinfB;
  level.errorLinf = errorLinf;
  level.errorL2A = errorL2A;
  level.errorL2B = errorL2B;
  level.errorL2 = errorL2;
  level.bandErrorLinfA = bandErrorLinfA;
  level.bandErrorLinfB = bandErrorLinfB;
  level.bandErrorLinf = max(bandErrorLinfA, bandErrorLinfB);
  level.bandErrorL2A = bandErrorL2A;
  level.bandErrorL2B = bandErrorL2B;
  level.bandErrorL2 = sqrt(mean([bandErrA; bandErrB].^2));
  level.surfaceEvalPointsA = surfaceEvalPointsA;
  level.surfaceEvalPointsB = surfaceEvalPointsB;
  level.crossRowsA = length(crossRowsA);
  level.crossRowsB = length(crossRowsB);
  level.innerBandA = length(ibandfullA);
  level.innerBandB = length(ibandfullB);
  level.outerBandA = length(obandfullA);
  level.outerBandB = length(obandfullB);
  level.maxExtensionRowSumError = maxRowSumError;
  level.maxSingularBranchDifference = maxSingularDiff;
  level.maxSingularExactError = singularExactError;


function u = solveIcpLinearSystem(LA, EAA, EAB, RA, LB, EBA, EBB, RB, rhs, ...
                                  solver, directMaxUnknowns, iterTol, iterMaxit, ...
                                  gmresRestart, showDiagnostics)

  solver = normalizeSolverName(solver);

  nA = size(LA, 1);
  nB = size(LB, 1);
  n = nA + nB;

  [LdiagA, LoffA] = splitLapsharpOperator(LA, RA);
  [LdiagB, LoffB] = splitLapsharpOperator(LB, RB);

  useDirect = strcmp(solver, 'direct') || ...
              (strcmp(solver, 'auto') && n <= directMaxUnknowns);

  if (showDiagnostics)
    fprintf('Linear solve unknowns A/B/total: %d / %d / %d\n', nA, nB, n);
    fprintf('Interpolation nnz EAA/EBB/EAB/EBA: %d / %d / %d / %d\n', ...
            nnz(EAA), nnz(EBB), nnz(EAB), nnz(EBA));
    fprintf('Estimated E sparse storage: %.1f MB\n', ...
            sparseStorageMB(EAA, EBB, EAB, EBA));
    fprintf('L off-diagonal nnz A/B: %d / %d\n', nnz(LoffA), nnz(LoffB));
  end

  if (useDirect)
    if (showDiagnostics)
      fprintf('Linear solver: direct sparse backslash\n');
    end

    MAA = spdiags(LdiagA, 0, nA, nA) + LoffA*EAA;
    MAB = LoffA*EAB;
    MBA = LoffB*EBA;
    MBB = spdiags(LdiagB, 0, nB, nB) + LoffB*EBB;
    M = [MAA MAB; MBA MBB];
    A = speye(n) - M;

    if (showDiagnostics)
      fprintf('Explicit M nnz: %d, estimated sparse storage: %.1f MB\n', ...
              nnz(M), sparseStorageMB(M));
    end

    u = A \ rhs;
  else
    if (showDiagnostics)
      fprintf('Linear solver: matrix-free restarted GMRES');
      if (strcmp(solver, 'auto'))
        fprintf(' (auto: direct threshold %d unknowns)', directMaxUnknowns);
      end
      fprintf('\n');
      fprintf('Explicit M is not assembled in this solve path.\n');
    end

    afun = @(v) applyIcpLinearSystem(v, LdiagA, LoffA, EAA, EAB, ...
                                    LdiagB, LoffB, EBA, EBB);
    precondDiag = 1 - [LdiagA; LdiagB];
    precondDiag(abs(precondDiag) < eps) = 1;
    mfun = @(v) v ./ precondDiag;

    [u, flag, relres, iter] = gmres(afun, rhs, gmresRestart, iterTol, ...
                                    iterMaxit, mfun);
    if (showDiagnostics)
      fprintf('GMRES flag: %d, relres: %g, iter: %s\n', ...
              flag, relres, mat2str(iter));
    end
    if (flag ~= 0)
      warning('example_two_sphere:gmresNoConvergence', ...
              'GMRES did not meet the requested tolerance; flag=%d, relres=%g.', ...
              flag, relres);
    end
  end


function y = applyIcpLinearSystem(v, LdiagA, LoffA, EAA, EAB, ...
                                  LdiagB, LoffB, EBA, EBB)

  nA = length(LdiagA);
  vA = v(1:nA);
  vB = v(nA+1:end);

  extA = EAA*vA + EAB*vB;
  extB = EBA*vA + EBB*vB;

  mvA = LdiagA.*vA + LoffA*extA;
  mvB = LdiagB.*vB + LoffB*extB;

  y = v - [mvA; mvB];


function [Ldiag, Loff] = splitLapsharpOperator(L, R)

  Ldiagpad = R .* L;
  Ldiag = full(sum(Ldiagpad, 2));
  Loff = L - Ldiagpad;


function solver = normalizeSolverName(solver)

  if (~ischar(solver))
    error('opts.solver must be ''auto'', ''direct'', or ''gmres''.');
  end

  solver = lower(solver);
  if (strcmp(solver, 'backslash'))
    solver = 'direct';
  elseif (strcmp(solver, 'iterative'))
    solver = 'gmres';
  end

  if (~strcmp(solver, 'auto') && ~strcmp(solver, 'direct') && ...
      ~strcmp(solver, 'gmres'))
    error('opts.solver must be ''auto'', ''direct'', or ''gmres''.');
  end


function mb = sparseStorageMB(varargin)

  bytes = 0;
  for k = 1:nargin
    A = varargin{k};
    bytes = bytes + 16*nnz(A) + 8*(size(A, 2) + 1);
  end
  mb = bytes / 1024^2;


function [cpx, cpy, cpz, dist, bdy] = cpSphereCap(x, y, z, R, cen, side)
% Closest point to a sphere cap cut by x = 0.
% side =  1 gives the cap x >= 0.
% side = -1 gives the cap x <= 0.

  xs = x - cen(1);
  ys = y - cen(2);
  zs = z - cen(3);
  rr = sqrt(xs.^2 + ys.^2 + zs.^2);

  rrSafe = rr;
  rrSafe(rrSafe == 0) = 1;

  cpx = cen(1) + R*xs ./ rrSafe;
  cpy = cen(2) + R*ys ./ rrSafe;
  cpz = cen(3) + R*zs ./ rrSafe;

  zeroRows = (rr == 0);
  if (any(zeroRows(:)))
    cpx(zeroRows) = cen(1) + R;
    cpy(zeroRows) = cen(2);
    cpz(zeroRows) = cen(3);
  end

  insideCap = (side*cpx >= 0);
  bdy = ~insideCap;

  if (any(bdy(:)))
    circleRadius = sqrt(R^2 - cen(1)^2);
    rho = sqrt(y.^2 + z.^2);
    rhoSafe = rho;
    rhoSafe(rhoSafe == 0) = 1;

    cpxCircle = zeros(size(x));
    cpyCircle = circleRadius*y ./ rhoSafe;
    cpzCircle = circleRadius*z ./ rhoSafe;

    axisRows = (rho == 0);
    if (any(axisRows(:)))
      cpyCircle(axisRows) = circleRadius;
      cpzCircle(axisRows) = 0;
    end

    cpx(bdy) = cpxCircle(bdy);
    cpy(bdy) = cpyCircle(bdy);
    cpz(bdy) = cpzCircle(bdy);
  end

  dist = sqrt((x - cpx).^2 + (y - cpy).^2 + (z - cpz).^2);


function [rotatedPoints, theta] = rotateBranchPoints(points, singularPoints, R, ...
                                                     fromCenter, toCenter, ...
                                                     fromSide, toSide)

  nFrom = normalizeRows(bsxfun(@minus, singularPoints, fromCenter) ./ R);
  nTo = normalizeRows(bsxfun(@minus, singularPoints, toCenter) ./ R);

  if (fromSide == 1)
    tau = normalizeRows(cross(nFrom, nTo, 2));
  else
    tau = normalizeRows(cross(nTo, nFrom, 2));
  end

  etaFromProbe = capConormal(nFrom, fromSide);
  etaTo = capConormal(nTo, toSide);
  targetDirection = -etaTo;
  theta = NaN(size(points, 1), 1);
  cpfFrom = @(x, y, z) cpSphereCap(x, y, z, R, fromCenter, fromSide);

  minProbeDistance = sqrt(eps)*max(1, R);
  for k = 1:size(points, 1)
    theta(k) = angle3D(points(k,1), points(k,2), points(k,3), cpfFrom, ...
                       singularPoints(k,:), tau(k,:), targetDirection(k,:));

    if (~isfinite(theta(k)))
      probeDistance = max(norm(points(k,:) - singularPoints(k,:)), minProbeDistance);
      probePoint = singularPoints(k,:) + probeDistance*etaFromProbe(k,:);
      theta(k) = angle3D(probePoint(1), probePoint(2), probePoint(3), cpfFrom, ...
                         singularPoints(k,:), tau(k,:), targetDirection(k,:));
    end
  end

  if (any(~isfinite(theta)))
    error('angle3D did not return finite branch rotation angles.');
  end

  rotatedPoints = rotateAboutAxis(points, singularPoints, tau, theta);


function eta = capConormal(normal, side)

  capOut = [-side*ones(size(normal,1),1) zeros(size(normal,1),1) zeros(size(normal,1),1)];
  eta = capOut - bsxfun(@times, dotRows(capOut, normal), normal);
  eta = normalizeRows(eta);


function rotatedPoints = rotateAboutAxis(points, origin, axis, theta)

  axis = normalizeRows(axis);
  v = points - origin;
  c = cos(theta);
  s = sin(theta);
  axisDotV = dotRows(axis, v);
  rotatedV = bsxfun(@times, v, c) + ...
             bsxfun(@times, cross(axis, v, 2), s) + ...
             bsxfun(@times, axis, axisDotV.*(1 - c));
  rotatedPoints = origin + rotatedV;


function u = exactMmsBranch(x, y, z, branch, cen, a, R)

  [s, ~, k] = unfoldedMeridionalCoordinate(x, branch, cen, a, R);
  u = sin(k*s);


function f = rhsMmsBranch(x, y, z, branch, cen, a, R)

  [s, theta, k] = unfoldedMeridionalCoordinate(x, branch, cen, a, R);

  u = sin(k*s);
  Us = k*cos(k*s);
  Uss = -k^2*u;

  sinTheta = sin(theta);
  regularRows = (abs(sinTheta) > sqrt(eps));
  cotTerm = zeros(size(theta));
  cotTerm(regularRows) = (cos(theta(regularRows)) ./ sinTheta(regularRows)) .* ...
                         Us(regularRows) / R;
  cotTerm(~regularRows) = Uss(~regularRows);

  laplacianExact = Uss + cotTerm;
  f = u - laplacianExact;


function [s, theta, k] = unfoldedMeridionalCoordinate(x, branch, cen, a, R)

  theta0 = acos(a/R);
  L = R*theta0;
  k = pi/(2*L);

  mu = (x - cen(1))/R;
  mu = min(1, max(-1, mu));
  theta = acos(mu);

  if (strcmp(branch, 'A'))
    s = R*(theta - theta0);
  elseif (strcmp(branch, 'B'))
    s = R*(theta - (pi - theta0));
  else
    error('Unknown MMS branch "%s".', branch);
  end


function [maxDiff, maxExactError] = singularCircleDiagnostic(x1d, y1d, z1d, ...
                                                            ibandA, ibandB, ...
                                                            uA, uB, p, ...
                                                            circleRadius, h, ...
                                                            cenA, a, R)

  ntheta = max(32, ceil(2*pi*circleRadius/(2*h)));
  th = linspace(0, 2*pi, ntheta+1).';
  th(end) = [];
  sx = zeros(size(th));
  sy = circleRadius*cos(th);
  sz = circleRadius*sin(th);

  EsingA = interp3_matrix(x1d, y1d, z1d, sx, sy, sz, p, ibandA);
  EsingB = interp3_matrix(x1d, y1d, z1d, sx, sy, sz, p, ibandB);

  valsA = EsingA*uA;
  valsB = EsingB*uB;
  exact = exactMmsBranch(sx, sy, sz, 'A', cenA, a, R);

  if (any(~isfinite([valsA; valsB; exact])))
    error('Non-finite singular-circle diagnostic values encountered.');
  end

  maxDiff = max(abs(valsA - valsB));
  maxExactError = max(abs(0.5*(valsA + valsB) - exact));


function [errorLinf, errorL2, surfaceWeight, numEval] = surfaceCapError( ...
      x1d, y1d, z1d, iband, u, p, branch, cen, a, R, h)

  theta0 = acos(a/R);
  if (strcmp(branch, 'A'))
    thetaMin = 0;
    thetaMax = theta0;
  elseif (strcmp(branch, 'B'))
    thetaMin = pi - theta0;
    thetaMax = pi;
  else
    error('Unknown MMS branch "%s".', branch);
  end

  circleRadius = sqrt(R^2 - a^2);
  ntheta = max(16, ceil(theta0*R/h*2));
  nphi = max(32, ceil(2*pi*circleRadius/h*2));
  dtheta = (thetaMax - thetaMin) / ntheta;
  dphi = 2*pi / nphi;

  theta = thetaMin + ((0:ntheta-1).' + 0.5)*dtheta;
  phi = ((0:nphi-1).' + 0.5)*dphi;
  [Theta, Phi] = ndgrid(theta, phi);

  sx = cen(1) + R*cos(Theta(:));
  sy = R*sin(Theta(:)).*cos(Phi(:));
  sz = R*sin(Theta(:)).*sin(Phi(:));
  weights = R^2*sin(Theta(:))*dtheta*dphi;

  Eeval = interp3_matrix(x1d, y1d, z1d, sx, sy, sz, p, iband);
  vals = Eeval*u;
  exact = exactMmsBranch(sx, sy, sz, branch, cen, a, R);
  err = vals - exact;

  if (any(~isfinite([vals; exact; err; weights])) || any(weights < 0))
    error('Non-finite or negative surface quadrature values encountered.');
  end

  surfaceWeight = sum(weights);
  if (~isfinite(surfaceWeight) || surfaceWeight <= 0)
    error('Invalid surface quadrature weight encountered.');
  end

  errorLinf = norm(err, inf);
  errorL2 = sqrt(sum(weights.*(err.^2)) / surfaceWeight);
  numEval = length(err);


function rates = convergenceRates(h, err)

  if (length(h) < 2)
    rates = [];
    return;
  end

  rates = log(err(1:end-1) ./ err(2:end)) ./ log(h(1:end-1) ./ h(2:end));


function printConvergenceTable(results)

  fprintf('\nConvergence summary\n');
  fprintf('       h    surface Linf      rate     surface L2       rate\n');
  for k = 1:length(results.h)
    if (k == 1)
      fprintf('%8.4g  %14.6e      --   %14.6e      --\n', ...
              results.h(k), results.errorsLinf(k), results.errorsL2(k));
    else
      fprintf('%8.4g  %14.6e  %6.3f   %14.6e  %6.3f\n', ...
              results.h(k), results.errorsLinf(k), results.ratesLinf(k-1), ...
              results.errorsL2(k), results.ratesL2(k-1));
    end
  end


function plotConvergenceSummary(results)

  [N, orderIdx] = sort(results.N);
  linfErr = results.errorsLinf(orderIdx);
  l2Err = results.errorsL2(orderIdx);

  finiteRows = isfinite(N) & isfinite(linfErr) & isfinite(l2Err) & ...
               (N > 0) & (linfErr > 0) & (l2Err > 0);
  N = N(finiteRows);
  linfErr = linfErr(finiteRows);
  l2Err = l2Err(finiteRows);

  if (isempty(N))
    return;
  end

  ref = l2Err(1)*(N/N(1)).^(-2);

  figure(2); clf;
  loglog(N, linfErr, 'o-', N, l2Err, 's-', N, ref, 'k--', 'LineWidth', 1.5);
  xlabel('N = 1/h');
  ylabel('surface error');
  title('two-sphere ICPM convergence');
  legend('L_\infty error', 'L_2 error', 'O(h^2)', 'Location', 'southwest');
  grid on;
  drawnow(); pause(0);


function plotLensSolution(x1d, y1d, z1d, ibandA, uA, ibandB, uB, ...
                          p, cenA, cenB, a, R, h)

  theta0 = acos(a/R);
  ntheta = 64;
  nphi = 64;
  phi = linspace(0, 2*pi, nphi+1);

  thetaA = linspace(0, theta0, ntheta+1).';
  thetaB = linspace(pi - theta0, pi, ntheta+1).';
  [ThetaA, PhiA] = ndgrid(thetaA, phi);
  [ThetaB, PhiB] = ndgrid(thetaB, phi);

  xpA = cenA(1) + R*cos(ThetaA);
  ypA = cenA(2) + R*sin(ThetaA).*cos(PhiA);
  zpA = cenA(3) + R*sin(ThetaA).*sin(PhiA);

  xpB = cenB(1) + R*cos(ThetaB);
  ypB = cenB(2) + R*sin(ThetaB).*cos(PhiB);
  zpB = cenB(3) + R*sin(ThetaB).*sin(PhiB);

  EplotA = interp3_matrix(x1d, y1d, z1d, xpA(:), ypA(:), zpA(:), p, ibandA);
  EplotB = interp3_matrix(x1d, y1d, z1d, xpB(:), ypB(:), zpB(:), p, ibandB);
  sphplotA = reshape(EplotA*uA, size(xpA));
  sphplotB = reshape(EplotB*uB, size(xpB));

  figure(1); clf;
  surf(xpA, ypA, zpA, sphplotA);
  hold on;
  surf(xpB, ypB, zpB, sphplotB);
  title(['soln for two-sphere ICPM, h = ' num2str(h)]);
  xlabel('x'); ylabel('y'); zlabel('z');
  axis equal; shading interp;
  colorbar;
  drawnow(); pause(0);


function v = normalizeRows(v)

  n = sqrt(sum(v.^2, 2));
  n(n == 0) = 1;
  v = bsxfun(@rdivide, v, n);


function d = dotRows(a, b)

  d = sum(a.*b, 2);


function value = getOption(opts, name, defaultValue)

  if (isfield(opts, name))
    value = opts.(name);
  else
    value = defaultValue;
  end


function value = minValue(x)

  if (isempty(x))
    value = NaN;
  else
    value = min(x);
  end


function value = maxValue(x)

  if (isempty(x))
    value = NaN;
  else
    value = max(x);
  end


function level = emptyLevelResult()

  level = struct('h', NaN, ...
                 'errorLinfA', NaN, ...
                 'errorLinfB', NaN, ...
                 'errorLinf', NaN, ...
                 'errorL2A', NaN, ...
                 'errorL2B', NaN, ...
                 'errorL2', NaN, ...
                 'bandErrorLinfA', NaN, ...
                 'bandErrorLinfB', NaN, ...
                 'bandErrorLinf', NaN, ...
                 'bandErrorL2A', NaN, ...
                 'bandErrorL2B', NaN, ...
                 'bandErrorL2', NaN, ...
                 'surfaceEvalPointsA', 0, ...
                 'surfaceEvalPointsB', 0, ...
                 'crossRowsA', 0, ...
                 'crossRowsB', 0, ...
                 'innerBandA', 0, ...
                 'innerBandB', 0, ...
                 'outerBandA', 0, ...
                 'outerBandB', 0, ...
                 'maxExtensionRowSumError', NaN, ...
                 'maxSingularBranchDifference', NaN, ...
                 'maxSingularExactError', NaN);


function reset_icpm2009bandingchecks(oldValue)

  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = oldValue;
