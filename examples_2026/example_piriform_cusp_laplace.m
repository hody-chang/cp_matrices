%% Laplace equation on the piriform curve y^2 = x^3 - x^4
% The piriform y^2 = x^3 - x^4 is a closed curve with a cusp singularity
% at the origin.  It is split into an upper branch (y >= 0) and a lower
% branch (y <= 0), each with its own duplicated grid band and unknowns.
% The branches meet at the singular cusp (0,0) and at the smooth point
% (1,0).  At both junctions the extension rows whose closest point is the
% junction are rotated into the other branch (angle rotation method), in
% the same style as example_v_shaped.m.
%
% PDE: the shifted Laplace(-Beltrami) equation on the curve,
%   u - u_ss = f,
% where s is arclength along the closed loop of total length L.  The
% shift keeps the operator invertible on a closed curve (u_ss = 0 alone
% only admits constants).  The exact solution is the two-mode function
%   u(s) = cos(2*pi*s/L + 1) + 0.5*sin(4*pi*s/L + 0.7),
% smooth and periodic in s, with f = u - u_ss assembled mode by mode.
% Reflection about the x axis maps s -> L - s; neither mode nor their
% sum is invariant, and the two phases rule out a mirror symmetry about
% any other point of the loop, so the solution has no symmetry between
% the branches and the junction coupling is genuinely exercised.
%
% Parametrization: gamma(phi) = (sin(phi)^2, sin(phi)^3*cos(phi)) traces
% the curve for phi in [0, pi].  The upper branch is phi in [0, pi/2],
% the lower branch is phi in [pi/2, pi].  The cusp is phi = 0 (and pi),
% the smooth junction is phi = pi/2 at the point (1,0).
%
% Observed convergence sits between O(h^1.5) and O(h^2): at the cusp
% the curvature blows up like 1/sqrt(x), so the straight-line unfolding
% behind the rotated extension carries an O(h^1.5) geometric error (and
% the angle2d tangent estimate has a matching O(sqrt(h)) tilt).  The
% smooth junction behaves second order: its estimated rotation angle
% goes to zero like O(h) and its branch mismatch like O(h^2).


%% Using cp_matrices

thisdir = fileparts(mfilename('fullpath'));
repodir = fileparts(thisdir);
addpath(thisdir);
addpath(fullfile(repodir, 'cp_matrices'));
addpath(fullfile(repodir, 'surfaces'));


global ICPM2009BANDINGCHECKS
oldICPM2009BANDINGCHECKS = ICPM2009BANDINGCHECKS;
cleanupICPM2009BANDINGCHECKS = ...
    onCleanup(@() reset_icpm2009bandingchecks(oldICPM2009BANDINGCHECKS));


%% Problem parameters

makePlots = true;

hvals = 1./[400 800 1600 3200];

p = 3;            % interpolation order
order = 2;        % Laplacian order

arc = buildArclengthTable();

fprintf('\nPiriform curve y^2 = x^3 - x^4, total arclength L = %g\n', arc.L);

levelResults = repmat(emptyLevelResult(), length(hvals), 1);


%% Convergence study

for levelNumber = 1:length(hvals)
  h = hvals(levelNumber);
  makeLevelPlots = makePlots && (levelNumber == length(hvals));

  levelResults(levelNumber) = solveOneLevel(h, p, order, arc, ...
                                            makeLevelPlots);

  fprintf(['h = %g, points = %d, L_inf error = %g, ' ...
           'L_2 error = %g\n'], ...
          h, levelResults(levelNumber).numPoints, ...
          levelResults(levelNumber).errorLinf, ...
          levelResults(levelNumber).errorL2);
end

printConvergenceTable(hvals, levelResults);

if (makePlots)
  plotConvergenceSummary(hvals, levelResults);
end


function level = solveOneLevel(h, p, order, arc, makePlots)

  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = 1;

  dim = 2;
  fdStenrad = order/2;
  bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ...
                   ((fdStenrad+(p+1)/2)^2));

  % the curve lives in [0,1] x [-3*sqrt(3)/16, 3*sqrt(3)/16]
  x1d = (-0.2:h:1.2).';
  y1d = (-0.5:h:0.5).';
  [xx, yy] = meshgrid(x1d, y1d);

  [cpxA, cpyA, distA, bdyA, phiA] = cpPiriformBranch(xx, yy, +1);
  [cpxB, cpyB, distB, bdyB, phiB] = cpPiriformBranch(xx, yy, -1);

  branchA = buildBranchMatrices2d(x1d, y1d, xx, yy, cpxA, cpyA, ...
                                  distA, bdyA, phiA, h, bw, p, order);
  branchB = buildBranchMatrices2d(x1d, y1d, xx, yy, cpxB, cpyB, ...
                                  distB, bdyB, phiB, h, bw, p, order);

  [EAA, EAB, EBA, EBB, junctionInfo] = ...
      buildJunctionExtensions(x1d, y1d, branchA, branchB, p);

  Lblock = blkdiag(branchA.L, branchB.L);
  Eblock = [EAA EAB; EBA EBB];
  Rblock = blkdiag(branchA.R, branchB.R);
  M = lapsharp_unordered(Lblock, Eblock, Rblock);

  ICPM2009BANDINGCHECKS = 0;

  [uExactInA, fInA] = piriformSolution(branchA.phiIn, arc);
  [uExactInB, fInB] = piriformSolution(branchB.phiIn, arc);
  rhs = [fInA; fInB];

  A = speye(size(M)) - M;
  u = A \ rhs;

  nA = length(branchA.innerband);
  uA = u(1:nA);
  uB = u(nA+1:end);

  bandErr = [uA - uExactInA; uB - uExactInB];

  % surface errors by arclength-weighted midpoint quadrature per branch
  nquad = 4000;
  [errA, sQA, uQA, exactQA, wA] = branchSurfaceValues(x1d, y1d, ...
      branchA, uA, [0 pi/2], nquad, arc, p);
  [errB, sQB, uQB, exactQB, wB] = branchSurfaceValues(x1d, y1d, ...
      branchB, uB, [pi/2 pi], nquad, arc, p);

  errorLinf = max(norm(errA, inf), norm(errB, inf));
  errorL2 = sqrt((sum(wA.*errA.^2) + sum(wB.*errB.^2)) / ...
                 (sum(wA) + sum(wB)));

  % duplicated branch values at the two junctions should agree
  EjnA = interp2_matrix(x1d, y1d, [0; 1], [0; 0], p, branchA.innerband);
  EjnB = interp2_matrix(x1d, y1d, [0; 1], [0; 0], p, branchB.innerband);
  jnMismatch = abs(EjnA*uA - EjnB*uB);

  rowSums = full(sum(Eblock, 2));
  maxRowSumErr = max(abs(rowSums - 1));

  fprintf('\nh = %g, grid %d x %d, unknowns A/B: %d / %d\n', ...
          h, length(x1d), length(y1d), nA, length(uB));
  fprintf('Cross rows at cusp A/B: %d / %d, at smooth junction A/B: %d / %d\n', ...
          junctionInfo(1).numCrossA, junctionInfo(1).numCrossB, ...
          junctionInfo(2).numCrossA, junctionInfo(2).numCrossB);
  fprintf('Rotation angle A->B at cusp: %.4f rad, at smooth junction: %.4f rad\n', ...
          junctionInfo(1).thetaAtoB, junctionInfo(2).thetaAtoB);
  fprintf('Junction mismatch cusp/smooth: %g / %g\n', ...
          jnMismatch(1), jnMismatch(2));
  fprintf('E row-sum max error: %g, band L_inf error: %g\n', ...
          maxRowSumErr, norm(bandErr, inf));

  if (makePlots)
    plotLevelFigures(h, branchA, branchB, uA, uB, ...
                     sQA, sQB, uQA, uQB, exactQA, exactQB, errA, errB);
  end

  level = emptyLevelResult();
  level.h = h;
  level.numPoints = nA + length(uB);
  level.errorLinf = errorLinf;
  level.errorL2 = errorL2;
  level.bandErrorLinf = norm(bandErr, inf);
  level.mismatchCusp = jnMismatch(1);
  level.mismatchSmooth = jnMismatch(2);
  level.maxRowSumErr = maxRowSumErr;
  level.thetaCuspAtoB = junctionInfo(1).thetaAtoB;
  level.thetaSmoothAtoB = junctionInfo(2).thetaAtoB;
end


function [gx, gy] = piriformPoint(phi)

  s = sin(phi);
  c = cos(phi);
  gx = s.^2;
  gy = s.^3.*c;
end


function [gx, gy, dgx, dgy, ddgx, ddgy] = piriformDerivs(phi)

  s = sin(phi);
  c = cos(phi);
  gx = s.^2;
  gy = s.^3.*c;
  dgx = 2*s.*c;
  dgy = 3*s.^2.*c.^2 - s.^4;
  ddgx = 2*(c.^2 - s.^2);
  ddgy = 6*s.*c.^3 - 10*s.^3.*c;
end


function [cpx, cpy, dist, bdy, phi] = cpPiriformBranch(x, y, branchSign)
%CPPIRIFORMBRANCH  Closest point to one branch of y^2 = x^3 - x^4.
%   branchSign = +1: upper branch (y >= 0), phi in [0, pi/2].
%   branchSign = -1: lower branch (y <= 0), phi in [pi/2, pi].
%   bdy = 1 marks closest points at the cusp (0,0), bdy = 2 at (1,0).
%   phi is the parameter on the full curve, phi in [0, pi].
%
%   The lower branch is the mirror image y -> -y of the upper branch, so
%   the search always runs on the upper branch with y mirrored.

  sz = size(x);
  x = x(:);
  y = branchSign*y(:);

  lo = 0;
  hi = pi/2;

  % dense samples give a global initial guess for Newton's method
  M = 600;
  phiSamples = linspace(lo, hi, M).';
  [xs, ys] = piriformPoint(phiSamples.');

  n = length(x);
  phi = zeros(n, 1);
  minddGuess = zeros(n, 1);
  chunk = 5000;
  for i1 = 1:chunk:n
    i2 = min(i1 + chunk - 1, n);
    dd = bsxfun(@minus, x(i1:i2), xs).^2 + ...
         bsxfun(@minus, y(i1:i2), ys).^2;
    [minddGuess(i1:i2), jmin] = min(dd, [], 2);
    phi(i1:i2) = phiSamples(jmin);
  end

  % clamped Newton iteration on g(phi) = (gamma(phi) - x) . gamma'(phi).
  % The step is capped at twice the sample spacing so a degenerate
  % g'(phi) (near the cusp or a focal point) cannot throw a point out of
  % the basin found by the dense sampling.
  maxStep = 2*(hi - lo)/(M - 1);
  for iter = 1:30
    [gx, gy, dgx, dgy, ddgx, ddgy] = piriformDerivs(phi);
    g = (gx - x).*dgx + (gy - y).*dgy;
    gp = dgx.^2 + dgy.^2 + (gx - x).*ddgx + (gy - y).*ddgy;
    step = g ./ gp;
    step(~isfinite(step)) = 0;
    step = max(min(step, maxStep), -maxStep);
    phi = min(max(phi - step, lo), hi);
  end

  % phi = 0 is a critical point of the distance for every query point
  % (gamma'(0) = 0), so cusp closest points are detected by proximity:
  % gamma(cuspTol) is within 1e-10 of the cusp, while genuine interior
  % closest points of band grid points have phi = O(sqrt(h)).
  cuspTol = 1e-5;
  clampTol = 1e-9;
  bdy = zeros(n, 1);
  bdy(phi <= lo + cuspTol) = 1;
  bdy(phi >= hi - clampTol) = 2;
  phi(bdy == 1) = lo;
  phi(bdy == 2) = hi;

  [cpx, cpy] = piriformPoint(phi);
  cpx(bdy == 1) = 0;  cpy(bdy == 1) = 0;
  cpx(bdy == 2) = 1;  cpy(bdy == 2) = 0;
  dist = sqrt((x - cpx).^2 + (y - cpy).^2);

  if (any(dist.^2 > minddGuess + 1e-9))
    error('cpPiriformBranch: Newton refinement worsened a closest point.');
  end

  % map back to the full-curve parameter and unmirror
  cpy = branchSign*cpy;
  if (branchSign < 0)
    phi = pi - phi;
  end

  cpx = reshape(cpx, sz);
  cpy = reshape(cpy, sz);
  dist = reshape(dist, sz);
  bdy = reshape(bdy, sz);
  phi = reshape(phi, sz);
end


function branch = buildBranchMatrices2d(x1d, y1d, xx, yy, cpx, cpy, ...
                                        dist, bdy, phi, h, bw, p, order)

  bandInit = find(abs(dist) <= bw*h);
  cpxInit = cpx(bandInit);  cpyInit = cpy(bandInit);
  xInit = xx(bandInit);     yInit = yy(bandInit);
  bdyInit = bdy(bandInit);  phiInit = phi(bandInit);

  Etemp = interp2_matrix(x1d, y1d, cpxInit, cpyInit, p);
  [~, j] = find(Etemp);
  innerband = unique(j);

  Ltemp = laplacian_2d_matrix(x1d, y1d, order, innerband, bandInit);
  [~, j] = find(Ltemp);
  outerbandtemp = unique(j);
  outerband = bandInit(outerbandtemp);

  cpxOut = cpxInit(outerbandtemp);  cpyOut = cpyInit(outerbandtemp);
  xOut = xInit(outerbandtemp);      yOut = yInit(outerbandtemp);
  bdyOut = bdyInit(outerbandtemp);  phiOut = phiInit(outerbandtemp);

  L = Ltemp(:, outerbandtemp);
  E = Etemp(outerbandtemp, innerband);
  clear Ltemp Etemp outerbandtemp

  R = sparse([], [], [], length(innerband), length(outerband), ...
             length(innerband));
  for k = 1:length(innerband)
    I = find(outerband == innerband(k));
    R(k, I) = 1;
  end

  branch.L = L;
  branch.E = E;
  branch.R = R;
  branch.innerband = innerband;
  branch.outerband = outerband;
  branch.cpxOut = cpxOut;
  branch.cpyOut = cpyOut;
  branch.xOut = xOut;
  branch.yOut = yOut;
  branch.bdyOut = bdyOut;
  branch.phiOut = phiOut;
  branch.phiIn = R*phiOut;
  branch.xIn = R*xOut;
  branch.yIn = R*yOut;
end


function [EAA, EAB, EBA, EBB, junctionInfo] = ...
    buildJunctionExtensions(x1d, y1d, branchA, branchB, p)

  EAA = branchA.E;
  EBB = branchB.E;
  EAB = sparse(size(EAA, 1), size(EBB, 2));
  EBA = sparse(size(EBB, 1), size(EAA, 2));

  cpfA = @(x, y) cpPiriformBranch(x, y, +1);
  cpfB = @(x, y) cpPiriformBranch(x, y, -1);

  junctions = [0 0; 1 0];   % the cusp, then the smooth junction
  Rpi = [-1 0; 0 -1];

  for jn = 1:2
    junction = junctions(jn, :);

    crossRowsA = find(branchA.bdyOut == jn);
    crossRowsB = find(branchB.bdyOut == jn);
    if (isempty(crossRowsA) || isempty(crossRowsB))
      error('No cross-extension rows found at junction (%g, %g).', ...
            junction(1), junction(2));
    end

    RA_out = angle2d(branchA.xOut(crossRowsA), ...
                     branchA.yOut(crossRowsA), cpfA, junction);
    RB_out = angle2d(branchB.xOut(crossRowsB), ...
                     branchB.yOut(crossRowsB), cpfB, junction);
    if (any(~isfinite([RA_out(:); RB_out(:)])))
      error('angle2d gave non-finite rotations at junction (%g, %g).', ...
            junction(1), junction(2));
    end

    RAtoB = RB_out * Rpi * RA_out.';
    RBtoA = RA_out * Rpi * RB_out.';

    x0 = branchA.cpxOut(crossRowsA);
    y0 = branchA.cpyOut(crossRowsA);
    dx0 = branchA.xOut(crossRowsA) - x0;
    dy0 = branchA.yOut(crossRowsA) - y0;

    xr = x0 + RAtoB(1,1)*dx0 + RAtoB(1,2)*dy0;
    yr = y0 + RAtoB(2,1)*dx0 + RAtoB(2,2)*dy0;

    [cpxAtoB, cpyAtoB] = cpPiriformBranch(xr, yr, -1);
    EAB(crossRowsA, :) = interp2_matrix(x1d, y1d, cpxAtoB, cpyAtoB, ...
                                        p, branchB.innerband);
    EAA(crossRowsA, :) = 0;

    x0 = branchB.cpxOut(crossRowsB);
    y0 = branchB.cpyOut(crossRowsB);
    dx0 = branchB.xOut(crossRowsB) - x0;
    dy0 = branchB.yOut(crossRowsB) - y0;

    xr = x0 + RBtoA(1,1)*dx0 + RBtoA(1,2)*dy0;
    yr = y0 + RBtoA(2,1)*dx0 + RBtoA(2,2)*dy0;

    [cpxBtoA, cpyBtoA] = cpPiriformBranch(xr, yr, +1);
    EBA(crossRowsB, :) = interp2_matrix(x1d, y1d, cpxBtoA, cpyBtoA, ...
                                        p, branchA.innerband);
    EBB(crossRowsB, :) = 0;

    junctionInfo(jn) = struct( ...
        'point', junction, ...
        'numCrossA', length(crossRowsA), ...
        'numCrossB', length(crossRowsB), ...
        'thetaAtoB', atan2(RAtoB(2,1), RAtoB(1,1)), ...
        'thetaBtoA', atan2(RBtoA(2,1), RBtoA(1,1)));
  end
end


function arc = buildArclengthTable()

  nFine = 200001;
  phiFine = linspace(0, pi, nFine).';
  [~, ~, dgx, dgy] = piriformDerivs(phiFine);
  speed = sqrt(dgx.^2 + dgy.^2);

  arc.phi = phiFine;
  arc.s = cumtrapz(phiFine, speed);
  arc.L = arc.s(end);
end


function [u, f] = piriformSolution(phi, arc)

% Two harmonics with incommensurate phases: no symmetry about the x axis
% (s -> L - s) and no mirror symmetry about any other point either.
  s = interp1(arc.phi, arc.s, phi, 'linear');
  kwave = 2*pi/arc.L;
  u1 = cos(kwave*s + 1);
  u2 = sin(2*kwave*s + 0.7);
  u = u1 + 0.5*u2;
  f = (1 + kwave^2)*u1 + 0.5*(1 + 4*kwave^2)*u2;
end


function [err, sQ, uQ, exactQ, w] = branchSurfaceValues(x1d, y1d, ...
    branch, u, phiRange, nquad, arc, p)

  dphi = (phiRange(2) - phiRange(1))/nquad;
  phiQ = phiRange(1) + ((1:nquad).' - 0.5)*dphi;

  [xq, yq] = piriformPoint(phiQ);
  Eq = interp2_matrix(x1d, y1d, xq, yq, p, branch.innerband);
  uQ = Eq*u;
  exactQ = piriformSolution(phiQ, arc);
  err = uQ - exactQ;

  [~, ~, dgx, dgy] = piriformDerivs(phiQ);
  w = sqrt(dgx.^2 + dgy.^2)*dphi;
  sQ = interp1(arc.phi, arc.s, phiQ, 'linear');
end


function plotLevelFigures(h, branchA, branchB, uA, uB, ...
                          sQA, sQB, uQA, uQB, exactQA, exactQB, errA, errB)

  nplot = 500;
  phiPlotA = linspace(0, pi/2, nplot).';
  phiPlotB = linspace(pi/2, pi, nplot).';
  [xpA, ypA] = piriformPoint(phiPlotA);
  [xpB, ypB] = piriformPoint(phiPlotB);

  figure(1);
  plot2d_compdomain([uA; uB], [branchA.xIn; branchB.xIn], ...
                    [branchA.yIn; branchB.yIn], h, h, 1);
  hold on;
  plot(xpA, ypA, 'k-', 'linewidth', 2);
  plot(xpB, ypB, 'k--', 'linewidth', 2);
  title(['embedded domain: piriform y^2 = x^3 - x^4, h = ' num2str(h)]);
  xlabel('x'); ylabel('y');

  figure(2); clf;
  hA = plot(sQA, uQA, 'b-');
  hold on;
  hB = plot(sQB, uQB, 'c-');
  hExact = plot([sQA; sQB], [exactQA; exactQB], 'r--');
  title(['soln of u - u_{ss} = f on the piriform, h = ' num2str(h)]);
  xlabel('arclength s'); ylabel('u');
  legend([hA hB hExact], 'upper branch iCPM', 'lower branch iCPM', ...
         'exact answer', 'Location', 'SouthWest');

  figure(3); clf;
  plot(sQA, errA, 'b-');
  hold on;
  plot(sQB, errB, 'c-');
  title(['error on the piriform, h = ' num2str(h)]);
  xlabel('arclength s'); ylabel('error');
  legend('upper branch', 'lower branch', 'Location', 'SouthWest');
end


function printConvergenceTable(hvals, levelResults)

  errsInf = [levelResults.errorLinf];
  errsL2 = [levelResults.errorL2];

  fprintf('\nConvergence summary\n');
  fprintf('       h     L_inf error    rate      L_2 error    rate\n');
  for k = 1:length(hvals)
    if (k == 1)
      fprintf('%8.4g  %14.6e      --  %14.6e      --\n', ...
              hvals(k), errsInf(k), errsL2(k));
    else
      rateInf = log(errsInf(k-1)/errsInf(k)) / log(hvals(k-1)/hvals(k));
      rateL2 = log(errsL2(k-1)/errsL2(k)) / log(hvals(k-1)/hvals(k));
      fprintf('%8.4g  %14.6e  %6.3f  %14.6e  %6.3f\n', ...
              hvals(k), errsInf(k), rateInf, errsL2(k), rateL2);
    end
  end
end


function plotConvergenceSummary(hvals, levelResults)

  N = [levelResults.numPoints];
  eInf = [levelResults.errorLinf];
  eL2 = [levelResults.errorL2];
  refScale = max(eInf(1), eL2(1));
  ref2 = refScale*(hvals/hvals(1)).^2;
  ref15 = refScale*(hvals/hvals(1)).^1.5;

  figure(100); clf;
  loglog(N, eInf, 'bo-', N, eL2, 'rs-', N, ref15, 'k-.', N, ref2, 'k--');
  grid on;
  xlabel('number of points');
  ylabel('error');
  title('piriform cusp: convergence of u - u_{ss} = f');
  legend('L_\infty error', 'L_2 error', 'O(h^{1.5})', 'O(h^2)', ...
         'Location', 'SouthWest');
end


function level = emptyLevelResult()

  level = struct('h', [], 'numPoints', [], 'errorLinf', [], ...
                 'errorL2', [], 'bandErrorLinf', [], ...
                 'mismatchCusp', [], 'mismatchSmooth', [], ...
                 'maxRowSumErr', [], 'thetaCuspAtoB', [], ...
                 'thetaSmoothAtoB', []);
end


function reset_icpm2009bandingchecks(oldValue)

  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = oldValue;
end
