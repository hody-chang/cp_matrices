%% Laplace equation on the piriform curve bent into 3D
% The piriform y^2 = x^3 - x^4 (see example_piriform_cusp_laplace.m) is
% lifted into 3D: the upper branch (y >= 0) stays in the x-y plane and
% the lower branch (y <= 0) is bent into the x-z plane by the map
% (x, y) -> (x, 0, -y).  The two branches meet at the cusp (0,0,0) and
% at the point (1,0,0).
%
% The bend is an isometry of the curve, so arclength and the exact
% solution are unchanged from the 2D example:
%   u - u_ss = f,   u(s) = cos(2*pi*s/L + 1) + 0.5*sin(4*pi*s/L + 0.7).
%
% The junction geometry changes: at the cusp both branches still leave
% tangentially along +x, but now in perpendicular planes, so the
% unfolding rotation is by about pi around an axis near (0,1,1)/sqrt(2)
% -- an axis determined only by the O(sqrt(h)) co-normal tilts of the
% two branches.  The formerly smooth junction (1,0,0) becomes a right
% angle bend (rotation by about pi/2 around the x axis).  Both rotation
% matrices are estimated numerically by angle3D from cp/cpbar averages,
% replacing the planar angle2d construction of the 2D example.
%
% Observed convergence over h = 1/40, 1/80, 1/160 sits between O(h^1.5)
% and O(h^2), as in the 2D cusp study: L_inf rates 1.8 -> 2.0 and L_2
% rates 1.7 -> 1.9.  The estimated cusp rotation angle approaches pi
% with an O(sqrt(h)) error (2.94, 2.96, 3.02 rad), while the bend angle
% converges quickly to pi/2 (within 1.2e-4 at the finest grid).


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

% 3D grids: much coarser than the 2D study for affordability
hvals = 1./[40 80 160 320 640];

p = 3;            % interpolation order
order = 2;        % Laplacian order

arc = buildArclengthTable();

fprintf('\nBent piriform curve in 3D, total arclength L = %g\n', arc.L);

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

  dim = 3;
  fdStenrad = order/2;
  bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ...
                   ((fdStenrad+(p+1)/2)^2));

  % the bent curve lives in [0,1] x [0, 3*sqrt(3)/16] x [0, 3*sqrt(3)/16]
  x1d = (-0.2:h:1.2).';
  y1d = (-0.2:h:0.5).';
  z1d = (-0.2:h:0.5).';
  [xx, yy, zz] = meshgrid(x1d, y1d, z1d);

  [cpxA, cpyA, cpzA, distA, bdyA, phiA] = cpPiriformBranch3d(xx, yy, zz, +1);
  branchA = buildBranchMatrices3d(x1d, y1d, z1d, xx, yy, zz, ...
                                  cpxA, cpyA, cpzA, distA, bdyA, phiA, ...
                                  h, bw, p, order);
  clear cpxA cpyA cpzA distA bdyA phiA

  [cpxB, cpyB, cpzB, distB, bdyB, phiB] = cpPiriformBranch3d(xx, yy, zz, -1);
  branchB = buildBranchMatrices3d(x1d, y1d, z1d, xx, yy, zz, ...
                                  cpxB, cpyB, cpzB, distB, bdyB, phiB, ...
                                  h, bw, p, order);
  clear cpxB cpyB cpzB distB bdyB phiB xx yy zz

  [EAA, EAB, EBA, EBB, junctionInfo] = ...
      buildJunctionExtensions(x1d, y1d, z1d, branchA, branchB, p);

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
  [errA, sQA, uQA, exactQA, wA] = branchSurfaceValues(x1d, y1d, z1d, ...
      branchA, uA, [0 pi/2], nquad, arc, p);
  [errB, sQB, uQB, exactQB, wB] = branchSurfaceValues(x1d, y1d, z1d, ...
      branchB, uB, [pi/2 pi], nquad, arc, p);

  errorLinf = max(norm(errA, inf), norm(errB, inf));
  errorL2 = sqrt((sum(wA.*errA.^2) + sum(wB.*errB.^2)) / ...
                 (sum(wA) + sum(wB)));

  % duplicated branch values at the two junctions should agree
  EjnA = interp3_matrix(x1d, y1d, z1d, [0; 1], [0; 0], [0; 0], p, ...
                        branchA.innerband);
  EjnB = interp3_matrix(x1d, y1d, z1d, [0; 1], [0; 0], [0; 0], p, ...
                        branchB.innerband);
  jnMismatch = abs(EjnA*uA - EjnB*uB);

  rowSums = full(sum(Eblock, 2));
  maxRowSumErr = max(abs(rowSums - 1));

  fprintf('\nh = %g, grid %d x %d x %d, unknowns A/B: %d / %d\n', ...
          h, length(x1d), length(y1d), length(z1d), nA, length(uB));
  fprintf('Cross rows at cusp A/B: %d / %d, at bend junction A/B: %d / %d\n', ...
          junctionInfo(1).numCrossA, junctionInfo(1).numCrossB, ...
          junctionInfo(2).numCrossA, junctionInfo(2).numCrossB);
  fprintf('Rotation angle A->B at cusp: %.4f rad, at bend junction: %.4f rad\n', ...
          junctionInfo(1).thetaAtoB, junctionInfo(2).thetaAtoB);
  fprintf('Junction mismatch cusp/bend: %g / %g\n', ...
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
  level.mismatchBend = jnMismatch(2);
  level.maxRowSumErr = maxRowSumErr;
  level.thetaCuspAtoB = junctionInfo(1).thetaAtoB;
  level.thetaBendAtoB = junctionInfo(2).thetaAtoB;
end


function [gx, gy] = piriformPoint(phi)

  s = sin(phi);
  c = cos(phi);
  gx = s.^2;
  gy = s.^3.*c;
end


function [px, py, pz] = piriformPoint3d(phi)

% upper branch (phi <= pi/2) in the x-y plane, bent lower branch in the
% x-z plane
  [gx, gy] = piriformPoint(phi);
  upper = (phi <= pi/2);
  px = gx;
  py = gy.*upper;
  pz = -gy.*(~upper);
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


function [cpx, cpy, bdy, phi] = cpPiriformUpper(x, y)
%CPPIRIFORMUPPER  Planar closest point to the upper piriform branch.
%   Upper branch of y^2 = x^3 - x^4, phi in [0, pi/2].  Column inputs.
%   bdy = 1 marks closest points at the cusp (0,0), bdy = 2 at (1,0).

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

  dist2 = (x - cpx).^2 + (y - cpy).^2;
  if (any(dist2 > minddGuess + 1e-9))
    error('cpPiriformUpper: Newton refinement worsened a closest point.');
  end
end


function [cpx, cpy, cpz, dist, bdy, phi] = cpPiriformBranch3d(x, y, z, ...
                                                              branchSign)
%CPPIRIFORMBRANCH3D  Closest point to one branch of the bent piriform.
%   branchSign = +1: upper branch in the x-y plane, phi in [0, pi/2].
%   branchSign = -1: bent lower branch in the x-z plane, phi in [pi/2, pi].
%   bdy = 1 marks closest points at the cusp (0,0,0), bdy = 2 at (1,0,0).
%   phi is the parameter on the full 2D curve, phi in [0, pi].
%
%   Both branches are the upper-branch shape in their own plane, so the
%   planar closest point in that plane is also the 3D closest point.

  sz = size(x);
  x = x(:);  y = y(:);  z = z(:);

  if (branchSign > 0)
    [cpx, cpy, bdy, phi] = cpPiriformUpper(x, y);
    cpz = zeros(size(cpx));
  else
    [cpx, cpz, bdy, phi] = cpPiriformUpper(x, z);
    cpy = zeros(size(cpx));
    phi = pi - phi;
  end
  dist = sqrt((x - cpx).^2 + (y - cpy).^2 + (z - cpz).^2);

  cpx = reshape(cpx, sz);
  cpy = reshape(cpy, sz);
  cpz = reshape(cpz, sz);
  dist = reshape(dist, sz);
  bdy = reshape(bdy, sz);
  phi = reshape(phi, sz);
end


function branch = buildBranchMatrices3d(x1d, y1d, z1d, xx, yy, zz, ...
                                        cpx, cpy, cpz, dist, bdy, phi, ...
                                        h, bw, p, order)

  bandInit = find(abs(dist) <= bw*h);
  cpxInit = cpx(bandInit);  cpyInit = cpy(bandInit);
  cpzInit = cpz(bandInit);
  xInit = xx(bandInit);     yInit = yy(bandInit);
  zInit = zz(bandInit);
  bdyInit = bdy(bandInit);  phiInit = phi(bandInit);

  Etemp = interp3_matrix(x1d, y1d, z1d, cpxInit, cpyInit, cpzInit, p);
  [~, j] = find(Etemp);
  innerband = unique(j);

  Ltemp = laplacian_3d_matrix(x1d, y1d, z1d, order, innerband, bandInit);
  [~, j] = find(Ltemp);
  outerbandtemp = unique(j);
  outerband = bandInit(outerbandtemp);

  cpxOut = cpxInit(outerbandtemp);  cpyOut = cpyInit(outerbandtemp);
  cpzOut = cpzInit(outerbandtemp);
  xOut = xInit(outerbandtemp);      yOut = yInit(outerbandtemp);
  zOut = zInit(outerbandtemp);
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
  branch.cpzOut = cpzOut;
  branch.xOut = xOut;
  branch.yOut = yOut;
  branch.zOut = zOut;
  branch.bdyOut = bdyOut;
  branch.phiOut = phiOut;
  branch.phiIn = R*phiOut;
  branch.xIn = R*xOut;
  branch.yIn = R*yOut;
  branch.zIn = R*zOut;
end


function [EAA, EAB, EBA, EBB, junctionInfo] = ...
    buildJunctionExtensions(x1d, y1d, z1d, branchA, branchB, p)

  EAA = branchA.E;
  EBB = branchB.E;
  EAB = sparse(size(EAA, 1), size(EBB, 2));
  EBA = sparse(size(EBB, 1), size(EAA, 2));

  cpfA = @(x, y, z) cpPiriformBranch3d(x, y, z, +1);
  cpfB = @(x, y, z) cpPiriformBranch3d(x, y, z, -1);

  junctions = [0 0 0; 1 0 0];   % the cusp, then the right-angle bend

  for jn = 1:2
    junction = junctions(jn, :);

    crossRowsA = find(branchA.bdyOut == jn);
    crossRowsB = find(branchB.bdyOut == jn);
    if (isempty(crossRowsA) || isempty(crossRowsB))
      error('No cross-extension rows found at junction (%g, %g, %g).', ...
            junction(1), junction(2), junction(3));
    end

    xA = branchA.xOut(crossRowsA);
    yA = branchA.yOut(crossRowsA);
    zA = branchA.zOut(crossRowsA);
    xB = branchB.xOut(crossRowsB);
    yB = branchB.yOut(crossRowsB);
    zB = branchB.zOut(crossRowsB);

    % each branch's outward co-normal at the junction, then the rotation
    % mapping it onto the other branch's incoming tangent (-conormal)
    [~, conormalA] = angle3D(xA, yA, zA, cpfA, junction);
    [~, conormalB] = angle3D(xB, yB, zB, cpfB, junction);

    axisAtoB = cross(conormalA, -conormalB);
    [RAtoB, ~, infoAtoB] = angle3D(xA, yA, zA, cpfA, junction, ...
                                   axisAtoB, -conormalB);
    axisBtoA = cross(conormalB, -conormalA);
    [RBtoA, ~, infoBtoA] = angle3D(xB, yB, zB, cpfB, junction, ...
                                   axisBtoA, -conormalA);
    if (any(~isfinite([RAtoB(:); RBtoA(:)])))
      error('angle3D gave non-finite rotations at junction (%g, %g, %g).', ...
            junction(1), junction(2), junction(3));
    end

    x0 = branchA.cpxOut(crossRowsA);
    y0 = branchA.cpyOut(crossRowsA);
    z0 = branchA.cpzOut(crossRowsA);
    dx0 = xA - x0;
    dy0 = yA - y0;
    dz0 = zA - z0;

    xr = x0 + RAtoB(1,1)*dx0 + RAtoB(1,2)*dy0 + RAtoB(1,3)*dz0;
    yr = y0 + RAtoB(2,1)*dx0 + RAtoB(2,2)*dy0 + RAtoB(2,3)*dz0;
    zr = z0 + RAtoB(3,1)*dx0 + RAtoB(3,2)*dy0 + RAtoB(3,3)*dz0;

    [cpxAtoB, cpyAtoB, cpzAtoB] = cpfB(xr, yr, zr);
    EAB(crossRowsA, :) = interp3_matrix(x1d, y1d, z1d, ...
                                        cpxAtoB, cpyAtoB, cpzAtoB, ...
                                        p, branchB.innerband);
    EAA(crossRowsA, :) = 0;

    x0 = branchB.cpxOut(crossRowsB);
    y0 = branchB.cpyOut(crossRowsB);
    z0 = branchB.cpzOut(crossRowsB);
    dx0 = xB - x0;
    dy0 = yB - y0;
    dz0 = zB - z0;

    xr = x0 + RBtoA(1,1)*dx0 + RBtoA(1,2)*dy0 + RBtoA(1,3)*dz0;
    yr = y0 + RBtoA(2,1)*dx0 + RBtoA(2,2)*dy0 + RBtoA(2,3)*dz0;
    zr = z0 + RBtoA(3,1)*dx0 + RBtoA(3,2)*dy0 + RBtoA(3,3)*dz0;

    [cpxBtoA, cpyBtoA, cpzBtoA] = cpfA(xr, yr, zr);
    EBA(crossRowsB, :) = interp3_matrix(x1d, y1d, z1d, ...
                                        cpxBtoA, cpyBtoA, cpzBtoA, ...
                                        p, branchA.innerband);
    EBB(crossRowsB, :) = 0;

    junctionInfo(jn) = struct( ...
        'point', junction, ...
        'numCrossA', length(crossRowsA), ...
        'numCrossB', length(crossRowsB), ...
        'conormalA', conormalA, ...
        'conormalB', conormalB, ...
        'thetaAtoB', infoAtoB.theta, ...
        'thetaBtoA', infoBtoA.theta);
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


function [err, sQ, uQ, exactQ, w] = branchSurfaceValues(x1d, y1d, z1d, ...
    branch, u, phiRange, nquad, arc, p)

  dphi = (phiRange(2) - phiRange(1))/nquad;
  phiQ = phiRange(1) + ((1:nquad).' - 0.5)*dphi;

  [xq, yq, zq] = piriformPoint3d(phiQ);
  Eq = interp3_matrix(x1d, y1d, z1d, xq, yq, zq, p, branch.innerband);
  uQ = Eq*u;
  exactQ = piriformSolution(phiQ, arc);
  err = uQ - exactQ;

  % the bend is an isometry, so the 2D speed is the 3D speed
  [~, ~, dgx, dgy] = piriformDerivs(phiQ);
  w = sqrt(dgx.^2 + dgy.^2)*dphi;
  sQ = interp1(arc.phi, arc.s, phiQ, 'linear');
end


function plotLevelFigures(h, branchA, branchB, uA, uB, ...
                          sQA, sQB, uQA, uQB, exactQA, exactQB, errA, errB)

  nplot = 500;
  phiPlotA = linspace(0, pi/2, nplot).';
  phiPlotB = linspace(pi/2, pi, nplot).';
  [xpA, ypA, zpA] = piriformPoint3d(phiPlotA);
  [xpB, ypB, zpB] = piriformPoint3d(phiPlotB);

  figure(1); clf;
  scatter3([branchA.xIn; branchB.xIn], [branchA.yIn; branchB.yIn], ...
           [branchA.zIn; branchB.zIn], 12, [uA; uB], 'filled');
  hold on;
  plot3(xpA, ypA, zpA, 'k-', 'linewidth', 2);
  plot3(xpB, ypB, zpB, 'k--', 'linewidth', 2);
  plot3([0; 1], [0; 0], [0; 0], 'ro', 'markerfacecolor', 'r');
  axis equal; grid on;
  xlabel('x'); ylabel('y'); zlabel('z');
  colorbar;
  title(['computational bands: bent piriform, h = ' num2str(h)]);

  figure(2); clf;
  hA = plot(sQA, uQA, 'b-');
  hold on;
  hB = plot(sQB, uQB, 'c-');
  hExact = plot([sQA; sQB], [exactQA; exactQB], 'r--');
  title(['soln of u - u_{ss} = f on the bent piriform, h = ' num2str(h)]);
  xlabel('arclength s'); ylabel('u');
  legend([hA hB hExact], 'x-y branch iCPM', 'x-z branch iCPM', ...
         'exact answer', 'Location', 'SouthWest');

  figure(3); clf;
  plot(sQA, errA, 'b-');
  hold on;
  plot(sQB, errB, 'c-');
  title(['error on the bent piriform, h = ' num2str(h)]);
  xlabel('arclength s'); ylabel('error');
  legend('x-y branch', 'x-z branch', 'Location', 'SouthWest');
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
  title('bent piriform cusp: convergence of u - u_{ss} = f');
  legend('L_\infty error', 'L_2 error', 'O(h^{1.5})', 'O(h^2)', ...
         'Location', 'SouthWest');
end


function level = emptyLevelResult()

  level = struct('h', [], 'numPoints', [], 'errorLinf', [], ...
                 'errorL2', [], 'bandErrorLinf', [], ...
                 'mismatchCusp', [], 'mismatchBend', [], ...
                 'maxRowSumErr', [], 'thetaCuspAtoB', [], ...
                 'thetaBendAtoB', []);
end


function reset_icpm2009bandingchecks(oldValue)

  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = oldValue;
end
