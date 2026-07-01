%% Heat equation on a V-shaped line with branch rotation at the vertex
% This example solves the heat equation on two line-segment branches joined
% at a vertex.  The vertex is duplicated, one value per branch.  Vertex
% extension rows are rotated into the other branch, and the two vertex
% values are averaged when a single physical vertex value is needed.
%
% The exact solution uses the unfolded arclength r in [0,2L],
%   u(t,r) = exp(-lambda*t)*cos(pi*r/(2L)),
% lambda = (pi/(2L))^2.  This has Neumann boundary conditions at the two
% physical endpoints r=0 and r=2L.


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

angleDegrees = [90 120];
hvals = 1./[200 400 800 1600];

vertex = [0.5 0.5];
branchLength = 0.4;

p = 3;        % interpolation order
order = 2;    % Laplacian order
Tf = 0.01;

allResults = struct([]);


%% Convergence study

for angleNumber = 1:length(angleDegrees)
  angleDeg = angleDegrees(angleNumber);
  levelResults = repmat(emptyLevelResult(), length(hvals), 1);

  fprintf('\nV-shaped line, angle = %g degrees\n', angleDeg);

  for levelNumber = 1:length(hvals)
    h = hvals(levelNumber);
    makeLevelPlots = makePlots && (levelNumber == length(hvals));

    levelResults(levelNumber) = solveOneLevel(h, angleDeg, vertex, ...
                                              branchLength, p, order, Tf, ...
                                              makeLevelPlots, angleNumber);

    fprintf(['h = %g, points = %d, L_inf error = %g, ' ...
             'L_2 error = %g\n'], ...
            h, levelResults(levelNumber).numPoints, ...
            levelResults(levelNumber).errorLinf, ...
            levelResults(levelNumber).errorL2);
  end

  allResults(angleNumber).angleDeg = angleDeg;
  allResults(angleNumber).h = hvals;
  allResults(angleNumber).numPoints = [levelResults.numPoints];
  allResults(angleNumber).errorsLinf = [levelResults.errorLinf];
  allResults(angleNumber).errorsL2 = [levelResults.errorL2];
  allResults(angleNumber).levels = levelResults;
end

if (makePlots)
  plotConvergenceSummary(allResults);
end


function level = solveOneLevel(h, angleDeg, vertex, branchLength, p, order, ...
                               Tf, makePlots, angleNumber)

  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = 1;

  alpha = angleDeg*pi/180;
  dA = [-sin(alpha/2) -cos(alpha/2)];
  dB = [ sin(alpha/2) -cos(alpha/2)];

  endpointA = vertex + branchLength*dA;
  endpointB = vertex + branchLength*dB;

  % Branch A is oriented from its physical endpoint to the vertex.
  % Branch B is oriented from the vertex to its physical endpoint.
  pA = endpointA;  qA = vertex;
  pB = vertex;     qB = endpointB;

  dim = 2;
  fdStenrad = order/2;
  bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ...
                   ((fdStenrad+(p+1)/2)^2));

  x1d = (0:h:1).';
  y1d = x1d;
  [xx, yy] = meshgrid(x1d, y1d);

  [cpxA, cpyA, distA, bdyA, sA] = cpLineSegment2d(xx, yy, pA, qA);
  [cpxB, cpyB, distB, bdyB, sB] = cpLineSegment2d(xx, yy, pB, qB);

  endpointCpfA = @(x, y) cpLineSegmentEndpointBoundary2d(x, y, pA, qA, 1);
  endpointCpfB = @(x, y) cpLineSegmentEndpointBoundary2d(x, y, pB, qB, 2);
  [cpxExtA, cpyExtA] = cpbar_2d(xx, yy, endpointCpfA);
  [cpxExtB, cpyExtB] = cpbar_2d(xx, yy, endpointCpfB);

  branchA = buildBranchMatrices2d(x1d, y1d, xx, yy, cpxA, cpyA, ...
                                  cpxExtA, cpyExtA, distA, bdyA, sA, ...
                                  h, bw, p, order);
  branchB = buildBranchMatrices2d(x1d, y1d, xx, yy, cpxB, cpyB, ...
                                  cpxExtB, cpyExtB, distB, bdyB, sB, ...
                                  h, bw, p, order);

  [EAA, EAB, EBA, EBB, crossRowsA, crossRowsB] = ...
      buildVertexExtension(x1d, y1d, branchA, branchB, pA, qA, pB, qB, p);

  Lblock = blkdiag(branchA.L, branchB.L);
  Eblock = [EAA EAB; EBA EBB];
  Rblock = blkdiag(branchA.R, branchB.R);
  M = lapsharp_unordered(Lblock, Eblock, Rblock);

  ICPM2009BANDINGCHECKS = 0;

  rA = branchLength*branchA.sIn;
  rB = branchLength + branchLength*branchB.sIn;

  u0A = exactHeatSolution(0, rA, branchLength);
  u0B = exactHeatSolution(0, rB, branchLength);
  u = [u0A; u0B];

  vertexA = setupVertexAveraging(x1d, y1d, xx, yy, branchA.innerband, ...
                                 vertex, p);
  vertexB = setupVertexAveraging(x1d, y1d, xx, yy, branchB.innerband, ...
                                 vertex, p);

  % Crank-Nicolson has O(dt^2) time error; dt=O(h) keeps it aligned with
  % the second-order spatial convergence study.
  dt = h/4;
  numtimesteps = ceil(Tf/dt);
  dt = Tf / numtimesteps;

  I = speye(size(M));
  A = I - 0.5*dt*M;
  B = I + 0.5*dt*M;

  for kt = 1:numtimesteps
    u = A \ (B*u);

    uA = u(1:length(u0A));
    uB = u(length(u0A)+1:end);
    [uA, uB] = averageVertexValues(uA, uB, vertexA, vertexB);
    u = [uA; uB];
  end

  t = numtimesteps*dt;
  uA = u(1:length(u0A));
  uB = u(length(u0A)+1:end);

  nplot = 500;
  [xpA, ypA] = paramLineSegment2d(nplot, pA, qA);
  [xpB, ypB] = paramLineSegment2d(nplot, pB, qB);
  sPlot = linspace(0, 1, nplot).';
  rPlotA = branchLength*sPlot;
  rPlotB = branchLength + branchLength*sPlot;

  EplotA = interp2_matrix(x1d, y1d, xpA, ypA, p, branchA.innerband);
  EplotB = interp2_matrix(x1d, y1d, xpB, ypB, p, branchB.innerband);

  plotA = EplotA*uA;
  plotB = EplotB*uB;
  initialPlotA = EplotA*u0A;
  initialPlotB = EplotB*u0B;

  vertexAvg = 0.5*(plotA(end) + plotB(1));
  plotA(end) = vertexAvg;
  plotB(1) = vertexAvg;

  exactPlotA = exactHeatSolution(t, rPlotA, branchLength);
  exactPlotB = exactHeatSolution(t, rPlotB, branchLength);

  err = [plotA - exactPlotA; plotB - exactPlotB];
  errorLinf = norm(err, inf);
  errorL2 = sqrt(mean(err.^2));

  if (makePlots)
    plotLevelFigures(angleNumber, angleDeg, h, t, branchA, branchB, ...
                     uA, uB, xpA, ypA, xpB, ypB, rPlotA, rPlotB, ...
                     plotA, plotB, initialPlotA, initialPlotB, ...
                     exactPlotA, exactPlotB, crossRowsA, crossRowsB);
  end

  level = emptyLevelResult();
  level.h = h;
  level.numPoints = length(uA) + length(uB);
  level.errorLinf = errorLinf;
  level.errorL2 = errorL2;
  level.crossRowsA = length(crossRowsA);
  level.crossRowsB = length(crossRowsB);
end


function branch = buildBranchMatrices2d(x1d, y1d, xx, yy, cpx, cpy, ...
                                        cpxExt, cpyExt, dist, bdy, s, ...
                                        h, bw, p, order)

  bandInit = find(abs(dist) <= bw*h);
  cpxInit = cpx(bandInit);  cpyInit = cpy(bandInit);
  cpxExtInit = cpxExt(bandInit);  cpyExtInit = cpyExt(bandInit);
  xInit = xx(bandInit);     yInit = yy(bandInit);
  bdyInit = bdy(bandInit);  sInit = s(bandInit);

  Etemp = interp2_matrix(x1d, y1d, cpxExtInit, cpyExtInit, p);
  [~, j] = find(Etemp);
  innerband = unique(j);

  Ltemp = laplacian_2d_matrix(x1d, y1d, order, innerband, bandInit);
  [~, j] = find(Ltemp);
  outerbandtemp = unique(j);
  outerband = bandInit(outerbandtemp);

  cpxOut = cpxInit(outerbandtemp);  cpyOut = cpyInit(outerbandtemp);
  xOut = xInit(outerbandtemp);      yOut = yInit(outerbandtemp);
  bdyOut = bdyInit(outerbandtemp);  sOut = sInit(outerbandtemp);

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
  branch.sOut = sOut;
  branch.sIn = R*sOut;
  branch.xIn = R*xOut;
  branch.yIn = R*yOut;
end


function [EAA, EAB, EBA, EBB, crossRowsA, crossRowsB] = ...
    buildVertexExtension(x1d, y1d, branchA, branchB, pA, qA, pB, qB, p)

  EAA = branchA.E;
  EBB = branchB.E;
  EAB = sparse(size(EAA, 1), size(EBB, 2));
  EBA = sparse(size(EBB, 1), size(EAA, 2));

  endpointTol = 100*eps(1);

  endpointIdA = branchA.bdyOut;
  endpointIdA(hypot(branchA.cpxOut - pA(1), branchA.cpyOut - pA(2)) <= ...
              endpointTol) = 1;
  endpointIdA(hypot(branchA.cpxOut - qA(1), branchA.cpyOut - qA(2)) <= ...
              endpointTol) = 2;

  endpointIdB = branchB.bdyOut;
  endpointIdB(hypot(branchB.cpxOut - pB(1), branchB.cpyOut - pB(2)) <= ...
              endpointTol) = 1;
  endpointIdB(hypot(branchB.cpxOut - qB(1), branchB.cpyOut - qB(2)) <= ...
              endpointTol) = 2;

  % Only the shared vertex is glued.  The two physical endpoints keep their
  % same-branch cpbar extension, giving Neumann endpoint conditions.
  crossRowsA = find(endpointIdA == 2);
  crossRowsB = find(endpointIdB == 1);

  cpfA = @(x, y) cpLineSegment2d(x, y, pA, qA);
  cpfB = @(x, y) cpLineSegment2d(x, y, pB, qB);
  RA_out = angle2d(branchA.xOut, branchA.yOut, cpfA, qA);
  RB_out = angle2d(branchB.xOut, branchB.yOut, cpfB, pB);
  Rpi = [-1 0; 0 -1];
  RAtoB = RB_out * Rpi * RA_out.';
  RBtoA = RA_out * Rpi * RB_out.';

  if (~isempty(crossRowsA))
    x0 = branchA.cpxOut(crossRowsA);
    y0 = branchA.cpyOut(crossRowsA);
    dx0 = branchA.xOut(crossRowsA) - x0;
    dy0 = branchA.yOut(crossRowsA) - y0;

    xr = x0 + RAtoB(1,1)*dx0 + RAtoB(1,2)*dy0;
    yr = y0 + RAtoB(2,1)*dx0 + RAtoB(2,2)*dy0;

    [cpxAtoB, cpyAtoB] = cpLineSegment2d(xr, yr, pB, qB);
    EAB(crossRowsA, :) = interp2_matrix(x1d, y1d, cpxAtoB, cpyAtoB, ...
                                        p, branchB.innerband);
    EAA(crossRowsA, :) = 0;
  end

  if (~isempty(crossRowsB))
    x0 = branchB.cpxOut(crossRowsB);
    y0 = branchB.cpyOut(crossRowsB);
    dx0 = branchB.xOut(crossRowsB) - x0;
    dy0 = branchB.yOut(crossRowsB) - y0;

    xr = x0 + RBtoA(1,1)*dx0 + RBtoA(1,2)*dy0;
    yr = y0 + RBtoA(2,1)*dx0 + RBtoA(2,2)*dy0;

    [cpxBtoA, cpyBtoA] = cpLineSegment2d(xr, yr, pA, qA);
    EBA(crossRowsB, :) = interp2_matrix(x1d, y1d, cpxBtoA, cpyBtoA, ...
                                        p, branchA.innerband);
    EBB(crossRowsB, :) = 0;
  end
end


function [cpx, cpy, dist, bdy] = ...
    cpLineSegmentEndpointBoundary2d(x, y, p, q, endpointId)

  [cpx, cpy, dist, bdy] = cpLineSegment2d(x, y, p, q);
  bdy(bdy ~= endpointId) = 0;
end


function vertexInfo = setupVertexAveraging(x1d, y1d, xx, yy, innerband, ...
                                           vertex, p)

  vertexInfo.E = interp2_matrix(x1d, y1d, vertex(1), vertex(2), p, ...
                                innerband);

  xInner = xx(innerband);
  yInner = yy(innerband);
  nodeTol = (x1d(2) - x1d(1))*1e-8;

  I = find((abs(xInner - vertex(1)) <= nodeTol) & ...
           (abs(yInner - vertex(2)) <= nodeTol));
  if (isempty(I))
    [dummy, I] = min(hypot(xInner - vertex(1), yInner - vertex(2)));
  end

  vertexInfo.writeIdx = I(1);
end


function [uA, uB] = averageVertexValues(uA, uB, vertexA, vertexB)

  vertexValueA = vertexA.E*uA;
  vertexValueB = vertexB.E*uB;
  vertexAvg = 0.5*(vertexValueA + vertexValueB);

  uA(vertexA.writeIdx) = vertexAvg;
  uB(vertexB.writeIdx) = vertexAvg;
end


function u = exactHeatSolution(t, r, branchLength)

  lambda = (pi/(2*branchLength))^2;
  u = exp(-lambda*t)*cos(pi*r/(2*branchLength));
end


function plotLevelFigures(angleNumber, angleDeg, h, t, branchA, branchB, ...
                          uA, uB, xpA, ypA, xpB, ypB, rPlotA, rPlotB, ...
                          plotA, plotB, initialPlotA, initialPlotB, ...
                          exactPlotA, exactPlotB, crossRowsA, crossRowsB)

  figBase = 10*(angleNumber - 1) + 1;

  figure(figBase);
  plot2d_compdomain([uA; uB], [branchA.xIn; branchB.xIn], ...
                    [branchA.yIn; branchB.yIn], h, h, figBase);
  hold on;
  plot(xpA, ypA, 'k-', 'linewidth', 2);
  plot(xpB, ypB, 'k--', 'linewidth', 2);
  title(['embedded domain: angle ' num2str(angleDeg) ...
         ' degrees, t = ' num2str(t)]);
  xlabel('x'); ylabel('y');

  figure(figBase + 1); clf;
  hA = plot(rPlotA, plotA, 'b-');
  hold on;
  hB = plot(rPlotB, plotB, 'c-');
  hExact = plot([rPlotA; rPlotB], [exactPlotA; exactPlotB], 'r--');
  hInitial = plot([rPlotA; rPlotB], [initialPlotA; initialPlotB], 'g-.');
  title(['soln at time ' num2str(t) ', V angle ' ...
         num2str(angleDeg) ' degrees']);
  xlabel('unfolded arclength r'); ylabel('u');
  legend([hA hB hExact hInitial], ...
         'branch A iCPM', 'branch B iCPM', 'exact answer', ...
         'initial condition', 'Location', 'SouthWest');

  figure(figBase + 2); clf;
  plot(rPlotA, plotA - exactPlotA, 'b-');
  hold on;
  plot(rPlotB, plotB - exactPlotB, 'c-');
  title(['error at time ' num2str(t) ', V angle ' ...
         num2str(angleDeg) ' degrees']);
  xlabel('unfolded arclength r'); ylabel('error');
  legend('branch A', 'branch B', 'Location', 'SouthWest');

  fprintf('Cross-branch extension rows A->B / B->A: %d / %d\n', ...
          length(crossRowsA), length(crossRowsB));
end


function plotConvergenceSummary(allResults)

  figure(100); clf;

  for k = 1:length(allResults)
    N = allResults(k).numPoints;
    eInf = allResults(k).errorsLinf;
    eL2 = allResults(k).errorsL2;
    refScale = max(eInf(1), eL2(1));
    ref = refScale*(N(1)./N).^2;

    subplot(1, length(allResults), k);
    loglog(N, eInf, 'bo-', N, eL2, 'rs-', N, ref, 'k--');
    grid on;
    xlabel('number of points');
    ylabel('error');
    title(['V angle ' num2str(allResults(k).angleDeg) ' degrees']);
    legend('L_\infty error', 'L_2 error', 'O(h^2)', ...
           'Location', 'SouthWest');
  end
end


function level = emptyLevelResult()

  level = struct('h', [], 'numPoints', [], 'errorLinf', [], ...
                 'errorL2', [], 'crossRowsA', [], 'crossRowsB', []);
end


function reset_icpm2009bandingchecks(oldValue)

  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = oldValue;
end
