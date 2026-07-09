%% Heat equation on a bent circle in 3D: two semicircles in different planes
% A closed 1D curve in 3D built from two unit-radius semicircular
% branches: branch A lies in the x-y plane (x >= 0) and branch B lies in
% the y-z plane (z >= 0).  The branches connect at the two vertices
% (0,1,0) and (0,-1,0), where the curve has a 90 degree kink.
%
% The curve is isometric to a unit circle: with arclength s in [0,2*pi]
% (s = 0 at the bottom vertex (0,-1,0), branch A is s in [0,pi], branch B
% is s in [pi,2*pi]), the heat equation with u0 = cos(s) has exact
% solution u(t) = exp(-t)*cos(s), as in example_heat_circle.m.
%
% Cross-branch extension rows are routed through the opposite branch
% after rotating the embedding point about the shared vertex.  The 3D
% rotation matrices are estimated numerically by angle3D from cp/cpbar
% averages near each vertex.
%
% A convergence study is run over the grids h = 1/20, 1/40, 1/80.  The
% time step dt = 0.2*dx^2 (implicit Euler is unconditionally stable, but
% this keeps the O(dt) time error at the order of the spatial error so
% the spatial convergence rate is visible).


%% Using cp_matrices

% Include the cp_matrices folder (edit as appropriate)
addpath('../cp_matrices');

% add functions for finding the closest points
addpath('../surfaces');


global ICPM2009BANDINGCHECKS
cleanupICPM2009BANDINGCHECKS = onCleanup(@() reset_icpm2009bandingchecks());


%% Parameters shared by all grid levels

doPlots = true;   % evolution plots on the finest level + convergence plot

hvals = 1./[20 40 80 100];   % grid sizes for the convergence study

RADIUS = 1;
cen = [0 0];

% Branch A: semicircle in the x-y plane with x >= 0.
% Branch B: semicircle in the y-z plane with z >= 0.
% cpArc angles are in the plane of each branch.
angleA1 = -pi/2;  angleA2 = pi/2;   % (x,y) angles: endpoints (0,-1), (0,1)
angleB1 = 0;      angleB2 = pi;     % (y,z) angles: endpoints (1,0), (-1,0)

cpfA = @(x,y,z) cpArcInXYPlane(x, y, z, RADIUS, cen, angleA1, angleA2);
cpfB = @(x,y,z) cpArcInYZPlane(x, y, z, RADIUS, cen, angleB1, angleB2);

% shared vertices where the two branches connect
vertices = [0  1  0; ...
            0 -1  0];
% cpArc endpoint ids (bdy=1 at angle1, bdy=2 at angle2) -> vertex row
vertexOfIdA = [2 1];
vertexOfIdB = [1 2];

% banding parameters
dim = 3;  % dimension
p = 3;    % interpolation order
order = 2;  % Laplacian order: bw will need to increase if changed
fd_stenrad = order/2;  % Finite difference stencil radius
bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ((fd_stenrad+(p+1)/2)^2));

Tf = 1.00;
uexactfn = @(t,s) exp(-t)*cos(s);

% plotting points on the curve, arclength parameterization:
% s in [0,pi] on A, [pi,2*pi] on B
nplot = 250;
splotA = linspace(0, pi, nplot)';
xpA = cos(splotA - pi/2);
ypA = sin(splotA - pi/2);
zpA = zeros(size(splotA));

splotB = linspace(pi, 2*pi, nplot)';
xpB = zeros(size(splotB));
ypB = cos(splotB - pi);
zpB = sin(splotB - pi);

errorsLinf = zeros(size(hvals));
errorsL2 = zeros(size(hvals));


%% Loop over grid levels

for level = 1:length(hvals)

  dx = hvals(level);
  plotLevel = doPlots && (level == length(hvals));
  fprintf('\n---- level %d of %d: dx = %g ----\n', level, length(hvals), dx);

  % this is a bit dangerous: will break other less tightly banded
  % codes, turn it off after construction
  ICPM2009BANDINGCHECKS = 1;

  % make vectors of x, y, z positions of the grid
  x1d = (-1.2:dx:1.2)';
  y1d = x1d;
  z1d = x1d;
  dy = dx;
  dz = dx;

  [xx, yy, zz] = meshgrid(x1d, y1d, z1d);


  %% Branch A two-band iCPM matrices

  [cpxA, cpyA, cpzA, distA, bdyA] = cpfA(xx, yy, zz);

  band_initA = find(abs(distA) <= bw*dx);
  cpxg_initA = cpxA(band_initA); cpyg_initA = cpyA(band_initA);
  cpzg_initA = cpzA(band_initA);
  xg_initA = xx(band_initA); yg_initA = yy(band_initA);
  zg_initA = zz(band_initA);
  bdyg_initA = bdyA(band_initA);
  clear cpxA cpyA cpzA distA bdyA

  disp('Constructing branch A interpolation matrix');
  EtempA = interp3_matrix(x1d, y1d, z1d, cpxg_initA, cpyg_initA, cpzg_initA, p);
  [iA,jA,SA] = find(EtempA);
  innerbandA = unique(jA);

  LtempA = laplacian_3d_matrix(x1d, y1d, z1d, order, innerbandA, band_initA);
  [iA,jA,SA] = find(LtempA);
  outerbandtempA = unique(jA);
  outerbandA = band_initA(outerbandtempA);

  cpxgoutA = cpxg_initA(outerbandtempA); cpygoutA = cpyg_initA(outerbandtempA);
  cpzgoutA = cpzg_initA(outerbandtempA);
  xgoutA = xg_initA(outerbandtempA); ygoutA = yg_initA(outerbandtempA);
  zgoutA = zg_initA(outerbandtempA);
  bdygoutA = bdyg_initA(outerbandtempA);

  LA = LtempA(:, outerbandtempA);
  EAA = EtempA(outerbandtempA, innerbandA);
  clear LtempA EtempA outerbandtempA

  innerInOuterA = zeros(size(innerbandA));
  RA = sparse([],[],[],length(innerbandA),length(outerbandA),length(innerbandA));
  for i=1:length(innerbandA)
    I = find(outerbandA == innerbandA(i));
    innerInOuterA(i) = I;
    RA(i,I) = 1;
  end

  cpxginA = RA*cpxgoutA;  cpyginA = RA*cpygoutA;  cpzginA = RA*cpzgoutA;
  xginA = RA*xgoutA;  yginA = RA*ygoutA;  zginA = RA*zgoutA;


  %% Branch B two-band iCPM matrices

  [cpxB, cpyB, cpzB, distB, bdyB] = cpfB(xx, yy, zz);

  band_initB = find(abs(distB) <= bw*dx);
  cpxg_initB = cpxB(band_initB); cpyg_initB = cpyB(band_initB);
  cpzg_initB = cpzB(band_initB);
  xg_initB = xx(band_initB); yg_initB = yy(band_initB);
  zg_initB = zz(band_initB);
  bdyg_initB = bdyB(band_initB);
  clear cpxB cpyB cpzB distB bdyB

  disp('Constructing branch B interpolation matrix');
  EtempB = interp3_matrix(x1d, y1d, z1d, cpxg_initB, cpyg_initB, cpzg_initB, p);
  [iB,jB,SB] = find(EtempB);
  innerbandB = unique(jB);

  LtempB = laplacian_3d_matrix(x1d, y1d, z1d, order, innerbandB, band_initB);
  [iB,jB,SB] = find(LtempB);
  outerbandtempB = unique(jB);
  outerbandB = band_initB(outerbandtempB);

  cpxgoutB = cpxg_initB(outerbandtempB); cpygoutB = cpyg_initB(outerbandtempB);
  cpzgoutB = cpzg_initB(outerbandtempB);
  xgoutB = xg_initB(outerbandtempB); ygoutB = yg_initB(outerbandtempB);
  zgoutB = zg_initB(outerbandtempB);
  bdygoutB = bdyg_initB(outerbandtempB);

  LB = LtempB(:, outerbandtempB);
  EBB = EtempB(outerbandtempB, innerbandB);
  clear LtempB EtempB outerbandtempB

  innerInOuterB = zeros(size(innerbandB));
  RB = sparse([],[],[],length(innerbandB),length(outerbandB),length(innerbandB));
  for i=1:length(innerbandB)
    I = find(outerbandB == innerbandB(i));
    innerInOuterB(i) = I;
    RB(i,I) = 1;
  end

  cpxginB = RB*cpxgoutB;  cpyginB = RB*cpygoutB;  cpzginB = RB*cpzgoutB;
  xginB = RB*xgoutB;  yginB = RB*ygoutB;  zginB = RB*zgoutB;


  %% Vertex rotation matrices estimated with angle3D

  % At each shared vertex, angle3D averages the cp/cpbar vectors of grid
  % points whose closest point is that vertex, giving each branch's
  % outward co-normal.  The A->B rotation maps branch A's outward
  % co-normal onto branch B's incoming tangent (-conormalB), rotating
  % about the axis perpendicular to both.  Exact values here: co-normals
  % (-1,0,0) and (0,0,-1), so |theta| = pi/2 at both vertices.
  RAtoB = cell(2,1);
  RBtoA = cell(2,1);
  for k = 1:2
    P = vertices(k,:);
    [dummy, conormalA] = angle3D(xgoutA, ygoutA, zgoutA, cpfA, P);
    [dummy, conormalB] = angle3D(xgoutB, ygoutB, zgoutB, cpfB, P);

    axisAtoB = cross(conormalA, -conormalB);
    [RAtoB{k}, dummy, infoAtoB] = angle3D(xgoutA, ygoutA, zgoutA, cpfA, P, ...
                                          axisAtoB, -conormalB);
    axisBtoA = cross(conormalB, -conormalA);
    [RBtoA{k}, dummy, infoBtoA] = angle3D(xgoutB, ygoutB, zgoutB, cpfB, P, ...
                                          axisBtoA, -conormalA);

    fprintf('Vertex %d: conormalA = [%g %g %g], conormalB = [%g %g %g]\n', ...
            k, conormalA(1), conormalA(2), conormalA(3), ...
            conormalB(1), conormalB(2), conormalB(3));
    fprintf('Vertex %d: theta A->B = %g, theta B->A = %g (exact pi/2 = %g)\n', ...
            k, infoAtoB.theta, infoBtoA.theta, pi/2);
  end


  %% Vertex ids for cross-branch extension rows

  % cpArc labels off-arc closest points with bdy=1 or bdy=2, but exact
  % endpoint-angle points can still have bdy=0.  Keep this local vertex id
  % for routing and rotation lookup.
  endpointTol = 100*eps(max(1,RADIUS));

  vertexIdA = zeros(size(bdygoutA));
  vertexIdA(bdygoutA ~= 0) = vertexOfIdA(bdygoutA(bdygoutA ~= 0));
  vertexIdB = zeros(size(bdygoutB));
  vertexIdB(bdygoutB ~= 0) = vertexOfIdB(bdygoutB(bdygoutB ~= 0));
  for k = 1:2
    dA = sqrt((cpxgoutA - vertices(k,1)).^2 + (cpygoutA - vertices(k,2)).^2 + ...
              (cpzgoutA - vertices(k,3)).^2);
    vertexIdA(dA <= endpointTol) = k;
    dB = sqrt((cpxgoutB - vertices(k,1)).^2 + (cpygoutB - vertices(k,2)).^2 + ...
              (cpzgoutB - vertices(k,3)).^2);
    vertexIdB(dB <= endpointTol) = k;
  end

  crossRowsA = find(vertexIdA ~= 0);
  crossRowsB = find(vertexIdB ~= 0);


  %% Cross-branch extension rows via vertex rotation

  EAB = sparse(length(outerbandA), length(innerbandB));
  EBA = sparse(length(outerbandB), length(innerbandA));

  for k = 1:2
    rows = crossRowsA(vertexIdA(crossRowsA) == k);
    if (~isempty(rows))
      Rk = RAtoB{k};
      x0 = cpxgoutA(rows);  y0 = cpygoutA(rows);  z0 = cpzgoutA(rows);
      dx0 = xgoutA(rows) - x0;
      dy0 = ygoutA(rows) - y0;
      dz0 = zgoutA(rows) - z0;
      xr = x0 + Rk(1,1)*dx0 + Rk(1,2)*dy0 + Rk(1,3)*dz0;
      yr = y0 + Rk(2,1)*dx0 + Rk(2,2)*dy0 + Rk(2,3)*dz0;
      zr = z0 + Rk(3,1)*dx0 + Rk(3,2)*dy0 + Rk(3,3)*dz0;
      [cpxAtoB, cpyAtoB, cpzAtoB] = cpfB(xr, yr, zr);
      EAB(rows,:) = interp3_matrix(x1d, y1d, z1d, cpxAtoB, cpyAtoB, cpzAtoB, ...
                                   p, innerbandB);
    end

    rows = crossRowsB(vertexIdB(crossRowsB) == k);
    if (~isempty(rows))
      Rk = RBtoA{k};
      x0 = cpxgoutB(rows);  y0 = cpygoutB(rows);  z0 = cpzgoutB(rows);
      dx0 = xgoutB(rows) - x0;
      dy0 = ygoutB(rows) - y0;
      dz0 = zgoutB(rows) - z0;
      xr = x0 + Rk(1,1)*dx0 + Rk(1,2)*dy0 + Rk(1,3)*dz0;
      yr = y0 + Rk(2,1)*dx0 + Rk(2,2)*dy0 + Rk(2,3)*dz0;
      zr = z0 + Rk(3,1)*dx0 + Rk(3,2)*dy0 + Rk(3,3)*dz0;
      [cpxBtoA, cpyBtoA, cpzBtoA] = cpfA(xr, yr, zr);
      EBA(rows,:) = interp3_matrix(x1d, y1d, z1d, cpxBtoA, cpyBtoA, cpzBtoA, ...
                                   p, innerbandA);
    end
  end

  EAA(crossRowsA,:) = 0;
  EBB(crossRowsB,:) = 0;

  fprintf('Cross-branch extension rows A->B: %d of %d\n', ...
          length(crossRowsA), length(outerbandA));
  fprintf('Cross-branch extension rows B->A: %d of %d\n', ...
          length(crossRowsB), length(outerbandB));


  %% Diagonal splitting for the combined branch operator

  % Cross-branch extension rows live in the off-diagonal blocks of Eblock.
  Lblock = blkdiag(LA, LB);
  Eblock = [EAA EAB; EBA EBB];
  Rblock = blkdiag(RA, RB);
  M = lapsharp_unordered(Lblock, Eblock, Rblock);

  % after building matrices, don't need this set
  ICPM2009BANDINGCHECKS = 0;


  %% Construct interpolation matrices for plotting on the curve

  EplotA = interp3_matrix(x1d, y1d, z1d, xpA, ypA, zpA, p, innerbandA);
  EplotB = interp3_matrix(x1d, y1d, z1d, xpB, ypB, zpB, p, innerbandB);


  %% Function u in the embedding space, initial conditions

  sgA = atan2(cpyginA, cpxginA) + pi/2;   % arclength in [0, pi]
  sgB = atan2(cpzginB, cpyginB) + pi;     % arclength in [pi, 2*pi]

  u0A = cos(sgA);
  u0B = cos(sgB);
  u = [u0A; u0B];

  curveplot0A = EplotA*u0A;
  curveplot0B = EplotB*u0B;


  %% Shared vertex evaluation and averaging

  sx = vertices(:,1);
  sy = vertices(:,2);
  sz = vertices(:,3);

  EvertexA = interp3_matrix(x1d, y1d, z1d, sx, sy, sz, p, innerbandA);
  EvertexB = interp3_matrix(x1d, y1d, z1d, sx, sy, sz, p, innerbandB);

  endpointNodeTol = dx*1e-8;
  xInnerA = xx(innerbandA);  yInnerA = yy(innerbandA);  zInnerA = zz(innerbandA);
  xInnerB = xx(innerbandB);  yInnerB = yy(innerbandB);  zInnerB = zz(innerbandB);
  clear xx yy zz
  vertexWriteIdxA = zeros(length(sx),1);
  vertexWriteIdxB = zeros(length(sx),1);
  vertexWriteExactA = true(length(sx),1);
  vertexWriteExactB = true(length(sx),1);

  for k = 1:length(sx)
    I = find((abs(xInnerA - sx(k)) <= endpointNodeTol) & ...
             (abs(yInnerA - sy(k)) <= endpointNodeTol) & ...
             (abs(zInnerA - sz(k)) <= endpointNodeTol));
    if (isempty(I))
      [dummy, I] = min(sqrt((xInnerA - sx(k)).^2 + (yInnerA - sy(k)).^2 + ...
                            (zInnerA - sz(k)).^2));
      vertexWriteExactA(k) = false;
    end
    vertexWriteIdxA(k) = I(1);

    I = find((abs(xInnerB - sx(k)) <= endpointNodeTol) & ...
             (abs(yInnerB - sy(k)) <= endpointNodeTol) & ...
             (abs(zInnerB - sz(k)) <= endpointNodeTol));
    if (isempty(I))
      [dummy, I] = min(sqrt((xInnerB - sx(k)).^2 + (yInnerB - sy(k)).^2 + ...
                            (zInnerB - sz(k)).^2));
      vertexWriteExactB(k) = false;
    end
    vertexWriteIdxB(k) = I(1);
  end

  if (any(~vertexWriteExactA) || any(~vertexWriteExactB))
    fprintf('Vertex averaging fallback used: nearest innerband DOF for missing endpoint node.\n');
  end


  if (plotLevel)
    figure(1); clf;
    figure(2); clf;
    figure(3); clf;
  end


  %% Time-stepping for the heat equation

  dt = 0.2*dx^2;
  numtimesteps = ceil(Tf/dt);
  dt = Tf / numtimesteps;

  I = speye(size(M));
  A = I - dt*M;
  % implicit Euler: factor once, back-substitute every step
  Afac = decomposition(A);

  plotgap = max(1, round(numtimesteps/10));

  for kt = 1:numtimesteps
    % implicit Euler timestepping
    u = Afac \ u;
    uA = u(1:length(innerbandA));
    uB = u(length(innerbandA)+1:end);

    vertexValuesA = EvertexA*uA;
    vertexValuesB = EvertexB*uB;
    vertexAvgValues = 0.5*(vertexValuesA + vertexValuesB);
    uA(vertexWriteIdxA) = vertexAvgValues;
    uB(vertexWriteIdxB) = vertexAvgValues;
    u = [uA; uB];

    t = kt*dt;

    % plotting
    if (plotLevel && ((mod(kt,plotgap) == 0) || (kt == numtimesteps)))

      % plot the computational bands in 3D, colored by u
      activateFigure(1);
      clf;
      scatter3([xginA; xginB], [yginA; yginB], [zginA; zginB], 12, ...
               [uA; uB], 'filled');
      hold on;
      plot3(xpA, ypA, zpA, 'k-', 'linewidth', 2);
      plot3(xpB, ypB, zpB, 'k--', 'linewidth', 2);
      plot3(sx, sy, sz, 'ro', 'markerfacecolor', 'r');
      axis equal; grid on;
      xlabel('x'); ylabel('y'); zlabel('z');
      colorbar;
      title( ['computational bands: soln at time ' num2str(t) ...
              ', timestep #' num2str(kt)] );

      % plot value on the curve against arclength
      activateFigure(2);
      clf;
      curveplotA = EplotA*uA;
      curveplotB = EplotB*uB;
      hA = plot(splotA, curveplotA, 'b-');
      hold on;
      hB = plot(splotB, curveplotB, 'c-');
      hExact = plot(splotA, uexactfn(t,splotA), 'r--');
      plot(splotB, uexactfn(t,splotB), 'r--');
      hInitial = plot(splotA, curveplot0A, 'g-.');
      plot(splotB, curveplot0B, 'g-.');
      title( ['soln at time ' num2str(t) ', on the bent circle'] );
      xlabel('arclength s'); ylabel('u');
      legend([hA hB hExact hInitial], ...
             'branch A (x-y plane)', 'branch B (y-z plane)', ...
             'exact answer', 'initial condition', 'Location', 'SouthEast');

      % plot error on the curve
      activateFigure(3);
      clf;
      plot(splotA, curveplotA - uexactfn(t,splotA), 'b-');
      hold on;
      plot(splotB, curveplotB - uexactfn(t,splotB), 'c-');
      title( ['error at time ' num2str(t) ', on the bent circle'] );
      xlabel('arclength s'); ylabel('error');

      drawnow();
    end
  end

  t = numtimesteps*dt;
  uA = u(1:length(innerbandA));
  uB = u(length(innerbandA)+1:end);

  curveplotA = EplotA*uA;
  curveplotB = EplotB*uB;

  curveErrA = curveplotA - uexactfn(t,splotA);
  curveErrB = curveplotB - uexactfn(t,splotB);
  errorsLinf(level) = max(max(abs(curveErrA)), max(abs(curveErrB)));
  errorsL2(level) = sqrt(2*pi*mean([curveErrA; curveErrB].^2));

  vertexValuesA = EvertexA*uA;
  vertexValuesB = EvertexB*uB;
  vertexDiffFinal = abs(vertexValuesA - vertexValuesB);

  fprintf('Max error on branch A at t=%g: %g\n', t, max(abs(curveErrA)));
  fprintf('Max error on branch B at t=%g: %g\n', t, max(abs(curveErrB)));
  fprintf('Final branch vertex differences [top bottom]: [%g %g]\n', ...
          vertexDiffFinal(1), vertexDiffFinal(2));
end


%% Convergence table and plot

ratesLinf = log(errorsLinf(1:end-1) ./ errorsLinf(2:end)) ./ ...
            log(hvals(1:end-1) ./ hvals(2:end));
ratesL2 = log(errorsL2(1:end-1) ./ errorsL2(2:end)) ./ ...
          log(hvals(1:end-1) ./ hvals(2:end));

fprintf('\nConvergence at Tf = %g (implicit Euler, dt = 0.2*dx^2):\n', Tf);
fprintf('        h     curve Linf      rate       curve L2      rate\n');
for k = 1:length(hvals)
  if (k == 1)
    fprintf('%9.5f   %12.4e       --   %12.4e       --\n', ...
            hvals(k), errorsLinf(k), errorsL2(k));
  else
    fprintf('%9.5f   %12.4e   %6.2f   %12.4e   %6.2f\n', ...
            hvals(k), errorsLinf(k), ratesLinf(k-1), ...
            errorsL2(k), ratesL2(k-1));
  end
end

if (doPlots)
  figure(4); clf;
  ref2 = errorsLinf(1) * (hvals/hvals(1)).^2;
  loglog(hvals, errorsLinf, 'bo-', hvals, errorsL2, 'cs-', ...
         hvals, ref2, 'k--', 'linewidth', 1.5);
  grid on;
  xlabel('h'); ylabel('error at Tf');
  legend('Linf error', 'L2 error', 'O(h^2) reference', 'Location', 'NorthWest');
  title('bent circle heat equation: convergence');
end


function reset_icpm2009bandingchecks()
  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = 0;
end


function activateFigure(n)
%ACTIVATEFIGURE  Make figure n current without stealing focus.
%   Like set(0,'CurrentFigure',n), but recreates the figure if it was
%   closed during the run (set(0,...) errors on a missing figure).
  if (ishghandle(n, 'figure'))
    set(0, 'CurrentFigure', n);
  else
    figure(n);
  end
end


function [cpx, cpy, cpz, dist, bdy] = cpArcInXYPlane(x, y, z, R, cen, angle1, angle2)
%CPARCINXYPLANE  Closest point of a circular arc lying in the x-y plane.
%   The planar closest point of (x,y) on the arc is also the 3D closest
%   point, since the arc has z = 0.
  [cpx, cpy, dist2d, bdy] = cpArc(x, y, R, cen, angle1, angle2);
  cpz = zeros(size(cpx));
  dist = sqrt((x - cpx).^2 + (y - cpy).^2 + (z - cpz).^2);
end


function [cpx, cpy, cpz, dist, bdy] = cpArcInYZPlane(x, y, z, R, cen, angle1, angle2)
%CPARCINYZPLANE  Closest point of a circular arc lying in the y-z plane.
%   In-plane coordinates are (y,z), and the arc has x = 0.
  [cpy, cpz, dist2d, bdy] = cpArc(y, z, R, cen, angle1, angle2);
  cpx = zeros(size(cpy));
  dist = sqrt((x - cpx).^2 + (y - cpy).^2 + (z - cpz).^2);
end
