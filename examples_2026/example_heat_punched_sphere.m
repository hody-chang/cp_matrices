%% Heat equation on a punched-in sphere via branch rotation (iCPM)
% The surface is a unit sphere whose spherical cap z > z0 has been
% reflected ("punched in") through the plane z = z0.  The folding map
%     F(x,y,z) = (x, y, z)          for z <= z0
%     F(x,y,z) = (x, y, 2*z0 - z)   for z >  z0
% is an intrinsic isometry from the unit sphere onto the punched
% sphere, so the Laplace-Beltrami operator is that of the unit sphere
% and the heat equation has the same solution as on the unit sphere.
% With initial condition u0 = cos(phi + pi/2) (pulled back through the
% isometry), the exact solution is u(t) = exp(-2*t)*u0, exactly as in
% examples/example_heat_sphere.m.
%
% The surface has a concave crease along the seam circle
% {x^2 + y^2 = 1 - z0^2, z = z0}.  Standard one-branch CPM fails
% there, so the surface is split into two branches:
%   branch A: the unit sphere with the open cap z > z0 removed
%   branch B: the punched-in cap (the reflected cap)
% and cross-branch extension rows are routed through the opposite
% branch after rotating the embedding point about the seam, as in
% example_heat_bent_circle_3d_rotation.m.  Unlike the bent circle,
% the singular set is a circle rather than isolated vertices: the
% rotation axis is the local seam tangent, which varies with azimuth.
% By axisymmetry the rotation angle is constant along the seam (exact
% value 2*acos(z0)), so angle3D estimates the rotation matrix R0 at
% one representative seam point (r0, 0, z0) and each cross-branch row
% uses the conjugated rotation R(phi) = Rz(phi)*R0*Rz(-phi) about the
% seam tangent at its own closest seam point.
%
% Settings follow examples/example_heat_sphere.m: degree 3
% interpolation, second-order Laplacian, dt = 0.1*dx^2, Tf = 2.
% (Time-stepping is implicit Euler on the iCPM operator, as in the
% bent circle example; dt = 0.1*dx^2 keeps the O(dt) time error at
% the order of the spatial error so the spatial rate is visible.)
%
% A convergence study is run over the grids in hvals.
%
% Output (plotted once, after time-stepping on the finest grid):
% figure 1 shows the final solution on the punched sphere (same style
% as example_heat_sphere.m), figure 2 shows the final relative error
% (normalized by the max exact solution magnitude) in the same style,
% and figure 3 shows the convergence plot.


%% Using cp_matrices

% Include the cp_matrices folder (edit as appropriate)
addpath('../cp_matrices');

% add functions for finding the closest points
addpath('../surfaces');


global ICPM2009BANDINGCHECKS
cleanupICPM2009BANDINGCHECKS = onCleanup(@() reset_icpm2009bandingchecks());


%% Parameters shared by all grid levels

doPlots = true;   % final plots on the finest level + convergence plot

hvals = 1./[20 40 60];   % grid sizes for the convergence study

z0 = 0.5;             % punch the cap z > z0 (angular radius pi/3)
r0 = sqrt(1 - z0^2);  % seam circle radius

cpfA = @(x,y,z) cpSphereMinusCap(x, y, z, z0);
cpfB = @(x,y,z) cpDentCap(x, y, z, z0);

% banding parameters
dim = 3;  % dimension
p = 3;    % interpolation order
order = 2;  % Laplacian order: bw will need to increase if changed
fd_stenrad = order/2;  % Finite difference stencil radius
bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ((fd_stenrad+(p+1)/2)^2));

Tf = 0.01;

% plotting grid: fold the standard sphere parameterization; points on
% the cap belong to branch B, the rest to branch A
[xp, yp, zp] = sphere(64);
capMask = (zp > z0);
zp_punch = zp;
zp_punch(capMask) = 2*z0 - zp(capMask);
% exact solution lives on the unpunched preimage points
[th_plot, phi_plot, r] = cart2sph(xp(:), yp(:), zp(:));
% area weights for the surface L2 norm on the lat-long plotting grid
wplot = cos(phi_plot);

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
  x1d = (-2.0:dx:2.0)';
  y1d = x1d;
  z1d = x1d;

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

  [tf, innerInOuterA] = ismember(innerbandA, outerbandA);
  if (~all(tf)), error('branch A innerband not contained in outerband'); end
  RA = sparse(1:length(innerbandA), innerInOuterA, 1, ...
              length(innerbandA), length(outerbandA));

  cpxginA = RA*cpxgoutA;  cpyginA = RA*cpygoutA;  cpzginA = RA*cpzgoutA;


  %% Branch B two-band iCPM matrices

  [cpxB, cpyB, cpzB, distB, bdyB] = cpfB(xx, yy, zz);

  band_initB = find(abs(distB) <= bw*dx);
  cpxg_initB = cpxB(band_initB); cpyg_initB = cpyB(band_initB);
  cpzg_initB = cpzB(band_initB);
  xg_initB = xx(band_initB); yg_initB = yy(band_initB);
  zg_initB = zz(band_initB);
  bdyg_initB = bdyB(band_initB);
  clear cpxB cpyB cpzB distB bdyB
  clear xx yy zz

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

  [tf, innerInOuterB] = ismember(innerbandB, outerbandB);
  if (~all(tf)), error('branch B innerband not contained in outerband'); end
  RB = sparse(1:length(innerbandB), innerInOuterB, 1, ...
              length(innerbandB), length(outerbandB));

  cpxginB = RB*cpxgoutB;  cpyginB = RB*cpygoutB;  cpzginB = RB*cpzgoutB;


  %% Seam rotation matrix estimated with angle3D

  % angle3D averages the cp/cpbar vectors of grid points whose closest
  % point is the representative seam point sstar = (r0, 0, z0), giving
  % each branch's outward co-normal there.  The A->B rotation maps
  % branch A's outward co-normal onto branch B's incoming tangent
  % (-conormalB), rotating about the axis perpendicular to both (the
  % seam tangent at sstar, i.e. the y-axis).  Exact rotation angle is
  % 2*acos(z0).
  sstar = [r0 0 z0];
  [dummy, conormalA] = angle3D(xgoutA, ygoutA, zgoutA, cpfA, sstar);
  [dummy, conormalB] = angle3D(xgoutB, ygoutB, zgoutB, cpfB, sstar);

  axisAtoB = cross(conormalA, -conormalB);
  [R0AtoB, dummy, infoAtoB] = angle3D(xgoutA, ygoutA, zgoutA, cpfA, sstar, ...
                                      axisAtoB, -conormalB);
  axisBtoA = cross(conormalB, -conormalA);
  [R0BtoA, dummy, infoBtoA] = angle3D(xgoutB, ygoutB, zgoutB, cpfB, sstar, ...
                                      axisBtoA, -conormalA);

  fprintf('Seam: conormalA = [%g %g %g], conormalB = [%g %g %g]\n', ...
          conormalA(1), conormalA(2), conormalA(3), ...
          conormalB(1), conormalB(2), conormalB(3));
  fprintf('Seam: theta A->B = %g, theta B->A = %g (exact 2*acos(z0) = %g)\n', ...
          infoAtoB.theta, infoBtoA.theta, 2*acos(z0));


  %% Seam ids for cross-branch extension rows

  % The branch cp functions label seam-clamped closest points with
  % bdy=1, but exact seam hits can still have bdy=0; catch those by
  % distance to the seam circle.
  seamTol = 100*eps;

  seamDistA = sqrt((sqrt(cpxgoutA.^2 + cpygoutA.^2) - r0).^2 + ...
                   (cpzgoutA - z0).^2);
  crossRowsA = find((bdygoutA ~= 0) | (seamDistA <= seamTol));
  seamDistB = sqrt((sqrt(cpxgoutB.^2 + cpygoutB.^2) - r0).^2 + ...
                   (cpzgoutB - z0).^2);
  crossRowsB = find((bdygoutB ~= 0) | (seamDistB <= seamTol));


  %% Cross-branch extension rows via conjugated seam rotation

  % For each cross row the closest point is a seam point s with azimuth
  % phi; the embedding point is rotated about the seam tangent at s by
  % R(phi) = Rz(phi)*R0*Rz(-phi) and re-projected onto the other branch.
  EAB = sparse(length(outerbandA), length(innerbandB));
  EBA = sparse(length(outerbandB), length(innerbandA));

  if (~isempty(crossRowsA))
    [xr, yr, zr] = rotateAboutSeam(xgoutA(crossRowsA), ygoutA(crossRowsA), ...
                                   zgoutA(crossRowsA), cpxgoutA(crossRowsA), ...
                                   cpygoutA(crossRowsA), cpzgoutA(crossRowsA), ...
                                   R0AtoB);
    [cpxAtoB, cpyAtoB, cpzAtoB] = cpfB(xr, yr, zr);
    EAB(crossRowsA,:) = interp3_matrix(x1d, y1d, z1d, cpxAtoB, cpyAtoB, ...
                                       cpzAtoB, p, innerbandB);
  end

  if (~isempty(crossRowsB))
    [xr, yr, zr] = rotateAboutSeam(xgoutB(crossRowsB), ygoutB(crossRowsB), ...
                                   zgoutB(crossRowsB), cpxgoutB(crossRowsB), ...
                                   cpygoutB(crossRowsB), cpzgoutB(crossRowsB), ...
                                   R0BtoA);
    [cpxBtoA, cpyBtoA, cpzBtoA] = cpfA(xr, yr, zr);
    EBA(crossRowsB,:) = interp3_matrix(x1d, y1d, z1d, cpxBtoA, cpyBtoA, ...
                                       cpzBtoA, p, innerbandA);
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


  %% Construct interpolation matrices for plotting on the punched sphere

  EplotA = interp3_matrix(x1d, y1d, z1d, xp(~capMask), yp(~capMask), ...
                          zp_punch(~capMask), p, innerbandA);
  EplotB = interp3_matrix(x1d, y1d, z1d, xp(capMask), yp(capMask), ...
                          zp_punch(capMask), p, innerbandB);


  %% Function u in the embedding space, initial conditions

  % initial condition cos(phi + pi/2) on the unit sphere, pulled back
  % through the isometry (unfold branch B before evaluating)
  [th, phi, r] = cart2sph(cpxginA, cpyginA, cpzginA);
  u0A = cos(phi + pi/2);
  [th, phi, r] = cart2sph(cpxginB, cpyginB, 2*z0 - cpzginB);
  u0B = cos(phi + pi/2);
  u = [u0A; u0B];


  %% Time-stepping for the heat equation

  dt = 0.1*dx^2;
  numtimesteps = ceil(Tf/dt);
  % adjust for integer number of steps
  dt = Tf / numtimesteps;

  I = speye(size(M));
  A = I - dt*M;
  % implicit Euler: factor once, back-substitute every step
  Afac = decomposition(A);

  tic
  for kt = 1:numtimesteps
    % implicit Euler timestepping
    u = Afac \ u;
  end
  t_implicit = toc;
  fprintf('Time-stepping done in %g s\n', t_implicit);


  %% Relative errors at Tf on the plotting grid

  t = numtimesteps*dt;
  uA = u(1:length(innerbandA));
  uB = u(length(innerbandA)+1:end);

  sphplot = zeros(size(xp));
  sphplot(~capMask) = EplotA*uA;
  sphplot(capMask) = EplotB*uB;
  exactplot = reshape(exp(-2*t)*cos(phi_plot + pi/2), size(xp));
  % relative error: normalize by the max exact solution magnitude
  % (pointwise division would blow up at the zeros of the solution)
  relerrplot = (sphplot - exactplot) / norm(exactplot(:),inf);
  relerr = relerrplot(:);

  errorsLinf(level) = norm(relerr,inf);
  % area-weighted surface L2 norm (lat-long grid clusters at the poles)
  errorsL2(level) = sqrt(sum(wplot.*relerr.^2) / sum(wplot));

  fprintf('Max relative error at t=%g: %g\n', t, errorsLinf(level));


  %% Final solution and error plots on the finest level

  if (plotLevel)
    % solution on the punched sphere
    figure(1); clf;
    surf(xp, yp, zp_punch, sphplot);
    title( ['soln at time ' num2str(t) ', dx = ' num2str(dx)] );
    xlabel('x'); ylabel('y'); zlabel('z');
    axis equal; shading interp;
    colorbar;

    % relative error, same style
    figure(2); clf;
    surf(xp, yp, zp_punch, relerrplot);
    title( ['relative error at time ' num2str(t) ', dx = ' num2str(dx)] );
    xlabel('x'); ylabel('y'); zlabel('z');
    axis equal; shading interp;
    colorbar;

    drawnow();
  end
end


%% Convergence table and plot

ratesLinf = log(errorsLinf(1:end-1) ./ errorsLinf(2:end)) ./ ...
            log(hvals(1:end-1) ./ hvals(2:end));
ratesL2 = log(errorsL2(1:end-1) ./ errorsL2(2:end)) ./ ...
          log(hvals(1:end-1) ./ hvals(2:end));

fprintf('\nConvergence at Tf = %g (implicit Euler, dt = 0.1*dx^2):\n', Tf);
fprintf('        h     rel Linf        rate       rel L2        rate\n');
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
  figure(3); clf;
  ref2 = errorsLinf(1) * (hvals/hvals(1)).^2;
  loglog(hvals, errorsLinf, 'bo-', hvals, errorsL2, 'cs-', ...
         hvals, ref2, 'k--', 'linewidth', 1.5);
  grid on;
  xlabel('h'); ylabel('relative error at Tf');
  legend('Linf error', 'L2 error', 'O(h^2) reference', 'Location', 'NorthWest');
  title('punched sphere heat equation: convergence');
end


function reset_icpm2009bandingchecks()
  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = 0;
end


function [xr, yr, zr] = rotateAboutSeam(x, y, z, sx, sy, sz, R0)
%ROTATEABOUTSEAM  Rotate embedding points about their seam points.
%   (sx,sy,sz) are seam-circle points; each (x,y,z) is rotated about
%   the seam tangent at its own seam point by the conjugated rotation
%   R(phi) = Rz(phi)*R0*Rz(-phi), where phi = atan2(sy,sx) is the
%   azimuth of the seam point and R0 is the rotation estimated at the
%   representative seam point (r0, 0, z0).
  phi = atan2(sy, sx);
  c = cos(phi);  s = sin(phi);

  dx0 = x - sx;  dy0 = y - sy;  dz0 = z - sz;

  % Rz(-phi)
  e1 =  c.*dx0 + s.*dy0;
  e2 = -s.*dx0 + c.*dy0;
  e3 = dz0;

  % R0
  f1 = R0(1,1)*e1 + R0(1,2)*e2 + R0(1,3)*e3;
  f2 = R0(2,1)*e1 + R0(2,2)*e2 + R0(2,3)*e3;
  f3 = R0(3,1)*e1 + R0(3,2)*e2 + R0(3,3)*e3;

  % Rz(phi)
  g1 = c.*f1 - s.*f2;
  g2 = s.*f1 + c.*f2;
  g3 = f3;

  xr = sx + g1;
  yr = sy + g2;
  zr = sz + g3;
end


function [cpx, cpy, cpz, dist, bdy] = cpSphereMinusCap(x, y, z, z0)
%CPSPHEREMINUSCAP  Closest point on the unit sphere minus the open cap z > z0.
%   Full-sphere closest points landing in the removed cap are clamped
%   to the nearest point of the seam circle (same azimuth) and
%   labelled with bdy = 1.
  [cpx, cpy, cpz] = cpSphere(x, y, z);
  bdy = zeros(size(cpx));
  I = (cpz > z0);
  bdy(I) = 1;
  r0 = sqrt(1 - z0^2);
  [sx, sy] = seamPoint(x(I), y(I), r0);
  cpx(I) = sx;  cpy(I) = sy;  cpz(I) = z0;
  dist = sqrt((x-cpx).^2 + (y-cpy).^2 + (z-cpz).^2);
end


function [cpx, cpy, cpz, dist, bdy] = cpDentCap(x, y, z, z0)
%CPDENTCAP  Closest point on the punched-in (reflected) cap.
%   Reflect the query through the plane z = z0, take the closest
%   point on the closed cap z >= z0 of the unit sphere (clamping to
%   the seam circle with bdy = 1 as in cpSphereMinusCap), and reflect
%   the closest point back.
  zr = 2*z0 - z;
  [cpx, cpy, cpz] = cpSphere(x, y, zr);
  bdy = zeros(size(cpx));
  I = (cpz < z0);
  bdy(I) = 1;
  r0 = sqrt(1 - z0^2);
  [sx, sy] = seamPoint(x(I), y(I), r0);
  cpx(I) = sx;  cpy(I) = sy;  cpz(I) = z0;
  cpz = 2*z0 - cpz;
  dist = sqrt((x-cpx).^2 + (y-cpy).^2 + (z-cpz).^2);
end


function [sx, sy] = seamPoint(x, y, r0)
%SEAMPOINT  Nearest point on the seam circle x^2+y^2 = r0^2 (in-plane).
%   Same azimuth as (x,y); an arbitrary seam point is used for queries
%   on the z-axis, where every seam point is equidistant.
  r = sqrt(x.^2 + y.^2);
  sx = r0*ones(size(x));   % arbitrary azimuth for on-axis points
  sy = zeros(size(x));
  J = (r > 0);
  sx(J) = r0*x(J)./r(J);
  sy(J) = r0*y(J)./r(J);
end
