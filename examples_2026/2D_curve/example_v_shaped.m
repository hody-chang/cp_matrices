%% Heat equation on a V-shaped line with branch rotation at the vertex
% This example solves the heat equation on two line-segment branches joined
% at a vertex.  The vertex is duplicated, one value per branch.  Vertex
% extension rows are rotated into the other branch, and the two vertex
% values are averaged when a single physical vertex value is needed.
%
% The exact solution uses the unfolded arclength r in [0,2L],
%
%   u(t,r) = exp(-lambda*t)*cos(pi*r/(2L)),
%
% lambda = (pi/(2L))^2.  This has Neumann boundary conditions at the two
% physical endpoints r=0 and r=2L.
%
% Run headlessly, from this directory:
%   matlab -batch "example_v_shaped"
% The angles, grid sizes and the plotting flag are edited below.


%% Using cp_matrices

% Resolve dependencies relative to this example.
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', '..', 'cp_matrices'));
addpath(fullfile(here, '..', '..', 'surfaces'));


global ICPM2009BANDINGCHECKS

% the banding checks are switched on around the two-band construction
% below and off again afterwards; this restores whatever the caller had
old_bandingchecks = ICPM2009BANDINGCHECKS;
cleanup_bandingchecks = ...
    onCleanup(@() reset_icpm2009bandingchecks(old_bandingchecks));


%% Problem parameters

makePlots = true;

% the opening angle at the vertex, in degrees
angle_list = [90 120];

% grid sizes for the convergence study
hvals = 1./[200 400 800 1600];

vertex = [0.5 0.5];
branchlen = 0.4;

p = 3;        % interpolation order
order = 2;    % Laplacian order
Tf = 0.01;

results = struct([]);


%% Convergence study

for ai = 1:length(angle_list)
  angdeg = angle_list(ai);
  lev = repmat(empty_level_result(), length(hvals), 1);

  fprintf('\nV-shaped line, angle = %g degrees\n', angdeg);

  for k = 1:length(hvals)
    h = hvals(k);
    % only the finest level is worth drawing
    plotthis = makePlots && (k == length(hvals));

    lev(k) = solve_one_level(h, angdeg, vertex, branchlen, p, order, Tf, ...
                             plotthis, ai);

    fprintf(['h = %g, points = %d, L_inf error = %g, ' ...
             'L_2 error = %g\n'], ...
            h, lev(k).numpts, lev(k).err_inf, lev(k).err_l2);
  end

  results(ai).angdeg = angdeg;
  results(ai).h = hvals;
  results(ai).numpts = [lev.numpts];
  results(ai).errs_inf = [lev.err_inf];
  results(ai).errs_l2 = [lev.err_l2];
  results(ai).levels = lev;
end

if (makePlots)
  plotconv(results);
end


%% ----------------------------------------------------------------------
%% local functions
%% ----------------------------------------------------------------------

function lev = solve_one_level(h, angdeg, vertex, branchlen, p, order, ...
                               Tf, makePlots, ai)
%SOLVE_ONE_LEVEL  build the two glued branches at one grid size and solve

  global ICPM2009BANDINGCHECKS

  % this is a bit dangerous: will break other less tightly banded
  % codes, turn it off later
  ICPM2009BANDINGCHECKS = 1;

  alpha = angdeg*pi/180;
  dA = [-sin(alpha/2) -cos(alpha/2)];
  dB = [ sin(alpha/2) -cos(alpha/2)];

  endptA = vertex + branchlen*dA;
  endptB = vertex + branchlen*dB;

  % Branch A is oriented from its physical endpoint to the vertex.
  % Branch B is oriented from the vertex to its physical endpoint.
  pA = endptA;     qA = vertex;
  pB = vertex;     qB = endptB;

  dim = 2;  % dimension
  fd_stenrad = order/2;  % finite difference stencil radius
  % The formula for bw is found in [Ruuth & Merriman 2008] and the 1.0002
  % is a safety factor.
  bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ...
                   ((fd_stenrad+(p+1)/2)^2));

  x1d = (0:h:1).';
  y1d = x1d;
  [xx, yy] = meshgrid(x1d, y1d);

  [cpxA, cpyA, distA, bdyA, sA] = cpLineSegment2d(xx, yy, pA, qA);
  [cpxB, cpyB, distB, bdyB, sB] = cpLineSegment2d(xx, yy, pB, qB);

  % cpbar of a cpf that reports only one endpoint as boundary: this is the
  % same-branch extension, used everywhere except at the shared vertex
  cpf_endptA = @(x, y) cp_endpoint_bdy(x, y, pA, qA, 1);
  cpf_endptB = @(x, y) cp_endpoint_bdy(x, y, pB, qB, 2);
  [cpbarxA, cpbaryA] = cpbar_2d(xx, yy, cpf_endptA);
  [cpbarxB, cpbaryB] = cpbar_2d(xx, yy, cpf_endptB);

  brA = build_branch(x1d, y1d, xx, yy, cpxA, cpyA, ...
                     cpbarxA, cpbaryA, distA, bdyA, sA, ...
                     h, bw, p, order);
  brB = build_branch(x1d, y1d, xx, yy, cpxB, cpyB, ...
                     cpbarxB, cpbaryB, distB, bdyB, sB, ...
                     h, bw, p, order);

  [EAA, EAB, EBA, EBB, crossrowsA, crossrowsB] = ...
      build_vertex_ext(x1d, y1d, brA, brB, pA, qA, pB, qB, p);

  Lblock = blkdiag(brA.L, brB.L);
  Eblock = [EAA EAB; EBA EBB];
  Rblock = blkdiag(brA.R, brB.R);
  M = lapsharp_unordered(Lblock, Eblock, Rblock);

  ICPM2009BANDINGCHECKS = 0;

  rA = branchlen*brA.sgin;
  rB = branchlen + branchlen*brB.sgin;

  u0A = uexact(0, rA, branchlen);
  u0B = uexact(0, rB, branchlen);
  u = [u0A; u0B];

  vertA = setup_vertex_avg(x1d, y1d, xx, yy, brA.innerband, vertex, p);
  vertB = setup_vertex_avg(x1d, y1d, xx, yy, brB.innerband, vertex, p);

  % Crank-Nicolson has O(dt^2) time error; dt=O(h) keeps it aligned with
  % the second-order spatial convergence study.
  dt = h/4;
  numtimesteps = ceil(Tf/dt);
  % adjust for integer number of steps
  dt = Tf / numtimesteps;

  I = speye(size(M));
  A = I - 0.5*dt*M;
  B = I + 0.5*dt*M;

  for kt = 1:numtimesteps
    % Crank-Nicolson timestepping
    u = A \ (B*u);

    uA = u(1:length(u0A));
    uB = u(length(u0A)+1:end);
    [uA, uB] = average_vertex(uA, uB, vertA, vertB);
    u = [uA; uB];
  end

  t = numtimesteps*dt;
  uA = u(1:length(u0A));
  uB = u(length(u0A)+1:end);


  %% Interpolation matrices for plotting on the two branches

  nplot = 500;
  [xpA, ypA] = paramLineSegment2d(nplot, pA, qA);
  [xpB, ypB] = paramLineSegment2d(nplot, pB, qB);
  splot = linspace(0, 1, nplot).';
  rplotA = branchlen*splot;
  rplotB = branchlen + branchlen*splot;

  EplotA = interp2_matrix(x1d, y1d, xpA, ypA, p, brA.innerband);
  EplotB = interp2_matrix(x1d, y1d, xpB, ypB, p, brB.innerband);

  uplotA = EplotA*uA;
  uplotB = EplotB*uB;
  initplotA = EplotA*u0A;
  initplotB = EplotB*u0B;

  % the two branches carry their own copy of the vertex; the physical
  % value there is their average
  vavg = 0.5*(uplotA(end) + uplotB(1));
  uplotA(end) = vavg;
  uplotB(1) = vavg;

  exactplotA = uexact(t, rplotA, branchlen);
  exactplotB = uexact(t, rplotB, branchlen);

  err = [uplotA - exactplotA; uplotB - exactplotB];
  err_inf = norm(err, inf);
  err_l2 = sqrt(mean(err.^2));

  if (makePlots)
    plotlevel(ai, angdeg, h, t, brA, brB, uA, uB, xpA, ypA, xpB, ypB, ...
              rplotA, rplotB, uplotA, uplotB, initplotA, initplotB, ...
              exactplotA, exactplotB, crossrowsA, crossrowsB);
  end

  lev = empty_level_result();
  lev.h = h;
  lev.numpts = length(uA) + length(uB);
  lev.err_inf = err_inf;
  lev.err_l2 = err_l2;
  lev.crossrowsA = length(crossrowsA);
  lev.crossrowsB = length(crossrowsB);
end


function br = build_branch(x1d, y1d, xx, yy, cpx, cpy, ...
                           cpbarx, cpbary, dist, bdy, s, ...
                           h, bw, p, order)
%BUILD_BRANCH  the two-band iCPM matrices L, E, R for one branch

  band_init = find(abs(dist) <= bw*h);
  cpxg_init = cpx(band_init);        cpyg_init = cpy(band_init);
  cpbarxg_init = cpbarx(band_init);  cpbaryg_init = cpbary(band_init);
  xg_init = xx(band_init);           yg_init = yy(band_init);
  bdyg_init = bdy(band_init);        sg_init = s(band_init);

  Etemp = interp2_matrix(x1d, y1d, cpbarxg_init, cpbaryg_init, p);
  [~, j] = find(Etemp);
  innerband = unique(j);

  Ltemp = laplacian_2d_matrix(x1d, y1d, order, innerband, band_init);
  [~, j] = find(Ltemp);
  outerbandtemp = unique(j);
  outerband = band_init(outerbandtemp);

  cpxgout = cpxg_init(outerbandtemp);  cpygout = cpyg_init(outerbandtemp);
  xgout = xg_init(outerbandtemp);      ygout = yg_init(outerbandtemp);
  bdygout = bdyg_init(outerbandtemp);  sgout = sg_init(outerbandtemp);

  L = Ltemp(:, outerbandtemp);
  E = Etemp(outerbandtemp, innerband);
  clear Ltemp Etemp outerbandtemp

  R = sparse([], [], [], length(innerband), length(outerband), ...
             length(innerband));
  for k = 1:length(innerband)
    I = find(outerband == innerband(k));
    R(k, I) = 1;
  end

  br.L = L;
  br.E = E;
  br.R = R;
  br.innerband = innerband;
  br.outerband = outerband;
  br.cpxgout = cpxgout;
  br.cpygout = cpygout;
  br.xgout = xgout;
  br.ygout = ygout;
  br.bdygout = bdygout;
  br.sgout = sgout;
  br.sgin = R*sgout;
  br.xgin = R*xgout;
  br.ygin = R*ygout;
end


function [EAA, EAB, EBA, EBB, crossrowsA, crossrowsB] = ...
    build_vertex_ext(x1d, y1d, brA, brB, pA, qA, pB, qB, p)
%BUILD_VERTEX_EXT  route the vertex extension rows through the other branch

  EAA = brA.E;
  EBB = brB.E;
  EAB = sparse(size(EAA, 1), size(EBB, 2));
  EBA = sparse(size(EBB, 1), size(EAA, 2));

  endpttol = 100*eps(1);

  endptidA = brA.bdygout;
  endptidA(hypot(brA.cpxgout - pA(1), brA.cpygout - pA(2)) <= endpttol) = 1;
  endptidA(hypot(brA.cpxgout - qA(1), brA.cpygout - qA(2)) <= endpttol) = 2;

  endptidB = brB.bdygout;
  endptidB(hypot(brB.cpxgout - pB(1), brB.cpygout - pB(2)) <= endpttol) = 1;
  endptidB(hypot(brB.cpxgout - qB(1), brB.cpygout - qB(2)) <= endpttol) = 2;

  % Only the shared vertex is glued.  The two physical endpoints keep their
  % same-branch cpbar extension, giving Neumann endpoint conditions.
  crossrowsA = find(endptidA == 2);
  crossrowsB = find(endptidB == 1);

  cpfA = @(x, y) cpLineSegment2d(x, y, pA, qA);
  cpfB = @(x, y) cpLineSegment2d(x, y, pB, qB);
  RA_out = angle2d(brA.xgout, brA.ygout, cpfA, qA);
  RB_out = angle2d(brB.xgout, brB.ygout, cpfB, pB);
  Rpi = [-1 0; 0 -1];
  RAtoB = RB_out * Rpi * RA_out.';
  RBtoA = RA_out * Rpi * RB_out.';

  if (~isempty(crossrowsA))
    x0 = brA.cpxgout(crossrowsA);
    y0 = brA.cpygout(crossrowsA);
    dx0 = brA.xgout(crossrowsA) - x0;
    dy0 = brA.ygout(crossrowsA) - y0;

    xr = x0 + RAtoB(1,1)*dx0 + RAtoB(1,2)*dy0;
    yr = y0 + RAtoB(2,1)*dx0 + RAtoB(2,2)*dy0;

    [cpxAtoB, cpyAtoB] = cpLineSegment2d(xr, yr, pB, qB);
    EAB(crossrowsA, :) = interp2_matrix(x1d, y1d, cpxAtoB, cpyAtoB, ...
                                        p, brB.innerband);
    EAA(crossrowsA, :) = 0;
  end

  if (~isempty(crossrowsB))
    x0 = brB.cpxgout(crossrowsB);
    y0 = brB.cpygout(crossrowsB);
    dx0 = brB.xgout(crossrowsB) - x0;
    dy0 = brB.ygout(crossrowsB) - y0;

    xr = x0 + RBtoA(1,1)*dx0 + RBtoA(1,2)*dy0;
    yr = y0 + RBtoA(2,1)*dx0 + RBtoA(2,2)*dy0;

    [cpxBtoA, cpyBtoA] = cpLineSegment2d(xr, yr, pA, qA);
    EBA(crossrowsB, :) = interp2_matrix(x1d, y1d, cpxBtoA, cpyBtoA, ...
                                        p, brA.innerband);
    EBB(crossrowsB, :) = 0;
  end
end


function [cpx, cpy, dist, bdy] = cp_endpoint_bdy(x, y, p, q, endptid)
%CP_ENDPOINT_BDY  cpLineSegment2d reporting only one endpoint as boundary

  [cpx, cpy, dist, bdy] = cpLineSegment2d(x, y, p, q);
  bdy(bdy ~= endptid) = 0;
end


function v = setup_vertex_avg(x1d, y1d, xx, yy, innerband, vertex, p)
%SETUP_VERTEX_AVG  how to read the vertex value off a branch, and where to
%   write the averaged one back

  v.E = interp2_matrix(x1d, y1d, vertex(1), vertex(2), p, innerband);

  xgin = xx(innerband);
  ygin = yy(innerband);
  nodetol = (x1d(2) - x1d(1))*1e-8;

  I = find((abs(xgin - vertex(1)) <= nodetol) & ...
           (abs(ygin - vertex(2)) <= nodetol));
  if (isempty(I))
    [~, I] = min(hypot(xgin - vertex(1), ygin - vertex(2)));
  end

  v.writeidx = I(1);
end


function [uA, uB] = average_vertex(uA, uB, vertA, vertB)
%AVERAGE_VERTEX  replace the two branch copies of the vertex by their mean

  vvalA = vertA.E*uA;
  vvalB = vertB.E*uB;
  vavg = 0.5*(vvalA + vvalB);

  uA(vertA.writeidx) = vavg;
  uB(vertB.writeidx) = vavg;
end


function u = uexact(t, r, branchlen)
%UEXACT  the exact solution in the unfolded arclength r

  lambda = (pi/(2*branchlen))^2;
  u = exp(-lambda*t)*cos(pi*r/(2*branchlen));
end


function plotlevel(ai, angdeg, h, t, brA, brB, uA, uB, ...
                   xpA, ypA, xpB, ypB, rplotA, rplotB, ...
                   uplotA, uplotB, initplotA, initplotB, ...
                   exactplotA, exactplotB, crossrowsA, crossrowsB)
%PLOTLEVEL  the band, the solution and the error at one grid size

  figbase = 10*(ai - 1) + 1;

  % plot over computation band
  figure(figbase);
  plot2d_compdomain([uA; uB], [brA.xgin; brB.xgin], ...
                    [brA.ygin; brB.ygin], h, h, figbase);
  hold on;
  plot(xpA, ypA, 'k-', 'linewidth', 2);
  plot(xpB, ypB, 'k--', 'linewidth', 2);
  title(['embedded domain: angle ' num2str(angdeg) ...
         ' degrees, t = ' num2str(t)]);
  xlabel('x'); ylabel('y');

  % plot value on the unfolded line
  figure(figbase + 1); clf;
  hA = plot(rplotA, uplotA, 'b-');
  hold on;
  hB = plot(rplotB, uplotB, 'c-');
  hexact = plot([rplotA; rplotB], [exactplotA; exactplotB], 'r--');
  hinit = plot([rplotA; rplotB], [initplotA; initplotB], 'g-.');
  title(['soln at time ' num2str(t) ', V angle ' ...
         num2str(angdeg) ' degrees']);
  xlabel('unfolded arclength r'); ylabel('u');
  legend([hA hB hexact hinit], ...
         'branch A iCPM', 'branch B iCPM', 'exact answer', ...
         'initial condition', 'Location', 'SouthWest');

  figure(figbase + 2); clf;
  plot(rplotA, uplotA - exactplotA, 'b-');
  hold on;
  plot(rplotB, uplotB - exactplotB, 'c-');
  title(['error at time ' num2str(t) ', V angle ' ...
         num2str(angdeg) ' degrees']);
  xlabel('unfolded arclength r'); ylabel('error');
  legend('branch A', 'branch B', 'Location', 'SouthWest');

  fprintf('Cross-branch extension rows A->B / B->A: %d / %d\n', ...
          length(crossrowsA), length(crossrowsB));
end


function plotconv(results)
%PLOTCONV  the error against the number of points, one panel per angle

  figure(100); clf;

  for k = 1:length(results)
    N = results(k).numpts;
    e_inf = results(k).errs_inf;
    e_l2 = results(k).errs_l2;
    refscale = max(e_inf(1), e_l2(1));
    ref = refscale*(N(1)./N).^2;

    subplot(1, length(results), k);
    loglog(N, e_inf, 'bo-', N, e_l2, 'rs-', N, ref, 'k--');
    grid on;
    xlabel('number of points');
    ylabel('error');
    title(['V angle ' num2str(results(k).angdeg) ' degrees']);
    legend('L_\infty error', 'L_2 error', 'O(h^2)', ...
           'Location', 'SouthWest');
  end
end


function lev = empty_level_result()

  lev = struct('h', [], 'numpts', [], 'err_inf', [], ...
               'err_l2', [], 'crossrowsA', [], 'crossrowsB', []);
end


function reset_icpm2009bandingchecks(oldvalue)

  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = oldvalue;
end
