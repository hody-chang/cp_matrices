%% Two sphere caps glued along a circle, with branch rotation
% Manufactured-solution convergence for two sphere caps glued along a
% circle.
%
% The surface is the boundary of the intersection of two unit spheres with
% centers [-a 0 0] and [a 0 0].  Branch A is the cap x >= 0 on the left
% sphere and branch B is the cap x <= 0 on the right sphere.  The branch
% extension rows whose closest points are on the shared singular circle are
% rotated into the opposite branch and placed in the off-diagonal block of
% the closest point extension matrix.
%
% Every level is solved twice, once with the rotation and once without, so
% that the second-order rate with it can be read against the first-order
% rate without it.
%
% Run headlessly, from this directory:
%   matlab -batch "example_two_sphere_intersection_icpm_rotation_convergence"
% The geometry, the grid sizes, the plotting flags and the solver settings
% are all edited below.  hvals is the one to shorten for a quick run.


%% Using cp_matrices

% Resolve dependencies relative to this example.
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', '..', 'cp_matrices'));
addpath(fullfile(here, '..', '..', 'surfaces'));


global ICPM2009BANDINGCHECKS

% this is a bit dangerous: will break other less tightly banded
% codes, restore whatever the caller had on the way out.
% NB: the onCleanup object is created before the global is set.  This is a
% script, so cleanup_bandingchecks survives in the workspace; re-running
% destroys the previous object, and its destructor must not fire after the
% line that switches the checks on.
old_bandingchecks = ICPM2009BANDINGCHECKS;
cleanup_bandingchecks = ...
    onCleanup(@() reset_icpm2009bandingchecks(old_bandingchecks));
ICPM2009BANDINGCHECKS = 1;


%% Problem parameters

R = 1;        % sphere radius
a = 0.5;      % half the distance between the two sphere centers
p = 3;        % interpolation degree
order = 2;    % Laplacian order
beta = 1.0;   % weight of the symmetry-breaking even mode in the MMS

% grid sizes for the convergence study
hvals = 1./[20 40 80];

makePlots = true;
showdiag = true;

% 'auto' assembles M and uses backslash while the system is small enough,
% and falls back on matrix-free GMRES above that
solver = 'auto';
direct_max_unknowns = 2e5;
itertol = 1e-8;
itermaxit = 100;
gmres_restart = 50;

if (a <= 0 || a >= R)
  error('Expected 0 < a < R.');
end

hvals = hvals(:).';
if (any(hvals <= 0))
  error('All h values must be positive.');
end


%% Convergence study
% Each level is solved twice: with the branch rotation, and without it.

lev = repmat(empty_level_result(), length(hvals), 1);
lev_norot = repmat(empty_level_result(), length(hvals), 1);

for k = 1:length(hvals)
  lev(k) = solve_one_level(hvals(k), R, a, p, order, ...
                           makePlots, showdiag, solver, ...
                           direct_max_unknowns, itertol, itermaxit, ...
                           gmres_restart, true, beta);
  lev_norot(k) = solve_one_level(hvals(k), R, a, p, order, ...
                                 false, showdiag, solver, ...
                                 direct_max_unknowns, itertol, itermaxit, ...
                                 gmres_restart, false, beta);
end

results.h = hvals;
results.N = 1 ./ hvals;
results.errs_inf = [lev.err_inf];
results.errs_l2 = [lev.err_l2];
results.rates_inf = conv_rates(results.h, results.errs_inf);
results.rates_l2 = conv_rates(results.h, results.errs_l2);
results.branch_errsA = [[lev.err_infA].' [lev.err_l2A].'];
results.branch_errsB = [[lev.err_infB].' [lev.err_l2B].'];
results.band_errs_inf = [lev.band_err_inf];
results.band_errs_l2 = [lev.band_err_l2];
results.crossrowsA = [lev.crossrowsA];
results.crossrowsB = [lev.crossrowsB];
results.max_rowsum_err = [lev.max_rowsum_err];
results.max_singular_diff = [lev.max_singular_diff];
results.levels = lev;

results.norot.h = hvals;
results.norot.N = 1 ./ hvals;
results.norot.errs_inf = [lev_norot.err_inf];
results.norot.errs_l2 = [lev_norot.err_l2];
results.norot.rates_inf = conv_rates(hvals, results.norot.errs_inf);
results.norot.rates_l2 = conv_rates(hvals, results.norot.errs_l2);
results.norot.levels = lev_norot;

if (showdiag)
  printconv(results);
end

if (makePlots)
  plotconv(results);
end


%% ----------------------------------------------------------------------
%% local functions
%% ----------------------------------------------------------------------

function lev = solve_one_level(h, R, a, p, order, makePlots, showdiag, ...
                               solver, direct_max_unknowns, itertol, ...
                               itermaxit, gmres_restart, userot, beta)
%SOLVE_ONE_LEVEL  build the two glued caps at one grid size and solve

  dim = 3;  % dimension
  fd_stenrad = order/2;  % finite difference stencil radius
  % The formula for bw is found in [Ruuth & Merriman 2008] and the 1.0002
  % is a safety factor.
  bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ...
                   ((fd_stenrad+(p+1)/2)^2));

  circrad = sqrt(R^2 - a^2);
  pad = (bw + fd_stenrad + (p+1)/2 + 2)*h;
  xmax = h*ceil((R - a + pad)/h);
  yzmax = h*ceil((circrad + pad)/h);
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

  band_initA = find(abs(distA) <= bw*h);
  band_initB = find(abs(distB) <= bw*h);

  cpxg_initA = cpxA(band_initA);  cpyg_initA = cpyA(band_initA);
  cpzg_initA = cpzA(band_initA);
  xg_initA = xx(band_initA);  yg_initA = yy(band_initA);  zg_initA = zz(band_initA);
  bdyg_initA = bdyA(band_initA);

  cpxg_initB = cpxB(band_initB);  cpyg_initB = cpyB(band_initB);
  cpzg_initB = cpzB(band_initB);
  xg_initB = xx(band_initB);  yg_initB = yy(band_initB);  zg_initB = zz(band_initB);
  bdyg_initB = bdyB(band_initB);

  if (showdiag)
    if (userot)
      methodlabel = 'with rotation';
    else
      methodlabel = 'without rotation';
    end
    fprintf('\nh = %g (%s), grid %d x %d x %d\n', ...
            h, methodlabel, length(x1d), length(y1d), length(z1d));
    fprintf('Initial bands A/B: %d / %d\n', length(band_initA), length(band_initB));
  end

  [LA, EAA, RA, ibandA, obandA, ibandfullA, obandfullA] = ...
      ops_and_bands3d(x1d, y1d, z1d, xg_initA, yg_initA, zg_initA, ...
                      cpxg_initA, cpyg_initA, cpzg_initA, band_initA, p, order);
  [LB, EBB, RB, ibandB, obandB, ibandfullB, obandfullB] = ...
      ops_and_bands3d(x1d, y1d, z1d, xg_initB, yg_initB, zg_initB, ...
                      cpxg_initB, cpyg_initB, cpzg_initB, band_initB, p, order);

  cpxgoutA = cpxg_initA(obandA);  cpygoutA = cpyg_initA(obandA);
  cpzgoutA = cpzg_initA(obandA);
  xgoutA = xg_initA(obandA);  ygoutA = yg_initA(obandA);  zgoutA = zg_initA(obandA);
  bdygoutA = bdyg_initA(obandA);

  cpxgoutB = cpxg_initB(obandB);  cpygoutB = cpyg_initB(obandB);
  cpzgoutB = cpzg_initB(obandB);
  xgoutB = xg_initB(obandB);  ygoutB = yg_initB(obandB);  zgoutB = zg_initB(obandB);
  bdygoutB = bdyg_initB(obandB);

  cpxginA = RA*cpxgoutA;  cpyginA = RA*cpygoutA;  cpzginA = RA*cpzgoutA;
  cpxginB = RB*cpxgoutB;  cpyginB = RB*cpygoutB;  cpzginB = RB*cpzgoutB;

  % the rows to send across: those whose closest point sits on the shared
  % singular circle x = 0
  circtol = 100*eps(max(1, R));
  crossrowsA = find(bdygoutA | (abs(cpxgoutA) <= circtol));
  crossrowsB = find(bdygoutB | (abs(cpxgoutB) <= circtol));

  EAB = sparse(size(EAA, 1), size(EBB, 2));
  EBA = sparse(size(EBB, 1), size(EAA, 2));

  thetaAtoB = [];
  thetaBtoA = [];

  if (~isempty(crossrowsA))
    ptsA = [xgoutA(crossrowsA) ygoutA(crossrowsA) zgoutA(crossrowsA)];
    if (userot)
      singA = [cpxgoutA(crossrowsA) cpygoutA(crossrowsA) cpzgoutA(crossrowsA)];
      [rotA, thetaAtoB] = rotate_branch_points(ptsA, singA, R, cenA, cenB, ...
                                               sideA, sideB);
    else
      rotA = ptsA;
    end
    [cpxAtoB, cpyAtoB, cpzAtoB] = cpSphereCap(rotA(:,1), rotA(:,2), rotA(:,3), ...
                                              R, cenB, sideB);
    EAB(crossrowsA,:) = interp3_matrix(x1d, y1d, z1d, ...
                                       cpxAtoB(:), cpyAtoB(:), cpzAtoB(:), ...
                                       p, ibandfullB);
    EAA(crossrowsA,:) = 0;
  end

  if (~isempty(crossrowsB))
    ptsB = [xgoutB(crossrowsB) ygoutB(crossrowsB) zgoutB(crossrowsB)];
    if (userot)
      singB = [cpxgoutB(crossrowsB) cpygoutB(crossrowsB) cpzgoutB(crossrowsB)];
      [rotB, thetaBtoA] = rotate_branch_points(ptsB, singB, R, cenB, cenA, ...
                                               sideB, sideA);
    else
      rotB = ptsB;
    end
    [cpxBtoA, cpyBtoA, cpzBtoA] = cpSphereCap(rotB(:,1), rotB(:,2), rotB(:,3), ...
                                              R, cenA, sideA);
    EBA(crossrowsB,:) = interp3_matrix(x1d, y1d, z1d, ...
                                       cpxBtoA(:), cpyBtoA(:), cpzBtoA(:), ...
                                       p, ibandfullA);
    EBB(crossrowsB,:) = 0;
  end

  rhsA = mms_rhs(cpxginA, cpyginA, cpzginA, 'A', cenA, a, R, beta);
  rhsB = mms_rhs(cpxginB, cpyginB, cpzginB, 'B', cenB, a, R, beta);
  rhs = [rhsA; rhsB];
  if (any(~isfinite(rhs)))
    error('Non-finite MMS right-hand side encountered at branch closest points.');
  end

  u = solve_icp_system(LA, EAA, EAB, RA, LB, EBA, EBB, RB, rhs, ...
                       solver, direct_max_unknowns, itertol, itermaxit, ...
                       gmres_restart, showdiag);
  if (any(~isfinite(u)))
    error('Non-finite ICPM solution encountered after solving the elliptic system.');
  end

  uA = u(1:length(rhsA));
  uB = u(length(rhsA)+1:end);


  %% Errors, on the band and on the surface

  band_exactA = mms_exact(cpxginA, cpyginA, cpzginA, 'A', cenA, a, R, beta);
  band_exactB = mms_exact(cpxginB, cpyginB, cpzginB, 'B', cenB, a, R, beta);
  band_errA = uA - band_exactA;
  band_errB = uB - band_exactB;

  if (any(~isfinite([band_exactA; band_exactB; band_errA; band_errB])))
    error('Non-finite MMS exact values or errors encountered; check closest points and RHS.');
  end

  band_err_infA = norm(band_errA, inf);
  band_err_infB = norm(band_errB, inf);
  band_err_l2A = sqrt(mean(band_errA.^2));
  band_err_l2B = sqrt(mean(band_errB.^2));

  [err_infA, err_l2A, swtA, nevalA] = ...
      surface_error(x1d, y1d, z1d, ibandfullA, uA, p, 'A', cenA, a, R, h, beta);
  [err_infB, err_l2B, swtB, nevalB] = ...
      surface_error(x1d, y1d, z1d, ibandfullB, uB, p, 'B', cenB, a, R, h, beta);
  swt = swtA + swtB;
  err_inf = max(err_infA, err_infB);
  err_l2 = sqrt((swtA*err_l2A^2 + swtB*err_l2B^2) / swt);

  if (any(~isfinite([err_infA; err_infB; err_inf; err_l2A; err_l2B; err_l2])))
    error('Non-finite surface error diagnostic encountered.');
  end

  % every extension row should still be an interpolation: its weights sum
  % to one whichever branch they were fetched from
  rowsums = full([sum(EAA, 2) + sum(EAB, 2); ...
                  sum(EBA, 2) + sum(EBB, 2)]);
  max_rowsum_err = max(abs(rowsums - 1));

  [max_singular_diff, singular_exact_err] = singular_circle_diag( ...
      x1d, y1d, z1d, ibandfullA, ibandfullB, uA, uB, p, circrad, h, ...
      cenA, a, R, beta);

  if (showdiag)
    fprintf('Inner bands A/B: %d / %d\n', length(ibandfullA), length(ibandfullB));
    fprintf('Outer bands A/B: %d / %d\n', length(obandfullA), length(obandfullB));
    fprintf('Cross rows A->B / B->A: %d / %d\n', length(crossrowsA), length(crossrowsB));
    if (userot)
      fprintf('Rotation angles A->B range: [%g %g]\n', minval(thetaAtoB), maxval(thetaAtoB));
      fprintf('Rotation angles B->A range: [%g %g]\n', minval(thetaBtoA), maxval(thetaBtoA));
    end
    fprintf('E row-sum max error: %g\n', max_rowsum_err);
    fprintf('Singular-circle branch max difference: %g\n', max_singular_diff);
    fprintf('Singular-circle exact max error: %g\n', singular_exact_err);
    fprintf('Surface eval points A/B: %d / %d\n', nevalA, nevalB);
    fprintf('Surface errors Linf A/B/combined: %g / %g / %g\n', ...
            err_infA, err_infB, err_inf);
    fprintf('Surface errors L2   A/B/combined: %g / %g / %g\n', ...
            err_l2A, err_l2B, err_l2);
    fprintf('Band errors Linf A/B/combined: %g / %g / %g\n', ...
            band_err_infA, band_err_infB, max(band_err_infA, band_err_infB));
    fprintf('Band errors RMS  A/B/combined: %g / %g / %g\n', ...
            band_err_l2A, band_err_l2B, sqrt(mean([band_errA; band_errB].^2)));
  end

  if (makePlots)
    plotlens(x1d, y1d, z1d, ibandfullA, uA, ibandfullB, uB, ...
             p, cenA, cenB, a, R, h);
  end

  lev = empty_level_result();
  lev.h = h;
  lev.err_infA = err_infA;
  lev.err_infB = err_infB;
  lev.err_inf = err_inf;
  lev.err_l2A = err_l2A;
  lev.err_l2B = err_l2B;
  lev.err_l2 = err_l2;
  lev.band_err_infA = band_err_infA;
  lev.band_err_infB = band_err_infB;
  lev.band_err_inf = max(band_err_infA, band_err_infB);
  lev.band_err_l2A = band_err_l2A;
  lev.band_err_l2B = band_err_l2B;
  lev.band_err_l2 = sqrt(mean([band_errA; band_errB].^2));
  lev.nevalA = nevalA;
  lev.nevalB = nevalB;
  lev.crossrowsA = length(crossrowsA);
  lev.crossrowsB = length(crossrowsB);
  lev.innerbandA = length(ibandfullA);
  lev.innerbandB = length(ibandfullB);
  lev.outerbandA = length(obandfullA);
  lev.outerbandB = length(obandfullB);
  lev.max_rowsum_err = max_rowsum_err;
  lev.max_singular_diff = max_singular_diff;
  lev.max_singular_exact_err = singular_exact_err;
end


function u = solve_icp_system(LA, EAA, EAB, RA, LB, EBA, EBB, RB, rhs, ...
                              solver, direct_max_unknowns, itertol, ...
                              itermaxit, gmres_restart, showdiag)
%SOLVE_ICP_SYSTEM  (I - M) u = rhs for the two coupled branches
%   Assembles M and uses backslash when the system is small enough,
%   otherwise applies M matrix-free under restarted GMRES.

  solver = normalize_solver(solver);

  nA = size(LA, 1);
  nB = size(LB, 1);
  n = nA + nB;

  [LdiagA, LoffA] = split_lapsharp(LA, RA);
  [LdiagB, LoffB] = split_lapsharp(LB, RB);

  usedirect = strcmp(solver, 'direct') || ...
              (strcmp(solver, 'auto') && n <= direct_max_unknowns);

  if (showdiag)
    fprintf('Linear solve unknowns A/B/total: %d / %d / %d\n', nA, nB, n);
    fprintf('Interpolation nnz EAA/EBB/EAB/EBA: %d / %d / %d / %d\n', ...
            nnz(EAA), nnz(EBB), nnz(EAB), nnz(EBA));
    fprintf('Estimated E sparse storage: %.1f MB\n', ...
            sparse_storage_mb(EAA, EBB, EAB, EBA));
    fprintf('L off-diagonal nnz A/B: %d / %d\n', nnz(LoffA), nnz(LoffB));
  end

  if (usedirect)
    if (showdiag)
      fprintf('Linear solver: direct sparse backslash\n');
    end

    MAA = spdiags(LdiagA, 0, nA, nA) + LoffA*EAA;
    MAB = LoffA*EAB;
    MBA = LoffB*EBA;
    MBB = spdiags(LdiagB, 0, nB, nB) + LoffB*EBB;
    M = [MAA MAB; MBA MBB];
    A = speye(n) - M;

    if (showdiag)
      fprintf('Explicit M nnz: %d, estimated sparse storage: %.1f MB\n', ...
              nnz(M), sparse_storage_mb(M));
    end

    u = A \ rhs;
  else
    if (showdiag)
      fprintf('Linear solver: matrix-free restarted GMRES');
      if (strcmp(solver, 'auto'))
        fprintf(' (auto: direct threshold %d unknowns)', direct_max_unknowns);
      end
      fprintf('\n');
      fprintf('Explicit M is not assembled in this solve path.\n');
    end

    afun = @(v) apply_icp_system(v, LdiagA, LoffA, EAA, EAB, ...
                                 LdiagB, LoffB, EBA, EBB);
    preconddiag = 1 - [LdiagA; LdiagB];
    preconddiag(abs(preconddiag) < eps) = 1;
    mfun = @(v) v ./ preconddiag;

    [u, flag, relres, iter] = gmres(afun, rhs, gmres_restart, itertol, ...
                                    itermaxit, mfun);
    if (showdiag)
      fprintf('GMRES flag: %d, relres: %g, iter: %s\n', ...
              flag, relres, mat2str(iter));
    end
    if (flag ~= 0)
      warning('example_two_sphere:gmresNoConvergence', ...
              'GMRES did not meet the requested tolerance; flag=%d, relres=%g.', ...
              flag, relres);
    end
  end
end


function y = apply_icp_system(v, LdiagA, LoffA, EAA, EAB, ...
                              LdiagB, LoffB, EBA, EBB)
%APPLY_ICP_SYSTEM  y = (I - M) v without assembling M

  nA = length(LdiagA);
  vA = v(1:nA);
  vB = v(nA+1:end);

  extA = EAA*vA + EAB*vB;
  extB = EBA*vA + EBB*vB;

  mvA = LdiagA.*vA + LoffA*extA;
  mvB = LdiagB.*vB + LoffB*extB;

  y = v - [mvA; mvB];
end


function [Ldiag, Loff] = split_lapsharp(L, R)
%SPLIT_LAPSHARP  the self-column of L, and everything else

  Ldiagpad = R .* L;
  Ldiag = full(sum(Ldiagpad, 2));
  Loff = L - Ldiagpad;
end


function solver = normalize_solver(solver)

  if (~ischar(solver))
    error('solver must be ''auto'', ''direct'', or ''gmres''.');
  end

  solver = lower(solver);
  if (strcmp(solver, 'backslash'))
    solver = 'direct';
  elseif (strcmp(solver, 'iterative'))
    solver = 'gmres';
  end

  if (~strcmp(solver, 'auto') && ~strcmp(solver, 'direct') && ...
      ~strcmp(solver, 'gmres'))
    error('solver must be ''auto'', ''direct'', or ''gmres''.');
  end
end


function mb = sparse_storage_mb(varargin)

  bytes = 0;
  for k = 1:nargin
    A = varargin{k};
    bytes = bytes + 16*nnz(A) + 8*(size(A, 2) + 1);
  end
  mb = bytes / 1024^2;
end


function [cpx, cpy, cpz, dist, bdy] = cpSphereCap(x, y, z, R, cen, side)
%CPSPHERECAP  closest point to a sphere cap cut by x = 0
%   side =  1 gives the cap x >= 0.
%   side = -1 gives the cap x <= 0.

  xs = x - cen(1);
  ys = y - cen(2);
  zs = z - cen(3);
  rr = sqrt(xs.^2 + ys.^2 + zs.^2);

  rr_safe = rr;
  rr_safe(rr_safe == 0) = 1;

  cpx = cen(1) + R*xs ./ rr_safe;
  cpy = cen(2) + R*ys ./ rr_safe;
  cpz = cen(3) + R*zs ./ rr_safe;

  zerorows = (rr == 0);
  if (any(zerorows(:)))
    cpx(zerorows) = cen(1) + R;
    cpy(zerorows) = cen(2);
    cpz(zerorows) = cen(3);
  end

  insidecap = (side*cpx >= 0);
  bdy = ~insidecap;

  if (any(bdy(:)))
    circrad = sqrt(R^2 - cen(1)^2);
    rho = sqrt(y.^2 + z.^2);
    rho_safe = rho;
    rho_safe(rho_safe == 0) = 1;

    cpx_circ = zeros(size(x));
    cpy_circ = circrad*y ./ rho_safe;
    cpz_circ = circrad*z ./ rho_safe;

    axisrows = (rho == 0);
    if (any(axisrows(:)))
      cpy_circ(axisrows) = circrad;
      cpz_circ(axisrows) = 0;
    end

    cpx(bdy) = cpx_circ(bdy);
    cpy(bdy) = cpy_circ(bdy);
    cpz(bdy) = cpz_circ(bdy);
  end

  dist = sqrt((x - cpx).^2 + (y - cpy).^2 + (z - cpz).^2);
end


function [rotpts, theta] = rotate_branch_points(pts, singpts, R, ...
                                                cen_from, cen_to, ...
                                                side_from, side_to)
%ROTATE_BRANCH_POINTS  turn the embedding points about the singular circle
%   The rotation axis is tau, the tangent of the singular circle.  The
%   estimated conormal of the branch the point came from is turned onto
%   minus the conormal of the branch it is going to.

  n_from = normalize_rows(bsxfun(@minus, singpts, cen_from) ./ R);
  n_to = normalize_rows(bsxfun(@minus, singpts, cen_to) ./ R);

  if (side_from == 1)
    tau = normalize_rows(cross(n_from, n_to, 2));
  else
    tau = normalize_rows(cross(n_to, n_from, 2));
  end

  eta_probe = cap_conormal(n_from, side_from);
  eta_to = cap_conormal(n_to, side_to);
  targetdir = -eta_to;
  Rk = cell(size(pts, 1), 1);
  theta = NaN(size(pts, 1), 1);
  cpf_from = @(x, y, z) cpSphereCap(x, y, z, R, cen_from, side_from);

  min_probedist = sqrt(eps)*max(1, R);
  for k = 1:size(pts, 1)
    [Rk{k}, ~, info] = angle3D(pts(k,1), pts(k,2), pts(k,3), cpf_from, ...
                               singpts(k,:), tau(k,:), targetdir(k,:));
    theta(k) = info.theta;

    % a degenerate cp - cpbar at this row: probe again from a point pushed
    % off along the estimated conormal instead
    if (any(~isfinite(Rk{k}(:))))
      probedist = max(norm(pts(k,:) - singpts(k,:)), min_probedist);
      probept = singpts(k,:) + probedist*eta_probe(k,:);
      [Rk{k}, ~, info] = angle3D(probept(1), probept(2), probept(3), cpf_from, ...
                                 singpts(k,:), tau(k,:), targetdir(k,:));
      theta(k) = info.theta;
    end
  end

  allfinite = true;
  for k = 1:size(pts, 1)
    if (any(~isfinite(Rk{k}(:))))
      allfinite = false;
      break;
    end
  end
  if (~allfinite)
    error('angle3D did not return finite branch rotation matrices.');
  end

  rotpts = zeros(size(pts));
  for k = 1:size(pts, 1)
    v = (pts(k,:) - singpts(k,:)).';
    rotpts(k,:) = singpts(k,:) + (Rk{k} * v).';
  end
end


function eta = cap_conormal(normal, side)
%CAP_CONORMAL  the outward conormal of a cap at its rim

  capout = [-side*ones(size(normal,1),1) zeros(size(normal,1),1) ...
            zeros(size(normal,1),1)];
  eta = capout - bsxfun(@times, dot_rows(capout, normal), normal);
  eta = normalize_rows(eta);
end


function u = mms_exact(x, y, z, branch, cen, a, R, beta)
%MMS_EXACT  the manufactured solution at a point of one branch

  [s, ~, k] = mms_coord(x, branch, cen, a, R);
  u = mms_profile(s, k, beta);
end


function [u, Us, Uss] = mms_profile(s, k, beta)
%MMS_PROFILE  smooth meridional profile in arc length s
%   The odd mode sin(k s) is the original antisymmetric manufactured
%   solution; the even mode beta*cos(2 k s) breaks the branch symmetry (and
%   the symmetry of |u|) while keeping the Neumann conditions u_s = 0 at
%   both poles s = +/- L = +/- pi/(2k).

  u = sin(k*s) + beta*cos(2*k*s);
  Us = k*cos(k*s) - 2*k*beta*sin(2*k*s);
  Uss = -k^2*sin(k*s) - 4*k^2*beta*cos(2*k*s);
end


function f = mms_rhs(x, y, z, branch, cen, a, R, beta)
%MMS_RHS  the right-hand side f = u - lap_S u of the manufactured problem

  [s, theta, k] = mms_coord(x, branch, cen, a, R);

  [u, Us, Uss] = mms_profile(s, k, beta);

  sintheta = sin(theta);
  regrows = (abs(sintheta) > sqrt(eps));
  cotterm = zeros(size(theta));
  cotterm(regrows) = (cos(theta(regrows)) ./ sintheta(regrows)) .* ...
                     Us(regrows) / R;
  % at a pole the cot term has the removable limit U_ss
  cotterm(~regrows) = Uss(~regrows);

  lapexact = Uss + cotterm;
  f = u - lapexact;
end


function [s, theta, k] = mms_coord(x, branch, cen, a, R)
%MMS_COORD  unfolded meridional arc length, polar angle and wavenumber

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
end


function [maxdiff, max_exact_err] = singular_circle_diag(x1d, y1d, z1d, ...
                                                         ibandA, ibandB, ...
                                                         uA, uB, p, ...
                                                         circrad, h, ...
                                                         cenA, a, R, beta)
%SINGULAR_CIRCLE_DIAG  how far apart the two branches are on the shared
%   circle, and how far their average is from the exact solution there

  ntheta = max(32, ceil(2*pi*circrad/(2*h)));
  th = linspace(0, 2*pi, ntheta+1).';
  th(end) = [];
  sx = zeros(size(th));
  sy = circrad*cos(th);
  sz = circrad*sin(th);

  EsingA = interp3_matrix(x1d, y1d, z1d, sx, sy, sz, p, ibandA);
  EsingB = interp3_matrix(x1d, y1d, z1d, sx, sy, sz, p, ibandB);

  valsA = EsingA*uA;
  valsB = EsingB*uB;
  exact = mms_exact(sx, sy, sz, 'A', cenA, a, R, beta);

  if (any(~isfinite([valsA; valsB; exact])))
    error('Non-finite singular-circle diagnostic values encountered.');
  end

  maxdiff = max(abs(valsA - valsB));
  max_exact_err = max(abs(0.5*(valsA + valsB) - exact));
end


function [err_inf, err_l2, swt, neval] = surface_error( ...
      x1d, y1d, z1d, iband, u, p, branch, cen, a, R, h, beta)
%SURFACE_ERROR  the error on the cap itself, by midpoint quadrature in
%   (theta, phi)

  theta0 = acos(a/R);
  if (strcmp(branch, 'A'))
    thmin = 0;
    thmax = theta0;
  elseif (strcmp(branch, 'B'))
    thmin = pi - theta0;
    thmax = pi;
  else
    error('Unknown MMS branch "%s".', branch);
  end

  circrad = sqrt(R^2 - a^2);
  ntheta = max(16, ceil(theta0*R/h*2));
  nphi = max(32, ceil(2*pi*circrad/h*2));
  dtheta = (thmax - thmin) / ntheta;
  dphi = 2*pi / nphi;

  theta = thmin + ((0:ntheta-1).' + 0.5)*dtheta;
  phi = ((0:nphi-1).' + 0.5)*dphi;
  [Theta, Phi] = ndgrid(theta, phi);

  sx = cen(1) + R*cos(Theta(:));
  sy = R*sin(Theta(:)).*cos(Phi(:));
  sz = R*sin(Theta(:)).*sin(Phi(:));
  weights = R^2*sin(Theta(:))*dtheta*dphi;

  Eeval = interp3_matrix(x1d, y1d, z1d, sx, sy, sz, p, iband);
  vals = Eeval*u;
  exact = mms_exact(sx, sy, sz, branch, cen, a, R, beta);
  err = vals - exact;

  if (any(~isfinite([vals; exact; err; weights])) || any(weights < 0))
    error('Non-finite or negative surface quadrature values encountered.');
  end

  swt = sum(weights);
  if (~isfinite(swt) || swt <= 0)
    error('Invalid surface quadrature weight encountered.');
  end

  err_inf = norm(err, inf);
  err_l2 = sqrt(sum(weights.*(err.^2)) / swt);
  neval = length(err);
end


function rates = conv_rates(h, err)

  if (length(h) < 2)
    rates = [];
    return;
  end

  rates = log(err(1:end-1) ./ err(2:end)) ./ log(h(1:end-1) ./ h(2:end));
end


function printconv(results)

  printconv_one('with rotation', results.h, ...
                results.errs_inf, results.rates_inf, ...
                results.errs_l2, results.rates_l2);
  printconv_one('without rotation', results.norot.h, ...
                results.norot.errs_inf, results.norot.rates_inf, ...
                results.norot.errs_l2, results.norot.rates_l2);
end


function printconv_one(label, h, errs_inf, rates_inf, errs_l2, rates_l2)

  fprintf('\nConvergence summary (%s)\n', label);
  fprintf('       h    surface Linf      rate     surface L2       rate\n');
  for k = 1:length(h)
    if (k == 1)
      fprintf('%8.4g  %14.6e      --   %14.6e      --\n', ...
              h(k), errs_inf(k), errs_l2(k));
    else
      fprintf('%8.4g  %14.6e  %6.3f   %14.6e  %6.3f\n', ...
              h(k), errs_inf(k), rates_inf(k-1), ...
              errs_l2(k), rates_l2(k-1));
    end
  end
end


function plotconv(results)
%PLOTCONV  the surface error against N = 1/h, with and without the rotation

  [N, ord] = sort(results.N);
  err_inf = results.errs_inf(ord);
  err_l2 = results.errs_l2(ord);
  err_inf_norot = results.norot.errs_inf(ord);
  err_l2_norot = results.norot.errs_l2(ord);

  ok = isfinite(N) & isfinite(err_inf) & isfinite(err_l2) & ...
       isfinite(err_inf_norot) & isfinite(err_l2_norot) & ...
       (N > 0) & (err_inf > 0) & (err_l2 > 0) & ...
       (err_inf_norot > 0) & (err_l2_norot > 0);
  N = N(ok);
  err_inf = err_inf(ok);
  err_l2 = err_l2(ok);
  err_inf_norot = err_inf_norot(ok);
  err_l2_norot = err_l2_norot(ok);

  if (isempty(N))
    return;
  end

  ref2 = err_l2(1)*(N/N(1)).^(-2);
  ref1 = err_l2_norot(1)*(N/N(1)).^(-1);

  figure(2); clf;
  loglog(N, err_inf, 'o-', N, err_l2, 's-', ...
         N, err_inf_norot, 'o--', N, err_l2_norot, 's--', ...
         N, ref2, 'k--', N, ref1, 'k:', 'LineWidth', 1.5);
  xlabel('N = 1/h');
  ylabel('surface error');
  title('two-sphere ICPM convergence');
  legend('L_\infty error (rotation)', 'L_2 error (rotation)', ...
         'L_\infty error (no rotation)', 'L_2 error (no rotation)', ...
         'O(h^2)', 'O(h)', 'Location', 'southwest');
  grid on;
  drawnow(); pause(0);
end


function plotlens(x1d, y1d, z1d, ibandA, uA, ibandB, uB, ...
                  p, cenA, cenB, a, R, h)
%PLOTLENS  the solution on the two caps

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
end


function v = normalize_rows(v)

  n = sqrt(sum(v.^2, 2));
  n(n == 0) = 1;
  v = bsxfun(@rdivide, v, n);
end


function d = dot_rows(a, b)

  d = sum(a.*b, 2);
end


function value = minval(x)

  if (isempty(x))
    value = NaN;
  else
    value = min(x);
  end
end


function value = maxval(x)

  if (isempty(x))
    value = NaN;
  else
    value = max(x);
  end
end


function lev = empty_level_result()

  lev = struct('h', NaN, ...
               'err_infA', NaN, ...
               'err_infB', NaN, ...
               'err_inf', NaN, ...
               'err_l2A', NaN, ...
               'err_l2B', NaN, ...
               'err_l2', NaN, ...
               'band_err_infA', NaN, ...
               'band_err_infB', NaN, ...
               'band_err_inf', NaN, ...
               'band_err_l2A', NaN, ...
               'band_err_l2B', NaN, ...
               'band_err_l2', NaN, ...
               'nevalA', 0, ...
               'nevalB', 0, ...
               'crossrowsA', 0, ...
               'crossrowsB', 0, ...
               'innerbandA', 0, ...
               'innerbandB', 0, ...
               'outerbandA', 0, ...
               'outerbandB', 0, ...
               'max_rowsum_err', NaN, ...
               'max_singular_diff', NaN, ...
               'max_singular_exact_err', NaN);
end


function reset_icpm2009bandingchecks(oldvalue)

  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = oldvalue;
end
