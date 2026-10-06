function results = example_half_twisted_rectangular_tube_rotation_convergence(hvals)
%EXAMPLE_HALF_TWISTED_RECTANGULAR_TUBE_ROTATION_CONVERGENCE  ICPM glue study
%   on the half-twisted rectangular tube, split into two strips along BOTH
%   closed corner curves, with exact and d_k2 branch rotations.
%
%   results = example_half_twisted_rectangular_tube_rotation_convergence()
%   results = example_half_twisted_rectangular_tube_rotation_convergence(hvals)
%
%   Run headlessly from any directory, e.g.
%     matlab -batch "example_half_twisted_rectangular_tube_rotation_convergence"
%     matlab -batch "example_half_twisted_rectangular_tube_rotation_convergence(1./[25 30 40])"
%   hvals defaults to 1./[25 30 40].  Rates use the actual h ratios.
%
%   Geometry (examples_2026/surfaces/halfTwistedRectTubeFrame.m).  R = 1, a = 0.45,
%   b = 0.20.  Branch A (= branch 1, wide strip) p = (s,b), s in [-a,a];
%   branch B (= branch 2, narrow strip) p = (a,s), s in [-b,b].  Each strip
%   is parametrized once by theta in [0,4*pi).  The strips meet along two
%   closed corner curves:
%     upper curve: A at (theta, +a) = B at (theta, +b),
%     lower curve: A at (theta, -a) = B at (theta+2*pi, -b)  (mod 4*pi).
%
%   Problem.  u - Lap_S u = f on the closed (torus) surface, with the
%   branchwise analytic manufactured solution halfTwistedRectTubeMMS.m
%   (shared traces and balanced conormal fluxes on both curves).
%
%   Discretization (per level, bands built once by halfTwistedRectTubeBands.m
%   and reused by both methods).  Degree p = 3 interpolation, 7-point
%   Laplacian, stabilized (diagonally split) ICPM operator
%       M = diag(Ldiag) + Loff*E,  Ldiag = sum(R.*L,2),  Loff = L - R.*L,
%   solved as (I - M) u = R*f_out.  Every complete stencil is required:
%   the 7-point Laplacian neighbours of each inner node must be initial
%   band nodes, and every interpolation stencil must lie inside the
%   relevant inner band; otherwise the run errors.  The inner band is the
%   union of the stencils of the initial band closest points and of every
%   surface evaluation sample (error grid and corner-curve trace samples).
%
%   Routing.  An outer row whose own-branch closest point is on an edge
%   (bdy ~= 0) is routed to the opposite branch: its self row is zeroed
%   and replaced by interpolation at the target-branch closest point of the
%   rotated grid node.  The right-hand side on a routed row is the TARGET
%   branch f evaluated at that target closest point (before applying R),
%   as in the lens example.  Both methods route identical rows; only the
%   rotation angle differs.  Bands and routing are closed together: the
%   inner band of each branch also contains the interpolation support of
%   the target closest points of rows routed into it by EITHER rotation,
%   and new outer rows created by that enlargement are routed in turn,
%   until nothing new is required (all within the conservative initial
%   band, else error).  Both methods therefore share bands and rows.
%
%   Rotation.  At the edge closest point c with analytic frame F, Fs, Ft:
%   tau = Ft/|Ft| (shared by both strips), source conormal
%   eta_s = sign(s)*unit(Fs - (Fs.tau)tau), and eta_t likewise on the
%   target strip at the glued parameters.  The grid node x is rotated about
%   the axis through c along tau by the signed angle taking the source
%   conormal onto minus the target conormal (Rodrigues formula).
%     'exact': the analytic eta_s and eta_t above.
%     'dk2'  : both conormals are pointwise second-order closest point
%              differences, projected onto the plane normal to the exact
%              tau and normalized; the axis is the exact tau in both
%              methods.  This is NOT a fully geometry-free estimator.
%       Source: from the original routed grid node x itself, with
%         v = c - x,  d = 1.5*c - 2*cp_s(c + v) + 0.5*cp_s(c + 2v)
%         (one-sided second-order difference along the reflected ray;
%         c + v = 2c - x, c + 2v = 3c - 2x).  The row is re-sampled with
%         a deterministic outward probe x' = c + h*eta_s, i.e.
%         v = -h*eta_s, if: |v| is tiny; the projected d is short
%         relative to |v| (x almost straight above the edge along the
%         surface normal: alpha = |d_perp|/|v| < alpha_min); a reflected
%         closest point is itself on an edge; it jumps to the opposite
%         sheet (theta differs by about 2*pi); or the estimate points
%         inward, d_perp.eta_s <= 0 (sign reversal from long reflected
%         samples on coarse grids).  The exact eta_s only decides whether
%         to probe and places the probe; the estimate is still the closest
%         point difference.  Probed rows are counted
%         (degenerate_source_rows, of which signflip_source_rows were sign
%         reversals).
%       Target: always the outward probe c_t + h*eta_t.  The exact eta_t
%         only places the ambient sample points c_t - h*eta_t and
%         c_t - 2h*eta_t, which point toward the target strip interior;
%         their target closest points are required to be interior, on the
%         same sheet, and to give an outward estimate.
%       If a probe is still unusable the run errors; there is no fallback
%       to the exact conormal.  Accuracy is only asymptotic and local:
%       when the reflected ray stays where the closest point map is smooth,
%       the error is O(|v|^2/alpha) with |v| <= bw*h, alpha >= alpha_min.
%       Accepted samples on coarse grids are not guaranteed to be in that
%       regime, so the measured max angle error is reported per level and
%       the rate is judged from fine levels, not assumed.  Directions are
%       row-specific:
%       no averaging across curve locations.
%
%   Solver.  Sparse backslash when the unknown count is <= 25000,
%   otherwise matrix-free restarted GMRES with the (I - Ldiag) diagonal
%   preconditioner (tol 1e-10, restart 80, maxit 200 outer cycles); a
%   GMRES failure is an ERROR.  The relative residual of the returned
%   solution is recomputed and stored.
%
%   Errors.  Surface sup error sampled on a uniform theta x s grid of each
%   strip (both edges included, about 2 samples per h), per branch and
%   combined, plus the trace discrepancy of the two branches on each
%   corner curve.
%
%   Output: results.h, results.params, results.exact and results.dk2 with
%   fields errs_inf, rates_inf, fit_rate, levels (per-level diagnostics:
%   err_infA/B, trace_diff, angle_error, routed row counts per branch and
%   curve, unknown counts, row sums, residual, ...).  A convergence PNG and
%   MAT are written to examples_2026/figs/
%   half_twisted_rectangular_tube_rotation_convergence.{png,mat}.

  here = fileparts(mfilename('fullpath'));
  addpath(fullfile(here, '..', '..', 'cp_matrices'));
  addpath(fullfile(here, '..', '..', 'surfaces'));
  addpath(fullfile(here, '..', 'surfaces'));
  addpath(here);

  %% Fixed, editable solver parameters (geometry/banding defaults live in
  %% halfTwistedRectTubeBands.m: R = 1, a = 0.45, b = 0.20, p = 3, ...)
  P.direct_max_unknowns = 25000;
  P.gmres_tol = 1e-10;
  P.gmres_restart = 80;
  P.gmres_maxit = 200;

  if (nargin < 1 || isempty(hvals))
    hvals = 1./[25 30 40];
  end
  hvals = hvals(:).';
  if (any(~isfinite(hvals)) || any(hvals <= 0))
    error('All h values must be positive and finite.');
  end
  % geometry/banding parameters (R, a, b, p, order, grid_shift,
  % alpha_min, eval_batch, max_closure_iter) are filled by
  % halfTwistedRectTubeBands, which also checks R > sqrt(a^2+b^2)

  methods = {'exact', 'dk2'};
  nlev = numel(hvals);
  levs = struct('exact', cell(1, nlev), 'dk2', cell(1, nlev));

  for k = 1:nlev
    h = hvals(k);
    tlev = tic;
    [geo, P] = halfTwistedRectTubeBands(h, P);
    fprintf('\nh = %.6g: grid %d x %d x %d, inner A/B %d / %d, outer A/B %d / %d, closure sweeps %d (bands %.1fs)\n', ...
            h, geo.nx, geo.ny, geo.nz, geo.B(1).ni, geo.B(2).ni, ...
            geo.B(1).no, geo.B(2).no, geo.closure_iter, toc(tlev));
    for m = 1:numel(methods)
      tm = tic;
      lev = solve_method(geo, methods{m}, P);
      lev.time = toc(tm);
      levs(k).(methods{m}) = lev;
      print_level(lev);
    end
  end

  results.h = hvals;
  results.params = P;
  for m = 1:numel(methods)
    name = methods{m};
    L = [levs.(name)];
    r.errs_inf = [L.err_inf];
    r.rates_inf = conv_rates(hvals, r.errs_inf);
    r.fit_rate = fit_rate(hvals, r.errs_inf);
    r.errs_infA = [L.err_infA];
    r.errs_infB = [L.err_infB];
    r.rates_infA = conv_rates(hvals, r.errs_infA);
    r.rates_infB = conv_rates(hvals, r.errs_infB);
    r.levels = L;
    results.(name) = r;
  end

  print_summary(results);

  figdir = fullfile(here, '..', 'figs');
  if (~exist(figdir, 'dir'))
    mkdir(figdir);
  end
  base = fullfile(figdir, 'half_twisted_rectangular_tube_rotation_convergence');
  plot_convergence(results, [base '.png']);
  save([base '.mat'], 'results');
  fprintf('Saved %s.png and %s.mat\n', base, base);
end


%% ----------------------------------------------------------------------
%% Interpolation (bands, routing and samples: halfTwistedRectTubeBands.m)
%% ----------------------------------------------------------------------

function E = interp_band(x1d, y1d, z1d, px, py, pz, p, band, what)
%INTERP_BAND  interpolation onto a band, requiring complete stencils

  [Ei, Ej, Es] = interp3_matrix(x1d, y1d, z1d, px(:), py(:), pz(:), p);
  [tf, jj] = ismember(Ej, band);
  if (~all(tf))
    error('Incomplete interpolation stencil (%s): %d stencil nodes outside the inner band.', ...
          what, nnz(~tf));
  end
  E = sparse(Ei, jj, Es, numel(px), numel(band));
end


%% ----------------------------------------------------------------------
%% One method on one level
%% ----------------------------------------------------------------------

function lev = solve_method(geo, method, P)
%SOLVE_METHOD  assemble from the cached routing targets, solve, measure

  h = geo.h;
  B = geo.B;
  Eself = cell(1, 2);  Ecross = cell(1, 2);  rhs = cell(1, 2);
  lev = empty_level(h, method);
  lev.closure_iter = geo.closure_iter;

  for src = 1:2
    tgt = 3 - src;
    S = B(src);  T = B(tgt);
    routed = (S.bdy ~= 0);
    rows = find(routed);
    [tf, loc] = ismember(S.oband(rows), geo.route{src}.node);
    if (~all(tf))
      error('Routed row of branch %d without a cached routing target.', src);
    end
    Rt = geo.route{src};
    if (strcmp(method, 'exact'))
      tc = Rt.cpE(loc, :);
      angerr = zeros(numel(rows), 1);
    else
      tc = Rt.cpD(loc, :);
      % the angle of the relative rotation, precomputed in
      % halfTwistedRectTubeBands from the (cos, sin) pairs rather than as a
      % rewrapped difference of two angles.  See ../rotations/rot2d_err.m.
      angerr = Rt.angerr_dk2(loc);
      lev.degenerate_source_rows(src) = nnz(Rt.probe(loc));
      lev.signflip_source_rows(src) = nnz(Rt.signflip(loc));
    end
    sg = Rt.sg(loc);

    Ec = interp_band(geo.x1d, geo.y1d, geo.z1d, tc(:,1), tc(:,2), tc(:,3), ...
                     P.p, T.iband, sprintf('routed rows %d->%d', src, tgt));
    [ii, jj, vv] = find(Ec);
    Ecross{src} = sparse(rows(ii), jj, vv, S.no, T.ni);
    Eself{src} = spdiags(double(~routed), 0, S.no, S.no) * S.E;

    % routed rows carry the TARGET branch f at the target closest point
    [~, fout] = halfTwistedRectTubeMMS(S.theta, S.s, src, P.R, P.a, P.b);
    [~, froute] = halfTwistedRectTubeMMS(tc(:,4), tc(:,5), tgt, P.R, P.a, P.b);
    fout(rows) = froute;
    rhs{src} = S.R * fout;

    up = (sg > 0);
    lev.routed(src, :) = [nnz(up) nnz(~up)];
    lev.angle_error(src, :) = [maxval(angerr(up)) maxval(angerr(~up))];
    lev.angle_exact_range(src, :) = [minval(Rt.phi_exact(loc)) maxval(Rt.phi_exact(loc))];
  end

  rowsum = full([sum(Eself{1}, 2) + sum(Ecross{1}, 2); ...
                 sum(Eself{2}, 2) + sum(Ecross{2}, 2)]);
  lev.max_rowsum_err = max(abs(rowsum - 1));
  if (~(lev.max_rowsum_err < 1e-10))
    error('Extension row sums deviate from one by %g.', lev.max_rowsum_err);
  end

  b = [rhs{1}; rhs{2}];
  if (any(~isfinite(b)))
    error('Non-finite right-hand side.');
  end
  [u, sol] = solve_system(B, Eself, Ecross, b, P);
  uA = u(1:B(1).ni);  uB = u(B(1).ni+1:end);

  [lev.err_infA, lev.nsurfA] = surface_error(geo, 1, uA, P);
  [lev.err_infB, lev.nsurfB] = surface_error(geo, 2, uB, P);
  lev.err_inf = max(lev.err_infA, lev.err_infB);
  lev.trace_diff = trace_discrepancy(geo, uA, uB, P);
  if (any(~isfinite([lev.err_inf lev.trace_diff])))
    error('Non-finite surface error diagnostics.');
  end

  lev.unknowns = [B(1).ni B(2).ni];
  lev.outer = [B(1).no B(2).no];
  lev.solver = sol.solver;
  lev.relres = sol.relres;
  lev.gmres_iter = sol.iter;
  lev.max_angle_error = max(lev.angle_error(:));
end


%% ----------------------------------------------------------------------
%% Linear solve
%% ----------------------------------------------------------------------

function [u, sol] = solve_system(B, Eself, Ecross, b, P)
%SOLVE_SYSTEM  (I - M) u = b, direct when small, else matrix-free GMRES

  nA = B(1).ni;  nB = B(2).ni;  n = nA + nB;
  afun = @(v) apply_system(v, B, Eself, Ecross);

  if (n <= P.direct_max_unknowns)
    M = [spdiags(B(1).Ldiag, 0, nA, nA) + B(1).Loff*Eself{1}, B(1).Loff*Ecross{1}; ...
         B(2).Loff*Ecross{2}, spdiags(B(2).Ldiag, 0, nB, nB) + B(2).Loff*Eself{2}];
    u = (speye(n) - M) \ b;
    sol.solver = 'direct';
    sol.iter = 0;
  else
    pdiag = 1 - [B(1).Ldiag; B(2).Ldiag];
    mfun = @(v) v ./ pdiag;
    [u, flag, relres, iter] = gmres(afun, b, P.gmres_restart, P.gmres_tol, ...
                                    P.gmres_maxit, mfun);
    if (flag ~= 0)
      error('GMRES failed: flag %d, relres %g, iter %s (n = %d).', ...
            flag, relres, mat2str(iter), n);
    end
    sol.solver = 'gmres';
    sol.iter = (iter(1) - 1)*P.gmres_restart + iter(end);
  end

  sol.relres = norm(b - afun(u)) / norm(b);
  if (any(~isfinite(u)) || ~isfinite(sol.relres))
    error('Non-finite solution or residual.');
  end
  if (sol.relres > 1e-8)
    error('Linear solve residual too large: %g.', sol.relres);
  end
end


function y = apply_system(v, B, Eself, Ecross)
%APPLY_SYSTEM  y = (I - M) v without assembling M

  nA = B(1).ni;
  vA = v(1:nA);  vB = v(nA+1:end);
  extA = Eself{1}*vA + Ecross{1}*vB;
  extB = Ecross{2}*vA + Eself{2}*vB;
  y = v - [B(1).Ldiag.*vA + B(1).Loff*extA; B(2).Ldiag.*vB + B(2).Loff*extB];
end


%% ----------------------------------------------------------------------
%% Errors on the surface
%% ----------------------------------------------------------------------

function [err_inf, npts] = surface_error(geo, br, u, P)
%SURFACE_ERROR  sup error on a uniform (theta, s) grid incl. both edges

  th = geo.samples{br}.theta;  s = geo.samples{br}.s;
  vals = eval_branch(geo, br, u, th, s, P);
  ex = halfTwistedRectTubeMMS(th, s, br, P.R, P.a, P.b);
  err_inf = max(abs(vals - ex));
  npts = numel(th);
end


function td = trace_discrepancy(geo, uA, uB, P)
%TRACE_DISCREPANCY  max |uA - uB| on the [upper lower] corner curves

  thv = geo.curve_theta;
  thl = mod(thv + 2*pi, 4*pi);

  FuA = halfTwistedRectTubeFrame(thv,  P.a, 1, P.R, P.a, P.b);
  FuB = halfTwistedRectTubeFrame(thv,  P.b, 2, P.R, P.a, P.b);
  FlA = halfTwistedRectTubeFrame(thv, -P.a, 1, P.R, P.a, P.b);
  FlB = halfTwistedRectTubeFrame(thl, -P.b, 2, P.R, P.a, P.b);
  if (max([vecnorm(FuA - FuB, 2, 2); vecnorm(FlA - FlB, 2, 2)]) > 1e-12)
    error('Corner curve parametrizations do not coincide.');
  end

  one = ones(size(thv));
  td = [max(abs(eval_branch(geo, 1, uA, thv,  P.a*one, P) - ...
                eval_branch(geo, 2, uB, thv,  P.b*one, P))), ...
        max(abs(eval_branch(geo, 1, uA, thv, -P.a*one, P) - ...
                eval_branch(geo, 2, uB, thl, -P.b*one, P)))];
end


function vals = eval_branch(geo, br, u, th, s, P)
%EVAL_BRANCH  interpolate branch data at F(theta,s), in batches

  n = numel(th);
  vals = zeros(n, 1);
  for i0 = 1:P.eval_batch:n
    idx = (i0:min(n, i0 + P.eval_batch - 1)).';
    F = halfTwistedRectTubeFrame(th(idx), s(idx), br, P.R, P.a, P.b);
    E = interp_band(geo.x1d, geo.y1d, geo.z1d, F(:,1), F(:,2), F(:,3), P.p, ...
                    geo.B(br).iband, sprintf('surface samples, branch %d', br));
    vals(idx) = E*u;
  end
end


%% ----------------------------------------------------------------------
%% Small vector helpers
%% ----------------------------------------------------------------------

function v = minval(x)
  if (isempty(x)), v = NaN; else, v = min(x); end
end

function v = maxval(x)
  if (isempty(x)), v = NaN; else, v = max(x); end
end


%% ----------------------------------------------------------------------
%% Bookkeeping, printing, plotting
%% ----------------------------------------------------------------------

function lev = empty_level(h, method)
%EMPTY_LEVEL  per-level diagnostics; rows of 2-by-2 fields are source
%   branch (A wide, B narrow), columns are [upper lower] curve.
  lev = struct('h', h, 'method', method, ...
               'err_inf', NaN, 'err_infA', NaN, 'err_infB', NaN, ...
               'trace_diff', [NaN NaN], ...
               'routed', zeros(2), ...
               'angle_error', NaN(2), 'max_angle_error', NaN, ...
               'angle_exact_range', NaN(2), ...
               'degenerate_source_rows', [0 0], ...
               'signflip_source_rows', [0 0], ...
               'closure_iter', 0, ...
               'max_rowsum_err', NaN, ...
               'unknowns', [0 0], 'outer', [0 0], ...
               'nsurfA', 0, 'nsurfB', 0, ...
               'solver', '', 'relres', NaN, 'gmres_iter', 0, ...
               'time', NaN);
end


function print_level(lev)
  fprintf('  [%s] unknowns A/B %d / %d, routed A->B up/low %d / %d, B->A up/low %d / %d\n', ...
          lev.method, lev.unknowns, lev.routed(1,:), lev.routed(2,:));
  fprintf('    Linf A/B/all %.4e / %.4e / %.4e, trace diff up/low %.3e / %.3e\n', ...
          lev.err_infA, lev.err_infB, lev.err_inf, lev.trace_diff);
  fprintf('    rowsum err %.2e, %s relres %.2e (iter %d), max angle err %.3e, probed src rows A/B %d / %d (sign reversals %d / %d), %.1fs\n', ...
          lev.max_rowsum_err, lev.solver, lev.relres, lev.gmres_iter, ...
          lev.max_angle_error, lev.degenerate_source_rows, ...
          lev.signflip_source_rows, lev.time);
end


function rates = conv_rates(h, err)
  if (numel(h) < 2)
    rates = [];
    return;
  end
  rates = log(err(1:end-1)./err(2:end)) ./ log(h(1:end-1)./h(2:end));
end


function r = fit_rate(h, err)
  if (numel(h) < 2)
    r = NaN;
    return;
  end
  c = polyfit(log(h), log(err), 1);
  r = c(1);
end


function print_summary(results)
  h = results.h;
  E = results.exact;  D = results.dk2;
  fprintf('\nSurface Linf convergence (combined over both strips)\n');
  fprintf('       h      exact Linf   rate      dk2 Linf    rate   dk2 max angle err\n');
  for k = 1:numel(h)
    if (k == 1)
      fprintf('%8.4g   %12.4e     --   %12.4e     --   %12.4e\n', ...
              h(k), E.errs_inf(k), D.errs_inf(k), D.levels(k).max_angle_error);
    else
      fprintf('%8.4g   %12.4e  %5.2f   %12.4e  %5.2f   %12.4e\n', ...
              h(k), E.errs_inf(k), E.rates_inf(k-1), D.errs_inf(k), ...
              D.rates_inf(k-1), D.levels(k).max_angle_error);
    end
  end
  fprintf('Least-squares fitted rates: exact %.3f, dk2 %.3f\n', E.fit_rate, D.fit_rate);
end


function plot_convergence(results, pngfile)
  h = results.h;
  E = results.exact.errs_inf;  D = results.dk2.errs_inf;
  ang = arrayfun(@(l) l.max_angle_error, results.dk2.levels);

  fig = figure('Color', 'w', 'Position', [100 100 1100 440]);
  subplot(1, 2, 1);
  ref = max([E D]);
  loglog(h, E, 'o-', h, D, 's-', ...
         h, ref*(h/h(1)), 'k:', h, ref*(h/h(1)).^2, 'k--', 'LineWidth', 1.5);
  grid on;  xlabel('h');  ylabel('surface L_\infty error');
  legend(sprintf('exact rotation (fit %.2f)', results.exact.fit_rate), ...
         sprintf('d_{k2} rotation (fit %.2f)', results.dk2.fit_rate), ...
         'O(h)', 'O(h^2)', 'Location', 'southeast');
  title('Half-twisted rectangular tube, both corner curves');

  subplot(1, 2, 2);
  if (all(ang > 0))
    loglog(h, ang, 's-', h, ang(1)*(h/h(1)).^2, 'k--', 'LineWidth', 1.5);
    legend('max |\phi_{dk2} - \phi_{exact}|', 'O(h^2)', 'Location', 'southeast');
  else
    semilogx(h, ang, 's-', 'LineWidth', 1.5);
  end
  grid on;  xlabel('h');  ylabel('max rotation angle error (rad)');
  title('d_{k2} angle error');
  drawnow;
  exportgraphics(fig, pngfile, 'Resolution', 150);
end
