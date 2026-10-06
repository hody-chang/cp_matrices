%% Curvature mismatch delta_kappa drives the angle-rotation iCPM error
%
% This script reimplements the angle-rotation closest point construction
% EXACTLY as it is defined in the draft
%
%     "Accuracy of the Angle-Rotation Closest Point Construction,
%      Poisson Equation Discretized by the Stabilized iCPM Operator",
%
% and measures the surface error it predicts.  The one claim under test is
% the draft's central result: with the STABILIZED iCPM operator
%
%     M = diag(L) + (L - diag(L))*E                         (draft eq. 18)
%
% the local cross-extension defect at a junction is controlled by the
% curvature-vector mismatch
%
%     Dc = c2 - R*c1,       ||Dc|| = |kappa1 + kappa2| =: delta_kappa
%                                                    (draft eqs. 174, 183)
%
% so that the surface solution error behaves like
%
%     ||u_h - U||_inf = C_curv * h + C2 * h^2 + o(h^2),     (draft eq. 228)
%     C_curv  proportional to  delta_kappa.
%
% Hence the scheme is O(h) whenever delta_kappa ~= 0, and drops to O(h^2)
% in the best case delta_kappa = 0 (the two branches share a curvature
% vector under the exact tangent rotation -- draft eqs. 176, 196).
%
% GEOMETRY (a controllable delta_kappa family).  A closed curve is built
% from a left and a right circular arc that share the two junction points
% V = (0, +H) and V = (0, -H).  The left radius Rleft is fixed; the right
% radius Rright is swept.  The signed cut-point curvatures are kappa =
% 1/R, so
%
%     delta_kappa = |1/Rleft - 1/Rright|
%
% sweeps through zero as Rright passes Rleft.  Rright = Rleft is the
% "perfect" symmetric lens: the exact rotation carries one arc's curvature
% vector onto the other's, Dc = 0, and the junction is second-order
% compatible.  Any other Rright opens a genuine O(1) curvature mismatch.
% The intrinsic 1-manifold is the same in every case (same total length,
% same arclength, same u(s) and f(s)); only the embedding, and with it the
% cross-extension the operator must perform, changes.  This is the two-arc
% analogue of the draft's cut-and-reflect ellipse experiment.
%
% FOUR GLUES, matching the draft's three extension matrices plus a control.
% A row whose closest point on its own arc is a junction V is turned about
% V so its outward tangent lands on the negative outward tangent of the
% other arc, then projected onto the other arc and interpolated there.
%
%     'exactR'  E_R  : the exact tangent rotation R.  No tangent-estimation
%                      error at all, so the ONLY thing left is the
%                      curvature mismatch Dc.  This is the clean test of
%                      eq. 228: O(h) for delta_kappa ~= 0, O(h^2) at 0.
%     'dk'      Ehat : rotation from the POINTWISE one-reflection tangent
%                      t_hat = d/|d|,  d = cp - cp(2*cp(x)-x)  (draft eq. 51).
%     'dk2'     Ehat : rotation from the POINTWISE three-point tangent
%                      G = 1.5*cp - 2*cp(2cp-x) + 0.5*cp(3cp-2x) (draft eq. 82).
%     'ideal'   E_*  : exact circular arclength continuation of one arc onto
%                      the other (the smooth-continuation extension of
%                      draft Section 5).  Always second order; the control.
%
% NO AVERAGING.  The draft is emphatic that "no averaging over captured
% points is used anywhere below" and derives the tangent error pointwise
% (eqs. 51, 82).  Unlike the earlier lens/ellipse scripts, cp_tangent_pt
% below therefore uses a SINGLE captured point per (branch, junction) --
% the one with the largest tangential offset xi, i.e. the "usable captured
% point with xi >= c0*h" of the draft's Step 1.6.  The pointwise tangent
% errors are then
%
%     one-reflection  eps ~ (kappa/2) xi          = O(h)     (eq. 53)
%     three-point     eps ~ 2 kappa^2 xi eta_N ...= O(h^2)    (eq. 83)
%
% so, after the operator chain of draft Section 4, 'dk' carries an O(1)
% local residual of its OWN first-order tangent error; its rotation-angle
% error is proportional to (kappa1*xi1 - kappa2*xi2) (draft eq. 113), so it
% is O(h) whenever the two branches are curvature-mismatched.  At the
% SYMMETRIC compatible lens (delta_kappa = 0, equal radii) that signed
% difference largely cancels, lifting 'dk' well above first order there
% (observed rate ~1.7; grid-phase asymmetry in xi1, xi2 leaves it short of
% a clean O(h^2)).  'dk2' has an O(h^2)
% tangent error and an O(h) rotation residual, so it tracks 'exactR' at
% every delta_kappa: O(h) for delta_kappa ~= 0, O(h^2) at 0
% (draft Step 5.10, "exact R and d_{k,2} have the same leading scale").
%
% EQUATION.  The manufactured problem is the screened surface Poisson
% equation
%
%     u - laplacian_S u = f
%
% on the closed curve, solved as (I - M) u = f.  This is the draft's
% "-laplacian_S u = f with the experiment's normalization included when
% needed" (eqs. 17-19): on a closed curve the pure Laplace-Beltrami is
% singular, and the +u term is the well-posed normalization.  It cancels
% in every operator difference (M_R - M_*, Mhat - M_R), so it does not
% touch the junction analysis being tested.
%
% Run headlessly, from this directory:
%   matlab -batch "example_curvature_mismatch_delta_kappa_convergence"

% Resolve dependencies relative to this example.
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', '..', 'cp_matrices'));
addpath(fullfile(here, '..', '..', 'surfaces'));
addpath(fullfile(here, '..', 'rotations'));


%% Parameters

H = 0.3;            % half chord: both junctions sit at (0, +-H)
Rleft = 0.5;        % fixed left radius; kappa_left = 1/Rleft = 2

% Right radii chosen so that delta_kappa = |1/Rleft - 1/Rright| lands on an
% even ladder {0, 0.2, ..., 1.0}, passing exactly through the compatible
% case delta_kappa = 0 at Rright = Rleft.  Every radius exceeds H, so each
% arc genuinely spans the chord.
Rright_values = 1 ./ ((1/Rleft) - [0 0.2 0.4 0.6 0.8 1.0]);

% Grid ladder.  Coarse enough to start above the arithmetic floor, fine
% enough to expose the competition between the C_curv*h and C2*h^2 terms.
% bw*h stays well under the smallest radius of curvature at every level.
hvals = 0.02*2.^-(0:5);

p = 3;      % interpolation degree
order = 2;  % Laplacian order (second-order Cartesian, q = 2)
dim = 2;
% bw as in [Ruuth & Merriman 2008]; 1.0002 is a safety factor.
bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ((order/2+(p+1)/2)^2));

glues = {'ideal', 'exactR', 'dk', 'dk2'};
gluelabels = {'ideal E_* (arclength)', 'exact R', ...
              'rotation from d_k (pointwise)', 'rotation from d_{k2} (pointwise)'};
nglue = numel(glues);

figdir = fullfile(here, '..', 'figs');   % .png output goes here


%% Sweep delta_kappa, refine h at each

results = struct('Rleft', {}, 'Rright', {}, 'kappa_left', {}, ...
                 'kappa_right', {}, 'delta_kappa', {}, 'h', {}, ...
                 'err', {}, 'rate', {}, 'fit', {}, 'xi_over_h', {});

for ir = 1:numel(Rright_values)
  Rright = Rright_values(ir);
  geo = lens_geometry(H, Rleft, Rright);

  % Same intrinsic problem for every geometry: u and f are functions of
  % global arclength only, so the exact solution is identical across the
  % sweep and the errors are directly comparable.
  omega = 2*pi/geo.total_length;
  ufun = @(s) sin(omega*s + 0.7) + 0.5*cos(2*omega*s - 0.3);
  ffun = @(s) (1 + omega^2)*sin(omega*s + 0.7) + ...
              0.5*(1 + 4*omega^2)*cos(2*omega*s - 0.3);

  nh = numel(hvals);
  err = nan(nglue, nh);
  xioh = nan(1, nh);   % offset xi/h of the pointwise captured point (dk2)

  fprintf(['\n==== lens: Rleft = %.6g, Rright = %.6g, ' ...
           'kappa = (%.4f, %.4f), delta_kappa = %.4f ====\n'], ...
          Rleft, Rright, 1/Rleft, 1/Rright, geo.delta_kappa);

  for ih = 1:nh
    h = hvals(ih);
    ncross_ref = [];
    for ig = 1:nglue
      out = solve_level(geo, h, p, order, bw, glues{ig}, ufun, ffun);
      err(ig, ih) = out.errsurf;
      if isempty(ncross_ref)
        ncross_ref = out.ncross;
      elseif any(out.ncross ~= ncross_ref)
        error('glue %s routed a different endpoint-row set at h = %g', ...
              glues{ig}, h);
      end
      if strcmp(glues{ig}, 'dk2')
        xioh(ih) = max(out.xi_over_h(:));
      end
    end
    fprintf('h = %.6g   ', h);
    for ig = 1:nglue
      fprintf('%-8s %.4e   ', glues{ig}, err(ig, ih));
    end
    fprintf('xi/h(dk2) = %.2f\n', xioh(ih));
  end

  % Fitted rates and the mixed-order coefficient C_curv (the leading O(h)
  % amplitude) for each glue.
  rate = nan(1, nglue);
  fit = cell(1, nglue);
  for ig = 1:nglue
    rate(ig) = fitrate(hvals, err(ig,:));
    fit{ig} = fit_mixed_order(hvals, err(ig,:));
  end

  print_refinement_table(hvals, err, glues);
  fprintf('fitted rates:');
  for ig = 1:nglue
    fprintf('  %s = %.3f', glues{ig}, rate(ig));
  end
  fprintf('\nC_curv (leading O(h) amplitude, e/h = C1 + C2 h):');
  for ig = 1:nglue
    fprintf('  %s C1 = %.4e', glues{ig}, fit{ig}.C1);
  end
  fprintf('\n');

  results(ir) = struct('Rleft', Rleft, 'Rright', Rright, ...
      'kappa_left', 1/Rleft, 'kappa_right', 1/Rright, ...
      'delta_kappa', geo.delta_kappa, 'h', hvals, 'err', err, ...
      'rate', rate, 'fit', {fit}, 'xi_over_h', xioh);
end


%% Verdict: is C_curv proportional to delta_kappa, and are the rates right?

igR   = find(strcmp(glues, 'exactR'));
igdk  = find(strcmp(glues, 'dk'));
igdk2 = find(strcmp(glues, 'dk2'));
igid  = find(strcmp(glues, 'ideal'));

dk_axis = [results.delta_kappa];
C_exactR = arrayfun(@(r) r.fit{igR}.C1,   results);
C_dk2    = arrayfun(@(r) r.fit{igdk2}.C1, results);

% Linear fit C_curv = slope * delta_kappa through the origin, using exactR
% (the clean curvature-only glue).  A tight fit is the numerical statement
% of draft eq. 228: the leading error amplitude is proportional to
% delta_kappa.
slope = (dk_axis*C_exactR.') / (dk_axis*dk_axis.');
pred = slope*dk_axis;
ss_res = sum((C_exactR - pred).^2);
ss_tot = sum((C_exactR - mean(C_exactR)).^2);
R2 = 1 - ss_res/max(ss_tot, eps);

fprintf('\n================ delta_kappa scaling of the leading error ================\n');
fprintf('%-12s %-12s %-14s %-14s %-14s %-14s %-14s\n', 'delta_kappa', ...
        'ideal rate', 'exactR rate', 'dk rate', 'dk2 rate', ...
        'C_curv(exactR)', 'C_curv(dk2)');
for ir = 1:numel(results)
  r = results(ir);
  fprintf('%-12.4f %-12.3f %-14.3f %-14.3f %-14.3f %-14.4e %-14.4e\n', ...
          r.delta_kappa, r.rate(igid), r.rate(igR), r.rate(igdk), ...
          r.rate(igdk2), r.fit{igR}.C1, r.fit{igdk2}.C1);
end
fprintf('-------------------------------------------------------------------------\n');
fprintf('C_curv(exactR) = slope * delta_kappa fit:  slope = %.4e,  R^2 = %.4f\n', ...
        slope, R2);
fprintf(['Reading of the table:\n' ...
         '  * ideal E_*  : ~2 at every delta_kappa (smooth continuation is the control).\n' ...
         '  * exact R    : ~1 when delta_kappa ~= 0, ~2 at delta_kappa = 0 (draft eq. 228).\n' ...
         '  * d_k2       : tracks exact R (pointwise 2nd-order tangent, draft Step 5.10).\n' ...
         '  * d_k        : ~1 when delta_kappa ~= 0 (own O(h) tangent error);\n' ...
         '                 its signed rotation error (eq. 113) largely cancels at\n' ...
         '                 the symmetric delta_kappa = 0 lens, lifting the rate\n' ...
         '                 toward 2 (observed ~1.7).\n' ...
         '  * C_curv proportional to delta_kappa: the leading amplitude vanishes with Dc.\n']);


%% Figures

if ~exist(figdir, 'dir'), mkdir(figdir); end

fig = figure('Color', 'w', 'Position', [100 100 1500 460]);

% (1) the geometry family
subplot(1,3,1); hold on; axis equal; box on;
colors = lines(numel(results));
for ir = 1:numel(results)
  geo = lens_geometry(H, results(ir).Rleft, results(ir).Rright);
  for j = 1:2
    s = linspace(geo.branches(j).s0, ...
                 geo.branches(j).s0 + geo.branches(j).length, 400)';
    xy = geo.branches(j).pointfun(s);
    plot(xy(:,1), xy(:,2), 'Color', colors(ir,:), 'LineWidth', 1.4);
  end
end
plot([0 0], [-H H], 'k.', 'MarkerSize', 15);
title('two-arc lenses: \Delta\kappa = |1/R_L - 1/R_R|');
xlabel('x'); ylabel('y');

% (2) convergence of exact R and the pointwise estimators
subplot(1,3,2); hold on; box on;
set(gca, 'XScale', 'log', 'YScale', 'log');
for ir = 1:numel(results)
  r = results(ir);
  loglog(r.h, r.err(igR,:), 'o-', 'Color', colors(ir,:), ...
         'DisplayName', sprintf('exact R, \\Delta\\kappa = %.1f (rate %.2f)', ...
                                r.delta_kappa, r.rate(igR)));
  loglog(r.h, r.err(igdk2,:), 's--', 'Color', colors(ir,:), ...
         'HandleVisibility', 'off');
end
hg = results(1).h;
loglog(hg, results(end).err(igR,1)*(hg/hg(1)), 'k:', 'DisplayName', 'O(h)');
loglog(hg, results(1).err(igR,1)*(hg/hg(1)).^2, 'k--', 'DisplayName', 'O(h^2)');
title('surface L_\infty error (solid: exact R, dashed: d_{k2})');
xlabel('h'); ylabel('error');
legend('Location', 'southeast', 'FontSize', 7);

% (3) the leading O(h) amplitude versus delta_kappa
subplot(1,3,3); hold on; box on;
plot(dk_axis, C_exactR, 'o-', 'Color', [0.2 0.35 0.75], 'LineWidth', 1.5, ...
     'DisplayName', 'C_{curv} (exact R)');
plot(dk_axis, C_dk2, 's--', 'Color', [0.8 0.35 0.2], 'LineWidth', 1.2, ...
     'DisplayName', 'C_{curv} (d_{k2})');
plot(dk_axis, pred, 'k:', 'DisplayName', ...
     sprintf('%.3g \\cdot \\Delta\\kappa (R^2 = %.3f)', slope, R2));
title('leading error amplitude \propto \Delta\kappa');
xlabel('\Delta\kappa'); ylabel('C_{curv}'); grid on;
legend('Location', 'northwest', 'FontSize', 8);

outfile = fullfile(figdir, 'curvature_mismatch_delta_kappa_convergence.png');
exportgraphics(fig, outfile, 'Resolution', 180);
fprintf('\nsaved %s\n', outfile);


%% ----------------------------------------------------------------------
%% local functions
%% ----------------------------------------------------------------------

function geo = lens_geometry(H, Rleft, Rright)
%LENS_GEOMETRY  Two circular arcs sharing V = (0, +-H); exact arclength maps.
%   Branches run top-to-bottom on the left and bottom-to-top on the right,
%   so a consistent traversal of the closed curve visits both.
  if Rleft <= H || Rright <= H
    error('each lens radius must satisfy R > H');
  end
  Vtop = [0 H]; Vbottom = [0 -H];
  left  = make_branch(Rleft,  [ sqrt(Rleft^2  - H^2) 0], ...
                      pi - asin(H/Rleft), -pi + asin(H/Rleft), 0, Vtop);
  right = make_branch(Rright, [-sqrt(Rright^2 - H^2) 0], ...
                      -asin(H/Rright), asin(H/Rright), left.length, Vbottom);
  geo.vertices = [Vtop; Vbottom];
  geo.branches = [left right];
  geo.total_length = left.length + right.length;
  geo.delta_kappa = abs(1/Rleft - 1/Rright);
  geo.bounds = [ -Rleft + sqrt(Rleft^2 - H^2), ...
                  Rright - sqrt(Rright^2 - H^2), -H, H];
end


function b = make_branch(R, cen, angle1, angle2, s0, vstart)
%MAKE_BRANCH  A circular arc with arclength, point, and outward-tangent maps.
%   vstart is the arc's starting vertex; tauout(iv,:) is the outward unit
%   tangent (pointing off the arc) at vertex iv, ordered top then bottom.
  span = mod(angle2 - angle1, 2*pi);
  if span <= 0, span = 2*pi; end
  b.R = R; b.cen = cen; b.angle1 = angle1; b.angle2 = angle2;
  b.s0 = s0; b.length = R*span;
  b.cpf = @(x,y) cpArc(x, y, R, cen, angle1, angle2);
  b.parfun = @(x,y) branch_arclength(x, y, b);
  b.pointfun = @(s) branch_points(s, b);
  % gamma'(theta) = R*(-sin, cos); the outward tangent leaves the arc, so
  % it is -gamma' at the start end and +gamma' at the far end.
  t1 = [-sin(angle1) cos(angle1)];
  t2 = [-sin(angle1 + span) cos(angle1 + span)];
  tauout_start = -t1; tauout_end = t2;
  if norm(vstart - [0 abs(vstart(2))]) <= 100*eps
    b.tauout = [tauout_start; tauout_end];      % starts at the top vertex
  else
    b.tauout = [tauout_end; tauout_start];      % starts at the bottom vertex
  end
end


function s = branch_arclength(x, y, b)
%BRANCH_ARCLENGTH  Global arclength of the closest points on one arc.
  theta = atan2(y - b.cen(2), x - b.cen(1));
  dtheta = mod(theta - b.angle1, 2*pi);
  tol = 100*eps(max(1, max(abs([x(:); y(:); b.cen(:)]))));
  span = b.length/b.R;
  dtheta(abs(dtheta - 2*pi) <= tol) = 0;
  dtheta(dtheta < tol) = 0;
  dtheta(abs(dtheta - span) <= tol) = span;
  if any(dtheta < -tol | dtheta > span + tol)
    error('closest-point angle lies outside its branch interval');
  end
  dtheta = min(max(dtheta, 0), span);
  s = b.s0 + b.R*dtheta;
end


function xy = branch_points(s, b)
%BRANCH_POINTS  Cartesian points for a column of global arclengths.
  if ~iscolumn(s), error('pointfun requires a column arclength input'); end
  q = min(max(s, b.s0), b.s0 + b.length);
  th = b.angle1 + (q - b.s0)/b.R;
  xy = [b.cen(1) + b.R*cos(th), b.cen(2) + b.R*sin(th)];
end


function out = solve_level(geo, h, p, order, bw, glue, ufun, ffun)
%SOLVE_LEVEL  Assemble and solve one two-band iCPM lens system at grid h.
  pad = (ceil(bw) + 4)*h;
  x1d = (geo.bounds(1) - pad + 0.317*h):h:(geo.bounds(2) + pad + 0.317*h);
  y1d = (geo.bounds(3) - pad + 0.211*h):h:(geo.bounds(4) + pad + 0.211*h);
  [xx, yy] = meshgrid(x1d, y1d);
  br(1) = setup_branch(x1d, y1d, xx, yy, h, p, order, bw, geo.branches(1), geo.vertices);
  br(2) = setup_branch(x1d, y1d, xx, yy, h, p, order, bw, geo.branches(2), geo.vertices);
  [out, ubranches] = assemble_solve(geo, br, x1d, y1d, p, glue, ffun);
  e1 = surface_error(x1d, y1d, p, br(1), ubranches{1}, ufun);
  e2 = surface_error(x1d, y1d, p, br(2), ubranches{2}, ufun);
  out.errsurf = max(e1, e2);
  if ~isfinite(out.errsurf), error('surface error is non-finite'); end
end


function s = setup_branch(x1d, y1d, xx, yy, h, p, order, bw, branch, vertices)
%SETUP_BRANCH  Checked initial, inner, and outer bands for one arc.
  switch order
    case 2, sten = 5;
    otherwise, error('order %d not implemented', order);
  end
  cand = (1:numel(xx))';
  [cpx, cpy, dist, bdy] = branch.cpf(xx(:), yy(:));
  keep = abs(dist) <= bw*h;
  s.dx = h;
  s.band = cand(keep); s.xinit = xx(keep); s.yinit = yy(keep);
  s.cpxinit = cpx(keep); s.cpyinit = cpy(keep); s.bdyinit = bdy(keep);
  [Ei, Ej, Es] = interp2_matrix(x1d, y1d, s.cpxinit, s.cpyinit, p);
  s.innerband = unique(Ej); s.nin = numel(s.innerband);
  s.inv_inner = make_invbandmap(numel(xx), s.innerband);
  Einit = sparse(Ei, s.inv_inner(Ej), Es, numel(s.band), s.nin);
  Ltemp = laplacian_2d_matrix(x1d, y1d, order, s.innerband, s.band);
  if nnz(Ltemp) ~= sten*s.nin, error('the Laplacian stencil leaves the initial band'); end
  [~, jj] = find(Ltemp); outertemp = unique(jj);
  [tf, loc] = ismember(s.innerband, s.band(outertemp));
  if ~all(tf), error('the inner band is not contained in the outer band'); end
  s.outerband = s.band(outertemp); s.nout = numel(outertemp);
  s.L = Ltemp(:, outertemp); s.E = Einit(outertemp,:);
  s.R = sparse(1:s.nin, loc, 1, s.nin, s.nout);
  s.xout = s.xinit(outertemp); s.yout = s.yinit(outertemp);
  s.cpxout = s.cpxinit(outertemp); s.cpyout = s.cpyinit(outertemp);
  s.bdyout = s.bdyinit(outertemp); s.geo = branch;
  tol = 100*eps(max(1, max(abs(vertices(:)))));
  s.vid = zeros(s.nout,1);
  for iv = 1:2
    s.vid(hypot(s.cpxout - vertices(iv,1), s.cpyout - vertices(iv,2)) <= tol) = iv;
  end
  if any((s.bdyout ~= 0) & (s.vid == 0))
    error('an endpoint closest point does not match a lens vertex');
  end
end


function [out, ubr] = assemble_solve(geo, br, x1d, y1d, p, glue, ffun)
%ASSEMBLE_SOLVE  Route endpoint rows through the chosen glue, build the
%   stabilized block operator M = diag(L) + (L - diag(L))*E, and solve
%   (I - M) u = f.  Each routed row stands for u at the target-arc
%   arclength the glue delivers, so its RHS uses f there.
  Eb = {br(1).E, sparse(br(1).nout, br(2).nin); ...
        sparse(br(2).nout, br(1).nin), br(2).E};
  ts = {br(1).geo.parfun(br(1).cpxout, br(1).cpyout), ...
        br(2).geo.parfun(br(2).cpxout, br(2).cpyout)};
  out.ncross = zeros(2,1);
  out.xi_over_h = zeros(2,2);

  % Pointwise estimated outward tangents, if this glue needs them.  ONE
  % captured point per (branch, vertex) -- the draft uses no averaging.
  need_est = strcmp(glue, 'dk') || strcmp(glue, 'dk2');
  if need_est
    scheme = 1 + double(strcmp(glue, 'dk2'));   % 1 = d_k, 2 = d_{k2}
    tauhat = zeros(2, 2, 2);                     % (branch, vertex, xy)
    for ib = 1:2
      for iv = 1:2
        [tauhat(ib,iv,:), xi] = cp_tangent_pt(br(ib), geo.vertices(iv,:), scheme);
        out.xi_over_h(ib,iv) = xi/br(ib).dx;
      end
    end
  end

  for source = 1:2
    target = 3 - source; rows = find(br(source).vid ~= 0);
    if isempty(rows), error('no rows to route from branch %d', source); end
    vids = br(source).vid(rows); out.ncross(source) = numel(rows);
    xq = [br(source).xout(rows), br(source).yout(rows)];
    qmap = zeros(numel(rows), 2);
    for iv = 1:2
      use = vids == iv;
      if ~any(use), continue; end
      switch glue
        case 'exactR'
          % Rodrigues about the out-of-plane axis: the rotation comes out
          % of the two tangents as a (cos, sin) pair and is applied as
          % built, with no angle formed in between.  See glue_rot2d.
          [cc, ss] = glue_rot2d(br(source).geo.tauout(iv,:), ...
                                br(target).geo.tauout(iv,:));
          qmap(use,:) = rotate_about2d(xq(use,:), geo.vertices(iv,:), cc, ss);
        case {'dk', 'dk2'}
          [cc, ss] = glue_rot2d(squeeze(tauhat(source,iv,:)).', ...
                                squeeze(tauhat(target,iv,:)).');
          qmap(use,:) = rotate_about2d(xq(use,:), geo.vertices(iv,:), cc, ss);
        case 'ideal'
          qmap(use,:) = map_by_arclength(xq(use,:), geo.vertices(iv,:), ...
              br(source).geo, br(target).geo, br(source).geo.tauout(iv,:), ...
              -br(target).geo.tauout(iv,:));
        otherwise
          error('unknown glue %s', glue);
      end
    end
    [cpx, cpy] = br(target).geo.cpf(qmap(:,1), qmap(:,2));
    [Ei, Ej, Es] = interp2_matrix(x1d, y1d, cpx, cpy, p);
    jj = br(target).inv_inner(Ej);
    if any(jj == 0), error('a cross-branch interpolation stencil leaves the inner band'); end
    Eb{source,target} = sparse(rows(Ei), jj, Es, br(source).nout, br(target).nin);
    Eself = Eb{source,source}; Eself(rows,:) = 0; Eb{source,source} = Eself;
    ts{source}(rows) = br(target).geo.parfun(cpx, cpy);
  end

  Eblk = [Eb{1,1} Eb{1,2}; Eb{2,1} Eb{2,2}];
  Lblk = blkdiag(br(1).L, br(2).L); Rblk = blkdiag(br(1).R, br(2).R);
  out.rowsum = max(abs(full(sum(Eblk,2)) - 1));
  if out.rowsum > 1e-10, error('extension row-sum defect is %g', out.rowsum); end
  M = lapsharp_unordered(Lblk, Eblk, Rblk);
  rhs = [ffun(br(1).R*ts{1}); ffun(br(2).R*ts{2})];
  u = (speye(size(M,1)) - M) \ rhs;
  if any(~isfinite(u)), error('the elliptic solve returned non-finite values'); end
  ubr = {u(1:br(1).nin), u(br(1).nin + (1:br(2).nin))};
end


function [tau, xi] = cp_tangent_pt(br, v, scheme)
%CP_TANGENT_PT  Pointwise outward tangent at a junction, from cp differences.
%   NO AVERAGING (draft eqs. 51, 82): a single captured point is used, the
%   clamped outer-band node with the largest offset from the vertex, i.e.
%   the "usable captured point with xi >= c0 h" of the draft's Step 1.6.
%   With cpbar = cp(2*cp(x)-x) and cp2bar = cp(3*cp(x)-2*x),
%
%     scheme 1:  d_k  = cp - cpbar                     (first order,  eq. 51)
%     scheme 2:  d_k2 = 1.5*cp - 2*cpbar + 0.5*cp2bar  (second order, eq. 82)
  tol = 100*eps(max(1, max(abs(v))));
  m = abs(br.cpxout - v(1)) <= tol & abs(br.cpyout - v(2)) <= tol;
  x = br.xout(m); y = br.yout(m);
  cx = br.cpxout(m); cy = br.cpyout(m);
  if isempty(x), error('no grid point has this junction as closest point'); end

  % pick the single farthest captured point (largest tangential offset)
  [~, isel] = max(hypot(x - v(1), y - v(2)));
  x = x(isel); y = y(isel); cx = cx(isel); cy = cy(isel);

  [b1x, b1y] = br.geo.cpf(2*cx - x, 2*cy - y);
  if scheme == 1
    dvx = cx - b1x; dvy = cy - b1y;
  else
    [b2x, b2y] = br.geo.cpf(3*cx - 2*x, 3*cy - 2*y);
    dvx = 1.5*cx - 2*b1x + 0.5*b2x;
    dvy = 1.5*cy - 2*b1y + 0.5*b2y;
  end
  nrm = hypot(dvx, dvy);
  if ~(isfinite(nrm) && nrm > tol)
    error('the pointwise cp difference at this junction was too small to use');
  end
  tau = [dvx dvy]/nrm;
  xi = (x - v(1))*tau(1) + (y - v(2))*tau(2);   % tangential offset of x
end


function q = map_by_arclength(xq, v, source, target, tauout_source, tauin_target)
%MAP_BY_ARCLENGTH  Exact circular arclength continuation (the E_* glue).
%   A source-arc endpoint offset is measured as an arclength from v along
%   the source circle and laid down as the same arclength along the target
%   circle -- the smooth continuation of Section 5, second order by design.
  rv = (v - source.cen)/source.R;
  rq = xq - source.cen;
  nrq = hypot(rq(:,1), rq(:,2));
  if any(nrq == 0), error('a source grid point equals the circle centre'); end
  rq = rq./nrq;
  angle = atan2(rv(1)*rq(:,2) - rv(2)*rq(:,1), rq*rv.');
  source_sign = sign(dot(tauout_source, [-rv(2), rv(1)]));
  xi = source.R*source_sign*angle;
  tol = 100*eps(max(1, max(abs([xq(:); v(:)]))));
  if any(xi < -tol), error('circular continuation produced a negative offset'); end
  xi(xi < 0) = 0;
  rvt = (v - target.cen)/target.R;
  target_sign = sign(dot(tauin_target, [-rvt(2), rvt(1)]));
  alpha = target_sign*xi/target.R;
  q = target.cen + target.R*[cos(alpha)*rvt(1) - sin(alpha)*rvt(2), ...
                              sin(alpha)*rvt(1) + cos(alpha)*rvt(2)];
end


function e = surface_error(x1d, y1d, p, br, u, ufun)
%SURFACE_ERROR  Max midpoint-sampled branch interpolation error in L_inf.
  nq = max(400, ceil(4*br.geo.length/br.dx));
  s = br.geo.s0 + ((0:nq-1)' + 0.5)*br.geo.length/nq;
  xy = br.geo.pointfun(s);
  [Ei, Ej, Es] = interp2_matrix(x1d, y1d, xy(:,1), xy(:,2), p);
  jj = br.inv_inner(Ej);
  if any(jj == 0), error('a surface sample stencil leaves the inner band'); end
  Eq = sparse(Ei, jj, Es, nq, br.nin);
  values = Eq*u; exact = ufun(s);
  if any(~isfinite(values)) || any(~isfinite(exact)), error('non-finite surface value'); end
  e = max(abs(values - exact));
end


function rate = fitrate(h, e)
%FITRATE  Pre-floor scalar log-log least-squares convergence rate.
  use = pre_floor_mask(e);
  if numel(use) < 2, rate = NaN; return; end
  q = polyfit(log(h(use)), log(e(use)), 1); rate = q(1);
end


function fit = fit_mixed_order(h, e)
%FIT_MIXED_ORDER  Nonnegative fit of e/h = C1 + C2*h on pre-floor levels.
%   C1 is the leading O(h) amplitude C_curv; C2 is the O(h^2) coefficient.
  use = pre_floor_mask(e);
  fit = struct('C1', NaN, 'C2', NaN, 'use', use, 'relres', NaN);
  if numel(use) < 2, return; end
  A = [ones(numel(use),1), h(use).']; b = e(use).'./h(use).';
  c = nnls2(A, b);
  fit.C1 = c(1); fit.C2 = c(2);
  fit.relres = norm(A*c - b)/max(norm(b), eps);
end


function c = nnls2(A, b)
%NNLS2  Dependency-free two-column nonnegative least-squares solve.
  candidates = [0 0; max(0, (A(:,1)'*b)/(A(:,1)'*A(:,1))) 0; ...
                0 max(0, (A(:,2)'*b)/(A(:,2)'*A(:,2)))];
  cu = A\b;
  if all(cu >= 0), candidates(end+1,:) = cu.'; end
  residuals = sum((A*candidates.' - b).^2, 1);
  [~, k] = min(residuals); c = candidates(k,:).';
end


function use = pre_floor_mask(e)
%PRE_FLOOR_MASK  Keep finite positive levels before the arithmetic floor.
  [~, imin] = min(e);
  cutoff = imin;
  if imin ~= numel(e), cutoff = imin - 1; end
  cutoff = max(cutoff, 2);
  use = find(isfinite(e) & e > 0 & (1:numel(e)) <= cutoff);
end


function print_refinement_table(h, err, glues)
%PRINT_REFINEMENT_TABLE  One block per glue: error and observed local rate.
  for ig = 1:numel(glues)
    fprintf('   %-28s', glues{ig});
  end
  fprintf('\n');
  for k = 1:numel(h)
    fprintf('h = %.6g  ', h(k));
    for ig = 1:numel(glues)
      if k == 1
        fprintf('%.4e (  --  )  ', err(ig,k));
      else
        fprintf('%.4e (%5.2f)  ', err(ig,k), ...
                log(err(ig,k-1)/err(ig,k))/log(2));
      end
    end
    fprintf('\n');
  end
end
