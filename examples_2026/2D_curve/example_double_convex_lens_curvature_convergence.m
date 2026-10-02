%% Double-convex lens: curvature mismatch and exact tangent rotation
%
% A closed curve is formed from a left and a right circular arc sharing
% $V_{top}=(0,H)$ and $V_{bottom}=(0,-H)$.  The `rotation` and `arclength`
% controls use exact tangents; `rotation_dk2` estimates them from
% second-order closest-point differences.  The experiment measures whether
% rigid rotation acquires $C_1 h$ as the branch curvatures differ, while
% circular arclength continuation remains second order.

% Resolve dependencies relative to this example.
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', '..', 'cp_matrices'));
addpath(fullfile(here, '..', '..', 'surfaces'));

%% Problem, geometry, and discretization
H = 0.4;
Rleft = 0.5;
Rright_values = [1.9, 1.95, 2.0, 2.05, 2.1]./4;
%hvals = 0.02*2.^-(0:8);
hvals = logspace(-2,-4,20);
% Put like 200 points for paper
p = 3;
order = 2;
dim = 2;
bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ((order/2+(p+1)/2)^2));

old_banding_checks = getenv('ICPM2009BANDINGCHECKS');
cleanup_banding_checks = onCleanup(@() setenv('ICPM2009BANDINGCHECKS', old_banding_checks)); %#ok<NASGU>
setenv('ICPM2009BANDINGCHECKS', '1');

results = struct('Rleft', {}, 'Rright', {}, 'kappa_left', {}, ...
                 'kappa_right', {}, 'delta_kappa', {}, 'h', {}, ...
                 'err_rot', {}, 'rate_rot', {}, 'fit_rot', {}, ...
                 'err_rot_dk2', {}, 'rate_rot_dk2', {}, ...
                 'err_arclength', {}, 'rate_arclength', {}, ...
                 'rowsum_rot', {}, 'rowsum_rot_dk2', {}, ...
                 'rowsum_arclength', {}, 'ncross', {});

%% Solve the intrinsic periodic manufactured problem
for ir = 1:numel(Rright_values)
  geo = lens_geometry(H, Rleft, Rright_values(ir));
  omega = 2*pi/geo.total_length;
  ufun = @(s) sin(omega*s + 0.7) + 0.5*cos(2*omega*s - 0.3);
  ffun = @(s) (1 + omega^2)*sin(omega*s + 0.7) + ...
              0.5*(1 + 4*omega^2)*cos(2*omega*s - 0.3);
  nh = numel(hvals);
  err_rot = nan(1, nh); err_dk2 = nan(1, nh); err_arc = nan(1, nh);
  rowsum_rot = nan(1, nh); rowsum_dk2 = nan(1, nh); rowsum_arc = nan(1, nh);
  ncross = zeros(2, nh);

  fprintf('\n==== double-convex lens: Rleft = %.16g, Rright = %.16g, delta_kappa = %.16g ====\n', ...
          geo.branches(1).R, geo.branches(2).R, geo.delta_kappa);
  for ih = 1:nh
    h = hvals(ih);
    out_rot = solve_level(geo, h, p, order, bw, 'rotation', ufun, ffun);
    out_dk2 = solve_level(geo, h, p, order, bw, 'rotation_dk2', ufun, ffun);
    out_arc = solve_level(geo, h, p, order, bw, 'arclength', ufun, ffun);
    err_rot(ih) = out_rot.errsurf;
    err_dk2(ih) = out_dk2.errsurf;
    err_arc(ih) = out_arc.errsurf;
    rowsum_rot(ih) = out_rot.rowsum;
    rowsum_dk2(ih) = out_dk2.rowsum;
    rowsum_arc(ih) = out_arc.rowsum;
    ncross(:,ih) = out_rot.ncross;
    if any(out_arc.ncross ~= out_rot.ncross) || any(out_dk2.ncross ~= out_rot.ncross)
      error('glue paths routed different endpoint-row sets at h = %g', h);
    end
    fprintf(['h = %.8g  rotation = %.8e  rotation_dk2 = %.8e  arclength = %.8e' ...
             '  cross = [%d %d]  rowsum = [%.3e %.3e %.3e]\n'], ...
            h, err_rot(ih), err_dk2(ih), err_arc(ih), ncross(1,ih), ncross(2,ih), ...
            rowsum_rot(ih), rowsum_dk2(ih), rowsum_arc(ih));
  end

  rate_rot = fitrate(hvals, err_rot);
  rate_dk2 = fitrate(hvals, err_dk2);
  rate_arc = fitrate(hvals, err_arc);
  fit_rot = fit_mixed_order(hvals, err_rot);
  print_refinement_table(hvals, err_rot, err_dk2, err_arc);
  fprintf('fitted rates: exact rotation = %.6f, d_{k2} rotation = %.6f, arclength = %.6f\n', ...
          rate_rot, rate_dk2, rate_arc);
  fprintf('rotation e/h = C1 + C2 h: C1 = %.8e, C2 = %.8e, relres = %.3e, bound = %.6f, effective-rate range = [%.6f, %.6f]\n', ...
          fit_rot.C1, fit_rot.C2, fit_rot.relres, fit_rot.bound_factor, ...
          min(fit_rot.effective_rate), max(fit_rot.effective_rate));

  results(ir) = struct('Rleft', Rleft, 'Rright', Rright_values(ir), ...
      'kappa_left', 1/Rleft, 'kappa_right', 1/Rright_values(ir), ...
      'delta_kappa', geo.delta_kappa, 'h', hvals, 'err_rot', err_rot, ...
      'rate_rot', rate_rot, 'fit_rot', fit_rot, 'err_rot_dk2', err_dk2, ...
      'rate_rot_dk2', rate_dk2, 'err_arclength', err_arc, ...
      'rate_arclength', rate_arc, 'rowsum_rot', rowsum_rot, ...
      'rowsum_rot_dk2', rowsum_dk2, 'rowsum_arclength', rowsum_arc, ...
      'ncross', ncross);
end

%% Requested validation summary
fprintf('\n%-8s %-8s %-12s %-18s %-18s %-19s %-12s %-12s %-12s %-12s %-21s\n', ...
        'Rleft', 'Rright', 'delta_kappa', 'exact rotation rate', ...
        'd_{k2} rotation rate', 'arclength fitted rate', 'C1', 'C2', ...
        'relres', 'bound', 'effective-rate range');
for ir = 1:numel(results)
  q = results(ir).fit_rot;
  fprintf('%-8.3g %-8.3g %-12.6g %-18.6f %-18.6f %-19.6f %-12.4e %-12.4e %-12.3e %-12.6f [%.6f, %.6f]\n', ...
          results(ir).Rleft, results(ir).Rright, results(ir).delta_kappa, ...
          results(ir).rate_rot, results(ir).rate_rot_dk2, results(ir).rate_arclength, ...
          q.C1, q.C2, q.relres, q.bound_factor, min(q.effective_rate), max(q.effective_rate));
end

%% Requested visualization
figdir = fullfile(here, '..', 'figs');
if ~exist(figdir, 'dir'), mkdir(figdir); end
fig = figure('Color', 'w', 'Position', [100 100 1500 460]);
subplot(1,3,1); hold on; axis equal; box on;
colors = lines(numel(results));
for ir = 1:numel(results)
  geo = lens_geometry(H, results(ir).Rleft, results(ir).Rright);
  for j = 1:2
    s = linspace(geo.branches(j).s0, geo.branches(j).s0 + geo.branches(j).length, 500)';
    xy = geo.branches(j).pointfun(s);
    plot(xy(:,1), xy(:,2), 'Color', colors(ir,:), 'LineWidth', 1.6);
  end
  text(0.02, H - 0.07*ir, sprintf('(\\kappa_L,\\kappa_R) = (%.2g, %.2g)', ...
       results(ir).kappa_left, results(ir).kappa_right), 'Color', colors(ir,:));
end
plot([0 0], [-H H], 'k.', 'MarkerSize', 15);
title('Double-convex circular lenses'); xlabel('x'); ylabel('y');

subplot(1,3,2); hold on; box on; set(gca, 'XScale', 'log', 'YScale', 'log');
for ir = 1:numel(results)
  r = results(ir); q = r.fit_rot; use = q.use;
  label = sprintf('R_R = %.2g, \\Delta\\kappa = %.2g, rate = %.2f', ...
                  r.Rright, r.delta_kappa, r.rate_rot);
  loglog(r.h, r.err_rot, 'o-', 'Color', colors(ir,:), 'DisplayName', label);
  loglog(r.h, r.err_arclength, '--', 'Color', 0.55*colors(ir,:) + 0.45, 'HandleVisibility', 'off');
end
hguide = results(1).h;
loglog(hguide, results(1).err_rot(1)*(hguide/hguide(1)), 'k:', 'DisplayName', 'O(h)');
loglog(hguide, results(1).err_rot(1)*(hguide/hguide(1)).^2, 'k--', 'DisplayName', 'O(h^2)');
title('Surface L_\infty error'); xlabel('h'); ylabel('error'); legend('Location', 'southwest', 'FontSize', 8);

subplot(1,3,3); hold on; box on;
dk = [results.delta_kappa]; c1 = arrayfun(@(r) r.fit_rot.C1, results);
plot(dk, c1, 'o-', 'Color', [0.2 0.35 0.75], 'LineWidth', 1.5);
for ir = 1:numel(results)
  text(dk(ir), c1(ir), sprintf('  R_R = %.2g', results(ir).Rright));
end
title('Rotation first-order coefficient'); xlabel('\Delta\kappa'); ylabel('C_1'); grid on;
exportgraphics(fig, fullfile(figdir, 'double_convex_lens_curvature_convergence.png'), 'Resolution', 180);

function geo = lens_geometry(H, Rleft, Rright)
%LENS_GEOMETRY Exact two-arc lens geometry and global arclength maps.
% Inputs are the half chord H and radii Rleft/Rright.  Output branches run
% top-to-bottom on the left and bottom-to-top on the right.
  if Rleft <= H || Rright <= H
    error('each lens radius must satisfy R > H');
  end
  Vtop = [0 H]; Vbottom = [0 -H];
  [left, phi_left, c_left] = make_branch(Rleft, [sqrt(Rleft^2-H^2) 0], ...
      pi - asin(H/Rleft), -pi + asin(H/Rleft), 0, Vtop, Vbottom);
  [right, phi_right, c_right] = make_branch(Rright, [-sqrt(Rright^2-H^2) 0], ...
      -asin(H/Rright), asin(H/Rright), left.length, Vbottom, Vtop);
  %#ok<NASGU> exact quantities retained to make the fixed-chord construction explicit.
  geo.vertices = [Vtop; Vbottom];
  geo.branches = [left right];
  geo.total_length = left.length + right.length;
  geo.delta_kappa = abs(1/Rleft - 1/Rright);
  geo.bounds = [c_left - Rleft, Rright - c_right, -H, H];
end

function [b, phi, c] = make_branch(R, cen, angle1, angle2, s0, vstart, vend)
%MAKE_BRANCH Circular branch with arclength, point, and outward-tangent maps.
  phi = asin(abs(vstart(2))/R); c = abs(cen(1));
  span = mod(angle2 - angle1, 2*pi);
  if span <= 0, span = 2*pi; end
  b.R = R; b.cen = cen; b.angle1 = angle1; b.angle2 = angle2;
  b.s0 = s0; b.length = R*span;
  b.cpf = @(x,y) cpArc(x, y, R, cen, angle1, angle2);
  b.parfun = @(x,y) branch_arclength(x, y, b);
  b.pointfun = @(s) branch_points(s, b);
  t1 = [-sin(angle1) cos(angle1)];
  t2 = [-sin(angle1 + span) cos(angle1 + span)];
  b.tauout_start = -t1; b.tauout_end = t2;
  % tauout rows follow geo.vertices: top then bottom.
  if norm(vstart - [0 abs(vstart(2))]) <= 100*eps
    b.tauout = [b.tauout_start; b.tauout_end];
  else
    b.tauout = [b.tauout_end; b.tauout_start];
  end
end

function s = branch_arclength(x, y, b)
%BRANCH_ARCLENGTH Global arclength of closest points on one circular branch.
% x and y may have any common shape; values are clamped only at roundoff-sized ends.
  theta = atan2(y - b.cen(2), x - b.cen(1));
  dtheta = mod(theta - b.angle1, 2*pi);
  tol = 100*eps(max(1, max(abs([x(:); y(:); b.cen(:)]))));
  dtheta(abs(dtheta - 2*pi) <= tol) = 0;
  dtheta(dtheta < tol) = 0;
  span = b.length/b.R;
  dtheta(abs(dtheta - span) <= tol) = span;
  if any(dtheta < -tol | dtheta > span + tol)
    error('closest-point angle lies outside its branch interval');
  end
  dtheta = min(max(dtheta, 0), span);
  s = b.s0 + b.R*dtheta;
end

function xy = branch_points(s, b)
%BRANCH_POINTS Cartesian branch points for a column of global arclengths.
  if ~iscolumn(s), error('pointfun requires a column arclength input'); end
  q = min(max(s, b.s0), b.s0 + b.length);
  th = b.angle1 + (q - b.s0)/b.R;
  xy = [b.cen(1) + b.R*cos(th), b.cen(2) + b.R*sin(th)];
end

function out = solve_level(geo, h, p, order, bw, glue_type, ufun, ffun)
%SOLVE_LEVEL Assemble and solve one two-band iCPM lens system.
% `glue_type` is rigid exact-tangent rotation or exact circular arclength continuation.
  pad = (ceil(bw) + 4)*h;
  x1d = (geo.bounds(1) - pad + 0.317*h):h:(geo.bounds(2) + pad + 0.317*h);
  y1d = (geo.bounds(3) - pad + 0.211*h):h:(geo.bounds(4) + pad + 0.211*h);
  [xx, yy] = meshgrid(x1d, y1d);
  br(1) = setup_branch(x1d, y1d, xx, yy, h, p, order, bw, geo.branches(1), geo.vertices);
  br(2) = setup_branch(x1d, y1d, xx, yy, h, p, order, bw, geo.branches(2), geo.vertices);
  [out, ubranches] = assemble_solve(geo, br, x1d, y1d, p, glue_type, ffun);
  e1 = surface_error(x1d, y1d, p, br(1), ubranches{1}, ufun);
  e2 = surface_error(x1d, y1d, p, br(2), ubranches{2}, ufun);
  out.errsurf = max(e1, e2);
  if ~isfinite(out.errsurf), error('surface error is non-finite'); end
end

function s = setup_branch(x1d, y1d, xx, yy, h, p, order, bw, branch, vertices)
%SETUP_BRANCH Build checked initial, inner, and outer bands for one lens arc.
% xx/yy are Cartesian grids; the returned maps index the branch inner unknowns.
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

function [out, ubr] = assemble_solve(geo, br, x1d, y1d, p, glue_type, ffun)
%ASSEMBLE_SOLVE Route endpoint rows, build the stabilized block operator, and solve.
% Each routed row represents its projected target-branch arclength in the RHS.
  Eb = {br(1).E, sparse(br(1).nout, br(2).nin); ...
        sparse(br(2).nout, br(1).nin), br(2).E};
  ts = {br(1).geo.parfun(br(1).cpxout, br(1).cpyout), ...
        br(2).geo.parfun(br(2).cpxout, br(2).cpyout)};
  out.ncross = zeros(2,1);
  if strcmp(glue_type, 'rotation_dk2')
    tauout = zeros(2, 2, 2);
    for ib = 1:2
      for iv = 1:2
        tauout(ib,iv,:) = cp_tangent_dk2(br(ib), geo.vertices(iv,:));
      end
    end
  end
  for source = 1:2
    target = 3-source; rows = find(br(source).vid ~= 0);
    if isempty(rows), error('no rows to route from branch %d', source); end
    vids = br(source).vid(rows); out.ncross(source) = numel(rows);
    xq = [br(source).xout(rows), br(source).yout(rows)];
    v = geo.vertices(vids,:);
    qmap = zeros(numel(rows),2);
    for iv = 1:2
      use = vids == iv;
      if ~any(use), continue; end
      if strcmp(glue_type, 'rotation') || strcmp(glue_type, 'rotation_dk2')
        if strcmp(glue_type, 'rotation_dk2')
          a = glue_angle(squeeze(tauout(source,iv,:)).', squeeze(tauout(target,iv,:)).');
        else
          a = glue_angle(br(source).geo.tauout(iv,:), br(target).geo.tauout(iv,:));
        end
        qmap(use,:) = rotate_about(xq(use,:), geo.vertices(iv,:), a);
      elseif strcmp(glue_type, 'arclength')
        qmap(use,:) = map_by_arclength(xq(use,:), geo.vertices(iv,:), ...
            br(source).geo, br(target).geo, br(source).geo.tauout(iv,:), ...
            -br(target).geo.tauout(iv,:));
      else
        error('glue_type must be rotation, rotation_dk2, or arclength');
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

function angle = glue_angle(tauout_source, tauout_target)
%GLUE_ANGLE Directed rotation from source outward tangent to negative target outward tangent.
  angle = atan2(-tauout_target(2), -tauout_target(1)) - ...
          atan2(tauout_source(2), tauout_source(1));
  angle = atan2(sin(angle), cos(angle));
end

function tau = cp_tangent_dk2(br, v)
%CP_TANGENT_DK2 Estimate an outward endpoint tangent from second-order CP differences.
% br is one branch band and v is one of its exact endpoint coordinates.
  tol = 100*eps(max(1, max(abs(v))));
  m = abs(br.cpxout - v(1)) <= tol & abs(br.cpyout - v(2)) <= tol;
  x = br.xout(m); y = br.yout(m);
  cx = br.cpxout(m); cy = br.cpyout(m);
  if isempty(x), error('no grid point has this endpoint as closest point'); end
  [b1x, b1y] = br.geo.cpf(2*cx - x, 2*cy - y);
  [b2x, b2y] = br.geo.cpf(3*cx - 2*x, 3*cy - 2*y);
  dvx = 1.5*cx - 2*b1x + 0.5*b2x;
  dvy = 1.5*cy - 2*b1y + 0.5*b2y;
  nrm = hypot(dvx, dvy);
  good = isfinite(nrm) & nrm > tol;
  if ~any(good), error('d_{k2} gave no usable endpoint tangent'); end
  tau = [mean(dvx(good)./nrm(good)), mean(dvy(good)./nrm(good))];
  tau = tau/norm(tau);
end

function q = rotate_about(x, v, angle)
%ROTATE_ABOUT Rigidly rotate row points x about vertex v by angle radians.
  c = cos(angle); s = sin(angle); d = x - v;
  q = v + [c*d(:,1) - s*d(:,2), s*d(:,1) + c*d(:,2)];
end

function q = map_by_arclength(xq, v, source, target, tauout_source, tauin_target)
%MAP_BY_ARCLENGTH Continue source-circle endpoint offsets onto the target circle.
% xq rows are source-band nodes; v is their shared vertex and tangents use outward/inward conventions.
  rv = (v - source.cen)/source.R;
  rq = xq - source.cen;
  nrq = hypot(rq(:,1), rq(:,2));
  if any(nrq == 0), error('a source grid point equals the circle centre'); end
  rq = rq./nrq;
  angle = atan2(rv(1)*rq(:,2) - rv(2)*rq(:,1), rq*rv.');
  source_sign = sign(dot(tauout_source, [-rv(2), rv(1)]));
  xi = source.R*source_sign*angle;
  tol = 100*eps(max(1, max(abs([xq(:); v(:)]))));
  if any(xi < -tol), error('circular continuation produced a negative endpoint offset'); end
  xi(xi < 0) = 0;
  rvt = (v - target.cen)/target.R;
  target_sign = sign(dot(tauin_target, [-rvt(2), rvt(1)]));
  alpha = target_sign*xi/target.R;
  q = target.cen + target.R*[cos(alpha)*rvt(1) - sin(alpha)*rvt(2), ...
                              sin(alpha)*rvt(1) + cos(alpha)*rvt(2)];
end

function e = surface_error(x1d, y1d, p, br, u, ufun)
%SURFACE_ERROR Maximum midpoint-sampled branch interpolation error.
% Samples cover this branch only and every interpolation stencil must remain in its inner band.
  nq = max(400, ceil(4*br.geo.length/br.dx));
  s = br.geo.s0 + ((0:nq-1)' + 0.5)*br.geo.length/nq;
  xy = br.geo.pointfun(s);
  [Ei, Ej, Es] = interp2_matrix(x1d, y1d, xy(:,1), xy(:,2), p);
  jj = br.inv_inner(Ej);
  if any(jj == 0), error('a surface sample stencil leaves the inner band'); end
  ns = (p+1)^2;
  Eq = sparse(Ei, jj, Es, nq, br.nin);
  values = Eq*u; exact = ufun(s);
  if any(~isfinite(values)) || any(~isfinite(exact)), error('non-finite surface value'); end
  e = max(abs(values - exact));
end

function rate = fitrate(h, e)
%FITRATE Pre-floor scalar log-log least-squares convergence rate.
  use = pre_floor_mask(e);
  if numel(use) < 2, rate = NaN; return; end
  q = polyfit(log(h(use)), log(e(use)), 1); rate = q(1);
end

function fit = fit_mixed_order(h, e)
%FIT_MIXED_ORDER Nonnegative fit of e/h = C1 + C2*h on pre-floor levels.
  use = pre_floor_mask(e);
  fit = struct('C1', NaN, 'C2', NaN, 'use', use, 'nuse', numel(use), ...
               'relres', NaN, 'model', [], 'effective_rate', [], 'bound_factor', NaN);
  if numel(use) < 2, return; end
  A = [ones(numel(use),1), h(use).']; b = e(use).'./h(use).';
  c = nnls2(A, b); model = h(use).'.*(c(1) + c(2)*h(use).');
  fit.C1 = c(1); fit.C2 = c(2); fit.model = model;
  fit.relres = norm(A*c - b)/max(norm(b), eps);
  fit.effective_rate = (c(1) + 2*c(2)*h(use).') ./ (c(1) + c(2)*h(use).');
  fit.bound_factor = max(model./e(use).');
end

function c = nnls2(A, b)
%NNLS2 Dependency-free two-column nonnegative least-squares solve.
% It compares the feasible unconstrained solution, two one-axis projections, and zero.
  candidates = [0 0; max(0, (A(:,1)'*b)/(A(:,1)'*A(:,1))) 0; ...
                0 max(0, (A(:,2)'*b)/(A(:,2)'*A(:,2)))];
  cu = A\b;
  if all(cu >= 0), candidates(end+1,:) = cu.'; end
  residuals = sum((A*candidates.' - b).^2, 1);
  [~, k] = min(residuals); c = candidates(k,:).';
end

function use = pre_floor_mask(e)
%PRE_FLOOR_MASK Existing convergence policy: keep finite positive levels before the floor.
  [~, imin] = min(e);
  cutoff = imin;
  if imin ~= numel(e), cutoff = imin - 1; end
  cutoff = max(cutoff, 2);
  use = find(isfinite(e) & e > 0 & (1:numel(e)) <= cutoff);
end

function print_refinement_table(h, er, ed, ea)
%PRINT_REFINEMENT_TABLE Report exact-rotation, d_{k2}-rotation, and control errors.
  fprintf('       h          exact rotation   rate       d_{k2} rotation  rate       arclength error  rate\n');
  for k = 1:numel(h)
    if k == 1
      fprintf('%12.5e  %12.5e    ---      %12.5e    ---      %12.5e    ---\n', ...
              h(k), er(k), ed(k), ea(k));
    else
      fprintf('%12.5e  %12.5e  %7.4f   %12.5e  %7.4f   %12.5e  %7.4f\n', ...
              h(k), er(k), log(er(k-1)/er(k))/log(2), ed(k), ...
              log(ed(k-1)/ed(k))/log(2), ea(k), log(ea(k-1)/ea(k))/log(2));
    end
  end
end
