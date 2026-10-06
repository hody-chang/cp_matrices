function [geo, P] = halfTwistedRectTubeBands(h, P)
%HALFTWISTEDRECTTUBEBANDS  two-strip narrow bands and branch routing of the
%   half-twisted rectangular tube on one grid level
%
%   [geo, P] = halfTwistedRectTubeBands(h)
%   [geo, P] = halfTwistedRectTubeBands(h, P)
%
%   Builds the grid, the inner/outer bands of branch A (= 1, wide strip
%   p = (s,b), s in [-a,a]) and branch B (= 2, narrow strip p = (a,s),
%   s in [-b,b]) and the exact and d_k2 routing targets of every routed
%   outer row (bdy ~= 0), closed under interpolation support: the inner
%   band of each branch contains the stencils of its initial-band closest
%   points, of the surface-error and corner-curve trace samples, and of
%   the target closest points of rows routed INTO it by either rotation.
%   No linear system is assembled or solved.  See
%   example_half_twisted_rectangular_tube_rotation_convergence.m for the
%   rotation and d_k2 conormal definitions.
%
%   P is optional; only MISSING geometry/banding fields are filled
%   (other fields, e.g. solver settings, are returned unchanged):
%     R = 1, a = 0.45, b = 0.20      tube geometry
%     p = 3                          interpolation degree
%     order = 2                      Laplacian order (7-point stencil)
%     grid_shift = [.317 .211 .173]  grid offsets (units of h)
%     alpha_min = 0.25               dk2 degeneracy threshold |d_perp|/|v|
%     eval_batch = 20000             points per interpolation batch
%     max_closure_iter = 20          band/routing closure sweeps
%
%   geo fields: h, bw, x1d, y1d, z1d, nx, ny, nz (meshgrid ordering),
%   B(br) per branch (branch, width, iband, oband, ni, no, R, Ldiag, Loff,
%   x, y, z, cpx, cpy, cpz, bdy, theta, s on the outer band, self E),
%   route{src} cached routing targets of the routed rows of branch src
%   (node = grid index, sg, cs_exact, cs_dk2, phi_exact, phi_dk2,
%    angerr_dk2, probe, signflip, cpE, cpD
%   with columns [cpx cpy cpz theta s] on the target branch),
%   closure_iter, samples{br} and traces{br} (structs with theta, s: the
%   surface-error grid and the corner-curve trace samples of branch br),
%   curve_theta (corner-curve theta grid) and params (= P).

  here = fileparts(mfilename('fullpath'));
  addpath(fullfile(here, '..', '..', 'cp_matrices'));
  addpath(fullfile(here, '..', '..', 'surfaces'));
  addpath(fullfile(here, '..', 'surfaces'));
  addpath(fullfile(here, '..', 'rotations'));
  addpath(here);

  if (nargin < 2 || isempty(P))
    P = struct();
  end
  defaults = struct('R', 1, 'a', 0.45, 'b', 0.20, 'p', 3, 'order', 2, ...
                    'grid_shift', [0.317 0.211 0.173], 'alpha_min', 0.25, ...
                    'eval_batch', 20000, 'max_closure_iter', 20);
  f = fieldnames(defaults);
  for k = 1:numel(f)
    if (~isfield(P, f{k}))
      P.(f{k}) = defaults.(f{k});
    end
  end

  if (~(isscalar(h) && isnumeric(h) && isreal(h) && isfinite(h) && h > 0))
    error('h must be a positive finite scalar.');
  end
  if (P.R <= sqrt(P.a^2 + P.b^2))
    error('Expected R > sqrt(a^2+b^2).');
  end

  geo = build_level(h, P);
end


%% ----------------------------------------------------------------------
%% Level construction
%% ----------------------------------------------------------------------

function geo = build_level(h, P)
%BUILD_LEVEL  grid, conservative prefilter, the two branch bands and the
%   routing targets of both methods, closed under interpolation support
%
%   The inner band of a branch must contain every interpolation stencil
%   used on it: its initial-band closest points, the surface evaluation
%   samples, and the target closest points of rows routed INTO it by
%   either rotation.  Enlarging an inner band enlarges its outer band,
%   which can create new routed rows, so routing and banding are iterated
%   until no new support appears.  Routing targets are cached per grid
%   node, so each node is rotated and projected only once.  All required
%   support must lie in the conservative initial band; otherwise error.

  rho = sqrt(P.a^2 + P.b^2);
  bw = 1.0002*sqrt(2*((P.p+1)/2)^2 + (P.order/2 + (P.p+1)/2)^2);
  pad = (ceil(bw) + 4)*h;

  Lxy = h*ceil((P.R + rho + pad)/h);
  Lz = h*ceil((rho + pad)/h);
  x1d = ((-Lxy:h:Lxy) + P.grid_shift(1)*h).';
  y1d = ((-Lxy:h:Lxy) + P.grid_shift(2)*h).';
  z1d = ((-Lz:h:Lz) + P.grid_shift(3)*h).';
  nx = numel(x1d);  ny = numel(y1d);  nz = numel(z1d);

  % Every strip point lies within rho of the centerline circle, so a node
  % within bw*h of either strip is within rho + bw*h of the circle.
  [X2, Y2] = meshgrid(x1d, y1d);
  rr2 = (sqrt(X2.^2 + Y2.^2) - P.R).^2;
  D2 = rr2 + reshape(z1d, 1, 1, []).^2;
  cand = find(D2 <= (rho + bw*h)^2);
  clear X2 Y2 rr2 D2;
  [iy, ix, iz] = ind2sub([ny nx nz], cand);
  xc = x1d(ix);  yc = y1d(iy);  zc = z1d(iz);

  geo.h = h;  geo.bw = bw;
  geo.x1d = x1d;  geo.y1d = y1d;  geo.z1d = z1d;
  geo.nx = nx;  geo.ny = ny;  geo.nz = nz;

  thv = curve_thetas(h, P);
  samples = cell(1, 2);  traces = cell(1, 2);
  ext = cell(1, 2);
  for br = 1:2
    I(br) = branch_init(br, cand, xc, yc, zc, geo, P); %#ok<AGROW>
    % surface-error grid and corner-curve trace samples (incl. the
    % theta+2*pi shifted lower-curve samples of branch B)
    [ths, ss] = surface_samples(br, h, thv, P);
    [tht, st] = trace_samples(br, thv, P);
    samples{br} = struct('theta', ths, 's', ss);
    traces{br} = struct('theta', tht, 's', st);
    ext{br} = stencil_support(geo, [ths; tht], [ss; st], br, P);
  end

  route = {empty_routes(), empty_routes()};
  rebuild = [true true];
  for it = 1:P.max_closure_iter
    for br = 1:2
      if (rebuild(br))
        B(br) = branch_ops(I(br), ext{br}, geo, P); %#ok<AGROW>
      end
    end
    rebuild = [false false];
    for src = 1:2
      tgt = 3 - src;
      rows = find(B(src).bdy ~= 0);
      rows = rows(~ismember(B(src).oband(rows), route{src}.node));
      if (isempty(rows))
        continue;
      end
      Rt = route_targets(B(src), rows, src, tgt, geo, P);
      route{src} = cat_routes(route{src}, Rt);
      sup = unique([support_of_points(geo, Rt.cpE(:,1:3), P); ...
                    support_of_points(geo, Rt.cpD(:,1:3), P)]);
      ext{tgt} = union(ext{tgt}, sup);
      if (~all(ismember(sup, B(tgt).iband)))
        rebuild(tgt) = true;
      end
    end
    if (~any(rebuild))
      break;
    end
  end
  if (any(rebuild))
    error('Band/routing closure did not converge in %d sweeps.', P.max_closure_iter);
  end

  geo.B = B;
  geo.route = route;
  geo.closure_iter = it;
  geo.samples = samples;
  geo.traces = traces;
  geo.curve_theta = thv;
  geo.params = P;
end


function I = branch_init(br, cand, xc, yc, zc, geo, P)
%BRANCH_INIT  conservative initial band of a branch and its CP stencils

  [cpx, cpy, cpz, dist, bdy, th, s] = ...
      cpHalfTwistedRectTube(xc, yc, zc, P.R, P.a, P.b, br);
  keep = abs(dist) <= geo.bw*geo.h;
  I.branch = br;
  I.width = P.a*(br == 1) + P.b*(br == 2);
  I.band_init = cand(keep);          % sorted meshgrid linear indices
  I.x = xc(keep);  I.y = yc(keep);  I.z = zc(keep);
  I.cpx = cpx(keep);  I.cpy = cpy(keep);  I.cpz = cpz(keep);
  I.bdy = bdy(keep);  I.theta = th(keep);  I.s = s(keep);
  if (any(~isfinite([I.cpx; I.cpy; I.cpz; I.theta; I.s])))
    error('Non-finite closest points on branch %d.', br);
  end
  I.iband0 = support_of_points(geo, [I.cpx I.cpy I.cpz], P);
end


function B = branch_ops(I, ext, geo, P)
%BRANCH_OPS  inner/outer bands, L, R and the self E of a branch, with the
%   inner band = CP stencils of the initial band U required extra support

  h = geo.h;
  ny = geo.ny;  nx = geo.nx;  nz = geo.nz;
  br = I.branch;

  iband = union(I.iband0, ext);
  iband = iband(:);
  if (~all(ismember(iband, I.band_init)))
    error('Branch %d: required inner band node outside the initial band (bandwidth too small).', br);
  end

  % complete 7-point Laplacian stencils of the inner band
  [jy, jx, jz] = ind2sub([ny nx nz], iband);
  if (any(jy < 2 | jy > ny-1 | jx < 2 | jx > nx-1 | jz < 2 | jz > nz-1))
    error('Branch %d: Laplacian stencil leaves the grid; increase padding.', br);
  end
  offs = [1, -1, ny, -ny, ny*nx, -ny*nx];
  nb = iband + offs;
  if (~all(ismember(nb(:), I.band_init)))
    error('Branch %d: incomplete 7-point Laplacian stencil in the initial band.', br);
  end
  oband = unique([iband; nb(:)]);
  [~, oloc] = ismember(oband, I.band_init);
  [~, selfcol] = ismember(iband, oband);
  [~, nbcol] = ismember(nb, oband);

  ni = numel(iband);  no = numel(oband);
  rows = repmat((1:ni).', 1, 7);
  vals = repmat([-6 1 1 1 1 1 1]/h^2, ni, 1);
  L = sparse(rows(:), [selfcol(:); nbcol(:)], vals(:), ni, no);
  Rm = sparse((1:ni).', selfcol, 1, ni, no);

  B.branch = br;
  B.width = I.width;
  B.iband = iband;  B.oband = oband;
  B.ni = ni;  B.no = no;
  B.R = Rm;
  B.Ldiag = full(sum(Rm.*L, 2));
  B.Loff = L - Rm.*L;
  B.x = I.x(oloc);  B.y = I.y(oloc);  B.z = I.z(oloc);
  B.cpx = I.cpx(oloc);  B.cpy = I.cpy(oloc);  B.cpz = I.cpz(oloc);
  B.bdy = I.bdy(oloc);  B.theta = I.theta(oloc);  B.s = I.s(oloc);
  % outer nodes are initial band nodes, whose CP stencils are in iband0
  B.E = interp_band(geo.x1d, geo.y1d, geo.z1d, B.cpx, B.cpy, B.cpz, P.p, iband, ...
                    sprintf('self E of branch %d', br));
end


function sup = stencil_support(geo, th, s, br, P)
%STENCIL_SUPPORT  grid support of the stencils at the strip points F(th,s)

  sup = zeros(0, 1);
  for i0 = 1:P.eval_batch:numel(th)
    idx = (i0:min(numel(th), i0 + P.eval_batch - 1)).';
    F = halfTwistedRectTubeFrame(th(idx), s(idx), br, P.R, P.a, P.b);
    sup = union(sup, support_of_points(geo, F, P));
  end
  sup = sup(:);
end


function sup = support_of_points(geo, X, P)
%SUPPORT_OF_POINTS  grid support of the stencils at the rows of X (n-by-3)

  sup = zeros(0, 1);
  for i0 = 1:P.eval_batch:size(X, 1)
    idx = (i0:min(size(X, 1), i0 + P.eval_batch - 1)).';
    [~, Ej] = interp3_matrix(geo.x1d, geo.y1d, geo.z1d, ...
                             X(idx,1), X(idx,2), X(idx,3), P.p);
    sup = union(sup, Ej);
  end
  sup = sup(:);
end


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
%% Surface evaluation samples
%% ----------------------------------------------------------------------

function [th, s] = surface_samples(br, h, thv, P)
%SURFACE_SAMPLES  uniform (theta, s) grid of a strip, both edges included,
%   about two samples per h in each direction (thv = curve_thetas(h, P))

  w = P.a*(br == 1) + P.b*(br == 2);
  ns = ceil(2*w*2/h) + 1;
  [TH, SS] = ndgrid(thv, linspace(-w, w, ns).');
  th = TH(:);  s = SS(:);
end


function thv = curve_thetas(h, P)
%CURVE_THETAS  uniform theta samples on [0,4*pi), about two per h of the
%   outermost circumference

  rho = sqrt(P.a^2 + P.b^2);
  nth = ceil(4*pi*(P.R + rho)*2/h);
  thv = (0:nth-1).' * (4*pi/nth);
end


function [th, s] = trace_samples(br, thv, P)
%TRACE_SAMPLES  the (theta, s) where the corner-curve traces of branch br
%   are evaluated: [upper; lower] curve (thv = curve_thetas(h, P))

  if (br == 1)
    th = [thv; thv];
    s = [P.a*ones(size(thv)); -P.a*ones(size(thv))];
  else
    th = [thv; mod(thv + 2*pi, 4*pi)];
    s = [P.b*ones(size(thv)); -P.b*ones(size(thv))];
  end
end


%% ----------------------------------------------------------------------
%% Routing targets (exact and d_k2 rotations)
%% ----------------------------------------------------------------------

function Rt = route_targets(S, rows, src, tgt, geo, P)
%ROUTE_TARGETS  exact and dk2 rotation of the routed outer rows `rows` of
%   branch S, and the target-branch closest points of the rotated nodes.
%   cpE/cpD columns: [cpx cpy cpz theta s] on the target branch.

  h = geo.h;
  sg = S.bdy(rows);
  c = [S.cpx(rows) S.cpy(rows) S.cpz(rows)];
  x = [S.x(rows) S.y(rows) S.z(rows)];
  th = S.theta(rows);
  wt = P.a*(tgt == 1) + P.b*(tgt == 2);

  % analytic frames: source edge point and glued target edge point
  [~, Fs, Ft] = halfTwistedRectTubeFrame(th, sg*S.width, src, P.R, P.a, P.b);
  tau = unit_rows(Ft);
  eta_s = sg .* unit_rows(perp(Fs, tau));
  th_t = mod(th + 2*pi*(sg < 0), 4*pi);
  [Fc_t, Fs_t] = halfTwistedRectTubeFrame(th_t, sg*wt, tgt, P.R, P.a, P.b);
  eta_t = sg .* unit_rows(perp(Fs_t, tau));
  if (max(vecnorm(Fc_t - c, 2, 2)) > 1e-9)
    error('Glued target edge point does not match the source edge point.');
  end
  % the glue rotation as a (cos, sin) pair, which is what Rodrigues'
  % formula consumes; the angles below are derived from the pairs for
  % reporting and are not used to rotate anything
  [ce, se] = signed_rot(eta_s, -eta_t, tau);

  [es_hat, probe, signflip] = dk2_source(c, x, th, tau, eta_s, src, h, P);
  et_hat = dk2_target(Fc_t, th_t, tau, eta_t, tgt, h, P);
  [cd, sd] = signed_rot(es_hat, -et_hat, tau);

  Rt.node = S.oband(rows);
  Rt.sg = sg;
  Rt.cs_exact = [ce se];
  Rt.cs_dk2 = [cd sd];
  Rt.phi_exact = atan2(se, ce);
  Rt.phi_dk2 = atan2(sd, cd);
  % the error of the estimated rotation against the exact one, as the angle
  % of the relative rotation about the shared axis tau.  Both rotations are
  % about the same axis, so this is the planar comparison rot2d_err makes:
  % it lands in (-pi, pi] by construction, with no difference of two angles
  % to rewrap.
  Rt.angerr_dk2 = abs(rot2d_err(cd, sd, ce, se));
  Rt.probe = probe;
  Rt.signflip = signflip;
  Rt.cpE = rotated_target_cp(x, c, tau, ce, se, tgt, P);
  Rt.cpD = rotated_target_cp(x, c, tau, cd, sd, tgt, P);
end


function cp = rotated_target_cp(x, c, tau, cth, sth, tgt, P)
%ROTATED_TARGET_CP  rotate x about the axis (c, tau) and project onto the
%   target branch
%   cth and sth are the cosine and sine of the rotation, as signed_rot
%   built them.

  xr = c + rodrigues(x - c, tau, cth, sth);
  [cx, cy, cz, ~, ~, tht, st] = ...
      cpHalfTwistedRectTube(xr(:,1), xr(:,2), xr(:,3), P.R, P.a, P.b, tgt);
  cp = [cx cy cz tht st];
  if (any(~isfinite(cp(:))))
    error('Non-finite target closest points of rotated nodes.');
  end
end


function R = empty_routes()
  R = struct('node', zeros(0, 1), 'sg', zeros(0, 1), ...
             'cs_exact', zeros(0, 2), 'cs_dk2', zeros(0, 2), ...
             'phi_exact', zeros(0, 1), 'phi_dk2', zeros(0, 1), ...
             'angerr_dk2', zeros(0, 1), ...
             'probe', false(0, 1), 'signflip', false(0, 1), ...
             'cpE', zeros(0, 5), 'cpD', zeros(0, 5));
end


function R = cat_routes(R, Rn)
  f = fieldnames(R);
  for k = 1:numel(f)
    R.(f{k}) = [R.(f{k}); Rn.(f{k})];
  end
end


function [es_hat, probe, signflip] = dk2_source(c, x, th, tau, eta_s, br, h, P)
%DK2_SOURCE  source conormal from the original grid node; documented
%   outward probe when that sampling is degenerate or reversed
%
%   signflip marks rows whose otherwise valid estimate points inward
%   (d_perp.eta_s <= 0): the long reflected samples of a coarse grid
%   misjudged the side.  Like degenerate rows they are re-sampled with
%   the outward h probe; the estimate stays a closest point difference.

  v = c - x;
  [d, ok] = cp_difference(c, v, th, tau, br, h, P);
  signflip = ok & (sum(d.*eta_s, 2) <= 0);
  probe = ~ok | signflip;
  bad = find(probe);
  if (~isempty(bad))
    vp = -h*eta_s(bad, :);    % outward probe c + h*eta_s, reflected
    [dp, okp] = cp_difference(c(bad, :), vp, th(bad), tau(bad, :), br, h, P);
    okp = okp & (sum(dp.*eta_s(bad, :), 2) > 0);
    if (~all(okp))
      error('dk2 source probe unusable on %d rows of branch %d.', nnz(~okp), br);
    end
    d(bad, :) = dp;
  end
  es_hat = unit_rows(d);
end


function et_hat = dk2_target(ct, th_t, tau, eta_t, br, h, P)
%DK2_TARGET  target conormal from the outward probe ct + h*eta_t

  [d, ok] = cp_difference(ct, -h*eta_t, th_t, tau, br, h, P);
  ok = ok & (sum(d.*eta_t, 2) > 0);
  if (~all(ok))
    error('dk2 target probe unusable on %d rows of branch %d.', nnz(~ok), br);
  end
  et_hat = unit_rows(d);
end


function [dperp, ok] = cp_difference(c, v, th, tau, br, h, P)
%CP_DIFFERENCE  d = 1.5 c - 2 cp(c+v) + 0.5 cp(c+2v), projected off tau
%   ok flags rows whose samples are interior, on the same sheet, and
%   whose projected difference is not degenerate.

  nv = vecnorm(v, 2, 2);
  q1 = c + v;  q2 = c + 2*v;
  [g1x, g1y, g1z, ~, b1, t1] = cpHalfTwistedRectTube(q1(:,1), q1(:,2), q1(:,3), ...
                                                     P.R, P.a, P.b, br);
  [g2x, g2y, g2z, ~, b2, t2] = cpHalfTwistedRectTube(q2(:,1), q2(:,2), q2(:,3), ...
                                                     P.R, P.a, P.b, br);
  d = 1.5*c - 2*[g1x g1y g1z] + 0.5*[g2x g2y g2z];
  dperp = perp(d, tau);
  alpha = vecnorm(dperp, 2, 2) ./ max(nv, realmin);
  samesheet = abs(wrap_4pi(t1 - th)) < 1 & abs(wrap_4pi(t2 - th)) < 1;
  ok = (nv > 1e-6*h) & (b1 == 0) & (b2 == 0) & samesheet & ...
       (alpha >= P.alpha_min) & all(isfinite(dperp), 2);
end


%% ----------------------------------------------------------------------
%% Small vector helpers
%% ----------------------------------------------------------------------

function v = unit_rows(v)
  n = vecnorm(v, 2, 2);
  if (any(n <= 0 | ~isfinite(n)))
    error('Zero or non-finite vector in normalization.');
  end
  v = v ./ n;
end

function w = perp(v, tau)
%PERP  component of v orthogonal to unit tau
  w = v - sum(v.*tau, 2).*tau;
end

function [c, s] = signed_rot(u, w, tau)
%SIGNED_ROT  the rotation about tau taking u onto w, as a (cos, sin) pair
%   u and w are rows orthogonal to the unit axes tau.  Rodrigues' formula
%   needs the cosine and sine of the angle, not the angle, and both are
%   already here: the cosine IS the dot product of u and w, and the sine IS
%   the component of their cross product along tau.  Forming
%   atan2(sin, cos) and then taking cos and sin of that again is a round
%   trip through three library transcendentals that cancel each other, and
%   it costs accuracy -- see ../rotations/rot2d_from_to.m, and
%   ../2D_curve/example_ellipse_cut_rotation_construction.m for the
%   measurement in the planar case.
%
%   The pair is normalized by its own hypot, since u and w are unit only to
%   within rounding.  An angle, where one is wanted for a report, is
%   atan2(s, c) of the pair.

  c = sum(u.*w, 2);
  s = sum(tau.*cross(u, w, 2), 2);
  n = hypot(c, s);
  c = c ./ n;
  s = s ./ n;
end

function vr = rodrigues(v, k, c, s)
%RODRIGUES  rotate rows of v about unit axes k, from a (cos, sin) pair
%   c and s come from signed_rot, which never forms the angle.  Taking the
%   pair rather than an angle is the point: an angle argument would have to
%   be turned back into a cosine and a sine here.

  vr = v.*c + cross(k, v, 2).*s + k.*(sum(k.*v, 2).*(1 - c));
end

function d = wrap_4pi(d)
  d = mod(d + 2*pi, 4*pi) - 2*pi;
end
