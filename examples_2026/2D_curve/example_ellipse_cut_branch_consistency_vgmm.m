%% How far apart do the two halves put the SAME grid point?  (vGMM)
%
% Companion to example_ellipse_cut_branch_consistency.m.  Same ellipse
%
%     gamma(t) = cen + [a*cos(t), b*sin(t)],     cen = (0.5, 0.5),
%
% cut along y = ycut, same three glue rotations (d_k, d_k2, exact R),
% same manufactured solution and analytic right-hand side for
% u - laplacian_S u = f, same four measures.  What changes is the
% discrete operator the two halves are glued into.
%
% THE OPERATOR.  The companion script uses the implicit CPM, the
% diagonal split of [Macdonald & Ruuth 2009],
%
%     M_icpm = diag(L) + (L - diag(L))*E,
%
% which applies the Cartesian Laplacian to the EXTENDED values, L*(E u),
% and pulls the diagonal back out for stability.  That needs two bands:
% E is read on an outer band and L lands back on an inner one, with a
% restriction R between them.
%
% Here it is the embedded method-of-lines operator of [vGMM 2013],
%
%     M_vgmm = E*L - gamma*(I - E),     gamma = 2*dim/dx^2,
%
% which is the other ordering.  The Laplacian acts on the band values,
% the RESULT is extended, and the penalty term -gamma*(I - E) is what
% holds u constant along the normals.  E and L are both square on ONE
% band, so there is no inner/outer split and no restriction here.
%
% The two are closer than they look, and it is worth saying which
% difference is the real one.  lapsharp_unordered(L, E, R, delta) with
% delta = 2*dim/dx^2 is NOT this operator: since the standard Laplacian
% has a constant diagonal, delta = -diag(L) and L*E - delta*(I - R*E) is
% algebraically the diagonal split all over again.  What makes this a
% different method is the ordering E*L, and with it the single band.
%
% THE GLUE is unchanged, and needs no new machinery.  A row whose
% closest point on its own half is a cut point is rotated about that
% point and interpolated on the other half, exactly as before, which
% puts its weights in the off-diagonal block E_{k,o}.  Both places E
% appears then pick the glue up on their own: L is block diagonal, so
% (E*L)_{k,o} = E_{k,o}*L_o extends the OTHER half's L u across the cut,
% and the penalty row u_i - (E u)_i becomes the statement that half k's
% value at a rotated node is half o's interpolant there.
%
% One degree p = 3 is used for both the E in E*L and the E in the
% penalty, so that p, bw, the grid and the manufactured solution are all
% exactly what the companion script uses and the operator is the only
% thing that moved.  [vGMM 2013] and Theorem 6.1 of [MMC 2026] use two,
% E_1*L - gamma*(I - E_3); that split is a separate experiment.
%
% TWO MANIFOLDS, chosen by the flag "flipped":
%
%   flipped = false   the plain ellipse.  The two halves join smoothly,
%                     the exact glue rotation is the identity, and any
%                     rotation an estimator produces is pure error.
%
%   flipped = true    the arc below the cut is reflected UP across the
%                     cut line, (x,y) -> (x, 2*ycut - y).  Both arcs then
%                     lie above the cut line and meet at the two cut
%                     points in a genuine corner, so the exact rotation
%                     is a definite nonzero angle -- something for the
%                     estimators to get right rather than a zero they can
%                     only overshoot.  This is the default.
%
% The reflection is an isometry, so the two curves are the SAME
% 1-manifold: same total length, same arclength parameter, same intrinsic
% geometry.
%
% Two things here are not node-for-node comparable with the companion
% script, both because there is one band instead of two.  gap and allgap
% range over the overlap of the two FULL bands, since that is where the
% unknowns live now and the iCPM inner band is a subset of it.  And
% cp_tangent averages over the band nodes clamped to a cut point rather
% than the outer-band ones, which is again the set the method actually
% rotates.  Same measures on slightly larger node sets: the rates are
% comparable, the individual numbers are not identical.
%
%
% References
%
% # [vGMM 2013]  Ingrid von Glehn, Thomas Marz, and Colin B. Macdonald.
%   An embedded method-of-lines approach to solving partial
%   differential equations on surfaces.  2013.
% # [MMC 2026]  Thomas Marz, Colin B. Macdonald, and Yujia Chen.
%   Consistency and stability of closest point iterations.  Draft.
% # examples/example_hole_ellipse.m is the in-repo template for the
%   steady vGMM solve, M = E*L - gamma*(I-E) on a single band.

% Resolve dependencies relative to this example.
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', '..', 'cp_matrices'));
addpath(fullfile(here, '..', '..', 'surfaces'));
addpath(fullfile(here, '..', 'surfaces'));


%% Parameters

cen = [0.5 0.5];
a = 1.4;    % semi-axis along x (the major one)
b = 0.6;    % semi-axis along y

% The manifold.  false: the plain ellipse, where the two halves join
% smoothly and the exact glue rotation is the identity.  true: the lower
% half is reflected up across the cut line, so the curve has a genuine
% corner at each cut point and the exact rotation is a known nonzero
% angle.  See the note at the top: the two curves are isometric as
% 1-manifolds, so the exact solution and f are the same function of
% arclength in both and the errors are directly comparable.
flipped = true;
if flipped
  geoname = 'flipped';
else
  geoname = 'ellipse';
end

% On the ellipse, y = cen(2) is the half cut and the second one is
% oblique to both the curve and the grid.  On the flipped curve the half
% cut is not available at all: reflecting the lower half of an ellipse
% across its own centre line lands it exactly on the upper half, so the
% "curve" would be one arc traversed twice, with the two bands on top of
% each other and no closest point well defined.  Two oblique cuts there
% instead, one of each curvature.
if flipped
  ycut_list = [0.3 0.1];
else
  ycut_list = [0.5 0.3];
end

hvals = 0.02*2.^-(0:7);

dim = 2;    % dimension
p = 3;      % interpolation degree
order = 2;  % Laplacian order
% The formula for bw is found in [Ruuth & Merriman 2008] and the 1.0002
% is a safety factor.  It is exactly what the single vGMM band needs:
% the (order/2 + (p+1)/2) in the normal direction is the interpolation
% stencil of a closest point plus one Laplacian stencil on top of it,
% which is the composition E*L reads.
bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ((order/2+(p+1)/2)^2));

% A band point has a unique closest point only while bw*dx stays under the
% smallest radius of curvature of the ellipse.
hvals = hvals(bw*hvals < 0.5*min(b^2/a, a^2/b));
nh = length(hvals);
if (nh < 2)
  error('no usable grid sizes for a = %g, b = %g', a, b);
end

dx_show = min(0.05, 0.5/(bw*max(a/b^2, b/a^2)));

figdir = fullfile(here, '..', 'figs');   % .png output goes here

if flipped
  exactlabel = 'exact R';
  % one band over the whole curve is no longer "exact" anything: it is
  % the closest point method applied to a corner as if it were smooth
  baselabel = 'uncut vGMM, corner ignored';
else
  exactlabel = 'exact R = I';
  baselabel = 'exact vGMM (uncut)';
end
varlabels = {'rotation from d_k', 'rotation from d_{k2}', exactlabel};
nvar = length(varlabels);
% the surface-error figure has the uncut curve alongside the three
varlabels4 = [{baselabel} varlabels];


%% Manufactured solution, in the ellipse parameter t

ufun = @(t) sin(t + 0.7) + 0.5*cos(2*t - 0.3);
utfun = @(t) cos(t + 0.7) - sin(2*t - 0.3);
uttfun = @(t) -sin(t + 0.7) - 2*cos(2*t - 0.3);

% right-hand side of u - laplacian_S u = f, all analytic
gfun = @(t) a^2*sin(t).^2 + b^2*cos(t).^2;
gtfun = @(t) (a^2 - b^2)*sin(2*t);
lapfun = @(t) uttfun(t)./gfun(t) - utfun(t).*gtfun(t)./(2*gfun(t).^2);
ffun = @(t) ufun(t) - lapfun(t);

%% One cut height at a time

results = struct('ycut', {}, 'kappa', {}, 'tcut', {}, 'dx', {}, ...
                 'gap', {}, 'vtx', {}, 'branch', {}, 'allgap', {}, ...
                 'nrot', {}, 'nshare', {}, 'thetamax', {}, ...
                 'rate', {}, 'ratev', {}, 'err', {}, 'rateerr', {});

for ci = 1:length(ycut_list)
  yc = ycut_list(ci);

  % where the cut meets the ellipse: t1 on the right, t2 on the left.
  % Branch 1 (A) is the upper arc [t1 t2], branch 2 (B) the lower arc.
  s0 = (yc - cen(2))/b;
  if (abs(s0) > 0.95)
    error('cut at y = %g misses the ellipse or grazes it (sin t = %g)', yc, s0);
  end
  if (flipped && abs(s0) < 0.05)
    error(['the flipped curve degenerates at y = %g: reflecting the lower ' ...
           'half across the centre line puts it on top of the upper half'], yc);
  end
  t1 = asin(s0);
  t2 = pi - t1;
  tlim = [t1 t2; t2 t1 + 2*pi];
  kappa = a*b/(a^2*sin(t1)^2 + b^2*cos(t1)^2)^1.5;

  % the two cut points, written exactly the way cpEllipseArc writes a
  % clamped closest point so that the match below is bit-for-bit.  They
  % sit on the cut line, so the reflection fixes them and they are the
  % same two points on either manifold.
  tv = [t1; t2];
  V = [cen(1) + a*cos(tv), cen(2) + b*sin(tv)];

  %% The two arcs, and the whole curve they make between them
  % geo{j} carries everything that depends on how arc j is embedded: its
  % closest-point function, the map from the ellipse parameter t to a
  % point of the arc, and the inverse of that map.  Reflecting arc 2 up
  % across the cut is an isometry, so only the embedding changes -- the
  % parameter t, the arclength and hence u(t) and f(t) do not.
  geo = cell(2,1);
  for j = 1:2
    cparc = @(xq,yq) cpEllipseArc(xq, yq, a, b, cen, tlim(j,1), tlim(j,2));
    if (flipped && j == 2)
      geo{j} = make_geo(@(xq,yq) cpflip(cparc, xq, yq, yc), -1, 2*yc, ...
                        tlim(j,:), a, b, cen);
    else
      geo{j} = make_geo(cparc, 1, 0, tlim(j,:), a, b, cen);
    end
  end
  geowhole = make_geo(@(xq,yq) cpnearest(geo{1}.cpf, geo{2}.cpf, xq, yq), ...
                      [], [], [], a, b, cen);
  geowhole.pieces = [geo{1}.pieces; geo{2}.pieces];
  geowhole.parfun = @(xq,yq) wholeparam(geo, xq, yq);

  %% Exact endpoint tangents, and the exact glue rotation
  % On the plain ellipse the two halves' outward tangents are exactly
  % opposite at each cut point and the exact rotation is the identity.
  % Reflecting arc 2 negates the y component of its tangents, which opens
  % a corner, and the exact rotation is then a definite nonzero angle --
  % the thing d_k and d_k2 are now being asked to estimate rather than a
  % zero they can only overshoot.
  tauex = cell(2,1);
  tauex{1} = [unit_tangent(t1, -1, a, b); unit_tangent(t2,  1, a, b)];
  tauex{2} = [unit_tangent(t1,  1, a, b); unit_tangent(t2, -1, a, b)];
  if flipped
    tauex{2}(:,2) = -tauex{2}(:,2);
  end
  thetaex = glue_angles(tauex{1}, tauex{2});

  err = nan(nvar+1, nh);    % surface L_inf error, baseline then the three
  gap = nan(nvar, nh);      % max |u_A - u_B| at the shared rotated nodes
  vtx = nan(nvar, nh);      % max |u_A(v) - u_B(v)|, on the manifold
  branch = nan(nvar, nh);   % max(|e_A|, |e_B|) at the same nodes
  allgap = nan(nvar, nh);   % the same gap over the whole overlap
  dcorner = nan(1, nh);     % where the baseline's worst point is
  nrot = nan(1, nh);        % how many shared rotated nodes there are
  nshare = nan(1, nh);      % how many shared nodes there are
  thetamax = nan(2, nh);    % max |theta|, for reference

  fprintf(['\n==== vGMM, %s, cut at y = %g: sin t = %.4f, cut-point kappa = ' ...
           '%.4f ====\n'], geoname, yc, s0, kappa);
  fprintf('exact glue angle at the two cut points: %.4f, %.4f rad\n', ...
          thetaex(1,1), thetaex(1,2));

  % |gamma'(t)| at the cut points, for turning a parameter interval there
  % into an arclength
  sqg = sqrt(gfun(tv));

  for k = 1:nh
    dx = hvals(k);

    %% Construct a grid, and the band and operators of the two halves
    [x1d, y1d, cand] = make_grid(geowhole.pieces, a, b, cen, dx, bw);
    br = cell(2,1);
    for j = 1:2
      br{j} = setup_band(x1d, y1d, dx, p, order, bw, cand, geo{j});
      br{j}.vid = vertex_ids(br{j}, V);
    end

    %% Baseline: vGMM on the whole curve, one band, no cut
    % On the ellipse this is the ordinary embedded method-of-lines solve
    % and the error the cut variants are trying to get back to.  On the
    % flipped curve it is that solve applied to a corner as if the corner
    % were not there, which is the other thing worth knowing.
    swhole = setup_band(x1d, y1d, dx, p, order, bw, cand, geowhole);
    outw = vgmm_solve({swhole}, [], x1d, y1d, p, dim, a, b, cen, ufun, ffun);
    err(1,k) = outw.errsurf;
    % how far, in arclength, the baseline's worst point is from the
    % nearest cut point: the test of whether a corner is what hurts it
    dcorner(k) = min(abs(wrapangle(outw.terr - tv)).*sqg);

    %% The nodes the two halves have in common
    sh = shared_points(br, V, geo, thetaex);
    nrot(k) = sum(sh.rot);
    nshare(k) = length(sh.i1);
    if (nrot(k) == 0)
      error(['at dx = %g no node clamped to a cut point by one half is ' ...
             'carried by the other; there is nothing to compare'], dx);
    end

    %% Endpoint tangents, and the glue rotation each scheme produces
    % thetamax is now the error in the angle, |theta - theta_exact|, not
    % |theta| itself: on the flipped curve the exact angle is not zero.
    theta = zeros(2, 2, nvar);   % (branch, vertex, variant)
    theta(:,:,nvar) = thetaex;   % the last variant is the exact rotation
    for scheme = 1:2
      tauA = [cp_tangent(br{1}, V(1,:), scheme); ...
              cp_tangent(br{1}, V(2,:), scheme)];
      tauB = [cp_tangent(br{2}, V(1,:), scheme); ...
              cp_tangent(br{2}, V(2,:), scheme)];
      theta(:,:,scheme) = glue_angles(tauA, tauB);
      thetamax(scheme,k) = ...
          max(max(abs(wrapangle(theta(:,:,scheme) - thetaex))));
    end

    %% Solve, once per variant, and compare the halves node by node
    for iv = 1:nvar
      out = vgmm_solve(br, theta(:,:,iv), x1d, y1d, p, dim, a, b, cen, ...
                       ufun, ffun);
      if (out.rowsum > 1e-10)
        error('extension matrix rows do not sum to one (%g)', out.rowsum);
      end
      err(iv+1,k) = out.errsurf;
      % each half against the value its own unknown stands for; on the
      % plain ellipse those are the same number and eA - eB is just
      % u_A - u_B
      eA = out.ubr{1}(sh.i1) - ufun(sh.tA);
      eB = out.ubr{2}(sh.i2) - ufun(sh.tB);

      gap(iv,k) = norm(eA(sh.rot) - eB(sh.rot), inf);
      branch(iv,k) = max(norm(eA(sh.rot), inf), norm(eB(sh.rot), inf));
      allgap(iv,k) = norm(eA - eB, inf);

      % the same comparison on the manifold: each half's interpolant at
      % the cut points, where both of them stand for u(v)
      uAv = interp_at(x1d, y1d, p, br{1}, out.ubr{1}, V);
      uBv = interp_at(x1d, y1d, p, br{2}, out.ubr{2}, V);
      vtx(iv,k) = norm(uAv - uBv, inf);
    end

    fprintf(['dx = %-8.4g shared %-4d rot %-3d both %-3d dth %8.2e %8.2e   ' ...
             'gap %9.3e %9.3e %9.3e   vtx %9.3e %9.3e %9.3e\n'], ...
            dx, nshare(k), nrot(k), sh.nboth, ...
            thetamax(1,k), thetamax(2,k), gap(:,k), vtx(:,k));
  end

  %% Convergence table
  % Every fitted rate stops short of the level where its column bottoms
  % out: see fitrate below.
  rate = nan(1, nvar);
  ratev = nan(1, nvar);
  fprintf('\n%s\n', repmat('-', 1, 78));

  % the surface error first: four columns, the baseline and the three
  % glued solves, which is the left-hand figure of the pair
  rateerr = nan(1, nvar+1);
  fprintf('\nerr    = surface L_inf error of u\n      dx    ');
  for iv = 1:nvar+1
    fprintf('%-22s', varlabels4{iv});
  end
  fprintf('\n');
  for k = 1:nh
    fprintf('%10.4g  ', hvals(k));
    for iv = 1:nvar+1
      fprintf('%12.3e', err(iv,k));
      if (k == 1)
        fprintf('   --  ');
      else
        fprintf(' %6.2f', ...
                log(err(iv,k-1)/err(iv,k))/log(hvals(k-1)/hvals(k)));
      end
      fprintf('   ');
    end
    fprintf('\n');
  end
  fprintf('fitted      ');
  for iv = 1:nvar+1
    [rateerr(iv), nuse] = fitrate(hvals, err(iv,:));
    fprintf('%12.2f (%dl)  ', rateerr(iv), nuse);
  end
  fprintf('\n');
  for imetric = 1:4
    switch imetric
      case 1
        cols = gap;
        name = 'gap    = max |u_A - u_B| at the shared rotated nodes';
      case 2
        cols = vtx;
        name = 'vtx    = max |u_A(v) - u_B(v)| on the manifold';
      case 3
        cols = branch;
        name = 'branch = max(|e_A|, |e_B|) at the same nodes';
      case 4
        cols = allgap;
        name = 'allgap = max |u_A - u_B| over the whole overlap';
    end
    fprintf('\n%s\n      dx    ', name);
    for iv = 1:nvar
      fprintf('%-22s', varlabels{iv});
    end
    fprintf('\n');
    for k = 1:nh
      fprintf('%10.4g  ', hvals(k));
      for iv = 1:nvar
        fprintf('%12.3e', cols(iv,k));
        if (k == 1)
          fprintf('   --  ');
        else
          fprintf(' %6.2f', ...
                  log(cols(iv,k-1)/cols(iv,k))/log(hvals(k-1)/hvals(k)));
        end
        fprintf('   ');
      end
      fprintf('\n');
    end
    fprintf('fitted      ');
    for iv = 1:nvar
      [r, nuse] = fitrate(hvals, cols(iv,:));
      if (imetric == 1)
        rate(iv) = r;
      elseif (imetric == 2)
        ratev(iv) = r;
      end
      fprintf('%12.2f (%dl)  ', r, nuse);
    end
    fprintf('\n');
  end
  fprintf('\nfitted rate of the angle error |theta - theta_exact|:');
  fprintf(' d_k %5.2f, d_k2 %5.2f\n', ...
          fitrate(hvals, thetamax(1,:)), fitrate(hvals, thetamax(2,:)));
  fprintf(['arclength from the nearest cut point to the uncut baseline''s ' ...
           'worst point,\n  as a multiple of dx:']);
  fprintf(' %.1f', dcorner./hvals);
  fprintf('\n');

  results(ci) = struct('ycut', yc, 'kappa', kappa, 'tcut', tv.', ...
                       'dx', hvals, 'gap', gap, 'vtx', vtx, ...
                       'branch', branch, 'allgap', allgap, 'nrot', nrot, ...
                       'nshare', nshare, 'thetamax', thetamax, ...
                       'rate', rate, 'ratev', ratev, ...
                       'err', err, 'rateerr', rateerr);

  %% Plot
  % Positions are set by hand: "axis equal" letterboxes the ellipse, and
  % where the leftover room ends up depends on its aspect ratio.
  [xs1d, ys1d, cands] = make_grid(geowhole.pieces, a, b, cen, dx_show, bw);
  brs = cell(2,1);
  for j = 1:2
    brs{j} = setup_band(xs1d, ys1d, dx_show, p, order, bw, cands, geo{j});
    brs{j}.vid = vertex_ids(brs{j}, V);
  end

  % the band nodes are coloured by the solution, so the picture needs one:
  % the exact rotation, solved on the display grid.  It is much coarser
  % than any level in the convergence study -- the point is to see the
  % nodes, not the accuracy -- so this solve is for the picture only and
  % none of the numbers above come from it.
  outshow = vgmm_solve(brs, thetaex, xs1d, ys1d, p, dim, a, b, cen, ...
                       ufun, ffun);
  ushow = outshow.ubr;
  cl = [min([ushow{1}; ushow{2}]) max([ushow{1}; ushow{2}])];

  if ~exist(figdir, 'dir')
    mkdir(figdir);
  end

  % Figure one: how accurate the solution is.  The uncut ellipse against
  % the three glued solves, which is the question the schemes exist to
  % answer.
  figure(2*ci - 1); clf;
  set(gcf, 'Position', [100 100 1150 480]);
  leftpanel(brs, ushow, cl, V, a, b, cen, yc, tv, dx_show, bw, kappa, geo, geoname);
  axes('Position', [0.56 0.13 0.40 0.74]);
  plotconv(hvals, err, rateerr, varlabels4, 'surface L_\infty error of u');
  title({sprintf('vGMM:  u - \\Delta_S u = f on the cut curve (%s)', geoname), ...
         sprintf('cut at y = %g, cut-point curvature \\kappa = %.3f', ...
                 yc, kappa)}, 'FontSize', 9);
  legend('Location', 'southeast', 'FontSize', 7);
  outfile = fullfile(figdir, ...
                     sprintf('%s_vgmm_cut_y_%g_error.png', geoname, yc));
  exportgraphics(gcf, outfile, 'Resolution', 150);
  fprintf('saved %s\n', outfile);

  % Figure two: how well the two halves agree, the three cross-branch
  % measures on one axis.  Only the glued variants have these, so the
  % uncut baseline has no curve here.
  figure(2*ci); clf;
  set(gcf, 'Position', [100 100 1150 480]);
  leftpanel(brs, ushow, cl, V, a, b, cen, yc, tv, dx_show, bw, kappa, geo, geoname);
  axes('Position', [0.56 0.13 0.40 0.74]);
  plotgaps(hvals, {gap, vtx}, {rate, ratev}, varlabels, ...
           {'gap (nodes)', 'vtx (u at v)'});
  title({'vGMM: the two halves against each other', ...
         sprintf('cut at y = %g, cut-point curvature \\kappa = %.3f', ...
                 yc, kappa)}, 'FontSize', 9);
  legend('Location', 'southeast', 'FontSize', 6, 'NumColumns', 2);
  outfile = fullfile(figdir, ...
                     sprintf('%s_vgmm_cut_y_%g_gaps.png', geoname, yc));
  exportgraphics(gcf, outfile, 'Resolution', 150);
  fprintf('saved %s\n', outfile);
end


%% Summary across the cut heights

sumname = {'gap (u, the grid nodes)', 'vtx (u, on the manifold)'};
sumfield = {'gap', 'vtx'};
sumrate = {'rate', 'ratev'};
for imetric = 1:2
  fprintf('\n\nSummary: fitted rate and %s at dx = %g, against the cut\n', ...
          sumname{imetric}, hvals(end));
  fprintf('    ycut     kappa ');
  for iv = 1:nvar
    fprintf('  %-18s', varlabels{iv});
  end
  fprintf('\n');
  for ci = 1:length(results)
    fprintf('%8.3g  %8.3f ', results(ci).ycut, results(ci).kappa);
    e = results(ci).(sumfield{imetric});
    r = results(ci).(sumrate{imetric});
    for iv = 1:nvar
      fprintf('  %5.2f  %9.3e', r(iv), e(iv,end));
    end
    fprintf('\n');
  end
end


%% One more figure: all the cuts together

if (length(results) > 1)
  figure(2*length(ycut_list) + 1); clf;
  set(gcf, 'Position', [100 100 1150 500]);
  cmap = lines(length(results));

  % left: every cut, and the curve each one produces.  When the lower
  % half is flipped the manifold itself changes with the cut height, so
  % there is one curve per cut rather than one shared ellipse.
  axes('Position', [0.06 0.14 0.36 0.75]);
  hold on;
  th = linspace(0, 2*pi, 2000)';
  hl = plot(cen(1) + a*cos(th), cen(2) + b*sin(th), '-', ...
            'Color', [0.7 0.7 0.7], 'LineWidth', 1.0);
  leg = {'the uncut ellipse'};
  for ci = 1:length(results)
    yc = results(ci).ycut;
    tvc = results(ci).tcut;
    if flipped
      % upper arc as it is, lower arc reflected up across y = yc
      tup = linspace(tvc(1), tvc(2), 1000)';
      tdn = linspace(tvc(2), tvc(1) + 2*pi, 1000)';
      hl(end+1) = plot([cen(1) + a*cos(tup); cen(1) + a*cos(tdn)], ...
                       [cen(2) + b*sin(tup); 2*yc - (cen(2) + b*sin(tdn))], ...
                       '-', 'Color', cmap(ci,:), 'LineWidth', 1.4);
    else
      hl(end+1) = plot(cen(1) + [-1.15*a 1.15*a], yc*[1 1], '-', ...
                       'Color', cmap(ci,:), 'LineWidth', 1.2);
    end
    plot(cen(1) + a*cos(tvc), cen(2) + b*sin(tvc), 'o', ...
         'Color', cmap(ci,:), 'MarkerFaceColor', cmap(ci,:), ...
         'MarkerSize', 7, 'HandleVisibility', 'off');
    leg{end+1} = sprintf('y = %g, \\kappa = %.2f', yc, results(ci).kappa);
  end
  axis equal; grid on; box on;
  xlabel('x'); ylabel('y');
  title(sprintf('the cuts (%s)', geoname), 'FontSize', 9);
  legend(hl, leg, 'Location', 'southoutside', 'FontSize', 7, 'NumColumns', 2);

  % right: d_k and d_k2 at every cut
  axes('Position', [0.55 0.16 0.41 0.72]);
  hold on;
  leg = {};
  style = {'-s', '--d'};
  for ci = 1:length(results)
    for iv = 1:2
      plot(hvals, results(ci).gap(iv,:), style{iv}, 'Color', cmap(ci,:), ...
           'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerFaceColor', cmap(ci,:));
      leg{end+1} = sprintf('%s, y = %g', varlabels{iv}, results(ci).ycut);
    end
  end
  base = 3*max(arrayfun(@(r) max(r.gap(1:2,1)), results));
  plot(hvals, base*(hvals/hvals(1)), 'k:', 'LineWidth', 1.2);
  leg{end+1} = 'O(dx)';
  plot(hvals, base*(hvals/hvals(1)).^2, 'k--', 'LineWidth', 1.2);
  leg{end+1} = 'O(dx^2)';
  set(gca, 'XScale', 'log', 'YScale', 'log');
  xticks(10.^(-6:1:-1));
  grid on; box on;
  xlabel('dx'); ylabel('max |u_A - u_B| at the rotated nodes');
  title('vGMM: the two halves disagreeing, at every cut height', 'FontSize', 9);
  legend(leg, 'Location', 'southwest', 'FontSize', 7);

  outfile = fullfile(figdir, sprintf('%s_vgmm_cut_summary.png', geoname));
  exportgraphics(gcf, outfile, 'Resolution', 150);
  fprintf('saved %s\n\n', outfile);
end


%% ----------------------------------------------------------------------
%% local functions
%% ----------------------------------------------------------------------

function g = make_geo(cpf, yflip, yoff, tlim, a, b, cen)
%MAKE_GEO  one embedded piece of curve: how to find it, and how it is
%   parametrized
%   Every arc here is the ellipse arc t in tlim, embedded either as it
%   stands (yflip = 1, yoff = 0) or reflected across y = yoff/2
%   (yflip = -1).  The reflection is an isometry, so t is still arclength
%   in the same sense and u(t), f(t) carry over unchanged.
%
%     g.cpf      closest point on this piece
%     g.yflip    +1, or -1 for the reflected arc
%     g.yoff     0, or 2*ycut for the reflected arc
%     g.pieces   rows [yflip yoff ta tb], one per arc, for sampling
%     g.parfun   a point of the curve -> its parameter t

  g.cpf = cpf;
  g.yflip = yflip;
  g.yoff = yoff;
  if isempty(tlim)
    g.pieces = zeros(0,4);
    g.parfun = [];
  else
    g.pieces = [yflip yoff tlim(1) tlim(2)];
    g.parfun = @(x,y) arcparam(yflip, yoff, x, y, a, b, cen);
  end
end


function t = arcparam(yflip, yoff, x, y, a, b, cen)
%ARCPARAM  the ellipse parameter of a point on an arc, reflected or not

  t = atan2((((y - yoff)/yflip) - cen(2))/b, (x - cen(1))/a);
end


function xy = arcpoint(piece, t, a, b, cen)
%ARCPOINT  the point of an arc at parameter t; piece is [yflip yoff ...]

  xy = [cen(1) + a*cos(t), piece(1)*(cen(2) + b*sin(t)) + piece(2)];
end


function [cx, cy, dist, bdy] = cpflip(cpf, x, y, yc)
%CPFLIP  closest point on an arc reflected across the line y = yc
%   A reflection is an isometry and is its own inverse, so the closest
%   point on the reflected arc is the reflection of the closest point of
%   the reflected query.

  [cx, cy0, dist, bdy] = cpf(x, 2*yc - y);
  cy = 2*yc - cy0;
end


function [cx, cy, dist, bdy] = cpnearest(cpf1, cpf2, x, y)
%CPNEAREST  closest point on the union of two arcs
%   The whole curve, for the uncut baseline.  Whichever arc is nearer
%   wins; the seam between them is the pair of cut points, where the two
%   agree.  bdy is zero throughout: the union is a closed curve and has
%   no boundary, even where it has a corner.

  [c1x, c1y] = cpf1(x, y);
  [c2x, c2y] = cpf2(x, y);
  d1 = hypot(x - c1x, y - c1y);
  d2 = hypot(x - c2x, y - c2y);
  take2 = d2 < d1;
  cx = c1x;  cx(take2) = c2x(take2);
  cy = c1y;  cy(take2) = c2y(take2);
  dist = min(d1, d2);
  bdy = zeros(size(x));
end


function t = wholeparam(geo, x, y)
%WHOLEPARAM  the parameter of a point of the whole curve
%   Which arc a point belongs to is settled by which one it lies on.  At
%   a cut point it lies on both and they return the same t, so the tie
%   does not matter.

  [c1x, c1y] = geo{1}.cpf(x, y);
  [c2x, c2y] = geo{2}.cpf(x, y);
  on2 = hypot(x - c2x, y - c2y) < hypot(x - c1x, y - c1y);
  t = geo{1}.parfun(x, y);
  t(on2) = geo{2}.parfun(x(on2), y(on2));
end


function tau = unit_tangent(t, sgn, a, b)
%UNIT_TANGENT  sgn * gamma'(t)/|gamma'(t)| on the ellipse
%   sgn = +1 at an arc's upper end and -1 at its lower end gives the
%   tangent pointing OUT of that arc.

  g = [-a*sin(t) b*cos(t)];
  tau = sgn*g/norm(g);
end


function th = glue_angles(tauA, tauB)
%GLUE_ANGLES  the rotation each half needs, from the two outward tangents
%   tauA(j,:) and tauB(j,:) are the outward unit tangents of the two
%   halves at cut point j -- outward meaning away from that half's own
%   arc.  A row of half A sitting past the cut point lies roughly in the
%   direction tauA from it, and the glue has to put it where arc B
%   continues, which is the direction -tauB.  So the rotation is the
%   angle from tauA to -tauB, and the other half is the mirror of that.
%   On a smooth join tauB = -tauA and both angles are zero.

  th = zeros(2, 2);
  for j = 1:2
    th(1,j) = wrapangle(atan2(-tauB(j,2), -tauB(j,1)) - ...
                        atan2( tauA(j,2),  tauA(j,1)));
    th(2,j) = wrapangle(atan2(-tauA(j,2), -tauA(j,1)) - ...
                        atan2( tauB(j,2),  tauB(j,1)));
  end
end

function sh = shared_points(br, V, geo, thetaex)
%SHARED_POINTS  the grid nodes carrying an unknown in BOTH halves
%   With one band per half the unknowns are the band itself, stored as
%   global linear indices into the grid, so the overlap is their
%   intersection.  sh.rot marks the nodes at least one half clamps to a
%   cut point, which is where the glue rotation acts.
%
%   On the plain ellipse that is always exactly one of the two halves:
%   the arcs share a tangent line at the cut point, so they share a
%   normal line there and their clamping regions are its two sides.  At a
%   corner the two normals are different lines, and the wedges between
%   them are clamped by both halves or by neither.  Both cases are
%   ordinary -- a node clamped by both simply has two rotated rows -- and
%   nothing below assumes otherwise.
%
%   The point of this function is sh.tA and sh.tB: the parameter each
%   half's unknown at a shared node stands for.  For a half that is not
%   clamping the node that is just the parameter of its own closest
%   point.  For the half that IS clamping it, the glue does not extend by
%   the constant u(v): it rotates the node about v by the EXACT angle and
%   interpolates on the other arc, so the unknown stands for u there.
%   On the plain ellipse the exact angle is zero and the two parameters
%   coincide -- both halves are after the same number and the difference
%   of the unknowns is itself the error.  At a corner they do not
%   coincide, the raw difference is O(dx) whatever the scheme does, and
%   what has to be compared is each half's own error.
%
%     sh.i1, sh.i2   positions in band 1 and band 2
%     sh.tA, sh.tB   the parameter each half's unknown stands for
%     sh.rot         clamped to a cut point by one of the two halves
%     sh.vid         which cut point, where rot holds
%     sh.x, sh.y     coordinates of the shared nodes

  [~, i1, i2] = intersect(br{1}.band, br{2}.band);
  sh.i1 = i1;
  sh.i2 = i2;
  sh.x = br{1}.x(i1);
  sh.y = br{1}.y(i1);

  cp = {[br{1}.cpx(i1) br{1}.cpy(i1)], [br{2}.cpx(i2) br{2}.cpy(i2)]};

  % the same bit-for-bit test vertex_ids uses
  tol = 100*eps(max(1, max(abs(V(:)))));
  vid = cell(2,1);
  for k = 1:2
    vid{k} = zeros(size(sh.x));
    for j = 1:size(V,1)
      vid{k}(hypot(cp{k}(:,1) - V(j,1), cp{k}(:,2) - V(j,2)) <= tol) = j;
    end
  end
  sh.rot = (vid{1} ~= 0) | (vid{2} ~= 0);
  sh.vid = max(vid{1}, vid{2});
  sh.nboth = sum((vid{1} ~= 0) & (vid{2} ~= 0));

  t = cell(2,1);
  for k = 1:2
    o = 3 - k;
    t{k} = geo{k}.parfun(cp{k}(:,1), cp{k}(:,2));
    m = find(vid{k} ~= 0);
    if ~isempty(m)
      th = thetaex(k, vid{k}(m)).';
      x0 = V(vid{k}(m), 1);
      y0 = V(vid{k}(m), 2);
      ddx = sh.x(m) - x0;
      ddy = sh.y(m) - y0;
      xr = x0 + cos(th).*ddx - sin(th).*ddy;
      yr = y0 + sin(th).*ddx + cos(th).*ddy;
      [cxr, cyr] = geo{o}.cpf(xr, yr);
      t{k}(m) = geo{o}.parfun(cxr, cyr);
    end
  end
  sh.tA = t{1};
  sh.tB = t{2};
end


function [x1d, y1d, cand] = make_grid(pieces, a, b, cen, dx, bw)
%MAKE_GRID  a tight grid around the curve, and the nodes near it
%   Only a sliver of the box is ever used, so the box is kept tight: it
%   is the bounding box of the pieces actually drawn, which for the
%   flipped curve is a good deal shorter than the ellipse's.  The offsets
%   by odd fractions of dx matter: a grid symmetric about the cut line
%   would give the two halves mirror-image tangent estimates, whose
%   errors would cancel in the glue rotation and flatter the estimator.

  rad = ceil(bw) + 3;      % see candidate_nodes
  pad = (rad + 2)*dx;

  % The box is exact, not sampled.  x and y along an arc are a*cos t and
  % +-b*sin t, so both are stationary only at multiples of pi/2, and the
  % extremes of a piece are at its ends or at whichever of those it
  % contains.  Sampling instead would put the box off by ~1e-6, which
  % shifts the whole grid by that much and moves the errors in the fourth
  % digit -- and it would do so differently for different cut heights,
  % making even the uncut baseline depend on where the cut was.
  lo = [inf inf];
  hi = [-inf -inf];
  for ip = 1:size(pieces,1)
    ta = pieces(ip,3);
    tb = pieces(ip,4);
    tc = [ta; tb; (ceil(ta/(pi/2)):floor(tb/(pi/2)))'*(pi/2)];
    xy = arcpoint(pieces(ip,:), tc, a, b, cen);
    lo = min(lo, min(xy, [], 1));
    hi = max(hi, max(xy, [], 1));
  end
  x1d = ((lo(1) - pad - 0.317*dx) : dx : (hi(1) + pad))';
  y1d = ((lo(2) - pad - 0.211*dx) : dx : (hi(2) + pad))';
  cand = candidate_nodes(x1d, y1d, dx, rad, a, b, cen, pieces);
end


function cand = candidate_nodes(x1d, y1d, dx, rad, a, b, cen, pieces)
%CANDIDATE_NODES  grid nodes that could be within bw*dx of the curve
%   A meshgrid over the whole box cannot be built at the finest dx, and
%   all but a sliver of it is discarded by the banding anyway.  So walk
%   the curve instead, piece by piece: sample so that consecutive samples
%   are at most dx/2 apart, snap each sample to the nearest grid node,
%   and keep every node within rad cells of one of those.

  nx = length(x1d);
  ny = length(y1d);
  mask = false(ny, nx);
  [oi, oj] = meshgrid(-rad:rad, -rad:rad);

  for ip = 1:size(pieces,1)
    % |gamma'| <= max(a,b), so this many samples are dx/2 apart at worst
    ta = pieces(ip,3);
    tb = pieces(ip,4);
    ns = ceil(2*(tb - ta)*max(a,b)/dx) + 1;
    xy = arcpoint(pieces(ip,:), linspace(ta, tb, ns)', a, b, cen);
    is = round((xy(:,1) - x1d(1))/dx) + 1;
    js = round((xy(:,2) - y1d(1))/dx) + 1;

    for c = 1:numel(oi)
      ii = is + oi(c);
      jj = js + oj(c);
      ok = (ii >= 1) & (ii <= nx) & (jj >= 1) & (jj <= ny);
      mask(sub2ind([ny nx], jj(ok), ii(ok))) = true;
    end
  end
  cand = find(mask);
end


function s = setup_band(x1d, y1d, dx, p, order, bw, cand, geo)
%SETUP_BAND  the band and the operators for one piece of curve
%   One band, not two: vGMM applies L to the band values and extends the
%   result, so both E and L are square on the same set of nodes and no
%   restriction operator is needed.  The band is the usual one, the nodes
%   within bw*dx of the curve, and every node of it carries an unknown.
%   geo says which curve this is and how it is parametrized -- one arc
%   for a half, both arcs for the uncut baseline.
%
%   s.lfull marks the rows of L whose whole stencil landed in the band.
%   Rows near the outer edge of the band lose part of theirs, which is
%   harmless as long as nothing reads them: in E*L a row of L is read
%   only if E has a nonzero in that column, and the bw above is chosen
%   precisely so those columns are interior.  That is checked here for E
%   and again in vgmm_solve for the cross-cut blocks.

  cpf = geo.cpf;

  switch (order)
    case 2
      sten = 5;
    case 4
      sten = 9;
    otherwise
      error('order %d not implemented', order);
  end

  nx = length(x1d);
  ny = length(y1d);
  [jc, ic] = ind2sub([ny nx], cand);
  xc = x1d(ic);
  yc = y1d(jc);
  [cpx, cpy, dist, bdy] = cpf(xc, yc);

  % the band: the nodes within bw*dx of the arc
  keep = abs(dist) <= bw*dx;
  band = cand(keep);
  n = length(band);
  invband = make_invbandmap(nx*ny, band);

  s.geo = geo;
  s.cpf = cpf;
  s.dx = dx;
  s.band = band;
  s.invband = invband;
  s.n = n;
  s.x = xc(keep);
  s.y = yc(keep);
  s.cpx = cpx(keep);
  s.cpy = cpy(keep);
  s.bdy = bdy(keep);

  % the closest point extension, square on the band
  [Ei, Ej, Es] = interp2_matrix(x1d, y1d, s.cpx, s.cpy, p);
  jj = invband(Ej);
  if any(jj == 0)
    error('a closest point interpolation stencil leaves the band');
  end
  s.E = sparse(Ei, jj, Es, n, n);

  % the Laplacian, square on the same band; entries outside it are
  % dropped by laplacian_2d_matrix, which is what lfull records
  s.L = laplacian_2d_matrix(x1d, y1d, order, band, band);
  s.lfull = (full(sum(s.L ~= 0, 2)) == sten);
  if ~all(s.lfull(unique(jj)))
    error('the Laplacian stencil of a row the extension reads leaves the band');
  end
end


function vid = vertex_ids(s, V)
%VERTEX_IDS  which band node sits at a cut point, 0 if none
%   A node with vid ~= 0 is one whose closest point on this arc is a cut
%   point: those are the ones that get rotated and extended through the
%   other half.  With a single band these are also exactly the unknowns
%   the glue acts on, so the same vector serves both the operator and
%   the picture.

  tol = 100*eps(max(1, max(abs(V(:)))));
  vid = zeros(size(s.cpx));
  for j = 1:size(V,1)
    vid(hypot(s.cpx - V(j,1), s.cpy - V(j,2)) <= tol) = j;
  end
  if any((s.bdy ~= 0) & (vid == 0))
    error('an endpoint node was not matched to a cut point');
  end
end


function tau = cp_tangent(s, v, scheme)
%CP_TANGENT  outward unit tangent at an arc endpoint, from cp differences
%   Uses the band nodes whose closest point on the arc is the endpoint v
%   -- the same nodes the glue then rotates, which with one band is a
%   slightly larger set than the companion script's outer-band rows.
%   With cpbar = cp(2*cp(x)-x) and cp2bar = cp(3*cp(x)-2*x),
%
%     scheme 1:  d_k  = cp - cpbar                          (first order)
%     scheme 2:  d_k2 = 1.5*cp - 2*cpbar + 0.5*cp2bar       (second order)

  tol = 100*eps(max(1, max(abs(v))));
  m = (abs(s.cpx - v(1)) <= tol) & (abs(s.cpy - v(2)) <= tol);
  x = s.x(m);  y = s.y(m);
  cx = s.cpx(m);  cy = s.cpy(m);
  if isempty(x)
    error('no grid points have this cut point as their closest point');
  end

  [b1x, b1y] = s.cpf(2*cx - x, 2*cy - y);
  if (scheme == 1)
    dvx = cx - b1x;
    dvy = cy - b1y;
  else
    [b2x, b2y] = s.cpf(3*cx - 2*x, 3*cy - 2*y);
    dvx = 1.5*cx - 2*b1x + 0.5*b2x;
    dvy = 1.5*cy - 2*b1y + 0.5*b2y;
  end

  nrm = hypot(dvx, dvy);
  ok = isfinite(nrm) & (nrm > tol);
  if ~any(ok)
    error('every cp difference at this cut point was too small to use');
  end
  avg = [mean(dvx(ok)./nrm(ok)) mean(dvy(ok)./nrm(ok))];
  tau = avg/norm(avg);
end


function out = vgmm_solve(br, theta, x1d, y1d, p, dim, a, b, cen, ufun, ffun)
%VGMM_SOLVE  assemble and solve u - laplacian_S u = f the vGMM way
%   br is one branch (the uncut curve, the baseline) or two (the glued
%   halves).  With two branches, the band nodes sitting at a cut point
%   are rotated about it by theta(k,j) -- k the branch, j the cut point
%   -- and then extended by interpolating on the other branch, which is
%   the same glue the iCPM script uses and lands in the same place: the
%   off-diagonal block E_{k,o} of the extension.
%
%   The operator is then the embedded method-of-lines one of [vGMM 2013],
%
%       M = E*L - gamma*(I - E),      gamma = 2*dim/dx^2,
%
%   and the glue needs no further handling.  L is block diagonal, so
%   (E*L)_{k,o} = E_{k,o}*L_o, which extends the OTHER half's L u across
%   the cut; and the penalty row u_i - (E u)_i at a rotated node i of
%   half k is exactly the statement that half k's value there agrees
%   with half o's interpolant at the rotated point.
%
%   Returns the solution split by branch in out.ubr, which is what the
%   two halves are compared through, and the surface L_inf error in
%   out.errsurf.

  nb = numel(br);
  dx = br{1}.dx;
  out.nclamp = 0;

  Eb = cell(nb, nb);
  for i = 1:nb
    for j = 1:nb
      if (i == j)
        Eb{i,j} = br{i}.E;
      else
        Eb{i,j} = sparse(br{i}.n, br{j}.n);
      end
    end
  end

  for k = 1:(nb - 1)*2      % nothing to route when there is one branch
    o = 3 - k;
    rows = find(br{k}.vid ~= 0);
    if isempty(rows)
      error('no rows to route across the cut on branch %d', k);
    end
    th = theta(k, br{k}.vid(rows)).';
    x0 = br{k}.cpx(rows);        % the cut point itself
    y0 = br{k}.cpy(rows);
    ddx = br{k}.x(rows) - x0;
    ddy = br{k}.y(rows) - y0;
    xr = x0 + cos(th).*ddx - sin(th).*ddy;
    yr = y0 + sin(th).*ddx + cos(th).*ddy;

    % A wrong rotation can push a row back across the normal line, and
    % its closest point on the other half is then clamped to the cut
    % point again.  That is the method doing what it does with a wrong
    % angle, not a failure: it is counted, not rejected.
    [cpxr, cpyr, ~, bdyr] = br{o}.cpf(xr, yr);
    out.nclamp = out.nclamp + sum(bdyr ~= 0);
    [Ei, Ej, Es] = interp2_matrix(x1d, y1d, cpxr, cpyr, p);
    jj = br{o}.invband(Ej);
    if any(jj == 0)
        error('a cross-cut interpolation stencil leaves the other band');
    end
    % E*L reads L on those columns, so they must be full-stencil rows
    if ~all(br{o}.lfull(unique(jj)))
      error('a cross-cut stencil reads a row whose Laplacian leaves the band');
    end
    Eb{k,o} = sparse(rows(Ei), jj, Es, br{k}.n, br{o}.n);
    % these rows are extended through the other half now, not this one
    Ekk = Eb{k,k};
    Ekk(rows,:) = 0;
    Eb{k,k} = Ekk;
  end
  if (nb == 2)
    Eblk = [Eb{1,1} Eb{1,2}; Eb{2,1} Eb{2,2}];
    Lblk = blkdiag(br{1}.L, br{2}.L);
  else
    Eblk = Eb{1,1};
    Lblk = br{1}.L;
  end

  % the interpolation weights of every row must still sum to one
  out.rowsum = max(abs(full(sum(Eblk, 2)) - 1));

  %% The vGMM operator, then the elliptic solve
  n = size(Eblk, 1);
  gamma = 2*dim/dx^2;
  M = Eblk*Lblk - gamma*(speye(n) - Eblk);

  rhs = zeros(n, 1);
  off = 0;
  for k = 1:nb
    t = br{k}.geo.parfun(br{k}.cpx, br{k}.cpy);
    rhs(off + (1:br{k}.n)) = ffun(t);
    off = off + br{k}.n;
  end
  u = (speye(n) - M) \ rhs;
  if any(~isfinite(u))
    error('the elliptic solve returned non-finite values');
  end

  %% Split by branch, and the surface error alongside
  out.ubr = cell(nb,1);
  out.errsurf = 0;
  out.ndrop = 0;
  off = 0;
  out.terr = NaN;
  for k = 1:nb
    out.ubr{k} = u(off + (1:br{k}.n));
    [es, nd, tg] = surface_error(x1d, y1d, p, br{k}, out.ubr{k}, ...
                                 a, b, cen, ufun);
    if (es > out.errsurf)
      out.errsurf = es;
      out.terr = tg;
    end
    out.ndrop = out.ndrop + nd;
    off = off + br{k}.n;
  end
  out.u = u;
  out.unknowns = n;
end


function [eLinf, ndrop, targ] = surface_error(x1d, y1d, p, s, u, a, b, cen, ufun)
%SURFACE_ERROR  L_inf error of the interpolant on the curve
%   Midpoints of a uniform partition of each arc's parameter interval, so
%   no sample sits exactly on a cut point.  A sample whose interpolation
%   stencil is not entirely inside the band is dropped and counted.
%   One arc for a half, both for the uncut baseline.

  eLinf = 0;
  ndrop = 0;
  targ = NaN;      % where the max sits, which is how the corner is caught
  for ip = 1:size(s.geo.pieces, 1)
    piece = s.geo.pieces(ip,:);
    ta = piece(3);
    tb = piece(4);
    nq = max(400, ceil(4*(tb - ta)*max(a,b)/s.dx));
    tq = ta + ((0:nq-1)' + 0.5)*(tb - ta)/nq;
    xy = arcpoint(piece, tq, a, b, cen);

    [~, Ej, Es] = interp2_matrix(x1d, y1d, xy(:,1), xy(:,2), p);
    ns = (p+1)^2;
    JJ = reshape(s.invband(Ej), nq, ns);
    SS = reshape(Es, nq, ns);
    good = all(JJ > 0, 2);
    ndrop = ndrop + nq - sum(good);

    ng = sum(good);
    JJ = JJ(good,:);
    SS = SS(good,:);
    ii = repmat((1:ng)', 1, ns);
    Eq = sparse(ii(:), JJ(:), SS(:), ng, s.n);
    tg = tq(good);
    [e, im] = max(abs(Eq*u - ufun(tg)));
    if (e > eLinf)
      eLinf = e;
      targ = tg(im);
    end
  end
end


function hh = drawband(brs, ushow, cl, V, a, b, cen, yc, tv, dx_show, bw, ...
                       msize, geo)
%DRAWBAND  the curve, the band coloured by the solution, and the rotations
%   One filled square per unknown, at the band node that carries it and
%   coloured by its value, both halves on the same colour scale.  The
%   band drawn here is the whole vGMM band, which is wider than the inner
%   band of the iCPM script because every node of it carries an unknown.
%   Two overlays mark where the glue acts: an open SQUARE on a node the
%   upper half rotates across the cut, an open CIRCLE on one the lower
%   half does.  On the plain ellipse those two sets are disjoint, because
%   the halves share a normal line at the cut point and a node falls on
%   one side of it or the other.  At a corner the normals are different
%   lines and a node in the wedge between them gets both marks, which is
%   what the two marker shapes are there to show.
%
%   The curve is drawn arc by arc from geo, so it comes out flipped when
%   the geometry is, and each arc gets its own normal at each cut point.

  hold on;

  % A node in both bands carries TWO unknowns and they are not the same
  % number -- each half's stands for u at its own reference point, which
  % at a corner are different points on the curve.  So one square cannot
  % honestly show a shared node.  Branch A's value fills the square,
  % branch B's is a smaller square inset on top of it, and where the two
  % agree the inset simply disappears into its surroundings.  On the
  % ellipse it always does; on the flipped curve it does not.
  [~, i1, i2] = intersect(brs{1}.band, brs{2}.band);
  onlyB = true(size(brs{2}.x));
  onlyB(i2) = false;

  hh(1) = scatter(brs{1}.x, brs{1}.y, msize^2, ushow{1}, 's', 'filled');
  scatter(brs{2}.x(onlyB), brs{2}.y(onlyB), msize^2, ...
          ushow{2}(onlyB), 's', 'filled');
  hh(8) = scatter(brs{1}.x(i1), brs{1}.y(i1), (0.5*msize)^2, ...
                  ushow{2}(i2), 's', 'filled');
  set(gca, 'CLim', cl);
  colormap(gca, parula);

  for ip = 1:2
    piece = geo{ip}.pieces;
    th = linspace(piece(3), piece(4), 1500)';
    xy = arcpoint(piece, th, a, b, cen);
    hh(2) = plot(xy(:,1), xy(:,2), 'k-', 'LineWidth', 1.2);
  end
  hh(3) = plot(cen(1) + [-1.2*a 1.2*a], yc*[1 1], '--', ...
               'Color', [0.4 0.4 0.4], 'LineWidth', 0.8);
  for j = 1:2
    for ip = 1:2
      nrm = [b*cos(tv(j)) geo{ip}.yflip*a*sin(tv(j))];
      nrm = nrm/norm(nrm);
      seg = 3*bw*dx_show*[-1; 1]*nrm + V(j,:);
      hh(4) = plot(seg(:,1), seg(:,2), '-.', 'Color', [0.5 0.2 0.6], ...
                   'LineWidth', 0.9);
    end
  end

  % the circle is drawn larger than the square so that a node carrying
  % both marks still shows both
  rA = (brs{1}.vid ~= 0);
  rB = (brs{2}.vid ~= 0);
  hh(5) = plot(brs{1}.x(rA), brs{1}.y(rA), 's', 'Color', [0.85 0 0], ...
               'MarkerSize', 1.5*msize, 'LineWidth', 1.0);
  hh(6) = plot(brs{2}.x(rB), brs{2}.y(rB), 'o', 'Color', [0 0 0], ...
               'MarkerSize', 2.1*msize, 'LineWidth', 1.0);
  hh(7) = plot(V(:,1), V(:,2), 'kp', 'MarkerFaceColor', 'y', ...
               'MarkerSize', 2.2*msize);
end


function leftpanel(brs, ushow, cl, V, a, b, cen, yc, tv, dx_show, bw, ...
                   kappa, geo, geoname)
%LEFTPANEL  the manifold, the two half bands, the cut and the cut points
%   Shared by both figures of a pair.  A zoom on a cut point is inset
%   rather than given a panel of its own: at the scale of the curve the
%   rotated rows are a few pixels wide, but the figure is meant to be two
%   panels.

  axes('Position', [0.045 0.30 0.32 0.60]);
  % 3 points is about one cell at this panel's scale, so the squares tile
  % the band instead of merging into a ribbon
  hh = drawband(brs, ushow, cl, V, a, b, cen, yc, tv, dx_show, bw, 3, geo);
  margin = (bw + 2)*dx_show;
  % the y extent is read off the curve: the flipped one sits entirely
  % above the cut line and is not centred on cen(2) at all
  ylo = inf;  yhi = -inf;
  for ip = 1:2
    piece = geo{ip}.pieces;
    xy = arcpoint(piece, linspace(piece(3), piece(4), 400)', a, b, cen);
    ylo = min(ylo, min(xy(:,2)));
    yhi = max(yhi, max(xy(:,2)));
  end
  axis equal; grid on; box on;
  xlim([cen(1)-a-margin cen(1)+a+margin]);
  ylim([ylo-margin yhi+margin]);
  xlabel('x'); ylabel('y');
  title({sprintf('%s:  a = %g, b = %g, cut at y = %g  (\\kappa = %.3f)', ...
                 geoname, a, b, yc, kappa), ...
         sprintf('vGMM band drawn at dx = %.4g', dx_show)}, 'FontSize', 9);
  legend(hh, {'band node, colour = u', 'the manifold', ...
              sprintf('cut line y = %g', yc), 'normals at the cut points', ...
              'rotated for branch A (upper)', ...
              'rotated for branch B (lower)', ...
              'cut points', 'inset: branch B where both carry a value'}, ...
         'FontSize', 6, 'NumColumns', 3, ...
         'Position', [0.045 0.02 0.32 0.15]);

  % The colour bar goes in the gutter, under the zoom.  Inside the axes it
  % would have to dodge the title, and how much room there is above the
  % curve depends on the geometry.  Setting the axes position back
  % afterwards stops the colour bar shrinking it.
  axpos = get(gca, 'Position');
  cb = colorbar;
  set(gca, 'Position', axpos);
  set(cb, 'Position', [0.388 0.32 0.011 0.17], 'FontSize', 6);
  cb.Label.String = 'u';
  cb.Label.FontSize = 7;

  % the zoom sits in the gutter between the two panels.  Inside the left
  % one it would have to dodge the curve, and where the curve leaves room
  % depends on the geometry: the ellipse is hollow in the middle, the
  % flipped lens is not, and the two have different aspect ratios.
  axes('Position', [0.385 0.55 0.095 0.24]);
  drawband(brs, ushow, cl, V, a, b, cen, yc, tv, dx_show, bw, 6, geo);
  w = 3.5*bw*dx_show;
  axis equal; box on;
  xlim([V(1,1)-w V(1,1)+w]);
  ylim([V(1,2)-w V(1,2)+w]);
  set(gca, 'XTick', [], 'YTick', []);
  title('zoom at a cut point', 'FontSize', 7);
end


function plotconv(hvals, err, rate, labels, ylab)
%PLOTCONV  one family of error curves against dx, with slope guides
%   Black for the uncut baseline when it is there, then one colour per
%   glue scheme.

  hold on;
  markers = {'o', 's', 'd', '^'};
  colors = [0 0 0; 0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19];
  n = size(err, 1);
  off = 4 - n;      % three curves means no baseline, so skip the black
  for iv = 1:n
    % the exact rotation lands on the uncut baseline to plotting accuracy
    % at a normal cut, so the baseline is drawn heavy enough to still be
    % seen underneath whatever covers it
    if (iv == 1 && off == 0)
      lw = 3.5;  ms = 11;
    else
      lw = 1.5;  ms = 6;
    end
    plot(hvals, err(iv,:), ['-' markers{iv+off}], ...
         'Color', colors(iv+off,:), 'LineWidth', lw, 'MarkerSize', ms, ...
         'MarkerFaceColor', colors(iv+off,:), ...
         'DisplayName', sprintf('%s, rate %.2f', labels{iv}, rate(iv)));
  end
  slopeguides(hvals, 3*max(err(:,1)));
  set(gca, 'XScale', 'log', 'YScale', 'log');
  xticks(10.^(-6:1:-1));
  grid on; box on;
  xlabel('dx'); ylabel(ylab);
end


function plotgaps(hvals, data, rates, varlabels, metriclabels)
%PLOTGAPS  the cross-branch measures on one axis
%   Colour is the glue scheme, line style is which measure: solid for the
%   grid-node gap, dashed for the value at the cut points, dash-dot for
%   the derivative there.  Reading it down a column of the legend
%   compares schemes; reading across compares what is being measured.

  hold on;
  markers = {'s', 'd', '^'};
  styles = {'-', '--', '-.'};
  widths = [1.5 1.2 1.2];
  colors = [0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19];
  top = 0;
  for im = 1:length(data)
    for iv = 1:size(data{im}, 1)
      plot(hvals, data{im}(iv,:), [styles{im} markers{iv}], ...
           'Color', colors(iv,:), 'LineWidth', widths(im), ...
           'MarkerSize', 5, 'MarkerFaceColor', mfc(colors(iv,:), im), ...
           'DisplayName', sprintf('%s, %s, rate %.2f', ...
                                  metriclabels{im}, varlabels{iv}, ...
                                  rates{im}(iv)));
    end
    top = max(top, max(data{im}(:,1)));
  end
  slopeguides(hvals, 3*top);
  set(gca, 'XScale', 'log', 'YScale', 'log');
  xticks(10.^(-6:1:-1));
  grid on; box on;
  xlabel('dx'); ylabel('disagreement between the two halves');
end


function c = mfc(col, im)
%MFC  filled markers for the first measure, open for the others

  if (im == 1)
    c = col;
  else
    c = 'none';
  end
end


function slopeguides(hvals, base)
%SLOPEGUIDES  the O(dx) and O(dx^2) reference lines

  plot(hvals, base*(hvals/hvals(1)), 'k:', 'LineWidth', 1.2, ...
       'DisplayName', 'O(dx)');
  plot(hvals, base*(hvals/hvals(1)).^2, 'k--', 'LineWidth', 1.2, ...
       'DisplayName', 'O(dx^2)');
end


function val = interp_at(x1d, y1d, p, s, u, pts)
%INTERP_AT  one half's interpolant evaluated at points of the surface
%   The same degree-p interpolation the operator itself uses, restricted
%   to this half's band.  Used at the cut points, where both halves have
%   a value and both of them stand for u(v).

  n = size(pts, 1);
  [~, Ej, Es] = interp2_matrix(x1d, y1d, pts(:,1), pts(:,2), p);
  ns = (p+1)^2;
  JJ = reshape(s.invband(Ej), n, ns);
  SS = reshape(Es, n, ns);
  if any(JJ(:) == 0)
    error('an interpolation stencil at a cut point leaves the band');
  end
  val = sum(SS.*u(JJ), 2);
end


function th = wrapangle(th)
%WRAPANGLE  put an angle into (-pi, pi]

  th = angle(exp(1i*th));
end


function [r, nuse] = fitrate(dx, e)
%FITRATE  slope of log(e) against log(dx), over the levels before the floor
%   The level where e bottoms out is the one where round-off has caught up
%   with the truncation error, so it and everything past it say nothing
%   about the order and are dropped.

  n = length(e);
  [~, imin] = min(e);
  if (imin < n)
    imin = imin - 1;
  end
  use = isfinite(e) & (e > 0) & ((1:n) <= max(imin, 2));
  nuse = sum(use);
  if (nuse < 2)
    r = NaN;
    return;
  end
  c = polyfit(log(dx(use)), log(e(use)), 1);
  r = c(1);
end
