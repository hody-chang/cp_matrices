%% How the glue rotation is built: Rodrigues against atan2
%
% Companion to example_ellipse_cut_tangent_schemes.m and
% example_ellipse_cut_branch_consistency.m.  Same ellipse
%
%     gamma(t) = cen + [a*cos(t), b*sin(t)],     cen = (0.5, 0.5),
%
% cut along y = ycut, same two bands, same d_k and d_k2 endpoint tangents,
% same two manifolds (the plain ellipse, where the exact glue is the
% identity, and the cut-and-reflect one, where it is a definite nonzero
% rotation).  What varies here is neither the geometry nor the tangent
% estimator but the ARITHMETIC that turns a pair of tangents into the
% rotation matrix the glue applies.
%
% THE TWO CONSTRUCTIONS.  A row of one half sitting past a cut point has
% to be turned so that its outward tangent u = tau_src lands on the inward
% tangent w = -tau_tgt of the other half.
%
%   atan2       theta = wrap(atan2(w(2), w(1)) - atan2(u(2), u(1)));
%               R     = [cos(theta) -sin(theta); sin(theta) cos(theta)];
%
%   Rodrigues   R = I + sin(theta) K + (1 - cos(theta)) K^2, with K the
%               skew matrix of the rotation axis.  In the plane the axis
%               is e_z, so K = [0 -1; 1 0], K^2 = -I, and the formula
%               collapses to R = cos(theta) I + sin(theta) K with
%
%                 cos(theta) = u . w,    sin(theta) = u1 w2 - u2 w1,
%
%               one division by the hypot of the two to absorb whatever
%               the inputs were off unit by, and R = [c -s; s c].
%
% The two produce the same matrix in exact arithmetic.  They are not the
% same computation: the first evaluates six library transcendentals (two
% atan2, the complex exponential and the atan2 inside the rewrap, then cos
% and sin) and the second evaluates none.  The angle in the first is an
% intermediate that cos and sin immediately undo, and it is the only place
% a branch cut enters the glue at all.
%
% WHAT IS MEASURED, in five parts:
%
%   1  the construction against an exactly known rotation.  With analytic
%      tangents the glue is algebraic -- on the reflected manifold the
%      tangent components give the cosine and sine of the corner rotation
%      by the double-angle identity -- so ||R - R_exact|| is available to
%      full precision and is a property of the arithmetic alone, with no
%      grid and no estimator anywhere near it.  Also the three identities
%      a rotation ought to satisfy exactly: R'R = I, det R = 1, and
%      R_AB R_BA = I, the two halves' glues being mutual inverses.
%
%   2  a sweep of the glue rotation over a full turn, which is where the
%      two features of the angle-first path that this ellipse does not
%      reach become visible: the cancellation in atan2(w) - atan2(u) as
%      the rotation approaches the identity, and the rewrap at +-pi.
%
%   3  the two constructions on ESTIMATED tangents, d_k and d_k2, over the
%      refinement.  Two questions: do the constructions differ by more
%      than rounding, and how does that difference compare with the error
%      the estimator itself commits?  This is the part that says where the
%      error budget of the glue actually sits.
%
%   4  the quantities the operator consumes: the rotated grid node, its
%      closest point on the far arc, and then the solution of the glued
%      iCPM system itself.  The same solve as the branch-consistency
%      script, run once per construction, differing in nothing else.
%
%   5  cost.
%
% WHAT TO EXPECT.  Parts 1 and 2 are the honest headline: Rodrigues comes
% out two to seven times more accurate in ||R - R_exact||, four times at
% the median over a full turn of glue angles, with both constructions
% within a few units in the last place.  Parts 3 and 4 are the other half
% of the story: at
% this precision the difference does not reach the solution, which is the
% result that makes the change safe rather than important.  The case for
% Rodrigues is not that it rescues an accuracy the angle-first path was
% losing.  It is that it is the cheaper computation, it is the one with no
% branch cut in it, it is the one IEEE 754 makes reproducible across
% platforms and libm versions, and it is the same formula angle3D.m
% already uses in three dimensions -- so the 2D and 3D glues stop being
% two different constructions of the same object.

% Resolve dependencies relative to this example.
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', '..', 'cp_matrices'));
addpath(fullfile(here, '..', '..', 'surfaces'));
addpath(fullfile(here, '..', 'surfaces'));
addpath(fullfile(here, '..', 'rotations'));


%% Parameters

cen = [0.5 0.5];
a = 1.4;    % semi-axis along x (the major one)
b = 0.6;    % semi-axis along y

dim = 2;    % dimension
p = 3;      % interpolation degree
order = 2;  % Laplacian order
% The formula for bw is found in [Ruuth & Merriman 2008] and the 1.0002
% is a safety factor.
bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ((order/2+(p+1)/2)^2));

% Four levels.  Part 3 reads the construction difference against the
% estimator error, and both are settled by the third level; the fifth
% level the sibling scripts use would lengthen part 4 considerably without
% changing a digit of the comparison.
hvals = 0.02*2.^-(0:3);
hvals = hvals(bw*hvals < 0.5*min(b^2/a, a^2/b));
nh = length(hvals);
if (nh < 2)
  error('no usable grid sizes for a = %g, b = %g', a, b);
end

% The two manifolds, as in the sibling scripts.  false: the plain ellipse,
% whose halves join smoothly, where the exact glue is the identity and
% every bit of rotation an estimator produces is error.  true: the arc
% below the cut reflected up across the cut line, so both arcs lie above
% it and meet in a genuine corner, and the exact glue is a known nonzero
% rotation.  Both are run: the identity case is where a construction can
% be exactly right, and the corner case is where it has real work to do.
FLIP = [false true];

METH = {'rodrigues', 'atan2'};
mlabel = {'Rodrigues', 'atan2'};
nm = numel(METH);

SCHEME = [1 2];                     % d_k and d_k2
slabel = {'d_k', 'd_{k2}'};
ns = numel(SCHEME);

figdir = fullfile(here, '..', 'figs');   % .png output goes here

% The manufactured solution of the sibling scripts, carried over unchanged
% so that part 4 is their solve and not a different one.
ufun = @(t) sin(t + 0.7) + 0.5*cos(2*t - 0.3);
utfun = @(t) cos(t + 0.7) - sin(2*t - 0.3);
uttfun = @(t) -sin(t + 0.7) - 2*cos(2*t - 0.3);
gfun = @(t) a^2*sin(t).^2 + b^2*cos(t).^2;
gtfun = @(t) (a^2 - b^2)*sin(2*t);
lapfun = @(t) uttfun(t)./gfun(t) - utfun(t).*gtfun(t)./(2*gfun(t).^2);
ffun = @(t) ufun(t) - lapfun(t);

fprintf('\n');
fprintf('===============================================================\n');
fprintf(' the glue rotation: Rodrigues against atan2, on the cut ellipse\n');
fprintf('===============================================================\n');


%% Part 1: the construction against an exactly known rotation
% The exact glue is available in closed form here, which is what makes a
% construction error measurable at all.  See exact_glue below: the
% reflected manifold's glue is cos and sin of twice the tangent's angle,
% a rational function of the tangent components with no trig in it.

exact = struct('flipped', {}, 'ycut', {}, 'theta', {}, 'Rerr', {}, ...
               'orth', {}, 'det', {}, 'inv', {}, 'tpose', {});

fprintf('\n');
fprintf('---- Part 1: ||R - R_exact|| with analytic tangents -----------\n');
fprintf('Every column is a property of the arithmetic alone: no grid and\n');
fprintf('no tangent estimator is involved.  ||dR|| is ||R - R_exact||_2\n');
fprintf('against the closed form in exact_glue; the rest are identities a\n');
fprintf('rotation ought to satisfy exactly -- orth = ||R''R - I||_inf,\n');
fprintf('det = |det R - 1|, inv = ||R_AB R_BA - I||_inf and tpose =\n');
fprintf('||R_AB - R_BA''||_inf, the two halves'' glues at a cut point being\n');
fprintf('inverse rotations of one another.\n\n');
fprintf('%-9s %7s %9s  %-10s %10s %10s %10s %10s %10s\n', ...
        'manifold', 'ycut', '|theta|', 'build', '||dR||', 'orth', 'det', ...
        'inv', 'tpose');

for fi = 1:numel(FLIP)
  flipped = FLIP(fi);
  [geoname, ycut_list] = manifold_cuts(flipped);

  for ci = 1:numel(ycut_list)
    yc = ycut_list(ci);
    [t1, t2] = cut_points(yc, a, b, cen, flipped);
    tauex = exact_tangents(t1, t2, a, b, flipped);
    csref = exact_glue(t1, t2, a, b, flipped, tauex);

    rec = struct('flipped', flipped, 'ycut', yc, ...
                 'theta', max(max(abs(cs_angle(csref)))), ...
                 'Rerr', zeros(1,nm), 'orth', zeros(1,nm), ...
                 'det', zeros(1,nm), 'inv', zeros(1,nm), ...
                 'tpose', zeros(1,nm));

    for im = 1:nm
      cs = glue_cs(tauex{1}, tauex{2}, METH{im});
      [~, dR] = rot2d_err(cs(:,:,1), cs(:,:,2), csref(:,:,1), csref(:,:,2));
      [orth, dt, iv, tp] = rot_identities(cs);

      rec.Rerr(im) = max(max(dR));
      rec.orth(im) = orth;
      rec.det(im) = dt;
      rec.inv(im) = iv;
      rec.tpose(im) = tp;

      if (im == 1)
        fprintf(['%-9s %7.3g %9.5f  %-10s %10.3e %10.3e %10.3e %10.3e ' ...
                 '%10.3e\n'], geoname, yc, rec.theta, mlabel{im}, ...
                rec.Rerr(im), orth, dt, iv, tp);
      else
        fprintf(['%-9s %7s %9s  %-10s %10.3e %10.3e %10.3e %10.3e ' ...
                 '%10.3e\n'], '', '', '', mlabel{im}, rec.Rerr(im), ...
                orth, dt, iv, tp);
      end
    end
    exact(end+1) = rec;  %#ok<SAGROW>
  end
end

fprintf('\nOn the plain ellipse the exact glue is the identity and both\n');
fprintf('constructions hit it exactly: tau_B = -tau_A bit for bit, so the\n');
fprintf('cross product is an exact zero and so is the atan2 difference.\n');
fprintf('The reflected manifold is where the two part.  Four cut points is\n');
fprintf('too small a sample to read the identity columns from; part 2 puts\n');
fprintf('them over many thousands of tangent pairs.\n');


%% Part 2: a sweep of the glue rotation over a full turn
% Part 1 samples only the corner angles these cut heights produce.  The
% sweep covers the whole range, which a sharper corner or a different
% manifold would reach.
%
% The reference is constructed, not estimated: take a unit u, take a
% (c, s) pair on the unit circle, and build w = R*u.  Then R is known to
% full precision and both constructions are asked to recover it.  The
% worst case over many directions u is what is reported, because the
% quantity of interest is how bad a construction can be, not how bad it
% is on average.

thsweep = [0, 10.^(-16:2:-2), 0.1 0.3 1 1.5 2 2.5 3 3.1, pi-1e-8, pi, ...
           pi+1e-8, 3.5, 4, 5, 6, 2*pi-1e-8, 2*pi];
nsw = numel(thsweep);
nsamp = 2000;
sweep = zeros(nm, nsw);

% a fixed stream, so the two constructions see identical input
rs = RandStream('twister', 'Seed', 20260101);
U = randn(rs, nsamp, 2);
U = U ./ hypot(U(:,1), U(:,2));

for it = 1:nsw
  th = thsweep(it);
  % the reference pair, renormalized once so that it is itself a rotation
  cth = cos(th);  sth = sin(th);
  nr = hypot(cth, sth);
  cth = cth/nr;  sth = sth/nr;
  W = [cth*U(:,1) - sth*U(:,2), sth*U(:,1) + cth*U(:,2)];

  for im = 1:nm
    [c, s] = rot2d_from_to(U, W, METH{im});
    [~, dR] = rot2d_err(c, s, cth, sth);
    sweep(im,it) = max(dR);
  end
end

fprintf('\n');
fprintf('---- Part 2: ||R - R_exact|| over a full turn -----------------\n');
fprintf('Worst case over %d directions at each angle, against a\n', nsamp);
fprintf('reference rotation built to full precision.  A ratio above one\n');
fprintf('favours Rodrigues.\n\n');
fprintf('%14s %12s %12s %8s\n', 'theta', mlabel{1}, mlabel{2}, 'ratio');
for it = 1:nsw
  fprintf('%14.6g %12.3e %12.3e %8.2f\n', thsweep(it), sweep(1,it), ...
          sweep(2,it), sweep(2,it)/max(sweep(1,it), realmin));
end
fprintf('\nworst over the sweep:  %-10s %.3e\n', mlabel{1}, max(sweep(1,:)));
fprintf('                       %-10s %.3e\n', mlabel{2}, max(sweep(2,:)));
fprintf('median ratio atan2/Rodrigues over the sweep: %.2f\n', ...
        median(sweep(2,:)./max(sweep(1,:), realmin)));

plot_sweep(figdir, thsweep, sweep, mlabel);

%% Part 2b: the rotation identities, over a sample large enough to read
% Part 1 has four cut points, which is not enough to tell one rounding
% from another.  Here the same identities are taken over many thousands of
% tangent PAIRS, built as a glue would see them: a source outward tangent
% and an unrelated target outward tangent, so the glue angle covers the
% whole range rather than the two values this ellipse supplies.

nid = 40000;
rs = RandStream('twister', 'Seed', 20260103);
TA = randn(rs, nid, 2);  TA = TA ./ hypot(TA(:,1), TA(:,2));
TB = randn(rs, nid, 2);  TB = TB ./ hypot(TB(:,1), TB(:,2));

ident = zeros(nm, 4);
for im = 1:nm
  [cA, sA] = glue_rot2d(TA, TB, METH{im});
  [cB, sB] = glue_rot2d(TB, TA, METH{im});

  % ||R'R - I||_inf and |det R - 1| are both |c^2 + s^2 - 1| here, since
  % R'R = (c^2 + s^2) I and det R = c^2 + s^2 for R = [c -s; s c]
  ident(im,1) = max(abs(cA.^2 + sA.^2 - 1));
  ident(im,2) = ident(im,1);
  % R_AB R_BA - I, componentwise: the diagonal is cA*cB - sA*sB - 1 and
  % the off-diagonal cA*sB + sA*cB, and the inf norm of a 2x2 is the
  % larger absolute row sum
  dg = cA.*cB - sA.*sB - 1;
  og = cA.*sB + sA.*cB;
  ident(im,3) = max(abs(dg) + abs(og));
  % R_AB - R_BA': the entries are cA - cB and -(sA + sB), twice each
  ident(im,4) = max(abs(cA - cB) + abs(sA + sB));
end

fprintf('\n');
fprintf('---- Part 2b: the rotation identities over %d tangent pairs ---\n', nid);
fprintf('%-12s %12s %12s %12s %12s\n', 'build', 'orth', 'det', 'inv', 'tpose');
for im = 1:nm
  fprintf('%-12s %12.3e %12.3e %12.3e %12.3e\n', mlabel{im}, ident(im,:));
end
fprintf(['\ntpose is ||R_AB - R_BA''||: the two halves'' glues at one cut\n' ...
         'point are inverse rotations, and Rodrigues makes them exact\n' ...
         'transposes because swapping the tangents negates the cross\n' ...
         'product bit for bit and leaves the dot product alone.  orth and\n' ...
         'det go the other way by one unit in the last place, cos and sin\n' ...
         'of a common angle agreeing slightly better than a pair\n' ...
         'normalized by its hypot.  See rot_identities.\n']);


%% Parts 3 and 4: estimated tangents, and what reaches the solution
% One grid family per manifold and cut height.  At each level the two
% endpoint tangent estimators are run once -- an estimate is a property of
% the band, not of the construction, so both constructions are handed the
% SAME tangents and nothing but the arithmetic differs.
%
% Part 3 reports, for each estimator:
%
%   dR(est)   ||R - R_exact||, the error the glue carries.  This is the
%             estimator's error and it converges at the estimator's rate.
%   dR(bld)   ||R_rodrigues - R_atan2||, the disagreement between the two
%             constructions on identical input.  It does not converge; it
%             is rounding.
%   ratio     dR(est)/dR(bld), how far apart the two are.
%
% Part 4 follows the glue downstream on the same levels: the rotated grid
% node, its closest point on the far arc, and the solution of the glued
% iCPM system for  u - laplacian_S u = f.

fprintf('\n');
fprintf('---- Parts 3 and 4: estimated tangents, and downstream -------\n');

grids = struct('flipped', {}, 'ycut', {}, 'Rerr_est', {}, 'Rdiff', {}, ...
               'node', {}, 'cp', {}, 'usol', {}, 'err', {});

for fi = 1:numel(FLIP)
  flipped = FLIP(fi);
  [geoname, ycut_list] = manifold_cuts(flipped);

  for ci = 1:numel(ycut_list)
    yc = ycut_list(ci);
    [t1, t2, V] = cut_points(yc, a, b, cen, flipped);
    tlim = [t1 t2; t2 t1 + 2*pi];
    tauex = exact_tangents(t1, t2, a, b, flipped);
    csref = exact_glue(t1, t2, a, b, flipped, tauex);

    % the two arcs, exactly as the sibling scripts embed them
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
    geowhole = make_geo([], [], [], [], a, b, cen);
    geowhole.pieces = [geo{1}.pieces; geo{2}.pieces];

    Rerr_est = nan(ns, nh);
    Rdiff = nan(ns, nh);
    nodediff = nan(ns, nh);
    cpdiff = nan(ns, nh);
    usol = nan(ns+1, nh);         % the two schemes, then the exact glue
    errm = nan(nm, ns+1, nh);

    fprintf('\n==== %s, cut at y = %g: exact glue |theta| = %.5f rad ====\n', ...
            geoname, yc, max(max(abs(cs_angle(csref)))));

    for k = 1:nh
      dx = hvals(k);
      [x1d, y1d, cand] = make_grid(geowhole.pieces, a, b, cen, dx, bw);
      br = cell(2,1);
      for j = 1:2
        br{j} = setup_band(x1d, y1d, dx, p, order, bw, cand, geo{j});
        br{j}.vid = vertex_ids(br{j}, V);
      end

      for is = 1:ns
        scheme = SCHEME(is);
        % one set of tangents, handed to both constructions
        tauA = [cp_tangent(br{1}, V(1,:), scheme); ...
                cp_tangent(br{1}, V(2,:), scheme)];
        tauB = [cp_tangent(br{2}, V(1,:), scheme); ...
                cp_tangent(br{2}, V(2,:), scheme)];

        cs = cell(nm,1);
        for im = 1:nm
          cs{im} = glue_cs(tauA, tauB, METH{im});
        end

        [~, dR] = rot2d_err(cs{1}(:,:,1), cs{1}(:,:,2), ...
                            csref(:,:,1), csref(:,:,2));
        Rerr_est(is,k) = max(max(dR));
        [~, dB] = rot2d_err(cs{1}(:,:,1), cs{1}(:,:,2), ...
                            cs{2}(:,:,1), cs{2}(:,:,2));
        Rdiff(is,k) = max(max(dB));

        % part 4a: the rotated node and its closest point on the far arc
        [nodediff(is,k), cpdiff(is,k)] = routed_points(br, geo, cs);

        % part 4b: the glued solve, once per construction
        [errm(:,is,k), usol(is,k)] = ...
            solve_both(br, cs, x1d, y1d, p, a, b, cen, ufun, ffun);
      end

      % the exact glue through the same solve, as the control
      csx = cell(nm,1);
      for im = 1:nm
        csx{im} = glue_cs(tauex{1}, tauex{2}, METH{im});
      end
      [errm(:,ns+1,k), usol(ns+1,k)] = ...
          solve_both(br, csx, x1d, y1d, p, a, b, cen, ufun, ffun);

      fprintf('dx = %-9.4g', dx);
      for is = 1:ns
        fprintf('  %-5s dR(est) %8.2e dR(bld) %8.2e', slabel{is}, ...
                Rerr_est(is,k), Rdiff(is,k));
      end
      fprintf('\n');
    end

    %% the tables for this cut
    fprintf('\nPart 3: the estimator error against the construction difference\n');
    fprintf('%-9s', 'dx');
    for is = 1:ns
      fprintf('%13s %13s %9s  ', ['dR(est) ' slabel{is}], ...
              ['dR(bld) ' slabel{is}], 'ratio');
    end
    fprintf('\n');
    for k = 1:nh
      fprintf('%-9.4g', hvals(k));
      for is = 1:ns
        fprintf('%13.3e %13.3e %9.1e  ', Rerr_est(is,k), Rdiff(is,k), ...
                Rerr_est(is,k)/max(Rdiff(is,k), realmin));
      end
      fprintf('\n');
    end
    fprintf('%-9s', 'rate');
    for is = 1:ns
      fprintf('%13.2f %13.2f %9s  ', fitrate(hvals, Rerr_est(is,:)), ...
              fitrate(hvals, Rdiff(is,:)), '');
    end
    fprintf('\n');
    fprintf(['The dR(est) columns converge at the estimator''s order.  The\n' ...
             'dR(bld) columns do not converge at all, because they are\n' ...
             'rounding and not discretization.\n']);

    fprintf('\nPart 4: how far the construction difference propagates\n');
    fprintf('%-9s %12s %12s %12s %12s\n', 'dx', 'd(node)', 'd(node)/dx', ...
            'd(cp)', 'd(u)/|u|');
    for k = 1:nh
      fprintf('%-9.4g %12.3e %12.3e %12.3e %12.3e\n', hvals(k), ...
              max(nodediff(:,k)), max(nodediff(:,k))/hvals(k), ...
              max(cpdiff(:,k)), max(usol(:,k)));
    end

    fprintf('\nsurface L_inf error of u, Rodrigues then atan2 in each pair\n');
    vl = [slabel {'exact R'}];
    fprintf('%-9s', 'dx');
    for iv = 1:numel(vl)
      fprintf('%-26s', vl{iv});
    end
    fprintf('\n');
    for k = 1:nh
      fprintf('%-9.4g', hvals(k));
      for iv = 1:numel(vl)
        fprintf('%11.4e %11.4e  ', errm(1,iv,k), errm(2,iv,k));
      end
      fprintf('\n');
    end
    fprintf('largest relative difference over the table: %.2e\n', ...
            max(abs(errm(1,:) - errm(2,:)))/max(abs(errm(1,:))));

    grids(end+1) = struct('flipped', flipped, 'ycut', yc, ...
                          'Rerr_est', Rerr_est, 'Rdiff', Rdiff, ...
                          'node', nodediff, 'cp', cpdiff, ...
                          'usol', usol, 'err', errm);  %#ok<SAGROW>
  end
end

plot_budget(figdir, hvals, grids, slabel);


%% Part 5: cost
% The glue is built a handful of times per solve, so this is not where the
% runtime goes.  It is reported because the operation count is one of the
% reasons to prefer the construction, and because it is the easiest of the
% claims to check.

ntime = 2e6;
rs = RandStream('twister', 'Seed', 20260102);
U = randn(rs, ntime, 2);  U = U ./ hypot(U(:,1), U(:,2));
W = randn(rs, ntime, 2);  W = W ./ hypot(W(:,1), W(:,2));

tm = zeros(nm, 1);
for im = 1:nm
  rot2d_from_to(U(1:1000,:), W(1:1000,:), METH{im});   % warm up
  tt = tic;
  rot2d_from_to(U, W, METH{im});
  tm(im) = toc(tt);
end

fprintf('\n');
fprintf('---- Part 5: cost of building %g rotations -------------------\n', ntime);
for im = 1:nm
  fprintf('  %-12s %8.4f s\n', mlabel{im}, tm(im));
end
fprintf('  atan2 / Rodrigues: %.2f\n', tm(2)/tm(1));
fprintf(['  transcendental calls per rotation: 0 for Rodrigues, 6 for\n' ...
         '  atan2 -- two atan2, the complex exponential and the atan2 in\n' ...
         '  the rewrap, then cos and sin\n']);


%% Summary

fprintf('\n');
fprintf('===============================================================\n');
fprintf(' Summary\n');
fprintf('===============================================================\n');

Re = reshape([exact.Rerr], nm, []);
fprintf('\n1. analytic tangents, worst ||R - R_exact|| over the four cuts\n');
fprintf('     %-12s %.3e\n', mlabel{1}, max(Re(1,:)));
fprintf('     %-12s %.3e\n', mlabel{2}, max(Re(2,:)));

fprintf('\n2. a full turn of glue angles, worst ||R - R_exact||\n');
fprintf('     %-12s %.3e\n', mlabel{1}, max(sweep(1,:)));
fprintf('     %-12s %.3e   (%.1fx)\n', mlabel{2}, max(sweep(2,:)), ...
        max(sweep(2,:))/max(sweep(1,:)));
fprintf('   and the two halves'' glues are exact transposes under\n');
fprintf('   Rodrigues (%.1e) but not under atan2 (%.1e).\n', ...
        ident(1,4), ident(2,4));

rmin = inf;
for i = 1:numel(grids)
  rmin = min(rmin, min(min(grids(i).Rerr_est ./ ...
                           max(grids(i).Rdiff, realmin))));
end
fprintf('\n3. estimated tangents: the estimator''s own error exceeds the\n');
fprintf('   difference between the two constructions by a factor of at\n');
fprintf('   least %.1e, at every level and on every cut.  The choice of\n', rmin);
fprintf('   construction is not where the glue''s error budget sits.\n');

un = 0;  nd = 0;  de = 0;
for i = 1:numel(grids)
  un = max(un, max(max(grids(i).usol)));
  nd = max(nd, max(max(grids(i).node ./ repmat(hvals, ...
                                        size(grids(i).node,1), 1))));
  e = grids(i).err;
  de = max(de, max(abs(e(1,:) - e(2,:)))/max(abs(e(1,:))));
end
fprintf('\n4. downstream: the two constructions move a rotated grid node\n');
fprintf('   by at most %.1e of a cell, the solution by at most %.1e\n', nd, un);
fprintf('   relative, and the surface error by at most %.1e relative --\n', de);
fprintf('   rounding, at every level tried.\n');

fprintf('\n5. cost: atan2/Rodrigues = %.2f, and 6 transcendentals against 0.\n', ...
        tm(2)/tm(1));

fprintf('\nConclusion.  The two constructions agree to rounding everywhere\n');
fprintf('this geometry reaches, so moving to Rodrigues changes none of\n');
fprintf('the numbers the sibling scripts report.  It is still the\n');
fprintf('construction to keep: it is more accurate against a known\n');
fprintf('rotation, it is cheaper, it has no branch cut to maintain, IEEE\n');
fprintf('754 makes it reproducible where atan2 and cos are not, and it is\n');
fprintf('what angle3D.m already does in three dimensions.\n\n');


%% ---------------------------------------------------------------- helpers
% The geometry, band and solve helpers from make_geo onwards are the ones
% example_ellipse_cut_branch_consistency.m uses, carried over unchanged so
% that part 4 runs that script's solve and not a different one.  Only the
% functions above make_geo are new here.

function cs = glue_cs(tauA, tauB, method)
%GLUE_CS  the rotation each half needs, from the two outward tangents
%   tauA(j,:) and tauB(j,:) are the outward unit tangents of the two halves
%   at cut point j.  cs(k,j,1) and cs(k,j,2) are the cosine and sine of the
%   rotation branch k applies at cut point j.  method selects the
%   arithmetic and is forwarded to glue_rot2d; it is the only thing this
%   script varies.

  n = size(tauA, 1);
  cs = zeros(2, n, 2);
  [c, s] = glue_rot2d(tauA, tauB, method);
  cs(1,:,1) = c;  cs(1,:,2) = s;
  [c, s] = glue_rot2d(tauB, tauA, method);
  cs(2,:,1) = c;  cs(2,:,2) = s;
end


function th = cs_angle(cs)
%CS_ANGLE  the angles of the rotations in cs, for printing only

  th = atan2(cs(:,:,2), cs(:,:,1));
end


function [geoname, ycut_list] = manifold_cuts(flipped)
%MANIFOLD_CUTS  the name and the two cut heights of each manifold
%   On the ellipse, y = cen(2) is the half cut and the second is oblique to
%   both the curve and the grid.  On the flipped curve the half cut is not
%   available: reflecting the lower half of an ellipse across its own
%   centre line lands it exactly on the upper half, so the "curve" would be
%   one arc traversed twice.  Two oblique cuts there instead, one of each
%   curvature.  Same choice as the sibling scripts.

  if flipped
    geoname = 'flipped';
    ycut_list = [0.3 0.1];
  else
    geoname = 'ellipse';
    ycut_list = [0.5 0.3];
  end
end


function [t1, t2, V] = cut_points(yc, a, b, cen, flipped)
%CUT_POINTS  where the cut y = yc meets the ellipse
%   t1 on the right, t2 on the left, and V the two points written exactly
%   the way cpEllipseArc writes a clamped closest point, so that the vertex
%   match in vertex_ids is bit for bit.  They lie on the cut line, so the
%   reflection fixes them and they are the same two points on either
%   manifold.

  s0 = (yc - cen(2))/b;
  if (abs(s0) > 0.95)
    error('cut at y = %g misses the ellipse or grazes it (sin t = %g)', ...
          yc, s0);
  end
  if (flipped && abs(s0) < 0.05)
    error(['the flipped curve degenerates at y = %g: reflecting the lower ' ...
           'half across the centre line puts it on top of the upper half'], ...
          yc);
  end
  t1 = asin(s0);
  t2 = pi - t1;
  tv = [t1; t2];
  V = [cen(1) + a*cos(tv), cen(2) + b*sin(tv)];
end


function tauex = exact_tangents(t1, t2, a, b, flipped)
%EXACT_TANGENTS  the analytic outward tangents of the two halves
%   Reflecting arc 2 negates the y component of its tangents, which is
%   what opens the corner.

  tauex = cell(2,1);
  tauex{1} = [unit_tangent(t1, -1, a, b); unit_tangent(t2,  1, a, b)];
  tauex{2} = [unit_tangent(t1,  1, a, b); unit_tangent(t2, -1, a, b)];
  if flipped
    tauex{2}(:,2) = -tauex{2}(:,2);
  end
end


function cs = exact_glue(t1, t2, a, b, flipped, tauex)
%EXACT_GLUE  the exact glue rotations, in closed form and without glue code
%   This is the reference part 1 measures both constructions against, so it
%   must not be either of them.
%
%   Let T be the unit tangent of the ellipse at a cut parameter.  Branch
%   one's outward tangent there is -T; branch two's is +T on the plain
%   ellipse and M*T with M = diag(1,-1) on the reflected manifold.  The
%   glue carries branch one's outward tangent onto w = -tau_B, so
%
%     plain       w = -(+T) = -T = tau_A, and (c, s) = (1, 0) exactly.
%     reflected   w = -M*T, and
%                   c = (-T) . (-M*T)          = T1^2 - T2^2,
%                   s = (-T1)(T2) - (-T2)(-T1) = -2*T1*T2,
%                 the cosine and sine of twice the angle of T by the
%                 double-angle identity, with
%                 c^2 + s^2 = (T1^2 + T2^2)^2 = 1.  So the exact rotation
%                 is a rational function of the tangent components and
%                 needs no trigonometry at all.
%
%   The expression is the same at both cut points.  The second one is the
%   other end of arc one, where the outward tangents are +T and -M*T rather
%   than -T and +M*T; negating both of them leaves the dot product alone
%   and leaves the cross product alone as well.  The change of sign the
%   turn really does undergo between the two cuts is already carried by
%   T2 = cos(t)/|gamma'|, which is positive at the first cut and negative
%   at the second.  The other branch's rotation is the inverse of this one,
%   so its sine is negated.
%
%   Passing tauex, the outward tangents, turns on a check that each
%   rotation really does carry its own branch's outward tangent onto minus
%   the other's.  It is cheap and it is worth having: a wrong reference
%   would be charged to both constructions alike and would read as a tie.

  tv = [t1 t2];
  cs = zeros(2, 2, 2);
  for j = 1:2
    g = [-a*sin(tv(j)) b*cos(tv(j))];
    T = g/norm(g);
    if flipped
      c = T(1)^2 - T(2)^2;
      s = -2*T(1)*T(2);
    else
      c = 1;
      s = 0;
    end
    cs(1,j,1) = c;  cs(1,j,2) =  s;
    cs(2,j,1) = c;  cs(2,j,2) = -s;
  end

  % A reference that is wrong is worse than no reference: it would be
  % charged to both constructions alike and would read as a tie.  So check
  % the defining property directly -- branch k's rotation carries its own
  % outward tangent onto minus the other branch's -- against the tangents,
  % which are built without reference to anything above.
  if (nargin >= 6) && ~isempty(tauex)
    for j = 1:2
      for k = 1:2
        o = 3 - k;
        R = [cs(k,j,1) -cs(k,j,2); cs(k,j,2) cs(k,j,1)];
        d = norm(R*tauex{k}(j,:).' + tauex{o}(j,:).');
        if ~(d < 1e-14)
          error(['the closed-form glue at cut point %d does not carry ' ...
                 'branch %d''s outward tangent onto minus branch %d''s ' ...
                 '(off by %g)'], j, k, o, d);
        end
      end
    end
  end
end


function [orth, dt, iv, tp] = rot_identities(cs)
%ROT_IDENTITIES  the identities a pair of glue rotations should satisfy
%   orth  max ||R'R - I||_inf over every rotation in cs
%   dt    max |det R - 1|
%   iv    max ||R_AB R_BA - I||_inf at a cut point.  The two halves' glues
%         there are inverse rotations, so their product is the identity.
%   tp    max ||R_AB - R_BA'||_inf, the sharper form of the same statement.
%
%   tp is the one that separates the two constructions, and it separates
%   them completely.  Rodrigues reads the second direction off the first:
%   swapping the tangents leaves the dot product's two summands unchanged
%   and swaps the order of the cross product's two, and floating-point
%   subtraction satisfies fl(p - q) = -fl(q - p) exactly, so the cosine
%   comes out bit for bit the same and the sine bit for bit negated.  The
%   two halves' glues are EXACT transposes.  The angle-first path computes
%   the two directions from four different atan2 calls, and
%   atan2(-y,-x) = atan2(y,x) -+ pi holds only to within a rounding, so
%   its two glues are transposes only approximately.
%
%   orth and dt go marginally the other way: the angle-first path gets
%   c^2 + s^2 = 1 to one unit in the last place, cos and sin being taken of
%   the same angle, where normalizing a pair by its hypot costs two.  That
%   is negligible beside tp, and it is reported rather than only the
%   columns that favour the conclusion.

  orth = 0;  dt = 0;  iv = 0;  tp = 0;
  for j = 1:size(cs, 2)
    RA = [cs(1,j,1) -cs(1,j,2); cs(1,j,2) cs(1,j,1)];
    RB = [cs(2,j,1) -cs(2,j,2); cs(2,j,2) cs(2,j,1)];
    for R = {RA, RB}
      orth = max(orth, norm(R{1}.'*R{1} - eye(2), inf));
      dt = max(dt, abs(det(R{1}) - 1));
    end
    iv = max(iv, norm(RA*RB - eye(2), inf));
    tp = max(tp, norm(RA - RB.', inf));
  end
end


function [nodediff, cpdiff] = routed_points(br, geo, cs)
%ROUTED_POINTS  how far apart the two constructions land a routed row
%   Every outer-band row clamped to a cut point has its grid node rotated
%   about that point by each construction's glue and projected onto the far
%   arc.  Returns the largest difference in the rotated node and in its
%   closest point: the two quantities the extension matrix is built from,
%   so this is the last place a difference can enter before it becomes a
%   matrix entry.

  nodediff = 0;
  cpdiff = 0;
  for k = 1:2
    o = 3 - k;
    rows = find(br{k}.vid ~= 0);
    if isempty(rows)
      continue;
    end
    vids = br{k}.vid(rows);
    x0 = br{k}.cpxout(rows);
    y0 = br{k}.cpyout(rows);
    ddx = br{k}.xout(rows) - x0;
    ddy = br{k}.yout(rows) - y0;

    nmeth = numel(cs);
    xr = cell(nmeth,1);  yr = cell(nmeth,1);
    cx = cell(nmeth,1);  cy = cell(nmeth,1);
    for im = 1:nmeth
      cc = cs{im}(k, vids, 1).';
      ss = cs{im}(k, vids, 2).';
      xr{im} = x0 + cc.*ddx - ss.*ddy;
      yr{im} = y0 + ss.*ddx + cc.*ddy;
      [cx{im}, cy{im}] = geo{o}.cpf(xr{im}, yr{im});
    end
    nodediff = max(nodediff, norm(hypot(xr{1} - xr{2}, yr{1} - yr{2}), inf));
    cpdiff = max(cpdiff, norm(hypot(cx{1} - cx{2}, cy{1} - cy{2}), inf));
  end
end


function [errs, udiff] = solve_both(br, cs, x1d, y1d, p, a, b, cen, ...
                                    ufun, ffun)
%SOLVE_BOTH  the glued iCPM solve, once per construction
%   errs is the surface L_inf error of each, and udiff the relative L_inf
%   difference between the two solution vectors.  Nothing differs between
%   the two solves except the arithmetic that built cs.

  nmeth = numel(cs);
  errs = zeros(nmeth, 1);
  uref = [];
  udiff = 0;
  for im = 1:nmeth
    gl.type = 'rot';
    gl.cs = cs{im};
    out = icpm_solve(br, gl, x1d, y1d, p, a, b, cen, ufun, ffun);
    errs(im) = out.errsurf;
    if isempty(uref)
      uref = out.u;
    else
      udiff = max(udiff, norm(out.u - uref, inf)/norm(uref, inf));
    end
  end
end


function plot_sweep(figdir, thsweep, sweep, mlabel)
%PLOT_SWEEP  the construction error across a full turn of glue angles
%   Left: every angle in the sweep, on a log axis, so the behaviour as the
%   rotation approaches the identity is visible.  Right: the same two
%   curves over [0, 2pi] on a linear axis, which is where a glue rotation
%   actually lives.

  f = figure('Position', [100 100 820 330]);

  subplot(1,2,1);
  keep = thsweep > 0;
  plot(thsweep(keep), max(sweep(1,keep), realmin), 'o-', ...
       'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerFaceColor', 'w');
  hold on;
  plot(thsweep(keep), max(sweep(2,keep), realmin), 's--', ...
       'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerFaceColor', 'w');
  set(gca, 'XScale', 'log', 'YScale', 'log');
  yline(eps, ':k');
  xlim([min(thsweep(keep))/2, 2*max(thsweep)]);
  xlabel('glue angle \theta'); ylabel('||R - R_{exact}||_2');
  title('the whole sweep, log scale');
  legend(mlabel, 'Location', 'northwest', 'FontSize', 8);
  grid on; box on;

  subplot(1,2,2);
  keep = (thsweep >= 0.05) & (thsweep <= 2*pi);
  plot(thsweep(keep), max(sweep(1,keep), realmin), 'o-', ...
       'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerFaceColor', 'w');
  hold on;
  plot(thsweep(keep), max(sweep(2,keep), realmin), 's--', ...
       'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerFaceColor', 'w');
  set(gca, 'YScale', 'log');
  yline(eps, ':k');
  xline(pi, ':k');
  text(pi, min(sweep(1,keep))*1.1, '  \pi', 'FontSize', 8);
  xlim([0 2*pi]);
  set(gca, 'XTick', (0:4)*pi/2, ...
           'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
  xlabel('glue angle \theta'); ylabel('||R - R_{exact}||_2');
  title('the range a glue rotation lives in');
  legend(mlabel, 'Location', 'southeast', 'FontSize', 8);
  grid on; box on;

  save_png(f, figdir, 'rotation_construction_sweep.png');
end


function plot_budget(figdir, hvals, grids, slabel)
%PLOT_BUDGET  the estimator error against the construction difference
%   The left panel is the point of the experiment: two curves that are
%   discretization and converge, and two that are rounding and do not.
%   Each curve is the worst case over the four cuts.
%
%   The right panel follows the difference downstream.  Several of its
%   values are exact zeros -- the two constructions returned the same
%   solution vector bit for bit -- which a log axis cannot show, so zeros
%   are drawn on the floor line and labelled there.

  if isempty(grids)
    return;
  end

  f = figure('Position', [100 100 820 330]);
  co = lines(4);
  flo = 1e-18;        % where an exact zero is drawn on the log axis

  subplot(1,2,1);
  nsl = numel(slabel);
  hl = gobjects(1, 2*nsl);
  lb = cell(1, 2*nsl);
  for is = 1:nsl
    e = -inf(1, numel(hvals));
    d = -inf(1, numel(hvals));
    for i = 1:numel(grids)
      e = max(e, grids(i).Rerr_est(is,:));
      d = max(d, grids(i).Rdiff(is,:));
    end
    hl(2*is-1) = plot(hvals, max(e, flo), 'o-', 'Color', co(is,:), ...
                      'LineWidth', 1.5, 'MarkerFaceColor', 'w');
    hold on;
    lb{2*is-1} = ['||R - R_{exact}||, ' slabel{is}];
    hl(2*is) = plot(hvals, max(d, flo), 's--', 'Color', co(is+2,:), ...
                    'LineWidth', 1.5, 'MarkerFaceColor', 'w');
    lb{2*is} = ['||R_{rod} - R_{atan2}||, ' slabel{is}];
  end
  set(gca, 'XScale', 'log', 'YScale', 'log');
  yline(eps, ':k');
  xlim([min(hvals)/1.5, max(hvals)*1.5]);
  xlabel('dx'); ylabel('error in the glue rotation');
  title('estimator error against construction difference');
  legend(hl, lb, 'Location', 'east', 'FontSize', 7);
  grid on; box on;

  subplot(1,2,2);
  u = -inf(1, numel(hvals));
  n = -inf(1, numel(hvals));
  for i = 1:numel(grids)
    u = max(u, max(grids(i).usol, [], 1));
    n = max(n, max(grids(i).node, [], 1));
  end
  n = n ./ hvals;
  plot(hvals, max(u, flo), 'o-', 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
  hold on;
  plot(hvals, max(n, flo), 's--', 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
  set(gca, 'XScale', 'log', 'YScale', 'log');
  yline(eps, ':k');
  ylim([flo/3, 1e-9]);
  xlim([min(hvals)/1.5, max(hvals)*1.5]);
  if any(u <= 0)
    text(min(hvals), flo*1.6, ' exact 0', 'FontSize', 7);
  end
  xlabel('dx'); ylabel('relative difference');
  title('what reaches the solution');
  legend({'||u_{rod} - u_{atan2}||_\infty / ||u||_\infty', ...
          'rotated node, in cells'}, 'Location', 'northwest', 'FontSize', 7);
  grid on; box on;

  save_png(f, figdir, 'rotation_construction_budget.png');
end


function save_png(f, figdir, name)
%SAVE_PNG  write a figure to figdir, on new MATLAB or old

  if ~exist(figdir, 'dir')
    mkdir(figdir);
  end
  fn = fullfile(figdir, name);
  if exist('exportgraphics', 'file')
    exportgraphics(f, fn, 'Resolution', 150);
  else
    print(f, fn, '-dpng', '-r150');
  end
  fprintf('saved %s\n', fn);
end


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


function tau = unit_tangent(t, sgn, a, b)
%UNIT_TANGENT  sgn * gamma'(t)/|gamma'(t)| on the ellipse
%   sgn = +1 at an arc's upper end and -1 at its lower end gives the
%   tangent pointing OUT of that arc.

  g = [-a*sin(t) b*cos(t)];
  tau = sgn*g/norm(g);
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
%SETUP_BAND  the two bands and the operators for one piece of curve
%   Same construction as example_ellipse_cut_tangent_schemes.m, plus xin
%   and yin: the coordinates of the inner-band nodes, which is where the
%   unknowns live and so where the two halves are compared.  geo says
%   which curve this is and how it is parametrized -- one arc for a half,
%   both arcs for the uncut baseline.

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

  % the initial band: the nodes within bw*dx of the arc
  keep = abs(dist) <= bw*dx;
  band = cand(keep);
  xinit = xc(keep);  yinit = yc(keep);
  cpxinit = cpx(keep);  cpyinit = cpy(keep);
  bdyinit = bdy(keep);

  % the inner band is the set of columns the closest point interpolation
  % touches
  [Ei, Ej, Es] = interp2_matrix(x1d, y1d, cpxinit, cpyinit, p);
  innerband = unique(Ej);
  nin = length(innerband);
  inv_inner = make_invbandmap(nx*ny, innerband);
  Einit = sparse(Ei, inv_inner(Ej), Es, length(band), nin);

  % the outer band is the set of columns the Laplacian on the inner band
  % touches
  Ltemp = laplacian_2d_matrix(x1d, y1d, order, innerband, band);
  if (nnz(Ltemp) ~= sten*nin)
    error('the Laplacian stencil of the inner band leaves the initial band');
  end
  [~, jj] = find(Ltemp);
  outertemp = unique(jj);

  [tf, loc] = ismember(innerband, band(outertemp));
  if ~all(tf)
    error('the inner band is not contained in the outer band');
  end

  s.geo = geo;
  s.cpf = cpf;
  s.dx = dx;
  s.band = band;
  s.innerband = innerband;
  s.outerband = band(outertemp);
  s.inv_inner = inv_inner;
  s.nin = nin;
  s.nout = length(outertemp);
  s.L = Ltemp(:, outertemp);
  s.E = Einit(outertemp, :);
  s.R = sparse(1:nin, loc, ones(nin,1), nin, s.nout);
  s.xout = xinit(outertemp);
  s.yout = yinit(outertemp);
  s.cpxout = cpxinit(outertemp);
  s.cpyout = cpyinit(outertemp);
  s.bdyout = bdyinit(outertemp);
  s.xin = s.R*s.xout;
  s.yin = s.R*s.yout;
  s.cpxin = s.R*s.cpxout;
  s.cpyin = s.R*s.cpyout;
end


function vid = vertex_ids(s, V)
%VERTEX_IDS  which cut point each outer-band row sits at, 0 if none
%   A row with vid ~= 0 is one whose closest point on this arc is a cut
%   point: those are the rows that get rotated and extended through the
%   other half.

  tol = 100*eps(max(1, max(abs(V(:)))));
  vid = zeros(size(s.cpxout));
  for j = 1:size(V,1)
    vid(hypot(s.cpxout - V(j,1), s.cpyout - V(j,2)) <= tol) = j;
  end
  if any((s.bdyout ~= 0) & (vid == 0))
    error('an endpoint row was not matched to a cut point');
  end
end


function tau = cp_tangent(s, v, scheme)
%CP_TANGENT  outward unit tangent at an arc endpoint, from cp differences
%   Uses the outer-band nodes whose closest point on the arc is the
%   endpoint v.  With cpbar = cp(2*cp(x)-x) and cp2bar = cp(3*cp(x)-2*x),
%
%     scheme 1:  d_k  = cp - cpbar                          (first order)
%     scheme 2:  d_k2 = 1.5*cp - 2*cpbar + 0.5*cp2bar       (second order)

  tol = 100*eps(max(1, max(abs(v))));
  m = (abs(s.cpxout - v(1)) <= tol) & (abs(s.cpyout - v(2)) <= tol);
  x = s.xout(m);  y = s.yout(m);
  cx = s.cpxout(m);  cy = s.cpyout(m);
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


function out = icpm_solve(br, gl, x1d, y1d, p, a, b, cen, ufun, ffun)
%ICPM_SOLVE Assemble and solve u - laplacian_S u with rigid rotation glue.
% For two branches, clamped outer rows are rotated about their cut point,
% projected onto the other arc, and interpolated there.

  nb = numel(br);
  out.nclamp = 0;

  Eb = cell(nb, nb);
  ts = cell(nb, 1);
  for i = 1:nb
    for j = 1:nb
      if (i == j)
        Eb{i,j} = br{i}.E;
      else
        Eb{i,j} = sparse(br{i}.nout, br{j}.nin);
      end
    end
    ts{i} = br{i}.geo.parfun(br{i}.cpxout, br{i}.cpyout);
  end

  for k = 1:(nb - 1)*2      % nothing to route when there is one branch
    o = 3 - k;
    rows = find(br{k}.vid ~= 0);
    if isempty(rows)
      error('no rows to route across the cut on branch %d', k);
    end
    vids = br{k}.vid(rows);
    x0 = br{k}.cpxout(rows);        % the cut point itself
    y0 = br{k}.cpyout(rows);
    xq = br{k}.xout(rows);
    yq = br{k}.yout(rows);
    % the glue pair is applied exactly as it was built: no angle in the
    % loop, and so no cos/sin undoing an atan2
    cc = gl.cs(k, vids, 1).';
    ss = gl.cs(k, vids, 2).';
    ddx = xq - x0;
    ddy = yq - y0;
    xr = x0 + cc.*ddx - ss.*ddy;
    yr = y0 + ss.*ddx + cc.*ddy;

    % A wrong glue can push a row back across the normal line, and its
    % closest point on the other half is then clamped to the cut point
    % again.  That is the method doing what it does with a wrong glue, not
    % a failure: it is counted, not rejected.
    [cpxr, cpyr, ~, bdyr] = br{o}.cpf(xr, yr);
    out.nclamp = out.nclamp + sum(bdyr ~= 0);
    [Ei, Ej, Es] = interp2_matrix(x1d, y1d, cpxr, cpyr, p);
    jj = br{o}.inv_inner(Ej);
    if any(jj == 0)
      error('a cross-cut interpolation stencil leaves the other inner band');
    end
    Eb{k,o} = sparse(rows(Ei), jj, Es, br{k}.nout, br{o}.nin);
    % these rows are extended through the other half now, not this one
    Ekk = Eb{k,k};
    Ekk(rows,:) = 0;
    Eb{k,k} = Ekk;
    % and they stand for u where the glue sent them
    ts{k}(rows) = br{o}.geo.parfun(cpxr, cpyr);
  end
  if (nb == 2)
    Eblk = [Eb{1,1} Eb{1,2}; Eb{2,1} Eb{2,2}];
    Lblk = blkdiag(br{1}.L, br{2}.L);
    Rblk = blkdiag(br{1}.R, br{2}.R);
  else
    Eblk = Eb{1,1};
    Lblk = br{1}.L;
    Rblk = br{1}.R;
  end

  % the interpolation weights of every row must still sum to one
  out.rowsum = max(abs(full(sum(Eblk, 2)) - 1));

  %% Diagonal splitting for iCPM, then the elliptic solve
  M = lapsharp_unordered(Lblk, Eblk, Rblk);
  n = size(M, 1);

  rhs = zeros(n, 1);
  off = 0;
  for k = 1:nb
    rhs(off + (1:br{k}.nin)) = ffun(br{k}.R*ts{k});
    off = off + br{k}.nin;
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
    out.ubr{k} = u(off + (1:br{k}.nin));
    [es, nd, tg] = surface_error(x1d, y1d, p, br{k}, out.ubr{k}, ...
                                 a, b, cen, ufun);
    if (es > out.errsurf)
      out.errsurf = es;
      out.terr = tg;
    end
    out.ndrop = out.ndrop + nd;
    off = off + br{k}.nin;
  end
  out.u = u;
  out.unknowns = n;
end


function [eLinf, ndrop, targ] = surface_error(x1d, y1d, p, s, u, a, b, cen, ufun)
%SURFACE_ERROR  L_inf error of the interpolant on the curve
%   Midpoints of a uniform partition of each arc's parameter interval, so
%   no sample sits exactly on a cut point.  A sample whose interpolation
%   stencil is not entirely inside the inner band is dropped and counted.
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
    JJ = reshape(s.inv_inner(Ej), nq, ns);
    SS = reshape(Es, nq, ns);
    good = all(JJ > 0, 2);
    ndrop = ndrop + nq - sum(good);

    ng = sum(good);
    JJ = JJ(good,:);
    SS = SS(good,:);
    ii = repmat((1:ng)', 1, ns);
    Eq = sparse(ii(:), JJ(:), SS(:), ng, s.nin);
    tg = tq(good);
    [e, im] = max(abs(Eq*u - ufun(tg)));
    if (e > eLinf)
      eLinf = e;
      targ = tg(im);
    end
  end
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
