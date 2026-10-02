%% How accurate is the glue rotation at an iCPM cut point?
%
% This script measures the ROTATION, not a PDE.  Nothing here solves
% anything: it builds the bands, finds the rows sitting at a cut point,
% runs the two tangent schemes on them, and compares the rotation they
% produce against the exact one.  What that rotation then does to a
% surface PDE is the companion script,
% example_ellipse_cut_branch_consistency.m.
%
% The surface is one fixed ellipse
%
%     gamma(t) = cen + [a*cos(t), b*sin(t)],     cen = (0.5, 0.5),
%
% cut along a horizontal line y = ycut.  Moving that line is how the
% curvature at the cut points is varied: the cut meets the ellipse at
% sin(t) = (ycut - cen(2))/b, where
%
%     kappa = a*b / (a^2*sin(t)^2 + b^2*cos(t)^2)^(3/2),
%
% the same value at both cut points (they are mirror images in x).  Note
% that the ellipse has to be WIDE for the cut height to matter: with the
% major axis horizontal, the cut through the centre lands on the ends of
% the major axis where kappa = a/b^2 is at its largest, and lowering the
% cut walks kappa down towards b/a^2 at the bottom.  On a tall ellipse the
% centre cut is already at the flattest point and the level barely moves
% kappa at all.
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
%                     only overshoot.
%
% The reflection is an isometry, so the two curves are the same
% 1-manifold and everything is written in the ellipse parameter t of the
% point's pre-image.  Only the embedding differs, and with it the closest
% point map the estimators see.  The half cut y = cen(2) is not available
% when flipped: reflecting the lower half of an ellipse across its own
% centre line lays it exactly on the upper half, and the "curve" would be
% one arc traversed twice.
%
% TWO TANGENT SCHEMES.  At a grid point x whose closest point on a half
% is the cut point v, with cpbar = cp(2*cp(x) - x) and
% cp2bar = cp(3*cp(x) - 2*x),
%
%     d_k  = cp - cpbar                        (first order)
%     d_k2 = 1.5*cp - 2*cpbar + 0.5*cp2bar     (second order)
%
% Each is normalized, averaged over the rows at that cut point, and
% normalized again.  d_k has tangent error ~ (kappa/2)*dx and d_k2
% ~ 2*kappa^2*dx^2.
%
% THREE ERRORS are reported at each dx, all maxima over the four
% (branch, cut point) pairs:
%
%     dtau   = |tau - tau_exact|,        the tangent the scheme estimates
%     dtheta = |theta - theta_exact|,    the angle that tangent gives
%     ||dR|| = ||R(theta) - R(theta_exact)||_2,   the rotation matrix
%
% The last is the one the operator actually applies, and it is what the
% figures plot.  For plane rotations ||R(a) - R(b)||_2 = 2|sin((a-b)/2)|,
% which is |a - b| to leading order, so the matrix error and the angle
% error have the same rate and differ only in the constant; both are
% tabulated so that is visible rather than assumed.
%
% One figure per cut height: left, the curve with its band, the cut and a
% zoom on a cut point; right, the rotation error against dx for the two
% schemes, with the tangent error dashed underneath it.  A last figure
% collects the cuts.
%
% Because there is no elliptic solve, this runs in seconds rather than
% minutes and finer levels are affordable if wanted.
%
% Run headlessly, from this directory:
%   matlab -batch "example_ellipse_cut_tangent_schemes"
% Set the "flipped" flag below to false for the ellipse and true for the
% corner.  Figures are named for the geometry, so the two do not overwrite
% each other.  The cut heights and grid sizes are edited below as well.

% Resolve dependencies relative to this example.
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', '..', 'cp_matrices'));
addpath(fullfile(here, '..', '..', 'surfaces'));
addpath(fullfile(here, '..', 'surfaces'));


%% Parameters

cen = [0.5 0.5];
a = 1.4;    % semi-axis along x (the major one: see the note above)
b = 0.6;    % semi-axis along y

% The manifold.
%
%   flipped = false   the plain ellipse.  The two halves join smoothly,
%                     the exact glue rotation is the identity, and any
%                     rotation an estimator produces is pure error.  This
%                     is the case the note above describes.
%
%   flipped = true    the arc below the cut is reflected UP across the
%                     cut line, (x,y) -> (x, 2*ycut - y).  Both arcs then
%                     lie above the cut line and meet at the two cut
%                     points in a genuine corner, so the exact rotation
%                     is a definite nonzero angle -- something for the
%                     estimators to get right rather than a zero they can
%                     only overshoot.
%
% The reflection is an isometry, so the two curves are the same
% 1-manifold and everything below is written in the ellipse parameter t
% of the point's pre-image: u(t), f(t) and the exact Laplace-Beltrami
% carry over unchanged, and the two runs solve the identical intrinsic
% problem.  Only the embedding, and so the closest point map, differs.
flipped = true;
if flipped
  geoname = 'flipped';
else
  geoname = 'ellipse';
end

% The heights of the horizontal cut; on the ellipse y = cen(2) cuts
% through the centre, which for this ellipse is the highest-curvature
% cut.  That one is not available when flipped: reflecting the lower half
% of an ellipse across its own centre line lays it exactly on the upper
% half, and the "curve" would be one arc traversed twice, with the two
% bands on top of each other and no closest point well defined.
if flipped
  ycut_list = [0.4 0.3 0.2 0.1];
else
  ycut_list = [0.5 0.4 0.3 0.2 0.1];
end

% Grid sizes.  The coarse end is where the rotation error stands out above
% the baseline; the fine end is where the baseline turns back up, the
% elliptic solve having run out of precision before it runs out of dx^2.
hvals = 0.02*2.^-(0:7);

dim = 2;    % dimension
p = 3;      % interpolation degree
order = 2;  % Laplacian order
% The formula for bw is found in [Ruuth & Merriman 2008] and the 1.0002
% is a safety factor.
bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ((order/2+(p+1)/2)^2));

% A band point has a unique closest point only while bw*dx stays under the
% smallest radius of curvature of the ellipse, min(b^2/a, a^2/b).  Levels
% that come anywhere near that are not a closest point method at all.
hvals = hvals(bw*hvals < 0.5*min(b^2/a, a^2/b));
nh = length(hvals);
if (nh < 2)
  error('no usable grid sizes for a = %g, b = %g', a, b);
end

% the band on the left of each figure is drawn at this spacing, coarse
% enough to see the individual nodes and fine enough to still be a band
dx_show = min(0.05, 0.5/(bw*max(a/b^2, b/a^2)));

figdir = fullfile(here, '..', 'figs');   % .png output goes here

schemelabels = {'d_k', 'd_{k2}'};


%% One cut height at a time
% No manufactured solution and no PDE here: nothing below solves
% anything.  The bands are built only so that the rows sitting at a cut
% point can be found, the two tangent schemes run on them, and the
% rotation they produce compared against the exact one.

results = struct('ycut', {}, 'kappa', {}, 'tcut', {}, 'dx', {}, ...
                 'Rerr', {}, 'therr', {}, 'tanerr', {}, 'nrot', {}, ...
                 'rate', {}, 'thetaex', {});

for ci = 1:length(ycut_list)
  yc = ycut_list(ci);

  % where the cut meets the ellipse: t1 on the right, t2 on the left.
  % Branch 1 is the upper arc [t1 t2], branch 2 the lower arc that closes
  % the ellipse.
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
  % point of the arc, and the inverse of that map.
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
  % the two arcs together, used only to size the grid and to draw
  geowhole = make_geo([], [], [], [], a, b, cen);
  geowhole.pieces = [geo{1}.pieces; geo{2}.pieces];

  % Exact outward unit tangents, +gamma' at an arc's upper end and
  % -gamma' at its lower end.  Vertex 1 = gamma(t1) is branch 1's lower
  % end and branch 2's upper end; vertex 2 = gamma(t2) is the other way
  % round.  On the ellipse they come out exactly opposite, which is the
  % statement that the exact glue rotation is the identity.  Reflecting
  % arc 2 negates the y component of its tangents, which opens the corner
  % and makes the exact rotation a definite nonzero angle.
  tauex = {[unit_tangent(t1, -1, a, b); unit_tangent(t2,  1, a, b)], ...
           [unit_tangent(t1,  1, a, b); unit_tangent(t2, -1, a, b)]};
  if flipped
    tauex{2}(:,2) = -tauex{2}(:,2);
  end
  thetaex = glue_angles(tauex{1}, tauex{2});

  Rerr = nan(2, nh);        % max ||R(theta) - R(theta_exact)||_2
  therr = nan(2, nh);       % max |theta - theta_exact|
  tanerr = nan(2, nh);      % max |tau - tau_exact|
  nrot = nan(2, nh);        % how many rows each estimate averages over

  fprintf(['\n==== %s, cut at y = %g: sin t = %.4f, cut-point kappa = ' ...
           '%.4f ====\n'], geoname, yc, s0, kappa);
  fprintf('exact glue angle at the two cut points: %.4f, %.4f rad\n', ...
          thetaex(1,1), thetaex(1,2));

  for k = 1:nh
    dx = hvals(k);

    %% Construct a grid, and the bands of the two halves
    [x1d, y1d, cand] = make_grid(geowhole.pieces, a, b, cen, dx, bw);
    br = cell(2,1);
    for j = 1:2
      br{j} = setup_band(x1d, y1d, dx, p, order, bw, cand, geo{j});
      br{j}.vid = vertex_ids(br{j}, V);
    end

    %% Endpoint tangents, and the glue rotation each scheme produces
    % Three errors, all maxima over the four (branch, cut point) pairs:
    % in the tangent the scheme estimates, in the angle that tangent
    % gives, and in the rotation matrix that angle gives.  The last is
    % the one the operator actually applies.
    for scheme = 1:2
      tauA = [cp_tangent(br{1}, V(1,:), scheme); ...
              cp_tangent(br{1}, V(2,:), scheme)];
      tauB = [cp_tangent(br{2}, V(1,:), scheme); ...
              cp_tangent(br{2}, V(2,:), scheme)];
      terr = 0;
      for j = 1:2
        terr = max([terr norm(tauA(j,:) - tauex{1}(j,:)) ...
                    norm(tauB(j,:) - tauex{2}(j,:))]);
      end
      dth = wrapangle(glue_angles(tauA, tauB) - thetaex);
      % ||R(a) - R(b)||_2 = 2|sin((a-b)/2)| for plane rotations, which is
      % |a - b| to leading order: the matrix error and the angle error
      % have the same rate, and the matrix one is what gets applied
      Rerr(scheme,k) = max(max(2*abs(sin(dth/2))));
      therr(scheme,k) = max(max(abs(dth)));
      tanerr(scheme,k) = terr;
    end
    nrot(1,k) = sum(br{1}.vid ~= 0);
    nrot(2,k) = sum(br{2}.vid ~= 0);

    fprintf(['dx = %-8.4g rows %-4d %-4d   ||dR|| %9.3e %9.3e   ' ...
             'dtheta %9.3e %9.3e   dtau %9.3e %9.3e\n'], ...
            dx, nrot(1,k), nrot(2,k), Rerr(:,k), therr(:,k), tanerr(:,k));
  end

  %% Convergence table
  % Every fitted rate here, and in the summary at the end, stops short of
  % the level where its column bottoms out: see fitrate below.
  rate = nan(1, 2);
  cols = [Rerr; therr; tanerr];
  labels = {'||dR|| d_k', '||dR|| d_{k2}', 'dtheta d_k', 'dtheta d_{k2}', ...
            'dtau d_k', 'dtau d_{k2}'};
  fprintf('\n      dx    ');
  for iv = 1:length(labels)
    fprintf('%-22s', labels{iv});
  end
  fprintf('\n');
  for k = 1:nh
    fprintf('%10.4g  ', hvals(k));
    for iv = 1:size(cols,1)
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
  for iv = 1:size(cols,1)
    [r, nuse] = fitrate(hvals, cols(iv,:));
    if (iv <= 2)
      rate(iv) = r;      % the rotation matrix error is what is plotted
    end
    fprintf('%12.2f (%dl)  ', r, nuse);
  end
  fprintf('\n');

  results(ci) = struct('ycut', yc, 'kappa', kappa, 'tcut', tv.', ...
                       'dx', hvals, 'Rerr', Rerr, 'therr', therr, ...
                       'tanerr', tanerr, 'nrot', nrot, ...
                       'rate', rate, 'thetaex', thetaex);

  %% Plot
  % Positions are set by hand: "axis equal" letterboxes the ellipse, and
  % where the leftover room ends up depends on its aspect ratio.
  [xs1d, ys1d, cands] = make_grid(geowhole.pieces, a, b, cen, dx_show, bw);
  brs = cell(2,1);
  for j = 1:2
    brs{j} = setup_band(xs1d, ys1d, dx_show, p, order, bw, cands, geo{j});
  end

  figure(ci); clf;
  set(gcf, 'Position', [100 100 1150 620]);

  % left, top: the curve, its band and the cut
  axes('Position', [0.055 0.56 0.40 0.35]);
  drawband(brs, V, a, b, cen, yc, tv, dx_show, bw, 8, geo);
  margin = (bw + 2)*dx_show;
  % the y extent is read off the curve: the flipped one sits entirely
  % above the cut line and is not centred on cen(2) at all
  ylo = inf;  yhi = -inf;
  for ip = 1:2
    piece = geo{ip}.pieces;
    xyshow = arcpoint(piece, linspace(piece(3), piece(4), 400)', a, b, cen);
    ylo = min(ylo, min(xyshow(:,2)));
    yhi = max(yhi, max(xyshow(:,2)));
  end
  axis equal; grid on; box on;
  xlim([cen(1)-a-margin cen(1)+a+margin]);
  ylim([ylo-margin yhi+margin]);
  xlabel('x'); ylabel('y');
  title({sprintf('%s:  a = %g, b = %g, cut at y = %g  (\\kappa = %.3f)', ...
                 geoname, a, b, yc, kappa), ...
         sprintf('band drawn at dx = %.4g', dx_show)}, 'FontSize', 9);

  % left, bottom: the same picture zoomed on one cut point, where at the
  % scale above the rows that get rotated are only a few pixels wide.  The
  % normal line there, not the cut line, is what separates the rows a half
  % keeps from the rows it has to fetch from the other half.
  axes('Position', [0.075 0.08 0.20 0.36]);
  hh = drawband(brs, V, a, b, cen, yc, tv, dx_show, bw, 11, geo);
  w = 4*bw*dx_show;
  axis equal; box on;
  xlim([V(1,1)-w V(1,1)+w]);
  ylim([V(1,2)-w V(1,2)+w]);
  set(gca, 'XTick', [], 'YTick', []);
  title('zoom at a cut point', 'FontSize', 9);
  legend(hh, {'upper half band', 'lower half band', 'the manifold', ...
              sprintf('cut line y = %g', yc), 'normals at the cut points', ...
              'upper-half rows sent across', 'lower-half rows sent across', ...
              'cut points'}, 'FontSize', 7, ...
         'Position', [0.300 0.10 0.16 0.32]);

  % right: the rotation error against dx.  Solid, the rotation matrix the
  % operator applies; dashed, the tangent it was built from.
  axes('Position', [0.55 0.32 0.41 0.58]);
  plotconv(hvals, Rerr, rate, tanerr, schemelabels);
  title({sprintf('the glue rotation on the cut curve (%s)', geoname), ...
         sprintf(['cut at y = %g, \\kappa = %.3f, exact angle ' ...
                  '%.3f rad'], yc, kappa, thetaex(1,1))}, 'FontSize', 9);
  legend('Location', 'southeast', 'FontSize', 7);

  if ~exist(figdir, 'dir')
    mkdir(figdir);
  end
  outfile = fullfile(figdir, ...
                     sprintf('%s_tangent_schemes_y_%g.png', geoname, yc));
  exportgraphics(gcf, outfile, 'Resolution', 150);
  fprintf('saved %s\n', outfile);
end


%% Summary across the cut heights

fprintf(['\n\nSummary: fitted rate and ||R - R_exact|| at dx = %g, ' ...
         'against the cut\n'], hvals(end));
fprintf('    ycut     kappa    exact angle');
for iv = 1:2
  fprintf('  %-18s', schemelabels{iv});
end
fprintf('\n');
for ci = 1:length(results)
  fprintf('%8.3g  %8.3f  %10.4f ', results(ci).ycut, results(ci).kappa, ...
          results(ci).thetaex(1,1));
  for iv = 1:2
    fprintf('  %5.2f  %9.3e', results(ci).rate(iv), results(ci).Rerr(iv,end));
  end
  fprintf('\n');
end


%% One more figure: all the cuts together
% This is where the dependence on the cut itself shows up: the centre cut
% keeps slope 2 and every oblique one drops to slope 1.

figure(length(ycut_list) + 1); clf;
set(gcf, 'Position', [100 100 1150 500]);
kap = arrayfun(@(r) r.kappa, results);
cmap = parula(length(results) + 1);

% left: every cut, and the curve each one produces.  When the lower half
% is flipped the manifold itself changes with the cut height, so there is
% one curve per cut rather than one shared ellipse.
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
    tup = linspace(tvc(1), tvc(2), 1000)';
    tdn = linspace(tvc(2), tvc(1) + 2*pi, 1000)';
    hl(end+1) = plot([cen(1) + a*cos(tup); cen(1) + a*cos(tdn)], ...
                     [cen(2) + b*sin(tup); 2*yc - (cen(2) + b*sin(tdn))], ...
                     '-', 'Color', cmap(ci,:), 'LineWidth', 1.4);
  else
    hl(end+1) = plot(cen(1) + [-1.15*a 1.15*a], yc*[1 1], '-', ...
                     'Color', cmap(ci,:), 'LineWidth', 1.2);
  end
  % the cut points ride along with their cut line, no legend entry of
  % their own
  plot(cen(1) + a*cos(tvc), cen(2) + b*sin(tvc), 'o', ...
       'Color', cmap(ci,:), 'MarkerFaceColor', cmap(ci,:), ...
       'MarkerSize', 7, 'HandleVisibility', 'off');
  leg{end+1} = sprintf('y = %g, \\kappa = %.2f', yc, results(ci).kappa);
end
axis equal; grid on; box on;
xlabel('x'); ylabel('y');
title(sprintf('the cuts (%s): kappa runs from %.2f down to %.2f', ...
              geoname, max(kap), min(kap)), 'FontSize', 9);
legend(hl, leg, 'Location', 'southoutside', 'FontSize', 7, 'NumColumns', 2);

% right: the rotation matrix error at every cut, both schemes
axes('Position', [0.55 0.16 0.41 0.72]);
hold on;
leg = {};
style = {'-s', '--d'};
for ci = 1:length(results)
  for iv = 1:2
    plot(hvals, results(ci).Rerr(iv,:), style{iv}, 'Color', cmap(ci,:), ...
         'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerFaceColor', cmap(ci,:));
    leg{end+1} = sprintf('%s, y = %g (\\kappa = %.2f)', ...
                         schemelabels{iv}, results(ci).ycut, ...
                         results(ci).kappa);
  end
end
base = 3*max(arrayfun(@(r) max(r.Rerr(:,1)), results));
plot(hvals, base*(hvals/hvals(1)), 'k:', 'LineWidth', 1.2);
leg{end+1} = 'O(dx)';
plot(hvals, base*(hvals/hvals(1)).^2, 'k--', 'LineWidth', 1.2);
leg{end+1} = 'O(dx^2)';
set(gca, 'XScale', 'log', 'YScale', 'log');
xticks(10.^(-6:1:-1));
grid on; box on;
xlabel('dx'); ylabel('||R - R_{exact}||_2');
title({'the glue rotation at every cut height', ...
       'solid d_k, dashed d_{k2}'}, 'FontSize', 9);
legend(leg, 'Location', 'southwest', 'FontSize', 6, 'NumColumns', 2);

outfile = fullfile(figdir, sprintf('%s_tangent_schemes_summary.png', geoname));
exportgraphics(gcf, outfile, 'Resolution', 150);
fprintf('saved %s\n\n', outfile);


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
  % digit -- and differently for different cut heights, making even the
  % uncut baseline depend on where the cut was.
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
%   and keep every node within rad cells of one of those.  A node within
%   bw*dx of the curve is within (bw + 3/4)*dx of the node snapped from
%   the sample nearest its closest point, so rad = ceil(bw) + 3 keeps all
%   of them (and some extras, which the |dist| <= bw*dx test discards).

  nx = length(x1d);
  ny = length(y1d);

  % a logical mask over the box is one byte per node, cheap even when a
  % meshgrid of doubles would not be
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
%   Same construction as example_two_bands.m, except that the columns are
%   mapped into band indices as they are built rather than by forming a
%   matrix with length(x1d)*length(y1d) columns.  geo says which curve
%   this is and how it is parametrized -- one arc for a half, both arcs
%   for the uncut baseline.

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
  % touches.  If the initial band were too thin the column restriction
  % inside laplacian_2d_matrix would silently drop coefficients, so count.
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
%
%   Both point out of the arc.  Each vector is normalized, the unit
%   vectors are averaged, and the average is normalized again.

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


function tau = unit_tangent(t, sgn, a, b)
%UNIT_TANGENT  sgn * gamma'(t)/|gamma'(t)| on the ellipse

  g = [-a*sin(t) b*cos(t)];
  tau = sgn*g/norm(g);
end


function hh = drawband(brs, V, a, b, cen, yc, tv, dx_show, bw, msize, geo)
%DRAWBAND  the curve, the two half bands, the cut and the rotated rows
%   The rows a half has to fetch from the other half are the ones on the
%   far side of the NORMAL line at the cut point, which coincides with
%   the cut line only for the cut through the centre of the ellipse.
%
%   The curve is drawn arc by arc from geo, so it comes out flipped when
%   the geometry is.  Each arc gets its own normal at each cut point:
%   they are the same line on the ellipse and two different lines at a
%   corner, which is why the rotated rows of the two halves are disjoint
%   in the one case and overlap in the other.

  hold on;
  hh(1) = plot(brs{1}.xout, brs{1}.yout, '.', 'Color', [0.55 0.55 0.55], ...
               'MarkerSize', msize);
  hh(2) = plot(brs{2}.xout, brs{2}.yout, '.', 'Color', [0.30 0.65 0.90], ...
               'MarkerSize', msize);
  for ip = 1:2
    piece = geo{ip}.pieces;
    th = linspace(piece(3), piece(4), 1500)';
    xy = arcpoint(piece, th, a, b, cen);
    hh(3) = plot(xy(:,1), xy(:,2), 'k-', 'LineWidth', 1.2);
  end
  hh(4) = plot(cen(1) + [-1.2*a 1.2*a], yc*[1 1], '--', ...
               'Color', [0.4 0.4 0.4], 'LineWidth', 0.8);
  for j = 1:2
    for ip = 1:2
      nrm = [b*cos(tv(j)) geo{ip}.yflip*a*sin(tv(j))];
      nrm = nrm/norm(nrm);
      seg = 3*bw*dx_show*[-1; 1]*nrm + V(j,:);
      hh(5) = plot(seg(:,1), seg(:,2), '-.', 'Color', [0.5 0.2 0.6], ...
                   'LineWidth', 0.9);
    end
  end
  mk = {'ro', 'ms'};
  for j = 1:2
    rows = find(vertex_ids(brs{j}, V) ~= 0);
    hh(5+j) = plot(brs{j}.xout(rows), brs{j}.yout(rows), mk{j}, ...
                   'MarkerSize', 0.6*msize, 'LineWidth', 1);
  end
  hh(8) = plot(V(:,1), V(:,2), 'kp', 'MarkerFaceColor', 'y', ...
               'MarkerSize', 1.3*msize);
end


function plotconv(hvals, Rerr, rate, tanerr, schemelabels)
%PLOTCONV  the rotation error against dx, with slope guides
%   Solid, the error in the rotation matrix the operator applies; dashed,
%   the error in the tangent it was built from, in the same colour.  The
%   two run parallel -- the angle is a smooth function of the tangent
%   near the exact one -- so the dashed curves are there to show that the
%   rotation error is inherited from the tangent estimate and not
%   introduced anywhere after it.

  hold on;
  markers = {'s', 'd'};
  colors = [0.85 0.33 0.10; 0.00 0.45 0.74];
  for iv = 1:size(Rerr,1)
    plot(hvals, Rerr(iv,:), ['-' markers{iv}], 'Color', colors(iv,:), ...
         'LineWidth', 1.5, 'MarkerSize', 6, ...
         'MarkerFaceColor', colors(iv,:), ...
         'DisplayName', sprintf('||dR||, %s, rate %.2f', ...
                                schemelabels{iv}, rate(iv)));
  end
  for iv = 1:size(tanerr,1)
    plot(hvals, tanerr(iv,:), ['--' markers{iv}], 'Color', colors(iv,:), ...
         'LineWidth', 1.2, 'MarkerSize', 5, ...
         'DisplayName', sprintf('tangent, %s', schemelabels{iv}));
  end
  base = 3*max([Rerr(:,1); tanerr(:,1)]);
  plot(hvals, base*(hvals/hvals(1)), 'k:', 'LineWidth', 1.2, ...
       'DisplayName', 'O(dx)');
  plot(hvals, base*(hvals/hvals(1)).^2, 'k--', 'LineWidth', 1.2, ...
       'DisplayName', 'O(dx^2)');
  set(gca, 'XScale', 'log', 'YScale', 'log');
  xticks(10.^(-6:1:-1));
  grid on; box on;
  xlabel('dx'); ylabel('error in the glue rotation');
end


function th = wrapangle(th)
%WRAPANGLE  put an angle into (-pi, pi]

  th = angle(exp(1i*th));
end


function [r, nuse] = fitrate(dx, e)
%FITRATE  slope of log(e) against log(dx), over the levels before the floor
%   The level where e bottoms out is the one where round-off has caught up
%   with the truncation error, so it and everything past it say nothing
%   about the order and are dropped.  If e is still falling at the finest
%   level, every level is used.

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
