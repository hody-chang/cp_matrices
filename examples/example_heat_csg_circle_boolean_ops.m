%% Heat equation on CSG manifolds from two circles using MMS
% This example solves
%
%   u_t = Delta_s u + f
%
% on three manifolds built from the same pair of overlapping circles:
%   1) union
%   2) intersection
%   3) difference
%
% The figures follow the same layout as example_heat_csg_two_circles:
%   Figure 1: solution in the embedding x-y plane
%   Figure 2: solution along the curve parameter theta
%   Figure 3: error along the curve parameter theta
%
% The exact solution is manufactured arc-by-arc from the local arclength
% coordinate, using a low-curvature polynomial profile with zero slope at
% the arc junctions.


%% Build the circle primitives and the CSG trees
radius = 1;
shift = 0.75;
cen1 = [-shift 0];
cen2 = [ shift 0];

left = csgLeaf(@(x, y) cpCircle(x, y, radius, cen1), 2);
right = csgLeaf(@(x, y) cpCircle(x, y, radius, cen2), 2);

trees = {csgUnion(left, right), ...
         csgIntersection(left, right), ...
         csgDifference(left, right)};
op_names = {'union', 'intersection', 'difference'};


%% Construct the embedding grid
dx = 0.004;
x1d = (-2.2:dx:2.2)';
y1d = (-1.8:dx:1.8)';
[xx, yy] = meshgrid(x1d, y1d);

dim = 2;
p = 3;
order = 2;
bw = rm_bandwidth(dim, p, order/2);


%% Time-stepping parameters
Tf = 0.2;
dt = 0.2*dx^2;
numtimesteps = ceil(Tf/dt);
dt = Tf / numtimesteps;


%% Plot setup
figure(1); clf;
set(gcf, 'color', 'w');

figure(2); clf;
set(gcf, 'color', 'w');

figure(3); clf;
set(gcf, 'color', 'w');


%% Solve the heat equation on each boolean manifold
nplot = 400;

for j = 1:length(trees)
  tree = trees{j};
  op_name = op_names{j};

  [cpx, cpy, dist] = cpCSG(xx, yy, tree);
  band = find(dist <= bw*dx);

  cpxb = cpx(band);
  cpyb = cpy(band);
  xb = xx(band);
  yb = yy(band);

  [arc_length1, arc_length2, total_length] = boolean_curve_lengths(op_name, radius, shift);
  s_cp = boolean_curve_arclength(op_name, cpxb, cpyb, radius, shift, cen1, cen2);

  u = manufactured_solution(s_cp, 0, arc_length1, arc_length2);

  E = interp2_matrix(x1d, y1d, cpxb, cpyb, p, band);
  L = laplacian_2d_matrix(x1d, y1d, order, band);

  for kt = 1:numtimesteps
    t = (kt - 1)*dt;
    forcing = manufactured_forcing(s_cp, t, arc_length1, arc_length2);
    unew = u + dt*(L*u + forcing);
    u = E*unew;
  end

  t = Tf;

  [s_plot, xp, yp] = boolean_curve_parameterization(op_name, radius, shift, cen1, cen2, nplot);
  theta_plot = 2*pi*s_plot / total_length;
  Eplot = interp2_matrix(x1d, y1d, xp, yp, p, band);
  uplot = Eplot*u;
  exactplot = manufactured_solution(s_plot, t, arc_length1, arc_length2);
  errplot = uplot - exactplot;
  max_err = max(abs(errplot));

  set(0, 'CurrentFigure', 1);
  subplot(length(trees), 1, j);
  plot_band_on_current_axes(u, xb, yb, dx, dx);
  hold on;
  plot(xp, yp, 'k-', 'linewidth', 2);
  hold off;
  axis([-2.2 2.2 -1.8 1.8]);
  xlabel('x');
  ylabel('y');
  title(sprintf('%s: embedded solution at time %0.2f', op_name, t));
  colorbar;

  set(0, 'CurrentFigure', 2);
  subplot(length(trees), 1, j);
  plot(theta_plot, uplot, 'b-');
  hold on;
  plot(theta_plot, exactplot, 'r--');
  hold off;
  xlabel('\theta');
  ylabel('u');
  title(sprintf('%s: solution on curve at time %0.2f', op_name, t));
  legend('CPM', 'exact', 'Location', 'SouthEast');

  set(0, 'CurrentFigure', 3);
  subplot(length(trees), 1, j);
  plot(theta_plot, errplot);
  xlabel('\theta');
  ylabel('error');
  title(sprintf('%s: error on curve at time %0.2f', op_name, t));

  fprintf('%s: %d band points, %d timesteps, max_err=%g\n', ...
          op_name, length(band), numtimesteps, max_err);
end


function u = manufactured_solution(s, t, arc_length1, arc_length2)
  [eta, arc_length] = local_arc_coordinate(s, arc_length1, arc_length2);
  xi = eta ./ arc_length;
  profile = 1 - 4*xi.^2 .* (1 - xi).^2;
  u = exp(-t) .* profile;
end


function f = manufactured_forcing(s, t, arc_length1, arc_length2)
  [eta, arc_length] = local_arc_coordinate(s, arc_length1, arc_length2);
  xi = eta ./ arc_length;
  profile = 1 - 4*xi.^2 .* (1 - xi).^2;
  profile_xixi = -8 + 48*xi - 48*xi.^2;
  f = exp(-t) .* (-profile - profile_xixi ./ (arc_length.^2));
end


function [s_plot, xp, yp] = boolean_curve_parameterization(op_name, radius, shift, cen1, cen2, nplot)
  alpha = acos(shift / radius);
  nseg = ceil(nplot / 2);

  switch op_name
   case 'union'
    s1 = linspace(0, 2*pi - 2*alpha, nseg)';
    s2 = linspace(2*pi - 2*alpha, 4*pi - 4*alpha, nseg)';

    th1 = s1 + alpha;
    th2 = s2 - (2*pi - 2*alpha) - pi + alpha;

    x1 = cen1(1) + radius*cos(th1);
    y1 = cen1(2) + radius*sin(th1);
    x2 = cen2(1) + radius*cos(th2);
    y2 = cen2(2) + radius*sin(th2);

   case 'intersection'
    s1 = linspace(0, 2*alpha, nseg)';
    s2 = linspace(2*alpha, 4*alpha, nseg)';

    th1 = s1 + (pi - alpha);
    th2 = s2 - 3*alpha;

    x1 = cen2(1) + radius*cos(th1);
    y1 = cen2(2) + radius*sin(th1);
    x2 = cen1(1) + radius*cos(th2);
    y2 = cen1(2) + radius*sin(th2);

   case 'difference'
    s1 = linspace(0, 2*pi - 2*alpha, nseg)';
    s2 = linspace(2*pi - 2*alpha, 2*pi, nseg)';

    th1 = s1 + alpha;
    th2 = pi + alpha - (s2 - (2*pi - 2*alpha));

    x1 = cen1(1) + radius*cos(th1);
    y1 = cen1(2) + radius*sin(th1);
    x2 = cen2(1) + radius*cos(th2);
    y2 = cen2(2) + radius*sin(th2);

   otherwise
    error('example_heat_csg_circle_boolean_ops:UnknownOperation', ...
          'unknown operation ''%s''', op_name);
  end

  s_plot = [s1; s2];
  xp = [x1; x2];
  yp = [y1; y2];
end


function [arc_length1, arc_length2, total_length] = boolean_curve_lengths(op_name, radius, shift)
  alpha = acos(shift / radius);

  switch op_name
   case 'union'
    arc_length1 = 2*pi - 2*alpha;
    arc_length2 = 2*pi - 2*alpha;
    total_length = 4*pi - 4*alpha;

   case 'intersection'
    arc_length1 = 2*alpha;
    arc_length2 = 2*alpha;
    total_length = 4*alpha;

   case 'difference'
    arc_length1 = 2*pi - 2*alpha;
    arc_length2 = 2*alpha;
    total_length = 2*pi;

   otherwise
    error('example_heat_csg_circle_boolean_ops:UnknownOperation', ...
          'unknown operation ''%s''', op_name);
  end
end


function s = boolean_curve_arclength(op_name, x, y, radius, shift, cen1, cen2)
  alpha = acos(shift / radius);
  tol = 1e-7;

  sdist1 = sqrt((x - cen1(1)).^2 + (y - cen1(2)).^2) - radius;
  sdist2 = sqrt((x - cen2(1)).^2 + (y - cen2(2)).^2) - radius;

  th1_pos = mod(atan2(y - cen1(2), x - cen1(1)), 2*pi);
  th1_raw = atan2(y - cen1(2), x - cen1(1));
  th2_raw = atan2(y - cen2(2), x - cen2(1));
  th2_pos = mod(th2_raw, 2*pi);

  switch op_name
   case 'union'
    s = (2*pi - 2*alpha) + (th2_raw - (-pi + alpha));
    use_left = (abs(sdist1) <= tol) & (sdist2 >= -tol);
    s(use_left) = th1_pos(use_left) - alpha;
    total_length = 4*pi - 4*alpha;

   case 'intersection'
    s = 2*alpha + (th1_raw + alpha);
    use_right = (abs(sdist2) <= tol) & (sdist1 <= tol);
    s(use_right) = th2_pos(use_right) - (pi - alpha);
    total_length = 4*alpha;

   case 'difference'
    s = (2*pi - 2*alpha) + (pi + alpha - th2_pos);
    use_left = (abs(sdist1) <= tol) & (sdist2 >= -tol);
    s(use_left) = th1_pos(use_left) - alpha;
    total_length = 2*pi;

   otherwise
    error('example_heat_csg_circle_boolean_ops:UnknownOperation', ...
          'unknown operation ''%s''', op_name);
  end

  s = mod(s, total_length);
end


function [eta, arc_length] = local_arc_coordinate(s, arc_length1, arc_length2)
  eta = s;
  arc_length = arc_length1*ones(size(s));

  on_second_arc = (s > arc_length1);
  eta(on_second_arc) = s(on_second_arc) - arc_length1;
  arc_length(on_second_arc) = arc_length2;
end


function plot_band_on_current_axes(u, x, y, dx, dy)
  cla;

  xpat = dx/2*[-1; 1; 1; -1];
  ypat = dy/2*[1; 1; -1; -1];
  X = repmat(x', 4, 1) + repmat(xpat, 1, length(u));
  Y = repmat(y', 4, 1) + repmat(ypat, 1, length(u));

  H = patch(X, Y, 'g');
  set(H, 'FaceColor', 'flat', 'FaceVertexCData', u);
  axis equal;
  axis tight;
end
