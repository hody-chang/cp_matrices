%% Convergence study for heat equation on CSG manifolds from two circles
% This example solves
%
%   u_t = Delta_s u + f
%
% on the union, intersection, and difference of two overlapping circles,
% and measures the error as the embedding grid is refined.
%
% Run examples/setupPaths before executing this script from a fresh MATLAB
% session, just as with the other examples in this repository.
%
% The exact solution is the same arc-by-arc manufactured solution used in
% example_heat_csg_circle_boolean_ops.  Here we repeat the solve over a
% sequence of dx values and report the observed convergence rates.  The
% short arcs are amplitude-scaled by their arc length so the intersection
% and difference cases are less stiff numerically than the long union arcs.


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


%% Convergence-study configuration
dx_list = [0.08 0.04 0.02 0.01 0.004 0.002];
Tf = 0.05;
nplot = 1200;
reference_arc_length = 2*pi - 2*acos(shift / radius);

dim = 2;
p = 3;
order = 2;
bw = rm_bandwidth(dim, p, order/2);

num_levels = length(dx_list);
num_ops = length(trees);

max_errs = zeros(num_levels, num_ops);
l2_errs = zeros(num_levels, num_ops);
max_rates = nan(num_levels, num_ops);
l2_rates = nan(num_levels, num_ops);
band_counts = zeros(num_levels, num_ops);
timesteps = zeros(num_levels, 1);


%% Run the convergence study
for ilev = 1:num_levels
  dx = dx_list(ilev);

  x1d = (-2.5:dx:2.5)';
  y1d = (-2.1:dx:2.1)';
  [xx, yy] = meshgrid(x1d, y1d);

  dt = 0.2*dx^2;
  numtimesteps = ceil(Tf/dt);
  dt = Tf / numtimesteps;
  timesteps(ilev) = numtimesteps;

  fprintf('dx=%g, timesteps=%d\n', dx, numtimesteps);

  for j = 1:num_ops
    tree = trees{j};
    op_name = op_names{j};

    [cpx, cpy, dist] = cpCSG(xx, yy, tree);
    band = find(dist <= bw*dx);
    band_counts(ilev, j) = length(band);

    cpxb = cpx(band);
    cpyb = cpy(band);

    [arc_length1, arc_length2, total_length] = ...
        boolean_curve_lengths(op_name, radius, shift);
    s_cp = boolean_curve_arclength(op_name, cpxb, cpyb, ...
        radius, shift, cen1, cen2);

    u = manufactured_solution(s_cp, 0, arc_length1, arc_length2, ...
                              reference_arc_length);

    E = interp2_matrix(x1d, y1d, cpxb, cpyb, p, band);
    L = laplacian_2d_matrix(x1d, y1d, order, band);

    for kt = 1:numtimesteps
      t = (kt - 1)*dt;
      forcing = manufactured_forcing(s_cp, t, arc_length1, arc_length2, ...
                                     reference_arc_length);
      unew = u + dt*(L*u + forcing);
      u = E*unew;
    end

    [s_plot, xp, yp] = boolean_curve_parameterization(op_name, radius, ...
        shift, cen1, cen2, nplot);
    Eplot = interp2_matrix(x1d, y1d, xp, yp, p, band);
    uplot = Eplot*u;
    exactplot = manufactured_solution(s_plot, Tf, arc_length1, arc_length2, ...
                                      reference_arc_length);
    errplot = uplot - exactplot;
    max_errs(ilev, j) = max(abs(errplot));
    l2_errs(ilev, j) = sqrt(total_length / length(s_plot)) * norm(errplot, 2);

    fprintf('  %-12s band=%6d max_err=%g l2_err=%g\n', ...
            op_name, band_counts(ilev, j), max_errs(ilev, j), l2_errs(ilev, j));
  end

  fprintf('\n');
end


%% Compute observed rates
for j = 1:num_ops
  for ilev = 2:num_levels
    max_rates(ilev, j) = log(max_errs(ilev-1, j) / max_errs(ilev, j)) / ...
                         log(dx_list(ilev-1) / dx_list(ilev));
    l2_rates(ilev, j) = log(l2_errs(ilev-1, j) / l2_errs(ilev, j)) / ...
                        log(dx_list(ilev-1) / dx_list(ilev));
  end
end


%% Print summary tables
fprintf('Convergence summary (max error on sampled curve)\n');
fprintf('%10s %14s %14s %14s\n', 'dx', op_names{1}, op_names{2}, op_names{3});
for ilev = 1:num_levels
  fprintf('%10.4f %14.6e %14.6e %14.6e\n', ...
          dx_list(ilev), max_errs(ilev, 1), max_errs(ilev, 2), max_errs(ilev, 3));
end
fprintf('\n');

fprintf('Convergence summary (sampled L2 error)\n');
fprintf('%10s %14s %14s %14s\n', 'dx', op_names{1}, op_names{2}, op_names{3});
for ilev = 1:num_levels
  fprintf('%10.4f %14.6e %14.6e %14.6e\n', ...
          dx_list(ilev), l2_errs(ilev, 1), l2_errs(ilev, 2), l2_errs(ilev, 3));
end
fprintf('\n');

fprintf('Observed max-error rates\n');
fprintf('%10s %14s %14s %14s\n', 'dx', op_names{1}, op_names{2}, op_names{3});
for ilev = 2:num_levels
  fprintf('%10.4f %14.6f %14.6f %14.6f\n', ...
          dx_list(ilev), max_rates(ilev, 1), max_rates(ilev, 2), max_rates(ilev, 3));
end
fprintf('\n');

fprintf('Observed L2-error rates\n');
fprintf('%10s %14s %14s %14s\n', 'dx', op_names{1}, op_names{2}, op_names{3});
for ilev = 2:num_levels
  fprintf('%10.4f %14.6f %14.6f %14.6f\n', ...
          dx_list(ilev), l2_rates(ilev, 1), l2_rates(ilev, 2), l2_rates(ilev, 3));
end


%% Plot the convergence results
figure(1); clf;
set(gcf, 'color', 'w');

loglog(dx_list, max_errs(:, 1), 'o-', 'linewidth', 1.5, 'markersize', 8);
hold on;
loglog(dx_list, max_errs(:, 2), 's-', 'linewidth', 1.5, 'markersize', 8);
loglog(dx_list, max_errs(:, 3), 'd-', 'linewidth', 1.5, 'markersize', 8);
loglog(dx_list, max_errs(1, 1)*(dx_list/dx_list(1)).^2, 'k--', 'linewidth', 1);
hold off;
grid on;
xlabel('\Deltax');
ylabel('max error');
title(sprintf('CSG circle heat equation convergence at time %0.2f', Tf));
legend('union', 'intersection', 'difference', 'O(\Deltax^2)', ...
       'Location', 'NorthWest');

figure(2); clf;
set(gcf, 'color', 'w');

loglog(dx_list, l2_errs(:, 1), 'o-', 'linewidth', 1.5, 'markersize', 8);
hold on;
loglog(dx_list, l2_errs(:, 2), 's-', 'linewidth', 1.5, 'markersize', 8);
loglog(dx_list, l2_errs(:, 3), 'd-', 'linewidth', 1.5, 'markersize', 8);
loglog(dx_list, l2_errs(1, 1)*(dx_list/dx_list(1)).^2, 'k--', 'linewidth', 1);
hold off;
grid on;
xlabel('\Deltax');
ylabel('sampled L2 error');
title(sprintf('CSG circle heat equation L2 convergence at time %0.2f', Tf));
legend('union', 'intersection', 'difference', 'O(\Deltax^2)', ...
       'Location', 'NorthWest');


function u = manufactured_solution(s, t, arc_length1, arc_length2, ...
                                   reference_arc_length)
  [eta, arc_length] = local_arc_coordinate(s, arc_length1, arc_length2);
  xi = eta ./ arc_length;
  amp = (arc_length ./ reference_arc_length).^2;
  profile = 1 - amp .* 4 .* xi.^2 .* (1 - xi).^2;
  u = exp(-t) .* profile;
end


function f = manufactured_forcing(s, t, arc_length1, arc_length2, ...
                                  reference_arc_length)
  [eta, arc_length] = local_arc_coordinate(s, arc_length1, arc_length2);
  xi = eta ./ arc_length;
  amp = (arc_length ./ reference_arc_length).^2;
  profile = 1 - amp .* 4 .* xi.^2 .* (1 - xi).^2;
  profile_xixi = amp .* (-8 + 48*xi - 48*xi.^2);
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
    error('example_heat_csg_circle_boolean_ops_convergence:UnknownOperation', ...
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
    error('example_heat_csg_circle_boolean_ops_convergence:UnknownOperation', ...
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
    error('example_heat_csg_circle_boolean_ops_convergence:UnknownOperation', ...
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
