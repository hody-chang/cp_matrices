%% Angle study for heat equation on the union of two circles
% This example investigates how bringing two circles together changes the
% observed convergence behavior for the heat equation solved with the
% closest point method on their CSG union.
%
% We start with two disjoint circles separated by a gap of 1 unit, then
% reduce the signed gap.  Positive gap means disjoint circles, negative gap
% means overlapping circles.  For overlapping cases, the seam angle is
% reported and the convergence rate is plotted against that angle.
%
% For disjoint circles, a level is skipped if the gap is smaller than twice
% the computational band width.  In that regime the two narrow bands begin
% to overlap before the circles touch, so the result is no longer a clean
% study of separated components.
%
% Run examples/setupPaths before executing this script from a fresh MATLAB
% session, just as with the other examples in this repository.
%
% The touching case (gap = 0) is omitted because the union is non-smooth at
% the single tangency point and does not have a unique seam angle there.


%% Study configuration
radius = 1;
gap_list = [1.0 0.5 0.25 0.10 0.02 -0.02 -0.10 -0.25 -0.50];
dx_list = [0.08 0.04 0.02 0.01];
Tf = 0.05;
nplot = 1200;

dim = 2;
p = 3;
order = 2;
bw = rm_bandwidth(dim, p, order/2);
reference_arc_length = 2*pi*radius;

num_gaps = length(gap_list);
num_levels = length(dx_list);

max_errs = zeros(num_levels, num_gaps);
l2_errs = zeros(num_levels, num_gaps);
band_counts = zeros(num_levels, num_gaps);
timesteps = zeros(num_levels, 1);
fit_rate_max = nan(1, num_gaps);
fit_rate_l2 = nan(1, num_gaps);
last_rate_max = nan(1, num_gaps);
last_rate_l2 = nan(1, num_gaps);
seam_angle_deg = nan(1, num_gaps);
shift_list = radius + gap_list/2;


%% Run the study
for igap = 1:num_gaps
  gap = gap_list(igap);
  shift = shift_list(igap);
  cen1 = [-shift 0];
  cen2 = [ shift 0];

  if (shift < radius)
    alpha = acos(shift / radius);
    seam_angle_deg(igap) = 2*alpha*180/pi;
  end

  left = csgLeaf(@(x, y) cpCircle(x, y, radius, cen1), 2);
  right = csgLeaf(@(x, y) cpCircle(x, y, radius, cen2), 2);
  tree = csgUnion(left, right);

  fprintf('gap=%6.3f, center distance=%6.3f', gap, 2*shift);
  if (shift < radius)
    fprintf(', seam angle=%6.2f deg\n', seam_angle_deg(igap));
  else
    fprintf(', disjoint\n');
  end

  for ilev = 1:num_levels
    dx = dx_list(ilev);

    x1d = (-3.2:dx:3.2)';
    y1d = (-2.2:dx:2.2)';
    [xx, yy] = meshgrid(x1d, y1d);

    dt = 0.2*dx^2;
    numtimesteps = ceil(Tf/dt);
    dt = Tf / numtimesteps;
    timesteps(ilev) = numtimesteps;

    if ((gap > 0) && (gap <= 2*bw*dx))
      max_errs(ilev, igap) = NaN;
      l2_errs(ilev, igap) = NaN;
      band_counts(ilev, igap) = 0;
      fprintf('  dx=%5.3f skipped (gap <= 2*bw*dx)\n', dx);
      continue;
    end

    [cpx, cpy, dist] = cpCSG(xx, yy, tree);
    band = find(dist <= bw*dx);
    band_counts(ilev, igap) = length(band);

    cpxb = cpx(band);
    cpyb = cpy(band);

    [arc_length1, arc_length2, total_length] = union_curve_lengths(radius, shift);
    s_cp = union_curve_arclength(cpxb, cpyb, radius, shift, cen1, cen2);

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

    [s_plot, xp, yp] = union_curve_parameterization(radius, shift, ...
                                                    cen1, cen2, nplot);
    Eplot = interp2_matrix(x1d, y1d, xp, yp, p, band);
    uplot = Eplot*u;
    exactplot = manufactured_solution(s_plot, Tf, arc_length1, arc_length2, ...
                                      reference_arc_length);
    errplot = uplot - exactplot;

    max_errs(ilev, igap) = max(abs(errplot));
    l2_errs(ilev, igap) = sqrt(total_length / length(s_plot)) * norm(errplot, 2);

    fprintf('  dx=%5.3f band=%6d max_err=%9.3e l2_err=%9.3e\n', ...
            dx, band_counts(ilev, igap), max_errs(ilev, igap), ...
            l2_errs(ilev, igap));
  end

  valid = ~isnan(l2_errs(:, igap));
  if (nnz(valid) >= 2)
    coeff = polyfit(log(dx_list(valid)), log(max_errs(valid, igap))', 1);
    fit_rate_max(igap) = coeff(1);
    coeff = polyfit(log(dx_list(valid)), log(l2_errs(valid, igap))', 1);
    fit_rate_l2(igap) = coeff(1);

    valid_idx = find(valid);
    i1 = valid_idx(end-1);
    i2 = valid_idx(end);
    last_rate_max(igap) = log(max_errs(i1, igap) / max_errs(i2, igap)) / ...
                          log(dx_list(i1) / dx_list(i2));
    last_rate_l2(igap) = log(l2_errs(i1, igap) / l2_errs(i2, igap)) / ...
                         log(dx_list(i1) / dx_list(i2));
  end

  fprintf('  fitted rates: max=%6.3f, l2=%6.3f\n', ...
          fit_rate_max(igap), fit_rate_l2(igap));
  fprintf('  last-step rates: max=%6.3f, l2=%6.3f\n\n', ...
          last_rate_max(igap), last_rate_l2(igap));
end


%% Print summary table
fprintf('Summary by signed gap\n');
fprintf(['%10s %12s %12s %14s %14s %14s %14s\n'], ...
        'gap', 'shift', 'angle(deg)', 'max err', 'l2 err', ...
        'fit rate', 'last l2');
for igap = 1:num_gaps
  fprintf('%10.3f %12.3f %12.3f %14.6e %14.6e %14.6f %14.6f\n', ...
          gap_list(igap), shift_list(igap), seam_angle_deg(igap), ...
          max_errs(end, igap), l2_errs(end, igap), ...
          fit_rate_l2(igap), last_rate_l2(igap));
end


%% Plot convergence curves for each separation
figure(1); clf;
set(gcf, 'color', 'w');
hold on;
plot_handles = [];
plot_labels = {};
for igap = 1:num_gaps
  valid = ~isnan(l2_errs(:, igap));
  if any(valid)
    h = loglog(dx_list(valid), l2_errs(valid, igap), 'o-', ...
               'linewidth', 1.2, 'markersize', 7);
    plot_handles(end+1) = h; %#ok<AGROW>
    plot_labels{end+1} = sprintf('gap = %0.2f', gap_list(igap)); %#ok<AGROW>
  end
end
href = loglog(dx_list, l2_errs(1, 1)*(dx_list/dx_list(1)).^2, ...
              'k--', 'linewidth', 1);
hold off;
grid on;
xlabel('\Deltax');
ylabel('sampled L2 error');
title(sprintf('Union of two circles: L2 convergence at time %0.2f', Tf));
legend([plot_handles href], [plot_labels {'O(\Deltax^2)'}], ...
       'Location', 'NorthWest');


%% Plot fitted rates against signed gap
figure(2); clf;
set(gcf, 'color', 'w');
plot(gap_list, fit_rate_max, 'o-', 'linewidth', 1.5, 'markersize', 8);
hold on;
plot(gap_list, fit_rate_l2, 's-', 'linewidth', 1.5, 'markersize', 8);
plot(gap_list, 2*ones(size(gap_list)), 'k--', 'linewidth', 1);
hold off;
grid on;
xlabel('signed gap between circles');
ylabel('fitted convergence rate');
title('Observed rate versus signed gap');
legend('max norm', 'L2 norm', 'second order', 'Location', 'SouthWest');


%% Plot fitted rates against seam angle for overlapping cases
overlap = ~isnan(seam_angle_deg);

figure(3); clf;
set(gcf, 'color', 'w');
plot(seam_angle_deg(overlap), fit_rate_max(overlap), 'o-', ...
     'linewidth', 1.5, 'markersize', 8);
hold on;
plot(seam_angle_deg(overlap), fit_rate_l2(overlap), 's-', ...
     'linewidth', 1.5, 'markersize', 8);
plot(seam_angle_deg(overlap), 2*ones(size(seam_angle_deg(overlap))), ...
     'k--', 'linewidth', 1);
hold off;
grid on;
xlabel('seam angle (degrees)');
ylabel('fitted convergence rate');
title('Observed rate versus seam angle for overlapping unions');
legend('max norm', 'L2 norm', 'second order', 'Location', 'SouthEast');


function u = manufactured_solution(s, t, arc_length1, arc_length2, reference_arc_length)
  [eta, arc_length] = local_arc_coordinate(s, arc_length1, arc_length2);
  xi = eta ./ arc_length;
  amp = (arc_length ./ reference_arc_length).^2;
  profile = 1 - amp .* 4 .* xi.^2 .* (1 - xi).^2;
  u = exp(-t) .* profile;
end


function f = manufactured_forcing(s, t, arc_length1, arc_length2, reference_arc_length)
  [eta, arc_length] = local_arc_coordinate(s, arc_length1, arc_length2);
  xi = eta ./ arc_length;
  amp = (arc_length ./ reference_arc_length).^2;
  profile = 1 - amp .* 4 .* xi.^2 .* (1 - xi).^2;
  profile_xixi = amp .* (-8 + 48*xi - 48*xi.^2);
  f = exp(-t) .* (-profile - profile_xixi ./ (arc_length.^2));
end


function [s_plot, xp, yp] = union_curve_parameterization(radius, shift, cen1, cen2, nplot)
  nseg = ceil(nplot / 2);

  if (shift >= radius)
    len = 2*pi*radius;

    s1 = linspace(0, len, nseg)';
    s2 = linspace(len, 2*len, nseg)';

    th1 = s1 / radius;
    th2 = (s2 - len) / radius;

    x1 = cen1(1) + radius*cos(th1);
    y1 = cen1(2) + radius*sin(th1);
    x2 = cen2(1) + radius*cos(th2);
    y2 = cen2(2) + radius*sin(th2);
  else
    alpha = acos(shift / radius);
    len = radius*(2*pi - 2*alpha);

    s1 = linspace(0, len, nseg)';
    s2 = linspace(len, 2*len, nseg)';

    th1 = s1 / radius + alpha;
    th2 = (s2 - len) / radius - pi + alpha;

    x1 = cen1(1) + radius*cos(th1);
    y1 = cen1(2) + radius*sin(th1);
    x2 = cen2(1) + radius*cos(th2);
    y2 = cen2(2) + radius*sin(th2);
  end

  s_plot = [s1; s2];
  xp = [x1; x2];
  yp = [y1; y2];
end


function [arc_length1, arc_length2, total_length] = union_curve_lengths(radius, shift)
  if (shift >= radius)
    arc_length1 = 2*pi*radius;
    arc_length2 = 2*pi*radius;
  else
    alpha = acos(shift / radius);
    arc_length1 = radius*(2*pi - 2*alpha);
    arc_length2 = arc_length1;
  end

  total_length = arc_length1 + arc_length2;
end


function s = union_curve_arclength(x, y, radius, shift, cen1, cen2)
  tol = 1e-7;

  sdist1 = sqrt((x - cen1(1)).^2 + (y - cen1(2)).^2) - radius;
  sdist2 = sqrt((x - cen2(1)).^2 + (y - cen2(2)).^2) - radius;

  th1_pos = mod(atan2(y - cen1(2), x - cen1(1)), 2*pi);
  th2_pos = mod(atan2(y - cen2(2), x - cen2(1)), 2*pi);
  th2_raw = atan2(y - cen2(2), x - cen2(1));

  if (shift >= radius)
    len = 2*pi*radius;
    use_left = (abs(sdist1) <= tol) & ((sdist1 <= sdist2 + tol) | (sdist2 > tol));
    s = len + radius*th2_pos;
    s(use_left) = radius*th1_pos(use_left);
  else
    alpha = acos(shift / radius);
    len = radius*(2*pi - 2*alpha);
    s = len + radius*(th2_raw - (-pi + alpha));
    use_left = (abs(sdist1) <= tol) & (sdist2 >= -tol);
    s(use_left) = radius*(th1_pos(use_left) - alpha);
    s = mod(s, 2*len);
  end
end


function [eta, arc_length] = local_arc_coordinate(s, arc_length1, arc_length2)
  eta = s;
  arc_length = arc_length1*ones(size(s));

  on_second_arc = (s > arc_length1);
  eta(on_second_arc) = s(on_second_arc) - arc_length1;
  arc_length(on_second_arc) = arc_length2;
end
