%% Heat equation on CSG manifolds from two spheres using MMS
% This example solves
%
%   u_t = Delta_s u + f
%
% Run examples/setupPaths before executing this script from a fresh MATLAB
% session, just as with the other examples in this repository.
%
% on three manifolds built from the same pair of overlapping spheres:
%   1) union
%   2) intersection
%   3) difference
%
% The manufactured solution is axisymmetric on each spherical patch, with
% zero meridional slope at every pole and seam circle.  This keeps the
% solution continuous across the CSG seam while still providing an exact
% solution and forcing for error plots.  The amplitude is scaled by the
% patch's angular span so the small intersection caps are less stiff
% numerically than the longer union patches.
%
% The figures use the same overall pattern as
% example_heat_csg_circle_boolean_ops:
%   Figure 1: solution on the embedded 3D surface
%   Figure 2: solution along a representative meridian
%   Figure 3: error along the representative meridian


%% Build the sphere primitives and the CSG trees
radius = 1;
shift = 0.75;
alpha = acos(shift / radius);
seam_radius = sqrt(radius^2 - shift^2);
reference_theta_span = pi - alpha;

cen1 = [-shift 0 0];
cen2 = [ shift 0 0];

left = csgLeaf(@(x, y, z) cpSphere(x, y, z, radius, cen1), 3);
right = csgLeaf(@(x, y, z) cpSphere(x, y, z, radius, cen2), 3);

trees = {csgUnion(left, right), ...
         csgIntersection(left, right), ...
         csgDifference(left, right)};
op_names = {'union', 'intersection', 'difference'};


%% Construct the embedding grid
dx = 0.05;
x1d = (-2.6:dx:2.6)';
y1d = (-2.0:dx:2.0)';
z1d = y1d;
[xx, yy, zz] = meshgrid(x1d, y1d, z1d);

dim = 3;
p = 3;
order = 2;
bw = rm_bandwidth(dim, p, order/2);


%% Time-stepping parameters
Tf = 0.1;
dt = 0.1*dx^2;
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
ntheta = 48;
nphi = 64;
nmeridian = 300;
numops = numel(trees);

for j = 1:numops
  tree = trees{j};
  op_name = op_names{j};
  patches = boolean_surface_patches(op_name, alpha, cen1, cen2);

  [cpx, cpy, cpz, dist] = cpCSG(xx, yy, zz, tree);
  band = find(dist <= bw*dx);

  cpxb = cpx(band);
  cpyb = cpy(band);
  cpzb = cpz(band);

  [theta_cp, theta_start_cp, theta_end_cp] = classify_surface_points( ...
      cpxb, cpyb, cpzb, radius, cen1, cen2, patches);

  u = manufactured_solution(theta_cp, theta_start_cp, theta_end_cp, ...
                            reference_theta_span, 0);

  E = interp3_matrix(x1d, y1d, z1d, cpxb, cpyb, cpzb, p, band);
  L = laplacian_3d_matrix(x1d, y1d, z1d, order, band, band);

  for kt = 1:numtimesteps
    t = (kt - 1)*dt;
    forcing = manufactured_forcing(theta_cp, theta_start_cp, theta_end_cp, ...
                                   reference_theta_span, radius, t);
    unew = u + dt*(L*u + forcing);
    u = E*unew;
  end

  t = Tf;

  surface_samples = sample_surface_patches(patches, radius, ntheta, nphi);
  [xp, yp, zp, theta_plot, theta_start_plot, theta_end_plot, counts] = ...
      flatten_surface_samples(surface_samples);

  Eplot = interp3_matrix(x1d, y1d, z1d, xp, yp, zp, p, band);
  uplot = Eplot*u;
  exactplot = manufactured_solution(theta_plot, theta_start_plot, ...
                                    theta_end_plot, reference_theta_span, t);
  surface_err = uplot - exactplot;

  [s_meridian, xm, ym, zm, theta_meridian, theta_start_meridian, ...
   theta_end_meridian] = boolean_meridian_parameterization(op_name, ...
      radius, alpha, cen1, cen2, nmeridian);

  Emeridian = interp3_matrix(x1d, y1d, z1d, xm, ym, zm, p, band);
  umeridian = Emeridian*u;
  exact_meridian = manufactured_solution(theta_meridian, ...
      theta_start_meridian, theta_end_meridian, reference_theta_span, t);
  err_meridian = umeridian - exact_meridian;
  max_err = max(abs(surface_err));

  set(0, 'CurrentFigure', 1);
  subplot(1, numops, j);
  cla;
  hold on;

  offset = 0;
  for k = 1:numel(surface_samples)
    idx = offset + (1:counts(k));
    uk = reshape(uplot(idx), size(surface_samples(k).X));
    surf(surface_samples(k).X, surface_samples(k).Y, ...
         surface_samples(k).Z, uk, 'EdgeColor', 'none');
    offset = offset + counts(k);
  end

  phi_seam = linspace(0, 2*pi, 200);
  plot3(zeros(size(phi_seam)), ...
        seam_radius*cos(phi_seam), ...
        seam_radius*sin(phi_seam), 'k-', 'LineWidth', 1.0);

  hold off;
  axis equal;
  axis([-2.2 2.2 -1.6 1.6 -1.6 1.6]);
  view([-35 24]);
  shading interp;
  xlabel('x');
  ylabel('y');
  zlabel('z');
  title(sprintf('%s: surface solution at time %0.2f', op_name, t));
  colorbar;

  set(0, 'CurrentFigure', 2);
  subplot(numops, 1, j);
  plot(s_meridian, umeridian, 'b-');
  hold on;
  plot(s_meridian, exact_meridian, 'r--');
  hold off;
  xlabel('s');
  ylabel('u');
  title(sprintf('%s: solution along a meridian at time %0.2f', ...
        op_name, t));
  legend('CPM', 'exact', 'Location', 'SouthEast');

  set(0, 'CurrentFigure', 3);
  subplot(numops, 1, j);
  plot(s_meridian, err_meridian);
  xlabel('s');
  ylabel('error');
  title(sprintf('%s: meridian error at time %0.2f', op_name, t));

  fprintf('%s: %d band points, %d timesteps, max_err=%g\n', ...
          op_name, numel(band), numtimesteps, max_err);
end


function patches = boolean_surface_patches(op_name, alpha, cen1, cen2)
  patches(1).center = cen1;
  patches(1).axis_sign = 1;
  patches(2).center = cen2;
  patches(2).axis_sign = -1;

  switch op_name
   case 'union'
    patches(1).theta_start = alpha;
    patches(1).theta_end = pi;
    patches(2).theta_start = alpha;
    patches(2).theta_end = pi;

   case 'intersection'
    patches(1).theta_start = 0;
    patches(1).theta_end = alpha;
    patches(2).theta_start = 0;
    patches(2).theta_end = alpha;

   case 'difference'
    patches(1).theta_start = alpha;
    patches(1).theta_end = pi;
    patches(2).theta_start = 0;
    patches(2).theta_end = alpha;

   otherwise
    error('example_heat_csg_sphere_boolean_ops:UnknownOperation', ...
          'unknown operation ''%s''', op_name);
  end
end


function [theta, theta_start, theta_end] = classify_surface_points( ...
    x, y, z, radius, cen1, cen2, patches)
  tol = 1e-10;

  sdist1 = sqrt((x - cen1(1)).^2 + (y - cen1(2)).^2 + (z - cen1(3)).^2) ...
           - radius;
  sdist2 = sqrt((x - cen2(1)).^2 + (y - cen2(2)).^2 + (z - cen2(3)).^2) ...
           - radius;

  use_left = abs(sdist1) <= (abs(sdist2) + tol);

  theta_left = safe_acos((x - cen1(1)) ./ radius);
  theta_right = safe_acos(-(x - cen2(1)) ./ radius);

  theta = theta_right;
  theta(use_left) = theta_left(use_left);

  theta_start = patches(2).theta_start*ones(size(theta));
  theta_end = patches(2).theta_end*ones(size(theta));
  theta_start(use_left) = patches(1).theta_start;
  theta_end(use_left) = patches(1).theta_end;
end


function u = manufactured_solution(theta, theta_start, theta_end, ...
                                   reference_theta_span, t)
  theta_span = theta_end - theta_start;
  xi = (theta - theta_start) ./ theta_span;
  amp = (theta_span ./ reference_theta_span).^2;

  profile = 1 - amp .* 4 .* xi.^2 .* (1 - xi).^2;
  u = exp(-t) .* profile;
end


function f = manufactured_forcing(theta, theta_start, theta_end, ...
                                  reference_theta_span, radius, t)
  theta_span = theta_end - theta_start;
  xi = (theta - theta_start) ./ theta_span;
  amp = (theta_span ./ reference_theta_span).^2;

  profile = 1 - amp .* 4 .* xi.^2 .* (1 - xi).^2;
  profile_xi = amp .* (-8*xi + 24*xi.^2 - 16*xi.^3);
  profile_xixi = amp .* (-8 + 48*xi - 48*xi.^2);

  ut = -exp(-t) .* profile;
  u_theta = exp(-t) .* profile_xi ./ theta_span;
  u_thetatheta = exp(-t) .* profile_xixi ./ (theta_span.^2);

  lap_theta = u_thetatheta + cos(theta) ./ sin(theta) .* u_theta;
  near_pole = abs(sin(theta)) < 1e-10;
  lap_theta(near_pole) = 2*u_thetatheta(near_pole);

  f = ut - lap_theta ./ (radius^2);
end


function samples = sample_surface_patches(patches, radius, ntheta, nphi)
  phi = linspace(0, 2*pi, nphi);

  for k = 1:numel(patches)
    theta = linspace(patches(k).theta_start, patches(k).theta_end, ntheta);
    [Theta, Phi] = meshgrid(theta, phi);

    [X, Y, Z] = axisymmetric_patch_coords(patches(k).center, ...
        patches(k).axis_sign, radius, Theta, Phi);

    samples(k).X = X;
    samples(k).Y = Y;
    samples(k).Z = Z;
    samples(k).theta = Theta;
    samples(k).theta_start = patches(k).theta_start;
    samples(k).theta_end = patches(k).theta_end;
  end
end


function [xp, yp, zp, theta_plot, theta_start_plot, theta_end_plot, counts] = ...
    flatten_surface_samples(samples)
  xp = [];
  yp = [];
  zp = [];
  theta_plot = [];
  theta_start_plot = [];
  theta_end_plot = [];
  counts = zeros(1, numel(samples));

  for k = 1:numel(samples)
    counts(k) = numel(samples(k).X);
    xp = [xp; samples(k).X(:)]; %#ok<AGROW>
    yp = [yp; samples(k).Y(:)]; %#ok<AGROW>
    zp = [zp; samples(k).Z(:)]; %#ok<AGROW>
    theta_plot = [theta_plot; samples(k).theta(:)]; %#ok<AGROW>
    theta_start_plot = [theta_start_plot; ...
        samples(k).theta_start*ones(counts(k), 1)]; %#ok<AGROW>
    theta_end_plot = [theta_end_plot; ...
        samples(k).theta_end*ones(counts(k), 1)]; %#ok<AGROW>
  end
end


function [s_plot, xp, yp, zp, theta_plot, theta_start_plot, theta_end_plot] = ...
    boolean_meridian_parameterization(op_name, radius, alpha, cen1, cen2, nplot)
  patches = boolean_surface_patches(op_name, alpha, cen1, cen2);
  nseg = ceil(nplot / 2);

  switch op_name
   case 'union'
    patch1 = patches(1);
    patch2 = patches(2);
    theta1 = linspace(patch1.theta_end, patch1.theta_start, nseg)';
    theta2 = linspace(patch2.theta_start, patch2.theta_end, nseg)';

   case 'intersection'
    patch1 = patches(2);
    patch2 = patches(1);
    theta1 = linspace(patch1.theta_start, patch1.theta_end, nseg)';
    theta2 = linspace(patch2.theta_end, patch2.theta_start, nseg)';

   case 'difference'
    patch1 = patches(1);
    patch2 = patches(2);
    theta1 = linspace(patch1.theta_end, patch1.theta_start, nseg)';
    theta2 = linspace(patch2.theta_end, patch2.theta_start, nseg)';

   otherwise
    error('example_heat_csg_sphere_boolean_ops:UnknownOperation', ...
          'unknown operation ''%s''', op_name);
  end

  s1 = radius*abs(theta1 - theta1(1));
  s2 = s1(end) + radius*abs(theta2 - theta2(1));

  theta2 = theta2(2:end);
  s2 = s2(2:end);

  phi_meridian = (pi/2)*ones(size(theta1));
  [x1, y1, z1] = axisymmetric_patch_coords(patch1.center, patch1.axis_sign, ...
      radius, theta1, phi_meridian);

  phi_meridian = (pi/2)*ones(size(theta2));
  [x2, y2, z2] = axisymmetric_patch_coords(patch2.center, patch2.axis_sign, ...
      radius, theta2, phi_meridian);

  xp = [x1; x2];
  yp = [y1; y2];
  zp = [z1; z2];
  s_plot = [s1; s2];

  theta_plot = [theta1; theta2];
  theta_start_plot = [patch1.theta_start*ones(size(theta1)); ...
                      patch2.theta_start*ones(size(theta2))];
  theta_end_plot = [patch1.theta_end*ones(size(theta1)); ...
                    patch2.theta_end*ones(size(theta2))];
end


function [x, y, z] = axisymmetric_patch_coords(center, axis_sign, ...
                                               radius, theta, phi)
  rho = radius*sin(theta);
  x = center(1) + axis_sign*radius*cos(theta);
  y = center(2) + rho .* cos(phi);
  z = center(3) + rho .* sin(phi);
end


function th = safe_acos(v)
  th = acos(max(-1, min(1, v)));
end
