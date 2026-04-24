%% Ellipse with a hole using a CSG-style trimmed curve
% This mirrors example_hole_ellipse.m, but constructs the open curve using
% a trimmed-curve closest point map driven by the same seam projection idea
% used in the CSG closest point algorithm.


%% Parameters
a = 1.6;   % semi-axis along x
b = 0.5;   % semi-axis along y
curve_leaf = csgLeaf(@(x, y) cpEllipse(x, y, a, b), 2);
paramf = @(N) paramEllipse(N, a, b);

dx = 0.1;
x1d = (-2.0:dx:2.0)';
y1d = x1d;


%% Build the hole geometry
theta_hole = pi/2;
hole_cen = [a*cos(theta_hole) b*sin(theta_hole)];
len_wanted = 2;
hole_rad = estimate_hole_radius(paramf, hole_cen, len_wanted);
trim_leaf = csgLeaf(@(x, y) cpCircle(x, y, hole_rad, hole_cen), 2);


%% Banding on the original ellipse
dim = 2;
p = 3;
order = 2;
bw = rm_bandwidth(dim, p, order/2);

[xx, yy] = meshgrid(x1d, y1d);
[cpx0, cpy0, dist0] = curve_leaf.cpf(xx, yy);
band = find(abs(dist0) <= bw*dx);

x = xx(band);
y = yy(band);


%% Closest points on the trimmed curve
[cpx, cpy, dist, bdy] = cpTrimmedCurve2d(x, y, curve_leaf, trim_leaf);


%% Extension and differential matrices
E = interp2_matrix(x1d, y1d, cpx, cpy, p, band);
E(bdy, :) = -E(bdy, :);

L = laplacian_2d_matrix(x1d, y1d, order, band, band);


%% Solve the same elliptic problem as example_hole_ellipse
d = 1;
D = -ones(length(E), 1) / d;

gamma = 2*dim/(dx^2);
I = speye(size(E));
M = E*L - gamma*(I - E);
u = M \ D;


%% Plot on the full ellipse parameterization
N = 512;
[xp, yp, thp] = paramf(N);
Eplot = interp2_matrix(x1d, y1d, xp(:), yp(:), p, band);
up = Eplot*u;

figure(1); clf;
plot(thp, up);
xlabel('theta');
ylabel('u');
title('solution on ellipse parameterization');

figure(2); clf;
plot2d_compdomain(u, x, y, dx, dx, 2)
hold on;
plot(xp, yp, 'k--', 'linewidth', 1.5);
plot(cpx(bdy), cpy(bdy), 'ro', 'markersize', 6, 'linewidth', 1.5);
thhole = linspace(0, 2*pi, 200);
plot(hole_cen(1) + hole_rad*cos(thhole), ...
     hole_cen(2) + hole_rad*sin(thhole), ':', ...
     'Color', [0.7 0 0], 'linewidth', 1.5);
hold off;
title('embedded solution and trimmed-curve boundary points');
xlabel('x');
ylabel('y');


function hole_rad = estimate_hole_radius(paramf, hole_cen, len_wanted)
  [xp, yp] = paramf(20000);
  ds = sqrt(diff(xp).^2 + diff(yp).^2);

  R_bar = len_wanted / 2;
  RR = linspace(0.7*R_bar, 1.02*R_bar, 150);
  err = inf(size(RR));

  for k = 1:length(RR)
    inside = ((xp(1:end-1) - hole_cen(1)).^2 + (yp(1:end-1) - hole_cen(2)).^2) < RR(k)^2;
    err(k) = abs(sum(ds(inside)) - len_wanted);
  end

  [junk, idx] = min(err); %#ok<ASGLU>
  hole_rad = RR(idx);
end
