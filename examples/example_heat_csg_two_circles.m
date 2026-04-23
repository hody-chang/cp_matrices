%% Heat equation on a manifold built from a CSG tree
% This example solves the surface heat equation on a disconnected
% manifold formed by the union of two disjoint circles.


%% Build the CSG tree
radius = 1;
cen1 = [-1.6 0];
cen2 = [ 1.6 0];

circle1 = csgLeaf(@(x, y) cpCircle(x, y, radius, cen1), 2);
circle2 = csgLeaf(@(x, y) cpCircle(x, y, radius, cen2), 2);
tree = csgUnion(circle1, circle2);


%% Construct a grid in the embedding space
dx = 0.1;
x1d = (-3.0:dx:3.0)';
y1d = (-2.0:dx:2.0)';

[xx, yy] = meshgrid(x1d, y1d);
[cpx, cpy, dist] = cpCSG(xx, yy, tree);


%% Banding
dim = 2;
p = 3;
order = 2;
bw = rm_bandwidth(dim, p, order/2);
band = find(dist <= bw*dx);

cpx = cpx(band);
cpy = cpy(band);
xg = xx(band);
yg = yy(band);


%% Initial condition on the disconnected manifold
dist1 = sqrt((cpx - cen1(1)).^2 + (cpy - cen1(2)).^2);
dist2 = sqrt((cpx - cen2(1)).^2 + (cpy - cen2(2)).^2);
left = (dist1 <= dist2);

th1 = atan2(cpy - cen1(2), cpx - cen1(1));
th2 = atan2(cpy - cen2(2), cpx - cen2(1));

u = zeros(size(cpx));
u(left) = cos(th1(left));
u(~left) = sin(th2(~left));


%% Interpolation and Laplacian matrices
disp('Constructing interpolation and laplacian matrices');
E = interp2_matrix(x1d, y1d, cpx, cpy, p, band);
L = laplacian_2d_matrix(x1d, y1d, order, band);


%% Interpolation matrices for plotting on each circle
thetas = linspace(-pi, pi, 200)';

xp1 = cen1(1) + radius*cos(thetas);
yp1 = cen1(2) + radius*sin(thetas);
xp2 = cen2(1) + radius*cos(thetas);
yp2 = cen2(2) + radius*sin(thetas);

Eplot1 = interp2_matrix(x1d, y1d, xp1, yp1, p, band);
Eplot2 = interp2_matrix(x1d, y1d, xp2, yp2, p, band);


%% Time-stepping
Tf = 1;
dt = 0.2*dx^2;
numtimesteps = ceil(Tf/dt);
dt = Tf / numtimesteps;

figure(1); clf;
figure(2); clf;
figure(3); clf;

for kt = 1:numtimesteps
  unew = u + dt*(L*u);
  u = E*unew;

  t = kt*dt;

  if ((kt < 10) || (mod(kt, 10) == 0) || (kt == numtimesteps))
    plot2d_compdomain(u, xg, yg, dx, dx, 1)
    title(sprintf('embedded domain: soln at time %g, timestep #%d', t, kt));
    xlabel('x');
    ylabel('y');
    hold on;
    plot(xp1, yp1, 'k-', 'linewidth', 2);
    plot(xp2, yp2, 'k-', 'linewidth', 2);
    hold off;

    circplot1 = Eplot1*u;
    circplot2 = Eplot2*u;
    exact1 = exp(-t)*cos(thetas);
    exact2 = exp(-t)*sin(thetas);
    max_err = max([max(abs(circplot1 - exact1)) ...
                   max(abs(circplot2 - exact2))]);

    set(0, 'CurrentFigure', 2);
    clf;
    subplot(2, 1, 1);
    plot(thetas, circplot1, 'b-');
    hold on;
    plot(thetas, exact1, 'r--');
    title(sprintf('left circle at time %g', t));
    xlabel('\theta');
    ylabel('u');
    legend('CPM', 'exact', 'Location', 'SouthEast');

    subplot(2, 1, 2);
    plot(thetas, circplot2, 'b-');
    hold on;
    plot(thetas, exact2, 'r--');
    title(sprintf('right circle at time %g', t));
    xlabel('\theta');
    ylabel('u');
    legend('CPM', 'exact', 'Location', 'SouthEast');

    set(0, 'CurrentFigure', 3);
    clf;
    subplot(2, 1, 1);
    plot(thetas, circplot1 - exact1);
    title(sprintf('left circle error at time %g', t));
    xlabel('\theta');
    ylabel('error');

    subplot(2, 1, 2);
    plot(thetas, circplot2 - exact2);
    title(sprintf('right circle error at time %g', t));
    xlabel('\theta');
    ylabel('error');

    fprintf('step %d of %d, max_err=%g\n', kt, numtimesteps, max_err);
    drawnow();
  end
end
