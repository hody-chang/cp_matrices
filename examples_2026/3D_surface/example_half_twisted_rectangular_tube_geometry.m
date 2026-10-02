%% Half-twisted rectangular tube: surface and its two closed edge curves
%
% C(theta) = R*(cos(theta), sin(theta), 0), with the rectangular boundary
% p in [-a,a] x [-b,b] rotated by phi = theta/2 in the (e_r, z) frame.
% F(p,theta + 2*pi) = F(-p,theta): opposite faces exchange after one loop.
% Plot each face pair ONCE on [0,4*pi], not all four faces on that interval.
% The wide pair uses p = (s,b), s in [-a,a]; the narrow pair uses
% p = (a,s), s in [-b,b]. Both have the red/blue curves as their boundaries.
% These are twisted face-pair strips; each is an annulus, not a literal
% Mobius strip. The complete boundary surface is an orientable torus.
% Cross-section sides are straight, but their swept faces are not planar.
%
% Run this script from any working directory. No CPM solve is performed.
% The interactive figure supports rotate3d; a PNG is saved beside the examples.

%% Geometry and sampling
R = 1;
a = 0.45;
b = 0.20;
theta = linspace(0, 4*pi, 257);

[Xwide, Ywide, Zwide] = tube_points(linspace(-a, a, 17)', b, theta, R);
[Xnarrow, Ynarrow, Znarrow] = tube_points(a, linspace(-b, b, 17)', theta, R);
[Xred, Yred, Zred] = tube_points(a, b, theta, R);
[Xblue, Yblue, Zblue] = tube_points(-a, b, theta, R);
center_theta = linspace(0, 2*pi, 257);

%% Assembled surface and separate face pairs
fig = figure('Color', 'w', 'Position', [80 100 1600 580]);
layout = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
wide_color = [0.88 0.75 0.48];
narrow_color = [0.52 0.76 0.79];
panel_titles = {'Complete surface', 'Wide face pair: p = (s,b)', ...
                'Narrow face pair: p = (a,s)'};
for panel = 1:3
  ax = nexttile(layout, panel);
  hold(ax, 'on');
  if panel ~= 3
    surf(ax, Xwide, Ywide, Zwide, 'FaceColor', wide_color, ...
         'EdgeColor', [0.35 0.30 0.20], 'EdgeAlpha', 0.22, ...
         'DisplayName', 'Wide face pair');
  end
  if panel ~= 2
    surf(ax, Xnarrow, Ynarrow, Znarrow, 'FaceColor', narrow_color, ...
         'EdgeColor', [0.20 0.35 0.38], 'EdgeAlpha', 0.22, ...
         'DisplayName', 'Narrow face pair');
  end
  plot3(ax, Xred, Yred, Zred, 'Color', [0.85 0.08 0.08], ...
        'LineWidth', 2.5, 'DisplayName', 'Edge 1: F((a,b),theta)');
  plot3(ax, Xblue, Yblue, Zblue, 'Color', [0.05 0.20 0.90], ...
        'LineWidth', 2.5, 'DisplayName', 'Edge 2: F((-a,b),theta)');
  plot3(ax, R*cos(center_theta), R*sin(center_theta), ...
        zeros(size(center_theta)), 'k--', 'LineWidth', 1, ...
        'DisplayName', 'Centerline');
  axis(ax, 'equal');
  axis(ax, 'vis3d');
  box(ax, 'on');
  grid(ax, 'on');
  view(ax, 38, 28);
  xlabel(ax, 'x'); ylabel(ax, 'y'); zlabel(ax, 'z');
  title(ax, panel_titles{panel});
  legend(ax, 'Location', 'southoutside', 'FontSize', 9);
end
title(layout, sprintf('Half-twisted rectangular tube: R = %.2f, a = %.2f, b = %.2f; theta in [0,4pi]', ...
                     R, a, b));
rotate3d(fig, 'on');
drawnow;

%% Save the geometry preview
figdir = fullfile(fileparts(mfilename('fullpath')), '..', 'figs');
if ~exist(figdir, 'dir'), mkdir(figdir); end
plotfile = fullfile(figdir, 'half_twisted_rectangular_tube_geometry.png');
exportgraphics(fig, plotfile, 'Resolution', 180);
fprintf('Geometry preview saved to %s\n', plotfile);

function [X, Y, Z] = tube_points(p1, p2, theta, R)
%TUBE_POINTS Rotate local cross-section coordinates in the (e_r,z) frame.
% A column of cross-section coordinates and a row of angles form a mesh.
  u = p1.*cos(theta/2) - p2.*sin(theta/2);
  v = p1.*sin(theta/2) + p2.*cos(theta/2);
  X = (R + u).*cos(theta);
  Y = (R + u).*sin(theta);
  Z = v;
end
