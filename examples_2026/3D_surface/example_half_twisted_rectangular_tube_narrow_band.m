function [fig, bands] = example_half_twisted_rectangular_tube_narrow_band(h)
%EXAMPLE_HALF_TWISTED_RECTANGULAR_TUBE_NARROW_BAND  3D plot of the
%   half-twisted rectangular tube and the computational narrow bands of
%   its two strips, with the nodes used for branch rotation highlighted.
%
%   [fig, bands] = example_half_twisted_rectangular_tube_narrow_band(h)
%
%   h is the grid spacing (positive finite scalar, default 1/25).  This is
%   a plotting-only example: the bands are built by the same builder as
%   example_half_twisted_rectangular_tube_rotation_convergence
%   (halfTwistedRectTubeBands, default geometry/banding parameters), and no
%   elliptic system is assembled or solved.
%
%   What is plotted.  For each branch (A = branch 1, wide strip; B =
%   branch 2, narrow strip) the OUTER band geo.B(br).oband is shown: the
%   grid nodes at which the closest point extension operator E is
%   evaluated (inner band plus its Laplacian stencil neighbours).  Unused
%   prefilter candidates outside those operator bands are not shown.
%   Every node is drawn at its exact grid position (meshgrid ordering of
%   geo.x1d/y1d/z1d); there is no subsampling and no jitter.
%
%   Rotation nodes.  An outer row of branch br whose own-branch closest
%   point lies on an edge (geo.B(br).bdy ~= 0) is routed to the other
%   branch through a rotation about the corner curve.  The rotation node
%   set is the union of the grid indices of the routed rows of BOTH
%   branches.  Colour priority: a node in this set is drawn ONLY in the
%   rotation colour (magenta, filled), never additionally as an A or B
%   node.  The remaining (non-rotation) nodes are drawn as branch A (blue
%   filled dots) and/or branch B (orange open rings); a node in both outer
%   bands gets both markers at the same centre, so both memberships stay
%   visible.  The surface is neutral semi-transparent gray with black
%   corner curves so that it does not compete with the node colours.
%
%   The legend counts are "displayed / total" for A and B (total = full
%   outer band size, displayed = those not drawn in the rotation colour),
%   and the number of unique rotation nodes.
%
%   bands fields:
%     h                       grid spacing
%     A, B                    N-by-3 coordinates of ALL outer band nodes
%                             of branch A / B (in oband order)
%     indicesA, indicesB      their meshgrid linear grid indices (oband)
%     rotatedA, rotatedB      logical masks over A / B rows: routed rows
%                             (bdy ~= 0)
%     rotation                M-by-3 coordinates of the unique rotation
%                             nodes
%     rotationIndices         their sorted unique grid indices
%     shownA, shownB          logical masks over A / B rows actually drawn
%                             in the branch colour (~ismember of rotation)
%     handles                 struct of graphics handles
%
%   The figure is saved to examples_2026/figs/
%   half_twisted_rectangular_tube_narrow_band.png.

  if (nargin < 1 || isempty(h))
    h = 1/25;
  end
  validateattributes(h, {'numeric'}, {'scalar', 'real', 'positive', 'finite'}, ...
                     mfilename, 'h');
  h = double(h);

  here = fileparts(mfilename('fullpath'));
  addpath(here);

  [geo, P] = halfTwistedRectTubeBands(h);

  dims = [numel(geo.y1d) numel(geo.x1d) numel(geo.z1d)];
  indA = geo.B(1).oband(:);
  indB = geo.B(2).oband(:);
  rotA = (geo.B(1).bdy(:) ~= 0);
  rotB = (geo.B(2).bdy(:) ~= 0);
  rotIdx = union(indA(rotA), indB(rotB));
  rotIdx = rotIdx(:);
  shownA = ~ismember(indA, rotIdx);
  shownB = ~ismember(indB, rotIdx);

  XA = grid_coords(geo, dims, indA);
  XB = grid_coords(geo, dims, indB);
  XR = grid_coords(geo, dims, rotIdx);

  nA = numel(indA);  nB = numel(indB);  nR = numel(rotIdx);
  fprintf('Half-twisted rectangular tube narrow bands, h = %g\n', h);
  fprintf('  outer band A: %d nodes (%d routed rows, %d shown in branch colour)\n', ...
          nA, nnz(rotA), nnz(shownA));
  fprintf('  outer band B: %d nodes (%d routed rows, %d shown in branch colour)\n', ...
          nB, nnz(rotB), nnz(shownB));
  fprintf('  nodes in both outer bands: %d\n', numel(intersect(indA, indB)));
  fprintf('  unique rotation nodes (union of routed A and B rows): %d\n', nR);

  %% Figure
  colA = [0.00 0.35 0.85];
  colB = [0.95 0.50 0.05];
  colR = [0.75 0.05 0.65];
  colS = [0.60 0.60 0.60];

  fig = figure('Color', 'w', 'Position', [100 100 1100 850]);
  ax = axes(fig);
  hold(ax, 'on');

  % surface: each strip once over theta in [0, 4*pi]
  th = linspace(0, 4*pi, 257).';
  hS = gobjects(1, 2);
  widths = [P.a P.b];
  for br = 1:2
    s = linspace(-widths(br), widths(br), 17);
    [TH, SS] = ndgrid(th, s);
    F = halfTwistedRectTubeFrame(TH(:), SS(:), br, P.R, P.a, P.b);
    hS(br) = surf(ax, reshape(F(:,1), size(TH)), reshape(F(:,2), size(TH)), ...
                  reshape(F(:,3), size(TH)), ...
                  'FaceColor', colS, 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
                  'HandleVisibility', 'off');
  end
  % the two corner curves (strip edges), shared by both strips
  hE = gobjects(1, 2);
  sg = [1 -1];
  for k = 1:2
    C = halfTwistedRectTubeFrame(th, sg(k)*P.a, 1, P.R, P.a, P.b);
    hE(k) = plot3(ax, C(:,1), C(:,2), C(:,3), 'k-', 'LineWidth', 1.5, ...
                  'HandleVisibility', 'off');
  end

  ms = 3;
  hA = scatter3(ax, XA(shownA,1), XA(shownA,2), XA(shownA,3), ms, colA, 'filled', ...
                'MarkerFaceAlpha', 0.12, 'MarkerEdgeColor', 'none', ...
                'DisplayName', sprintf('branch A outer band (%d shown / %d total)', ...
                                       nnz(shownA), nA));
  hB = scatter3(ax, XB(shownB,1), XB(shownB,2), XB(shownB,3), 3*ms, colB, ...
                'MarkerEdgeAlpha', 0.25, 'LineWidth', 0.4, ...
                'DisplayName', sprintf('branch B outer band (%d shown / %d total)', ...
                                       nnz(shownB), nB));
  hR = scatter3(ax, XR(:,1), XR(:,2), XR(:,3), 2*ms, colR, 'filled', ...
                'MarkerFaceAlpha', 0.45, 'MarkerEdgeColor', 'none', ...
                'DisplayName', sprintf('rotation nodes (%d unique; routed A %d, B %d)', ...
                                       nR, nnz(rotA), nnz(rotB)));

  axis(ax, 'equal');
  view(ax, 38, 28);
  grid(ax, 'on');
  xlabel(ax, 'x');  ylabel(ax, 'y');  zlabel(ax, 'z');
  title(ax, {sprintf('Half-twisted rectangular tube, h = %g', h), ...
             'operator outer bands (E evaluation nodes); rotation nodes take colour priority'});
  hL = legend(ax, [hA hB hR], 'Location', 'southoutside', 'FontSize', 10);
  hold(ax, 'off');
  rotate3d(fig, 'on');

  figdir = fullfile(here, '..', 'figs');
  if (~exist(figdir, 'dir'))
    mkdir(figdir);
  end
  pngfile = fullfile(figdir, 'half_twisted_rectangular_tube_narrow_band.png');
  drawnow;
  exportgraphics(fig, pngfile, 'Resolution', 150);
  fprintf('Saved %s\n', pngfile);

  bands = struct('h', h, ...
                 'A', XA, 'B', XB, 'rotation', XR, ...
                 'indicesA', indA, 'indicesB', indB, 'rotationIndices', rotIdx, ...
                 'rotatedA', rotA, 'rotatedB', rotB, ...
                 'shownA', shownA, 'shownB', shownB, ...
                 'handles', struct('axes', ax, 'surface', hS, 'edges', hE, ...
                                   'A', hA, 'B', hB, 'rotation', hR, 'legend', hL));
end


function X = grid_coords(geo, dims, ind)
%GRID_COORDS  coordinates of meshgrid linear grid indices
  [iy, ix, iz] = ind2sub(dims, ind(:));
  X = [geo.x1d(ix) geo.y1d(iy) geo.z1d(iz)];
  X = reshape(X, [], 3);
end
