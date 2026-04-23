function varargout = cpCSG(varargin)
%CPCSG  Closest point function for a manifold built from a CSG tree.
%   [cpx, cpy, dist] = cpCSG(x, y, tree)
%   [cpx, cpy, cpz, dist] = cpCSG(x, y, z, tree)
%
%   The tree is composed from csgLeaf, csgUnion, csgIntersection, and
%   csgDifference.  Leaves must provide a closest point projection onto a
%   manifold together with a signed distance to the associated solid.  The
%   resulting manifold is the boundary of the solid defined by the boolean
%   tree.
%
%   The returned 'dist' is the unsigned Euclidean distance to the manifold.
%
%   This routine implements the recursive candidate filtering and
%   alternating-projection seam search described in the accompanying
%   Python reference algorithm.

  if ((nargin ~= 3) && (nargin ~= 4))
    error('cpCSG:InvalidInputCount', ...
          'Use cpCSG(x, y, tree) or cpCSG(x, y, z, tree)');
  end

  tree = varargin{end};
  dim = nargin - 1;
  grids = varargin(1:dim);

  if (~isstruct(tree) || ~isfield(tree, 'dim'))
    error('cpCSG:InvalidTree', ...
          'tree must be a CSG leaf or CSG node');
  end
  if (tree.dim ~= dim)
    error('cpCSG:DimensionMismatch', ...
          'coordinate inputs and tree dimension do not match');
  end

  sz = size(grids{1});
  for k = 2:dim
    if (~isequal(size(grids{k}), sz))
      error('cpCSG:CoordinateSizeMismatch', ...
            'all coordinate inputs must have the same size');
    end
  end

  pts = zeros(numel(grids{1}), dim);
  for k = 1:dim
    pts(:, k) = grids{k}(:);
  end

  cp = zeros(size(pts));
  dist = zeros(size(pts, 1), 1);

  for k = 1:size(pts, 1)
    y = csgClosestCandidate(tree, pts(k, :));
    if (isempty(y))
      error('cpCSG:ProjectionFailed', ...
            'failed to find a closest point at index %d', k);
    end
    cp(k, :) = y;
    dist(k) = norm(pts(k, :) - y);
  end

  for k = 1:dim
    varargout{k} = reshape(cp(:, k), sz);
  end
  varargout{dim+1} = reshape(dist, sz);


function y = csgClosestCandidate(node, x)
  candidates = csgAllCandidates(node, x);

  if (isempty(candidates))
    y = [];
    return;
  end

  sqdist = sum((candidates - repmat(x, size(candidates, 1), 1)).^2, 2);
  [junk, idx] = min(sqdist); %#ok<ASGLU>
  y = candidates(idx, :);


function candidates = csgAllCandidates(node, x)
  if (strcmp(node.type, 'leaf'))
    candidates = csgLeafProject(node, x);
    return;
  end

  SA = csgAllCandidates(node.left, x);
  SB = csgAllCandidates(node.right, x);

  switch node.operation
   case 'intersection'
    VA = csgFilterCandidates(SA, @(y) csgContains(node.right, y));
    VB = csgFilterCandidates(SB, @(y) csgContains(node.left, y));

   case 'union'
    VA = csgFilterCandidates(SA, @(y) ~csgInInterior(node.right, y));
    VB = csgFilterCandidates(SB, @(y) ~csgInInterior(node.left, y));

   case 'difference'
    VA = csgFilterCandidates(SA, @(y) ~csgInInterior(node.right, y));
    VB = csgFilterCandidates(SB, @(y) csgInInterior(node.left, y));

   otherwise
    error('cpCSG:UnknownOperation', ...
          'unknown CSG operation ''%s''', node.operation);
  end

  candidates = [VA; VB];

  if (isempty(candidates))
    y = csgSeamAltProj(node.left, node.right, x);
    if (~isempty(y))
      candidates = y;
    end
  end


function candidates = csgFilterCandidates(candidates_in, predicate)
  candidates = zeros(0, size(candidates_in, 2));

  if (isempty(candidates_in))
    return;
  end

  keep = false(size(candidates_in, 1), 1);
  for k = 1:size(candidates_in, 1)
    keep(k) = predicate(candidates_in(k, :));
  end
  candidates = candidates_in(keep, :);


function tf = csgContains(node, x, tol)
  if (nargin < 3)
    tol = 1e-9;
  end

  tf = csgSdf(node, x) <= tol;


function tf = csgInInterior(node, x, tol)
  if (nargin < 3)
    tol = 1e-9;
  end

  tf = csgSdf(node, x) < -tol;


function sdist = csgSdf(node, x)
  if (strcmp(node.type, 'leaf'))
    sdist = csgLeafSdf(node, x);
    return;
  end

  lsdf = csgSdf(node.left, x);
  rsdf = csgSdf(node.right, x);

  switch node.operation
   case 'union'
    sdist = min(lsdf, rsdf);

   case 'intersection'
    sdist = max(lsdf, rsdf);

   case 'difference'
    sdist = max(lsdf, -rsdf);

   otherwise
    error('cpCSG:UnknownOperation', ...
          'unknown CSG operation ''%s''', node.operation);
  end


function y = csgLeafProject(node, x)
  args = num2cell(x);
  out = cell(1, node.dim);
  [out{:}] = node.cpf(args{:});
  y = zeros(1, node.dim);
  for k = 1:node.dim
    y(k) = out{k};
  end


function sdist = csgLeafSdf(node, x)
  args = num2cell(x);

  if (isempty(node.sdf_fun))
    out = cell(1, node.dim + 1);
    [out{:}] = node.cpf(args{:});
    sdist = out{node.dim + 1};
  else
    sdist = node.sdf_fun(args{:});
  end

  sdist = double(sdist);


function y = csgSeamAltProj(left, right, x, max_iter, tol, n_starts)
  if (nargin < 6)
    n_starts = 8;
  end
  if (nargin < 5)
    tol = 1e-8;
  end
  if (nargin < 4)
    max_iter = 200;
  end

  stream = RandStream('mt19937ar', 'Seed', 0);
  dim = length(x);
  directions = zeros(n_starts, dim);
  for k = 2:n_starts
    d = randn(stream, 1, dim);
    dn = norm(d);
    if (dn < eps)
      d = zeros(1, dim);
      d(1) = 1;
      dn = 1;
    end
    directions(k, :) = d / dn;
  end

  seam_tol = 1e-4;
  best_dist = inf;
  y = [];

  for k = 1:n_starts
    y0 = x + 0.5*directions(k, :);
    ycand = csgRunAltProj(left, right, y0, max_iter, tol);

    if (abs(csgSdf(left, ycand)) < seam_tol && ...
        abs(csgSdf(right, ycand)) < seam_tol)
      mydist = norm(x - ycand);
      if (mydist < best_dist)
        best_dist = mydist;
        y = ycand;
      end
    end
  end


function y = csgRunAltProj(left, right, y0, max_iter, tol)
  y = csgClosestCandidate(left, y0);
  if (isempty(y))
    y = y0;
  end

  for k = 1:max_iter
    if (mod(k, 2) == 1)
      ynext = csgClosestCandidate(right, y);
    else
      ynext = csgClosestCandidate(left, y);
    end

    if (isempty(ynext))
      ynext = y;
    end

    if (norm(ynext - y) < tol)
      y = ynext;
      return;
    end
    y = ynext;
  end
