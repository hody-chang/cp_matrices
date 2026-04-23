function sdist = csgSignedDistance(varargin)
%CSGSIGNEDDISTANCE  Signed distance to the solid defined by a CSG tree.
%   sdist = csgSignedDistance(tree, x, y)
%   sdist = csgSignedDistance(tree, x, y, z)
%
%   The sign convention matches the closest point functions used by the
%   CPM code: negative inside the associated solid, positive outside.
%
%   Leaves are expected to provide either:
%     * a closest point function whose final output is signed distance, or
%     * an explicit signed distance function through csgLeaf(..., sdf_fun).

  if ((nargin ~= 3) && (nargin ~= 4))
    error('csgSignedDistance:InvalidInputCount', ...
          'Use csgSignedDistance(tree, x, y) or csgSignedDistance(tree, x, y, z)');
  end

  tree = varargin{1};
  coords = varargin(2:end);

  if (~isstruct(tree) || ~isfield(tree, 'dim'))
    error('csgSignedDistance:InvalidTree', ...
          'tree must be a CSG leaf or CSG node');
  end
  if (tree.dim ~= length(coords))
    error('csgSignedDistance:DimensionMismatch', ...
          'coordinate inputs and tree dimension do not match');
  end

  sz = size(coords{1});
  for k = 2:length(coords)
    if (~isequal(size(coords{k}), sz))
      error('csgSignedDistance:CoordinateSizeMismatch', ...
            'all coordinate inputs must have the same size');
    end
  end

  sdist = csgSignedDistanceNode(tree, coords);


function sdist = csgSignedDistanceNode(node, coords)
  if (strcmp(node.type, 'leaf'))
    sdist = csgLeafSignedDistance(node, coords);
    return;
  end

  lsdf = csgSignedDistanceNode(node.left, coords);
  rsdf = csgSignedDistanceNode(node.right, coords);

  switch node.operation
   case 'union'
    sdist = min(lsdf, rsdf);

   case 'intersection'
    sdist = max(lsdf, rsdf);

   case 'difference'
    sdist = max(lsdf, -rsdf);

   otherwise
    error('csgSignedDistance:UnknownOperation', ...
          'unknown CSG operation ''%s''', node.operation);
  end


function sdist = csgLeafSignedDistance(node, coords)
  if (isempty(node.sdf_fun))
    out = cell(1, node.dim + 1);
    [out{:}] = node.cpf(coords{:});
    sdist = out{node.dim + 1};
  else
    sdist = node.sdf_fun(coords{:});
  end

  sdist = double(sdist);
