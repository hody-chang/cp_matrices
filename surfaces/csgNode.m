function node = csgNode(operation, left, right)
%CSGNODE  Construct a binary CSG node for a manifold tree.
%   node = csgNode(operation, left, right)
%
%   operation must be one of:
%     'union'
%     'intersection'
%     'difference'
%
%   Leaves should be created with csgLeaf, and child nodes may themselves
%   be other CSG nodes.

  operation = lower(operation);
  if (~any(strcmp(operation, {'union', 'intersection', 'difference'})))
    error('csgNode:UnknownOperation', ...
          'operation must be union, intersection, or difference');
  end
  if (~isstruct(left) || ~isstruct(right) || ...
      ~isfield(left, 'dim') || ~isfield(right, 'dim'))
    error('csgNode:InvalidChildren', ...
          'left and right must be CSG leaves or CSG nodes');
  end
  if (left.dim ~= right.dim)
    error('csgNode:DimensionMismatch', ...
          'left and right must have the same dimension');
  end

  node.type = 'node';
  node.operation = operation;
  node.left = left;
  node.right = right;
  node.dim = left.dim;
