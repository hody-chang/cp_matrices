function leaf = csgLeaf(cpf, dim, sdf_fun)
%CSGLEAF  Wrap a closest point function as a CSG leaf primitive.
%   leaf = csgLeaf(cpf, dim)
%   leaf = csgLeaf(cpf, dim, sdf_fun)
%
%   The closest point function 'cpf' should accept either
%
%     [cpx, cpy, sdist] = cpf(x, y)
%
%   or
%
%     [cpx, cpy, cpz, sdist] = cpf(x, y, z)
%
%   where 'sdist' is the signed distance to the associated solid
%   (negative inside).  If 'cpf' does not return a signed distance, pass
%   one separately using 'sdf_fun'.
%
%   This leaf representation is used by cpCSG to build closest point maps
%   for manifolds defined by a CSG tree.

  if (nargin < 3)
    sdf_fun = [];
  end

  if (~isa(cpf, 'function_handle'))
    error('csgLeaf:InvalidClosestPointFunction', ...
          'cpf must be a function handle');
  end
  if (~isscalar(dim) || ~any(dim == [2 3]))
    error('csgLeaf:InvalidDimension', ...
          'dim must be either 2 or 3');
  end
  if (~isempty(sdf_fun) && ~isa(sdf_fun, 'function_handle'))
    error('csgLeaf:InvalidSignedDistanceFunction', ...
          'sdf_fun must be empty or a function handle');
  end

  leaf.type = 'leaf';
  leaf.dim = dim;
  leaf.cpf = cpf;
  leaf.sdf_fun = sdf_fun;
