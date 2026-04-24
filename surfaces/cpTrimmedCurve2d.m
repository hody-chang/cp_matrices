function [cpx, cpy, dist, bdy] = cpTrimmedCurve2d(x, y, curve_leaf, trim_leaf, tol)
%CPTRIMMEDCURVE2D  Closest point map for a curve trimmed by a solid.
%   [cpx, cpy, dist, bdy] = cpTrimmedCurve2d(x, y, curve_leaf, trim_leaf)
%
%   The kept manifold is the portion of curve_leaf outside the interior of
%   trim_leaf.  Points whose closest points land in the removed section are
%   projected to the seam using alternating projection, which gives the
%   boundary points of the trimmed open curve.
%
%   curve_leaf and trim_leaf should be created with csgLeaf and have
%   dimension 2.

  if (nargin < 5)
    tol = 1e-10;
  end

  if (~isstruct(curve_leaf) || ~isstruct(trim_leaf) || ...
      curve_leaf.dim ~= 2 || trim_leaf.dim ~= 2)
    error('cpTrimmedCurve2d:InvalidLeaf', ...
          'curve_leaf and trim_leaf must be 2D leaves created with csgLeaf');
  end

  [cpx, cpy] = leaf_project(curve_leaf, x, y);
  bdy = false(size(cpx));

  sdist_trim = leaf_sdf(trim_leaf, cpx, cpy);
  inside = (sdist_trim < 0);

  if (any(inside(:)))
    xv = cpx(inside);
    yv = cpy(inside);

    for k = 1:length(xv)
      [xv(k), yv(k)] = seam_projection(curve_leaf, trim_leaf, xv(k), yv(k), tol);
    end

    cpx(inside) = xv;
    cpy(inside) = yv;
    bdy(inside) = true;
  end

  dist = sqrt((cpx - x).^2 + (cpy - y).^2);
end


function [cpx, cpy] = leaf_project(leaf, x, y)
  [cpx, cpy] = leaf.cpf(x, y);
end


function sdist = leaf_sdf(leaf, x, y)
  if (isempty(leaf.sdf_fun))
    [junk1, junk2, sdist] = leaf.cpf(x, y); %#ok<ASGLU>
  else
    sdist = leaf.sdf_fun(x, y);
  end
end


function [xs, ys] = seam_projection(curve_leaf, trim_leaf, x0, y0, tol)
  max_iter = 200;

  xs = x0;
  ys = y0;

  for k = 1:max_iter
    [x1, y1, sdist_trim] = trim_leaf.cpf(xs, ys);
    [x2, y2, sdist_curve] = curve_leaf.cpf(x1, y1);

    if (abs(sdist_trim) < tol && abs(sdist_curve) < tol)
      xs = x2;
      ys = y2;
      return;
    end

    if (norm([x2 - xs, y2 - ys]) < tol)
      xs = x2;
      ys = y2;
      return;
    end

    xs = x2;
    ys = y2;
  end
end
