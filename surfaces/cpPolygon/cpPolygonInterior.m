function [cpx, cpy, dist, bdy] = cpPolygonInterior(x, y, poly, inOutToggle)
%CPPOLYGONINTERIOR  Closest Point function for a solid polygon
%   The polygon does not need to be convex.
%
%   [cpx, cpy] = cpPolygon(x, y, poly)
%   [cpx, cpy, sdist] = cpPolygon(x, y, poly)
%
%   You can generate 'poly' interactively by calling
%
%      poly = makePolyInteractive()
%
%   If you leave off poly it will use a default one.
%
%   Note: returns signed distance (with negative inside).
%
%   Uses code by Tom Maerz.
%
%   The sign of the underlying signed distance no longer depends on
%   the orientation of 'poly', so 'inOutToggle' should not be needed;
%   it is kept because it is still a handy way to swap inside for
%   outside.
%
%   TODO: what should paramPolygonInterior do?

  % defaults
  if (nargin < 3)
    poly = makePolyDefault();
  end
  if (nargin < 4)
    inOutToggle = 0;
  end

  [cpx,cpy] = helper_CPopPoly(x, y, poly);
  sdist = helper_sDistPoly(x, y, poly);

  sdist = (-1)^(inOutToggle+1) * sdist;

  bdy = (sdist > 0);

  cpx = (sdist > 0) .* cpx  +  (sdist <= 0) .* x;
  cpy = (sdist > 0) .* cpy  +  (sdist <= 0) .* y;

  dist = (sdist >= 0) .* sdist;

