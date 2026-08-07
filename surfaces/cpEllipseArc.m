function [cpx, cpy, dist, bdy] = cpEllipseArc(x, y, a, b, cen, t1, t2)
%CPELLIPSEARC  Closest Point function for an arc of an ellipse.
%   [cpx, cpy, dist, bdy] = cpEllipseArc(x, y, a, b, cen, t1, t2)
%      The arc gamma(t) = cen + [a*cos(t), b*sin(t)] for t in [t1, t2].
%      't2 - t1' may be anything in (0, 2*pi]; t2 = t1 + 2*pi gives the
%      full closed ellipse (and then bdy is identically zero).
%
%      The full-ellipse closest point is computed first (Newton, via
%      cpParamCurveClosed, exactly as in cpEllipse).  The parameter of
%      that closest point is recovered from
%          t = atan2((cpy-cen(2))/b, (cpx-cen(1))/a)
%      and wrapped into the window [t1, t1+2*pi).  Points whose closest
%      point falls inside [t1, t2] keep it with bdy = 0; the remaining
%      points are clamped to the nearer of the two arc endpoints, with
%      bdy = 1 for gamma(t1) and bdy = 2 for gamma(t2) (same convention
%      as cpArc).
%
%      'dist' is the unsigned distance to the returned closest point.
%
%   Note: as with cpArc, the endpoint clamping is only the true closest
%   point sufficiently near the arc (which is all a narrow computational
%   band needs).
%
%   Code is vectorized: any size/shape for x, y should work.

  % defaults
  if (nargin < 3) || isempty(a)
    a = 0.75;
  end
  if (nargin < 4) || isempty(b)
    b = 1.25;
  end
  if (nargin < 5) || isempty(cen)
    cen = [0 0];
  end
  if (nargin < 6) || isempty(t1)
    t1 = 0;
  end
  if (nargin < 7) || isempty(t2)
    t2 = 2*pi;
  end

  span = t2 - t1;
  if (span <= 0 || span > 2*pi + 100*eps)
    error('cpEllipseArc: expected 0 < t2 - t1 <= 2*pi');
  end

  sz = size(x);
  xv = x(:) - cen(1);
  yv = y(:) - cen(2);

  % parametrised curve and its derivatives (as in cpEllipse)
  xs = @(t) a*cos(t);
  ys = @(t) b*sin(t);
  xp = @(t) -a*sin(t);
  yp = @(t) b*cos(t);
  xpp = @(t) -a*cos(t);
  ypp = @(t) -b*sin(t);

  [cx, cy, dd, fail] = ...
      cpParamCurveClosed(xv, yv, xs, ys, xp, yp, xpp, ypp, [0 2*pi]);
  cx = cx(:);  cy = cy(:);  dd = dd(:);

  if (any(fail(:)))
    warning('cpEllipseArc: some full-ellipse Newton solves failed');
  end

  % recover the parameter of the full-ellipse closest point exactly
  t = atan2(cy/b, cx/a);
  w = mod(t - t1, 2*pi);
  inside = (w <= span);

  bdy = zeros(size(cx));
  dist = dd;

  if (~all(inside))
    e1x = a*cos(t1);  e1y = b*sin(t1);
    e2x = a*cos(t2);  e2y = b*sin(t2);
    d1 = hypot(xv - e1x, yv - e1y);
    d2 = hypot(xv - e2x, yv - e2y);

    use1 = ~inside & (d1 <= d2);
    use2 = ~inside & ~use1;

    cx(use1) = e1x;  cy(use1) = e1y;  dist(use1) = d1(use1);  bdy(use1) = 1;
    cx(use2) = e2x;  cy(use2) = e2y;  dist(use2) = d2(use2);  bdy(use2) = 2;
  end

  cpx = reshape(cx + cen(1), sz);
  cpy = reshape(cy + cen(2), sz);
  dist = reshape(dist, sz);
  bdy = reshape(bdy, sz);
