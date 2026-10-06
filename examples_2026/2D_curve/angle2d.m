function [R, tangent, info] = angle2d(x, y, cpf, singpt, varargin)
%ANGLE2D  Estimate a 2D endpoint rotation matrix from cp/cpbar averages.
%   [R, tangent, info] = angle2d(x, y, cpf, singpt, ...)
%      computes cp(x), then cpbar(x) = cp(2*cp(x)-x), using the closest
%      point function handle cpf.  Points are selected either by closest
%      point proximity to singpt = [sx sy], or, when singpt is omitted
%      or empty, by bdy ~= 0 from cpf.
%
%      The vector cp - cpbar is formed at selected points.  Tiny vectors
%      are discarded, the remaining vectors are normalized, and tangent is
%      the normalized average direction.  R is the 2x2 rotation matrix
%      that maps [1; 0] to tangent, built directly from the tangent
%      components:  R = [t(1) -t(2); t(2) t(1)].  That is Rodrigues'
%      formula about the out-of-plane axis with [1; 0] as the source
%      direction, which makes the cosine t(1) and the sine t(2); no angle
%      is formed and no transcendental is evaluated.  The equivalent for an
%      arbitrary pair of directions is rot2d_from_to in
%      ../rotations, and info.theta below, which is
%      atan2(tangent(2), tangent(1)), is for reporting only.
%
%      Extra inputs are forwarded to cpf.
%
%   If there are no usable vectors, R is NaN(2), tangent is NaN, and
%   info.numValid is zero.
%
%   Code is vectorized: any size/shape for x should work, provided the
%   function handle cpf is vectorized as well.

  if (nargin < 4)
    singpt = [];
  end

  [cpx, cpy, dist, bdy] = cpf(x, y, varargin{:});
  [cpbarx, cpbary] = cpf(2*cpx - x, 2*cpy - y, varargin{:});

  % the tolerances scale with the coordinates the cp function works in
  vals = [x(:); y(:); cpx(:); cpy(:); cpbarx(:); cpbary(:)];
  if (~isempty(singpt))
    vals = [vals; singpt(:)];
  end
  vals = vals(isfinite(vals));
  if isempty(vals)
    scale = 1;
  else
    scale = max(1, max(abs(vals)));
  end

  tol = 100*eps*scale;
  singtol = tol;
  vectol = tol;

  if isempty(singpt)
    mask = (bdy ~= 0);
  else
    sx = singpt(1);
    sy = singpt(2);
    mask = sqrt((cpx - sx).^2 + (cpy - sy).^2) <= singtol;
  end

  vx = cpx - cpbarx;
  vy = cpy - cpbary;
  vnorm = sqrt(vx.^2 + vy.^2);
  validmask = mask & isfinite(vnorm) & (vnorm > vectol);
  zeromask = mask & isfinite(vnorm) & (vnorm <= vectol);

  ncand = sum(mask(:));
  nvalid = sum(validmask(:));
  nzero = sum(zeromask(:));

  tangent = [NaN NaN];
  R = NaN(2);
  theta = NaN;
  avg = [NaN NaN];
  avgnorm = NaN;

  if (nvalid > 0)
    ux = vx(validmask) ./ vnorm(validmask);
    uy = vy(validmask) ./ vnorm(validmask);
    avg = [mean(ux(:)) mean(uy(:))];
    avgnorm = norm(avg);

    if (avgnorm > vectol)
      tangent = avg ./ avgnorm;
      % the rotation taking [1; 0] onto tangent, read straight off the
      % tangent's components; theta is derived from it for reporting and
      % nothing is computed from theta
      R = [tangent(1) -tangent(2); tangent(2) tangent(1)];
      theta = atan2(tangent(2), tangent(1));
    end
  end

  % NB: the info field names below are the output interface of this
  % function; the tests in surfaces/tests depend on them.
  info.theta = theta;
  info.numCandidates = ncand;
  info.numValid = nvalid;
  info.numZero = nzero;
  info.cpx = cpx;
  info.cpy = cpy;
  info.dist = dist;
  info.bdy = bdy;
  info.cpbarx = cpbarx;
  info.cpbary = cpbary;
  info.mask = mask;
  info.validMask = validmask;
  info.zeroMask = zeromask;
  info.vectors = [vx(:) vy(:)];
  info.vectorNorms = vnorm;
  info.average = avg;
  info.averageNorm = avgnorm;
  info.tol = tol;
  info.singularTol = singtol;
  info.vectorTol = vectol;
end
