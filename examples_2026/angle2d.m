function [theta, tangent, info] = angle2d(x, y, cpf, singularPoint, varargin)
%ANGLE2D  Estimate a 2D endpoint tangent angle from cp/cpbar averages.
%   [theta, tangent, info] = angle2d(x, y, cpf, singularPoint, ...)
%      computes cp(x), then cpbar(x) = cp(2*cp(x)-x), using the closest
%      point function handle cpf.  Points are selected either by closest
%      point proximity to singularPoint = [sx sy], or, when singularPoint
%      is omitted or empty, by bdy ~= 0 from cpf.
%
%      The vector cp - cpbar is formed at selected points.  Tiny vectors
%      are discarded, the remaining vectors are normalized, and tangent is
%      the normalized average direction.  theta = atan2(tangent(2),
%      tangent(1)).  This direction estimates cp - cpbar, i.e. the
%      outward endpoint direction or -incoming tangent.
%
%      Extra inputs are forwarded to cpf.
%
%   If there are no usable vectors, theta and tangent are NaN and
%   info.numValid is zero.

  if (nargin < 4)
    singularPoint = [];
  end

  [cpx, cpy, dist, bdy] = cpf(x, y, varargin{:});
  [cpbarx, cpbary] = cpf(2*cpx - x, 2*cpy - y, varargin{:});

  vals = [x(:); y(:); cpx(:); cpy(:); cpbarx(:); cpbary(:)];
  if (~isempty(singularPoint))
    vals = [vals; singularPoint(:)];
  end
  vals = vals(isfinite(vals));
  if isempty(vals)
    scale = 1;
  else
    scale = max(1, max(abs(vals)));
  end

  tol = 100*eps*scale;
  singularTol = tol;
  vectorTol = tol;

  if isempty(singularPoint)
    mask = (bdy ~= 0);
  else
    sx = singularPoint(1);
    sy = singularPoint(2);
    mask = sqrt((cpx - sx).^2 + (cpy - sy).^2) <= singularTol;
  end

  vx = cpx - cpbarx;
  vy = cpy - cpbary;
  vectorNorms = sqrt(vx.^2 + vy.^2);
  validMask = mask & isfinite(vectorNorms) & (vectorNorms > vectorTol);
  zeroMask = mask & isfinite(vectorNorms) & (vectorNorms <= vectorTol);

  numCandidates = sum(mask(:));
  numValid = sum(validMask(:));
  numZero = sum(zeroMask(:));

  tangent = [NaN NaN];
  theta = NaN;
  average = [NaN NaN];
  averageNorm = NaN;

  if (numValid > 0)
    ux = vx(validMask) ./ vectorNorms(validMask);
    uy = vy(validMask) ./ vectorNorms(validMask);
    average = [mean(ux(:)) mean(uy(:))];
    averageNorm = norm(average);

    if (averageNorm > vectorTol)
      tangent = average ./ averageNorm;
      theta = atan2(tangent(2), tangent(1));
    end
  end

  info.numCandidates = numCandidates;
  info.numValid = numValid;
  info.numZero = numZero;
  info.cpx = cpx;
  info.cpy = cpy;
  info.dist = dist;
  info.bdy = bdy;
  info.cpbarx = cpbarx;
  info.cpbary = cpbary;
  info.mask = mask;
  info.validMask = validMask;
  info.zeroMask = zeroMask;
  info.vectors = [vx(:) vy(:)];
  info.vectorNorms = vectorNorms;
  info.average = average;
  info.averageNorm = averageNorm;
  info.tol = tol;
  info.singularTol = singularTol;
  info.vectorTol = vectorTol;
