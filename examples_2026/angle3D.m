function [R, conormal, info] = angle3D(x, y, z, cpf, singularPoint, axis, targetDirection, varargin)
%ANGLE3D  Estimate a 3D branch rotation matrix from cp/cpbar averages.
%   [R, conormal, info] = angle3D(x, y, z, cpf, singularPoint, axis,
%      targetDirection, ...) computes cp(x), then
%      cpbar(x) = cp(2*cp(x)-x), using the closest point function handle
%      cpf.  Points are selected either by closest point proximity to
%      singularPoint = [sx sy sz], or, when singularPoint is omitted or
%      empty, by bdy ~= 0 from cpf.  If cpf does not return bdy and no
%      singularPoint is supplied, all points are candidates.
%
%      The vector cp - cpbar is formed at selected points.  Tiny vectors
%      are discarded, the remaining vectors are normalized, and conormal is
%      the normalized average direction.  This direction estimates the
%      branch outward co-normal or -incoming tangent analogue.
%
%      When axis and targetDirection are both supplied and non-empty, R is
%      the 3x3 rotation matrix (Rodrigues' formula) about axis that maps
%      conormal to targetDirection (both projected perpendicular to axis).
%      The angle theta = atan2(sin,cos) is stored in info.theta.  When
%      axis or targetDirection is omitted, R is NaN(3).
%
%      Extra inputs are forwarded to cpf.
%
%   If there are no usable vectors, R is NaN(3), conormal is NaN, and
%   info.numValid is zero.

  if (nargin < 5)
    singularPoint = [];
  end
  if (nargin < 6)
    axis = [];
  end
  if (nargin < 7)
    targetDirection = [];
  end

  [cpx, cpy, cpz, dist, bdy, hasBdy] = angle3D_callCpf(cpf, x, y, z, varargin{:});
  [cpbarx, cpbary, cpbarz] = angle3D_callCpf(cpf, 2*cpx - x, 2*cpy - y, 2*cpz - z, varargin{:});

  vals = [x(:); y(:); z(:); cpx(:); cpy(:); cpz(:); cpbarx(:); cpbary(:); cpbarz(:)];
  if (~isempty(singularPoint))
    vals = [vals; singularPoint(:)];
  end
  if (~isempty(axis))
    vals = [vals; axis(:)];
  end
  if (~isempty(targetDirection))
    vals = [vals; targetDirection(:)];
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
    if (hasBdy && ~isempty(bdy))
      mask = (bdy ~= 0);
    else
      mask = true(size(cpx));
    end
  else
    sx = singularPoint(1);
    sy = singularPoint(2);
    sz = singularPoint(3);
    mask = sqrt((cpx - sx).^2 + (cpy - sy).^2 + (cpz - sz).^2) <= singularTol;
  end

  vx = cpx - cpbarx;
  vy = cpy - cpbary;
  vz = cpz - cpbarz;
  vectorNorms = sqrt(vx.^2 + vy.^2 + vz.^2);
  validMask = mask & isfinite(vectorNorms) & (vectorNorms > vectorTol);
  zeroMask = mask & isfinite(vectorNorms) & (vectorNorms <= vectorTol);

  numCandidates = sum(mask(:));
  numValid = sum(validMask(:));
  numZero = sum(zeroMask(:));

  conormal = [NaN NaN NaN];
  R = NaN(3);
  theta = NaN;
  average = [NaN NaN NaN];
  averageNorm = NaN;

  if (numValid > 0)
    ux = vx(validMask) ./ vectorNorms(validMask);
    uy = vy(validMask) ./ vectorNorms(validMask);
    uz = vz(validMask) ./ vectorNorms(validMask);
    average = [mean(ux(:)) mean(uy(:)) mean(uz(:))];
    averageNorm = norm(average);

    if (averageNorm > vectorTol)
      conormal = average ./ averageNorm;
    end
  end

  angleSource = [NaN NaN NaN];
  angleTarget = [NaN NaN NaN];
  axisUnit = [NaN NaN NaN];
  if (~isempty(axis) && ~isempty(targetDirection) && all(isfinite(conormal)))
    axisVec = axis(:).';
    targetVec = targetDirection(:).';
    axisNorm = norm(axisVec);
    targetNorm = norm(targetVec);

    if (length(axisVec) == 3 && length(targetVec) == 3 && ...
        isfinite(axisNorm) && isfinite(targetNorm) && ...
        axisNorm > vectorTol && targetNorm > vectorTol)
      axisUnit = axisVec ./ axisNorm;
      angleSource = conormal - dot(conormal, axisUnit)*axisUnit;
      angleTarget = targetVec - dot(targetVec, axisUnit)*axisUnit;
      sourceNorm = norm(angleSource);
      targetProjectedNorm = norm(angleTarget);

      if (sourceNorm > vectorTol && targetProjectedNorm > vectorTol)
        angleSource = angleSource ./ sourceNorm;
        angleTarget = angleTarget ./ targetProjectedNorm;
        cosTheta = dot(angleSource, angleTarget);
        sinTheta = dot(axisUnit, cross(angleSource, angleTarget));
        K = [0 -axisUnit(3) axisUnit(2); ...
             axisUnit(3) 0 -axisUnit(1); ...
             -axisUnit(2) axisUnit(1) 0];
        R = eye(3) + sinTheta*K + (1 - cosTheta)*(K*K);
        theta = atan2(sinTheta, cosTheta);
      end
    end
  end

  info.theta = theta;
  info.numCandidates = numCandidates;
  info.numValid = numValid;
  info.numZero = numZero;
  info.cpx = cpx;
  info.cpy = cpy;
  info.cpz = cpz;
  info.dist = dist;
  info.bdy = bdy;
  info.hasBdy = hasBdy;
  info.cpbarx = cpbarx;
  info.cpbary = cpbary;
  info.cpbarz = cpbarz;
  info.mask = mask;
  info.validMask = validMask;
  info.zeroMask = zeroMask;
  info.vectors = [vx(:) vy(:) vz(:)];
  info.vectorNorms = vectorNorms;
  info.average = average;
  info.averageNorm = averageNorm;
  info.tol = tol;
  info.singularTol = singularTol;
  info.vectorTol = vectorTol;
  if (~isempty(axis))
    info.axis = axis;
  end
  if (~isempty(targetDirection))
    info.targetDirection = targetDirection;
  end
  if (~isempty(axis) && ~isempty(targetDirection))
    info.axisUnit = axisUnit;
    info.angleSource = angleSource;
    info.angleTarget = angleTarget;
  end


function [cpx, cpy, cpz, dist, bdy, hasBdy] = angle3D_callCpf(cpf, x, y, z, varargin)

  try
    [cpx, cpy, cpz, dist, bdy] = cpf(x, y, z, varargin{:});
    hasBdy = true;
  catch ME
    tooManyOutputs = strcmp(ME.identifier, 'MATLAB:TooManyOutputs') || ...
      ~isempty(strfind(ME.message, 'Too many output'));
    if ~tooManyOutputs
      rethrow(ME);
    end

    [cpx, cpy, cpz, dist] = cpf(x, y, z, varargin{:});
    bdy = [];
    hasBdy = false;
  end
