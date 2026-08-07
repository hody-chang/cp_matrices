function [R, conormal, info] = angle3D(x, y, z, cpf, singpt, axis, targetdir, varargin)
%ANGLE3D  Estimate a 3D branch rotation matrix from cp/cpbar averages.
%   [R, conormal, info] = angle3D(x, y, z, cpf, singpt, axis, targetdir,
%      ...) computes cp(x), then cpbar(x) = cp(2*cp(x)-x), using the
%      closest point function handle cpf.  Points are selected either by
%      closest point proximity to singpt = [sx sy sz], or, when singpt
%      is omitted or empty, by bdy ~= 0 from cpf.  If cpf does not return
%      bdy and no singpt is supplied, all points are candidates.
%
%      The vector cp - cpbar is formed at selected points.  Tiny vectors
%      are discarded, the remaining vectors are normalized, and conormal is
%      the normalized average direction.  This direction estimates the
%      branch outward co-normal or -incoming tangent analogue.
%
%      When axis and targetdir are both supplied and non-empty, R is
%      the 3x3 rotation matrix (Rodrigues' formula) about axis that maps
%      conormal to targetdir (both projected perpendicular to axis).
%      The angle theta = atan2(sin,cos) is stored in info.theta.  When
%      axis or targetdir is omitted, R is NaN(3).
%
%      Extra inputs are forwarded to cpf.
%
%   If there are no usable vectors, R is NaN(3), conormal is NaN, and
%   info.numValid is zero.
%
%   Code is vectorized: any size/shape for x should work, provided the
%   function handle cpf is vectorized as well.

  if (nargin < 5)
    singpt = [];
  end
  if (nargin < 6)
    axis = [];
  end
  if (nargin < 7)
    targetdir = [];
  end

  [cpx, cpy, cpz, dist, bdy, hasbdy] = ...
      angle3D_call_cpf(cpf, x, y, z, varargin{:});
  [cpbarx, cpbary, cpbarz] = ...
      angle3D_call_cpf(cpf, 2*cpx - x, 2*cpy - y, 2*cpz - z, varargin{:});

  % the tolerances scale with the coordinates the cp function works in
  vals = [x(:); y(:); z(:); cpx(:); cpy(:); cpz(:); ...
          cpbarx(:); cpbary(:); cpbarz(:)];
  if (~isempty(singpt))
    vals = [vals; singpt(:)];
  end
  if (~isempty(axis))
    vals = [vals; axis(:)];
  end
  if (~isempty(targetdir))
    vals = [vals; targetdir(:)];
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
    if (hasbdy && ~isempty(bdy))
      mask = (bdy ~= 0);
    else
      mask = true(size(cpx));
    end
  else
    sx = singpt(1);
    sy = singpt(2);
    sz = singpt(3);
    mask = sqrt((cpx - sx).^2 + (cpy - sy).^2 + (cpz - sz).^2) <= singtol;
  end

  vx = cpx - cpbarx;
  vy = cpy - cpbary;
  vz = cpz - cpbarz;
  vnorm = sqrt(vx.^2 + vy.^2 + vz.^2);
  validmask = mask & isfinite(vnorm) & (vnorm > vectol);
  zeromask = mask & isfinite(vnorm) & (vnorm <= vectol);

  ncand = sum(mask(:));
  nvalid = sum(validmask(:));
  nzero = sum(zeromask(:));

  conormal = [NaN NaN NaN];
  R = NaN(3);
  theta = NaN;
  avg = [NaN NaN NaN];
  avgnorm = NaN;

  if (nvalid > 0)
    ux = vx(validmask) ./ vnorm(validmask);
    uy = vy(validmask) ./ vnorm(validmask);
    uz = vz(validmask) ./ vnorm(validmask);
    avg = [mean(ux(:)) mean(uy(:)) mean(uz(:))];
    avgnorm = norm(avg);

    if (avgnorm > vectol)
      conormal = avg ./ avgnorm;
    end
  end

  %% Rodrigues rotation about axis, taking conormal onto targetdir

  angsrc = [NaN NaN NaN];
  angtgt = [NaN NaN NaN];
  axisunit = [NaN NaN NaN];
  if (~isempty(axis) && ~isempty(targetdir) && all(isfinite(conormal)))
    axisvec = axis(:).';
    targetvec = targetdir(:).';
    axisnorm = norm(axisvec);
    targetnorm = norm(targetvec);

    if (length(axisvec) == 3 && length(targetvec) == 3 && ...
        isfinite(axisnorm) && isfinite(targetnorm) && ...
        axisnorm > vectol && targetnorm > vectol)
      axisunit = axisvec ./ axisnorm;
      % both directions projected perpendicular to the axis
      angsrc = conormal - dot(conormal, axisunit)*axisunit;
      angtgt = targetvec - dot(targetvec, axisunit)*axisunit;
      srcnorm = norm(angsrc);
      tgtnorm = norm(angtgt);

      if (srcnorm > vectol && tgtnorm > vectol)
        angsrc = angsrc ./ srcnorm;
        angtgt = angtgt ./ tgtnorm;
        costh = dot(angsrc, angtgt);
        sinth = dot(axisunit, cross(angsrc, angtgt));
        K = [0 -axisunit(3) axisunit(2); ...
             axisunit(3) 0 -axisunit(1); ...
             -axisunit(2) axisunit(1) 0];
        R = eye(3) + sinth*K + (1 - costh)*(K*K);
        theta = atan2(sinth, costh);
      end
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
  info.cpz = cpz;
  info.dist = dist;
  info.bdy = bdy;
  info.hasBdy = hasbdy;
  info.cpbarx = cpbarx;
  info.cpbary = cpbary;
  info.cpbarz = cpbarz;
  info.mask = mask;
  info.validMask = validmask;
  info.zeroMask = zeromask;
  info.vectors = [vx(:) vy(:) vz(:)];
  info.vectorNorms = vnorm;
  info.average = avg;
  info.averageNorm = avgnorm;
  info.tol = tol;
  info.singularTol = singtol;
  info.vectorTol = vectol;
  if (~isempty(axis))
    info.axis = axis;
  end
  if (~isempty(targetdir))
    info.targetDirection = targetdir;
  end
  if (~isempty(axis) && ~isempty(targetdir))
    info.axisUnit = axisunit;
    info.angleSource = angsrc;
    info.angleTarget = angtgt;
  end


function [cpx, cpy, cpz, dist, bdy, hasbdy] = ...
    angle3D_call_cpf(cpf, x, y, z, varargin)
%ANGLE3D_CALL_CPF  call cpf, tolerating one that does not return bdy

  try
    [cpx, cpy, cpz, dist, bdy] = cpf(x, y, z, varargin{:});
    hasbdy = true;
  catch ME
    toomany = strcmp(ME.identifier, 'MATLAB:TooManyOutputs') || ...
      ~isempty(strfind(ME.message, 'Too many output'));
    if ~toomany
      rethrow(ME);
    end

    [cpx, cpy, cpz, dist] = cpf(x, y, z, varargin{:});
    bdy = [];
    hasbdy = false;
  end
