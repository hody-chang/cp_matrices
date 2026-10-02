function [F,Fs,Ft,Fss,Fst,Ftt] = halfTwistedRectTubeFrame(theta,s,branch,R,a,b)
%HALFTWISTEDRECTTUBEFRAME  Parametrization and analytic derivatives of
%   one face-pair strip of the half-twisted rectangular tube.
%
%   [F,Fs,Ft,Fss,Fst,Ftt] = halfTwistedRectTubeFrame(theta,s,branch,R,a,b)
%
%   The tube is
%      F(t,p) = (R+u)*[cos t, sin t, 0] + v*[0 0 1],
%      u = p1*cos(t/2) - p2*sin(t/2),  v = p1*sin(t/2) + p2*cos(t/2),
%   with t (theta) 4*pi-periodic.  The two strips (branches) are
%      branch = 1 (wide):   p = (s, b),  s in [-a, a]
%      branch = 2 (narrow): p = (a, s),  s in [-b, b]
%   Each strip covers its face pair exactly once over t in [0,4*pi)
%   (t+2*pi maps a face onto the opposite face).
%
%   theta and s are equally sized vectors (either may be scalar and is
%   expanded); they are treated as columns.  Outputs are N-by-3:
%   F, dF/ds, dF/dt, d2F/ds2 (identically zero, strips are straight in s),
%   d2F/dsdt, d2F/dt2.  All derivatives are analytic.
%
%   Metric: E = |Fs|^2 = 1, H = Fs.Ft = -b/2 (wide), +a/2 (narrow),
%   G = |Ft|^2 = (R+u)^2 + (p1^2+p2^2)/4.
%
%   Edge gluing (s = +-width): wide upper (s=a) at t equals narrow upper
%   (s=b) at t; wide lower (s=-a) at t equals narrow lower (s=-b) at
%   t+2*pi (mod 4*pi).
%
%   See also cpHalfTwistedRectTube.

  theta = theta(:);
  s = s(:);
  if numel(theta) == 1 && numel(s) > 1
    theta = repmat(theta, size(s));
  elseif numel(s) == 1 && numel(theta) > 1
    s = repmat(s, size(theta));
  elseif numel(s) ~= numel(theta)
    error('halfTwistedRectTubeFrame:size', ...
          'theta and s must have equal numbers of elements (or be scalar)');
  end
  n = numel(theta);

  c  = cos(theta/2);
  sn = sin(theta/2);
  ct = cos(theta);
  st = sin(theta);

  switch branch
    case 1
      p1 = s;
      p2 = b;
      us = c;        % rotation of dp = (1,0)
      vs = sn;
    case 2
      p1 = a;
      p2 = s;
      us = -sn;      % rotation of dp = (0,1)
      vs = c;
    otherwise
      error('halfTwistedRectTubeFrame:branch', 'branch must be 1 or 2');
  end

  u = p1.*c - p2.*sn;
  v = p1.*sn + p2.*c;
  Ru = R + u;

  % t-derivatives of (u,v): d/dt rot(t/2) = (1/2) J rot(t/2), J(u,v) = (-v,u)
  ut  = -v/2;    vt  = u/2;
  utt = -u/4;    vtt = -v/4;
  ust = -vs/2;   vst = us/2;

  F   = [Ru.*ct, Ru.*st, v];
  Fs  = [us.*ct, us.*st, vs];
  Ft  = [ut.*ct - Ru.*st, ut.*st + Ru.*ct, vt];
  Fss = zeros(n,3);
  Fst = [ust.*ct - us.*st, ust.*st + us.*ct, vst];
  Ftt = [(utt - Ru).*ct - 2*ut.*st, (utt - Ru).*st + 2*ut.*ct, vtt];
end
