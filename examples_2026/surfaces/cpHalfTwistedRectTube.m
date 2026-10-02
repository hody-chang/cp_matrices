function [cpx,cpy,cpz,dist,bdy,theta,s] = cpHalfTwistedRectTube(x,y,z,R,a,b,branch)
%CPHALFTWISTEDRECTTUBE  Euclidean closest points to one face-pair strip of
%   the half-twisted rectangular tube.
%
%   [cpx,cpy,cpz,dist,bdy,theta,s] = ...
%       cpHalfTwistedRectTube(x,y,z,R,a,b,branch)
%
%   Surface (see halfTwistedRectTubeFrame):
%      F(t,p) = (R+u)*[cos t, sin t, 0] + v*[0 0 1],
%      u = p1*cos(t/2) - p2*sin(t/2),  v = p1*sin(t/2) + p2*cos(t/2),
%   branch = 1 (wide):   p = (s,b), s in [-a,a]
%   branch = 2 (narrow): p = (a,s), s in [-b,b]
%   t = theta is 4*pi-periodic; each strip covers its face pair once.
%   R, a, b default to 1, 0.45, 0.20 if empty.  Requires
%   R > sqrt(a^2+b^2) (embedded tube).
%
%   Outputs have the shape of x.  (cpx,cpy,cpz) = F(theta,s) is the true
%   Euclidean closest point of the BOUNDED strip (edges included), dist is
%   the unsigned Euclidean distance, theta in [0,4*pi), s in [-W,W]
%   (W = a or b), bdy = -1 / +1 when the closest point lies on the edge
%   s = -W / s = +W (within 1e-12, then s is snapped to -W / +W), 0
%   otherwise.
%
%   Algorithm.  For fixed theta the strip cross-section is a straight
%   segment in the (e_r, e_z) half-plane, so the optimal s is the exact
%   clamped projection onto that segment.  This leaves the 1D squared
%   distance g(theta), which is C^1 (envelope theorem) with analytic
%   g' = -2 (X-F).Ft and g'' = 2 (G - (X-F).Ftt - s'^2), where
%   s' = (X-F).Fst - H on interior points and s' = 0 on clamped ones.
%
%   Seeding policy (numerical multistart, not certified global
%   minimization).  g is sampled on a uniform periodic grid of NT = 256
%   points over [0,4*pi) and up to K = 4 of the lowest discrete local
%   minima are refined (always including the lowest sample, and letting
%   the sheets met at azimuth theta and theta+2*pi compete).  Each
%   candidate bracket [t_i - dt, t_i + dt] is refined by safeguarded
%   Newton/bisection on g' to |dtheta| <= 1e-12.  A refined root is
%   accepted only if g' changed sign over the bracket and the refined g
%   does not exceed the sampled bracket minimum (rejecting local maxima);
%   otherwise the bracket is resampled (16 points) and re-centred, up to
%   4 levels.  The best accepted candidate is returned; a candidate that
%   fails all levels raises an error.  A minimum missed by the coarse
%   samples, or ranked below the K best, is not detected.  This policy was
%   validated against brute-force minimization for narrow-band and
%   reflection-range queries of the convergence experiment, not proven
%   for arbitrary queries.  Theta accuracy ~1e-12 gives closest-point
%   accuracy ~1e-12.
%
%   Queries are processed in chunks of 4096 points; no toolboxes needed.
%
%   See also halfTwistedRectTubeFrame.

  if nargin < 7
    error('cpHalfTwistedRectTube:nargin', ...
          'usage: cpHalfTwistedRectTube(x,y,z,R,a,b,branch)');
  end
  if isempty(R), R = 1; end
  if isempty(a), a = 0.45; end
  if isempty(b), b = 0.20; end
  if ~(branch == 1 || branch == 2)
    error('cpHalfTwistedRectTube:branch', 'branch must be 1 or 2');
  end
  if ~(a > 0 && b > 0 && R > sqrt(a^2 + b^2))
    error('cpHalfTwistedRectTube:geometry', ...
          'require a > 0, b > 0, R > sqrt(a^2+b^2)');
  end
  if ~isequal(size(x), size(y), size(z))
    error('cpHalfTwistedRectTube:size', 'x, y, z must have equal sizes');
  end

  sz = size(x);
  X = [x(:), y(:), z(:)];
  N = size(X,1);

  NT = 256;
  K = 4;
  CHUNK = 4096;
  dt = 4*pi/NT;
  tg = (0:NT-1)*dt;
  cg = cos(tg/2);  sg = sin(tg/2);
  ctg = cos(tg);   stg = sin(tg);

  thetaAll = zeros(N,1);
  for i0 = 1:CHUNK:N
    idx = (i0:min(i0+CHUNK-1, N))';
    Xc = X(idx,:);
    n = numel(idx);

    % --- coarse global sampling of g(theta) -----------------------------
    W1 = bsxfun(@times, Xc(:,1), ctg) + bsxfun(@times, Xc(:,2), stg) - R;
    Q  = bsxfun(@times, Xc(:,2), ctg) - bsxfun(@times, Xc(:,1), stg);
    WP1 = bsxfun(@times, W1, cg) + bsxfun(@times, Xc(:,3), sg);
    WP2 = bsxfun(@times, Xc(:,3), cg) - bsxfun(@times, W1, sg);
    if branch == 1
      E1 = WP1 - min(max(WP1, -a), a);
      E2 = WP2 - b;
    else
      E1 = WP1 - a;
      E2 = WP2 - min(max(WP2, -b), b);
    end
    Gs = Q.^2 + E1.^2 + E2.^2;
    clear W1 Q WP1 WP2 E1 E2

    mask = (Gs <= circshift(Gs, [0 1])) & (Gs < circshift(Gs, [0 -1]));
    [gmin, imin] = min(Gs, [], 2);
    Mv = Gs;
    Mv(~mask) = Inf;
    [vals, ord] = sort(Mv, 2);
    vals = vals(:,1:K);
    ord = ord(:,1:K);
    none = ~isfinite(vals(:,1));          % flat rows: use argmin
    vals(none,1) = gmin(none);
    ord(none,1) = imin(none);
    valid = isfinite(vals);
    clear Gs Mv mask

    rowIdx = repmat((1:n)', 1, K);
    rows = rowIdx(valid);
    t0 = tg(ord(valid));
    t0 = t0(:);

    % --- refine candidates ---------------------------------------------
    [tc, gc, ok] = refineCandidates(Xc(rows,:), t0, vals(valid), dt, ...
                                    branch, R, a, b);
    if ~all(ok)
      bad = unique(idx(rows(~ok)));
      error('cpHalfTwistedRectTube:noconv', ...
            ['closest point refinement failed for %d query point(s); ' ...
             'first failing point (%g, %g, %g)'], numel(bad), ...
            X(bad(1),1), X(bad(1),2), X(bad(1),3));
    end

    Gbest = inf(n, K);
    Tbest = zeros(n, K);
    Gbest(valid) = gc;
    Tbest(valid) = tc;
    [tilde, j] = min(Gbest, [], 2); %#ok<ASGLU>
    thetaAll(idx) = Tbest(sub2ind([n K], (1:n)', j));
  end

  % --- final evaluation -------------------------------------------------
  th = mod(thetaAll, 4*pi);
  th(th >= 4*pi) = th(th >= 4*pi) - 4*pi;
  [tilde1, tilde2, tilde3, sv, sraw] = evalG(X, th, branch, R, a, b); %#ok<ASGLU>
  W = a*(branch == 1) + b*(branch == 2);
  % edge flag with roundoff tolerance; flagged points are snapped to s=+-W
  EDGETOL = 1e-12;
  bv = zeros(N,1);
  bv(sraw >= W - EDGETOL) = 1;
  bv(sraw <= -W + EDGETOL) = -1;
  sv(bv ~= 0) = W*bv(bv ~= 0);
  F = halfTwistedRectTubeFrame(th, sv, branch, R, a, b);
  D = X - F;
  dist = reshape(sqrt(sum(D.^2, 2)), sz);
  cpx = reshape(F(:,1), sz);
  cpy = reshape(F(:,2), sz);
  cpz = reshape(F(:,3), sz);
  bdy = reshape(bv, sz);
  theta = reshape(th, sz);
  s = reshape(sv, sz);
end


function [t, g, ok] = refineCandidates(X, t0, gref, dt, branch, R, a, b)
% Bracketed refinement with resampling fallback (see main help).
  m = numel(t0);
  t = t0;
  g = gref;
  ok = false(m,1);
  lo = t0 - dt;
  hi = t0 + dt;
  MSUB = 16;
  MAXLEVEL = 4;
  for level = 1:MAXLEVEL
    act = find(~ok);
    if isempty(act), break; end
    Xa = X(act,:);
    [tilde, gpl] = evalG(Xa, lo(act), branch, R, a, b); %#ok<ASGLU>
    [tilde, gph] = evalG(Xa, hi(act), branch, R, a, b); %#ok<ASGLU>
    br = (gpl <= 0) & (gph >= 0);
    good = false(numel(act),1);
    if any(br)
      ib = act(br);
      [tt, gg, conv] = rtsafe(X(ib,:), t(ib), lo(ib), hi(ib), gpl(br), ...
                              gph(br), branch, R, a, b);
      acc = conv & (gg <= gref(ib) + 1e-12*(1 + gref(ib)));
      t(ib(acc)) = tt(acc);
      g(ib(acc)) = gg(acc);
      good(br) = acc;
    end
    ok(act(good)) = true;
    rest = act(~good);
    if isempty(rest) || level == MAXLEVEL, break; end

    % resample the bracket and re-centre on the discrete minimum
    nr = numel(rest);
    w = linspace(0, 1, MSUB);
    step = (hi(rest) - lo(rest))/(MSUB-1);
    T = bsxfun(@plus, lo(rest), bsxfun(@times, hi(rest) - lo(rest), w));
    Xr = X(rest,:);
    Xrep = Xr(repmat((1:nr)', MSUB, 1), :);
    Gsub = reshape(evalG(Xrep, T(:), branch, R, a, b), nr, MSUB);
    [gm, jm] = min(Gsub, [], 2);
    tc = T(sub2ind([nr MSUB], (1:nr)', jm));
    t(rest) = tc;
    gref(rest) = gm;
    lo(rest) = tc - step;
    hi(rest) = tc + step;
  end
end


function [t, g, conv] = rtsafe(X, t, lo, hi, gpl, gph, branch, R, a, b)
% Safeguarded Newton/bisection for g'(t) = 0 on brackets with
% g'(lo) <= 0 <= g'(hi).  Vectorized over rows.
  TOL = 1e-12;
  MAXIT = 100;
  m = numel(t);
  conv = false(m,1);
  % endpoint roots
  z = (gpl == 0);  t(z) = lo(z);  conv(z) = true;
  z = (gph == 0) & ~conv;  t(z) = hi(z);  conv(z) = true;
  outside = ~conv & ~(t > lo & t < hi);
  t(outside) = 0.5*(lo(outside) + hi(outside));
  dxold = hi - lo;
  for it = 1:MAXIT
    act = find(~conv);
    if isempty(act), break; end
    [tilde, gp, gpp] = evalG(X(act,:), t(act), branch, R, a, b); %#ok<ASGLU>
    ta = t(act); la = lo(act); ha = hi(act);
    neg = gp < 0;  pos = gp > 0;
    la(neg) = ta(neg);
    ha(pos) = ta(pos);
    dxn = -gp./gpp;
    tn = ta + dxn;
    bis = ~(gpp > 0) | ~(tn > la & tn < ha) | ...
          (abs(dxn) > 0.5*abs(dxold(act)));
    tn(bis) = 0.5*(la(bis) + ha(bis));
    dx = tn - ta;
    dx(gp == 0) = 0;
    tn(gp == 0) = ta(gp == 0);
    dxold(act) = dx;
    t(act) = tn;
    lo(act) = la;
    hi(act) = ha;
    conv(act) = (abs(dx) <= TOL) | ((ha - la) <= TOL);
  end
  g = evalG(X, t, branch, R, a, b);
end


function [g, gp, gpp, s, sraw, W, F] = evalG(X, t, branch, R, a, b)
% Squared distance from X to the strip cross-section at t (s eliminated by
% clamped projection), and its first two theta-derivatives.
  c = cos(t/2);  sn = sin(t/2);
  w1 = X(:,1).*cos(t) + X(:,2).*sin(t) - R;
  w2 = X(:,3);
  if branch == 1
    sraw = w1.*c + w2.*sn;
    W = a;
  else
    sraw = w2.*c - w1.*sn;
    W = b;
  end
  s = min(max(sraw, -W), W);
  [F, Fs, Ft, tilde, Fst, Ftt] = halfTwistedRectTubeFrame(t, s, branch, R, a, b); %#ok<ASGLU>
  D = X - F;
  g = sum(D.^2, 2);
  if nargout > 1
    gp = -2*sum(D.*Ft, 2);
    G = sum(Ft.^2, 2);
    H = sum(Fs.*Ft, 2);
    sp = (sum(D.*Fst, 2) - H) .* (abs(sraw) < W);
    gpp = 2*(G - sum(D.*Ftt, 2) - sp.^2);
  end
end
