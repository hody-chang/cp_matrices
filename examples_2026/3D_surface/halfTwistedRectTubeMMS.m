function [u, f, us, ut] = halfTwistedRectTubeMMS(theta, s, branch, R, a, b)
%HALFTWISTEDRECTTUBEMMS Manufactured solution for -Lap u + u = f on the
% two branches of the half-twisted rectangular tube boundary.
%
%   [u,f,us,ut] = halfTwistedRectTubeMMS(theta,s,branch,R,a,b)
%
% Parametrization (per branch, theta 4*pi periodic):
%   F = (R+ug)*[cos t, sin t, 0] + vg*[0,0,1],
%   ug = p1*cos(t/2) - p2*sin(t/2),  vg = p1*sin(t/2) + p2*cos(t/2),
%   branch 1 (wide):   p = (s, b),  s in [-a, a]
%   branch 2 (narrow): p = (a, s),  s in [-b, b]
% Metric: E = 1, H = Fs.Ft = -b/2 (wide) or +a/2 (narrow),
%   G = (R+ug)^2 + (p1^2+p2^2)/4,  D = G - H^2.
% Laplace-Beltrami (divergence form, H constant):
%   Lap u = 1/sqrt(D) * [ d_s((G u_s - H u_t)/sqrt(D)) + d_t((-H u_s + u_t)/sqrt(D)) ].
% Outward conormal derivative at s = +-w:
%   d_nu u = sign(s)*sqrt(G/D)*(u_s - (H/G) u_t).
%
% Construction: u is the cubic Hermite interpolant in s between edge data
% (trace P, slope M) at s=-w and s=+w with theta-dependent coefficients.
% Edge traces / outward fluxes:
%   wide   upper (red,  s=+a): P = wR(t),        flux  gR(t)
%   wide   lower (blue, s=-a): P = wB(t),        flux  gB(t)
%   narrow upper (red,  s=+b): P = wR(t),        flux -gR(t)
%   narrow lower (blue, s=-b): P = wB(t+2pi),    flux -gB(t+2pi)
% Slopes are chosen to realize those fluxes exactly:
%   M = (H/G) P_t + sign(s) sqrt(D/G) g    (G evaluated on the edge).
% Hence traces agree across both glued curves and the outgoing conormal
% derivatives sum to zero there. All data are 4*pi periodic in theta.
% Every derivative is analytic; f = u - Lap u holds to roundoff.
%
% Inputs theta, s: equally sized arrays (scalar expansion allowed).
% Outputs have the common shape of the inputs. us = du/ds, ut = du/dtheta.

if ~(isequal(branch, 1) || isequal(branch, 2))
    error('halfTwistedRectTubeMMS:branch', 'branch must be 1 (wide) or 2 (narrow).');
end

if isscalar(theta) && ~isscalar(s)
    sz = size(s);
    theta = repmat(theta, sz);
elseif isscalar(s) && ~isscalar(theta)
    sz = size(theta);
    s = repmat(s, sz);
else
    if ~isequal(size(theta), size(s))
        error('halfTwistedRectTubeMMS:size', 'theta and s must have equal size.');
    end
    sz = size(theta);
end
t = theta(:);
s = s(:);

if branch == 1
    w = a;  H = -b/2;
else
    w = b;  H = a/2;
end

% ---- edge data ------------------------------------------------------
% Upper edge (s = +w): red curve.
[P1, P1t, P1tt, P1ttt] = traceR(t);
[g1, g1t, g1tt] = fluxR(t);
% Lower edge (s = -w): blue curve.
if branch == 1
    [P0, P0t, P0tt, P0ttt] = traceB(t);
    [g0, g0t, g0tt] = fluxB(t);
else
    [P0, P0t, P0tt, P0ttt] = traceB(t + 2*pi);
    [g0, g0t, g0tt] = fluxB(t + 2*pi);
    g1 = -g1;  g1t = -g1t;  g1tt = -g1tt;
    g0 = -g0;  g0t = -g0t;  g0tt = -g0tt;
end

[M1, M1t, M1tt] = edgeSlope(t,  w, +1, P1t, P1tt, P1ttt, g1, g1t, g1tt, branch, R, a, b, H);
[M0, M0t, M0tt] = edgeSlope(t, -w, -1, P0t, P0tt, P0ttt, g0, g0t, g0tt, branch, R, a, b, H);

% ---- cubic Hermite in s ---------------------------------------------
h  = 2*w;
xi = (s + w) / h;
x2 = xi.^2;  x3 = xi.^3;
h00 = 2*x3 - 3*x2 + 1;   h10 = x3 - 2*x2 + xi;
h01 = -2*x3 + 3*x2;      h11 = x3 - x2;
d00 = 6*x2 - 6*xi;       d10 = 3*x2 - 4*xi + 1;
d01 = -6*x2 + 6*xi;      d11 = 3*x2 - 2*xi;
e00 = 12*xi - 6;         e10 = 6*xi - 4;
e01 = -12*xi + 6;        e11 = 6*xi - 2;

comb = @(c00, c10, c01, c11, A0, B0, A1, B1) ...
    c00.*A0 + h*c10.*B0 + c01.*A1 + h*c11.*B1;

uu   = comb(h00, h10, h01, h11, P0,   M0,   P1,   M1);
uut  = comb(h00, h10, h01, h11, P0t,  M0t,  P1t,  M1t);
uutt = comb(h00, h10, h01, h11, P0tt, M0tt, P1tt, M1tt);
uus  = comb(d00, d10, d01, d11, P0,   M0,   P1,   M1)  / h;
uust = comb(d00, d10, d01, d11, P0t,  M0t,  P1t,  M1t) / h;
uuss = comb(e00, e10, e01, e11, P0,   M0,   P1,   M1)  / h^2;

% ---- Laplace-Beltrami -------------------------------------------------
[G, Gs, Gt] = metricG(t, s, branch, R, a, b);
D  = G - H^2;
q  = 1 ./ sqrt(D);
qs = -0.5 * Gs ./ D.^1.5;   % D_s = G_s
qt = -0.5 * Gt ./ D.^1.5;   % D_t = G_t
Js = G.*uus - H*uut;        % flux components (times sqrt(D))
Jt = -H*uus + uut;
divS = qs.*Js + q.*(Gs.*uus + G.*uuss - H*uust);
divT = qt.*Jt + q.*(-H*uust + uutt);
lap  = q .* (divS + divT);

u  = reshape(uu, sz);
f  = reshape(uu - lap, sz);
us = reshape(uus, sz);
ut = reshape(uut, sz);
end

% ======================================================================
function [M, Mt, Mtt] = edgeSlope(t, sEdge, sgn, Pt, Ptt, Pttt, g, gt, gtt, branch, R, a, b, H)
% M = H*r*P_t + sgn*phi*g, r = 1/G, phi = sqrt(1 - H^2 r) = sqrt(D/G).
[G, ~, Gt, Gtt] = metricG(t, sEdge + zeros(size(t)), branch, R, a, b);
r   = 1 ./ G;
rt  = -Gt ./ G.^2;
rtt = -Gtt ./ G.^2 + 2*Gt.^2 ./ G.^3;
phi   = sqrt(1 - H^2*r);
phit  = -H^2*rt ./ (2*phi);
phitt = -(0.5*H^2*rtt + phit.^2) ./ phi;
M   = H*r.*Pt + sgn*phi.*g;
Mt  = H*(rt.*Pt + r.*Ptt) + sgn*(phit.*g + phi.*gt);
Mtt = H*(rtt.*Pt + 2*rt.*Ptt + r.*Pttt) + sgn*(phitt.*g + 2*phit.*gt + phi.*gtt);
end

function [G, Gs, Gt, Gtt] = metricG(t, s, branch, R, a, b)
% G = (R+ug)^2 + (p1^2+p2^2)/4 and its derivatives.
c  = cos(t/2);  sn = sin(t/2);
if branch == 1
    p1 = s;  p2 = b + zeros(size(s));
    ugs = c;
else
    p1 = a + zeros(size(s));  p2 = s;
    ugs = -sn;
end
ug   = p1.*c - p2.*sn;
ugt  = -(p1.*sn + p2.*c)/2;
ugtt = -ug/4;
Rp   = R + ug;
G    = Rp.^2 + (p1.^2 + p2.^2)/4;
Gs   = 2*Rp.*ugs + s/2;
Gt   = 2*Rp.*ugt;
Gtt  = 2*ugt.^2 + 2*Rp.*ugtt;
end

function [w, wt, wtt, wttt] = traceR(t)
w    = sin(t/2 + 0.3) + 0.3*cos(t);
wt   = 0.5*cos(t/2 + 0.3) - 0.3*sin(t);
wtt  = -0.25*sin(t/2 + 0.3) - 0.3*cos(t);
wttt = -0.125*cos(t/2 + 0.3) + 0.3*sin(t);
end

function [w, wt, wtt, wttt] = traceB(t)
w    = 0.6*cos(t/2 - 0.4) + 0.2*sin(2*t);
wt   = -0.3*sin(t/2 - 0.4) + 0.4*cos(2*t);
wtt  = -0.15*cos(t/2 - 0.4) - 0.8*sin(2*t);
wttt = 0.075*sin(t/2 - 0.4) - 1.6*cos(2*t);
end

function [g, gt, gtt] = fluxR(t)
g   = 0.25*cos(t/2 - 0.2) + 0.2;
gt  = -0.125*sin(t/2 - 0.2);
gtt = -0.0625*cos(t/2 - 0.2);
end

function [g, gt, gtt] = fluxB(t)
g   = 0.2*sin(t/2 + 0.4) - 0.1;
gt  = 0.1*cos(t/2 + 0.4);
gtt = -0.05*sin(t/2 + 0.4);
end
