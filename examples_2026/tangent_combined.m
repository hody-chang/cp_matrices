% tangent_combined.m
% ---------------------------------------------------------------------------
% Prescribed probe points and real narrow-band grid points, on the same axes,
% over eight decades of h.
%
%   curves    circular arcs of radius 1/kappa,  kappa = 10.^(-2:1:1)
%   h         10.^(-1:-0.5:-8)
%
%   PRESCRIBED   x = v + a t + b n  with  a = h,  b = a
%   BAND         grid of spacing dx = h, narrow band |dist| <= bw*dx, then the
%                points with cp(x) = v; reported as the max over that cloud
%                (clean, see below) and at the single nearest point (jagged)
%
% Both use the same estimators and the same exact cp (cpArc):
%   d_k  = cp(x) - cpbar(x)
%   d_k2 = (3/2)cp(x) - 2 cpbar(x) + (1/2)cp2bar(x)
%
% Predicted for the prescribed points, where a = b = h and kappa' = 0:
%   |dhat_k  - t| = (kappa/2) h
%   |dhat_k2 - t| = 2 kappa^2 h^2
%
% Two things the eight decades expose that a three-decade sweep cannot:
%
%  1. ROUNDOFF.  Both estimators divide a difference of closest points by a
%     length ~h.  cpArc works in coordinates of size R, so each closest point
%     carries ~eps*R of rounding and the direction inherits ~C*eps*R/h.  That
%     floor GROWS as h shrinks.  d_k2's signal is O(h^2) against d_k's O(h),
%     so d_k2 meets the floor first -- near h ~ (eps R/kappa^2)^(1/3) ~ 1e-5
%     for kappa = 1, while d_k keeps converging to h ~ sqrt(eps R/kappa) ~ 3e-8.
%     Below its floor each curve turns and RISES like 1/h.
%
%  2. The band's per-point curve stays jagged at every scale, because a rank-k
%     grid node is not a fixed geometric object: v is not a lattice site, so
%     halving dx reshuffles which node is nearest and its a/dx and b/a jump.
%     The max over the captured cloud has no such labelling problem.
%
% The band grid is built in a window of half-width 3*bw*dx about v, not over
% the whole curve: every point with cp(x) = v satisfies |x-v| = dist <= bw*dx,
% so that window provably contains the entire captured set.  (A full-curve grid
% at dx = 1e-8 is ~1e15 nodes.)  The left subplot still draws the band over the
% wide window at a fixed dxShow, for context.
%
% Run headlessly:
%   cd cp_matrices/examples_2026 && matlab -batch "tangent_combined"
% ---------------------------------------------------------------------------
function tangent_combined()
  here = fileparts(mfilename('fullpath'));
  addpath(fullfile(here, '..', 'surfaces'));   % cpArc

  kappas = 10.^(-2:1:1);
  hs     = 10.^(-1:-0.5:-8);
  beta   = 0.7;  cen = [0 0];  span = 3*pi/2;
  dxShow = 0.0125;  Wshow = 0.3;

  p = 3; order = 2; dim = 2;
  bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + (order/2 + (p+1)/2)^2);

  fprintf('\n=====================================================================\n');
  fprintf(' prescribed (a = b = h) vs narrow-band grid points,  h = 1e-1 .. 1e-8\n');
  fprintf('=====================================================================\n');

  for kj = 1:numel(kappas)
    kap = kappas(kj);  R = 1/kap;
    a1 = beta - span;  a2 = beta;
    cpf = @(q) arcCP(q, R, cen, a1, a2);
    v = R*[cos(beta) sin(beta)];
    t = [-sin(beta) cos(beta)];
    n = -[cos(beta) sin(beta)];        % INWARD normal
    S = max(1, R);                     % coordinate scale seen by cpArc

    N = numel(hs);
    O1 = nan(1,N); O2 = O1;            % prescribed
    X1 = nan(1,N); X2 = O1;            % band, max over the captured cloud
    N1 = nan(1,N); N2 = O1;            % band, nearest captured point
    ncap = nan(1,N); nres = zeros(1,N);

    fprintf('\n=== kappa = %g  (R = %g) ===\n', kap, R);
    fprintf('    h      kappa h |  prescribed a=b=h        |  band (dx = h)\n');
    fprintf('                   |  d_k        d_k2         | ncap res  max d_k    max d_k2   near d_k2\n');
    fprintf('  -----------------------------------------------------------------------------------------\n');
    for i = 1:N
      h = hs(i);

      % ---- prescribed point: a = h, b = a ----
      x = v + h*t + h*n;
      [O1(i), O2(i)] = probeErrors(cpf, x, v, t);

      % ---- band points at dx = h ----
      Xall = captureBandPoints(v, R, cen, a1, a2, h, bw);
      ncap(i) = size(Xall,1);
      if ~isempty(Xall)
        % Each cloud point has its OWN noise floor ~ C eps S / a_m, and a_m can
        % be far below dx (a node may sit just inside the capture boundary
        % a = 0).  Such a point has a tiny true error and huge noise, so its
        % measurement is pure roundoff -- taking a max over the cloud without
        % excluding it reports the worst-conditioned node, not the worst error.
        % Keep only points that can resolve their own predicted signal.
        M = size(Xall,1);
        ea = nan(M,2); res = false(M,2);
        for m = 1:M
          am = (Xall(m,:) - v)*t(:);  bm = (Xall(m,:) - v)*n(:);
          [ea(m,1), ea(m,2)] = probeErrors(cpf, Xall(m,:), v, t);
          nz = 4*eps*S/max(am, realmin);
          res(m,1) = (kap/2)*abs(am)    > 10*nz;
          res(m,2) = 2*kap^2*abs(am*bm) > 10*nz;
        end
        g1 = res(:,1) & isfinite(ea(:,1));
        g2 = res(:,2) & isfinite(ea(:,2));
        if any(g1), X1(i) = max(ea(g1,1)); end
        if any(g2), X2(i) = max(ea(g2,2)); end
        nres(i) = nnz(g2); %#ok<AGROW>
        if res(1,2) && isfinite(ea(1,2)), N2(i) = ea(1,2); end  % sorted by |x-v|
        if res(1,1) && isfinite(ea(1,1)), N1(i) = ea(1,1); end
      end
      fprintf('  %8.2e  %7.1e | %.3e  %.3e  | %4d %3d  %.3e  %.3e  %.3e\n', ...
              h, kap*h, O1(i), O2(i), ncap(i), nres(i), X1(i), X2(i), N2(i));
    end

    reportOrders(hs, kap, S, O1, O2, X1, X2, N1, N2);
    makeFigure(kap, R, S, hs, O1, O2, X1, X2, N1, N2, ...
               v, t, n, cen, a1, a2, dxShow, Wshow, bw);
  end
  fprintf('\n');
end


%% ==== orders, over the window where the term is above the roundoff floor ===
function reportOrders(hs, kap, S, O1, O2, X1, X2, N1, N2)
  % in regime (kappa h small) and signal comfortably above the ~C eps S/h floor
  floor2 = 4*eps*S./hs;
  reg  = (kap*hs <= 0.1);
  us1  = reg & isfinite(O1) & O1 > 10*floor2;
  us2  = reg & isfinite(O2) & O2 > 10*floor2;
  fprintf('\n  fitted order (kappa h <= 0.1 and signal > 10x roundoff floor):\n');
  fprintf('    prescribed a=b=h    d_k %s   d_k2 %s\n', ...
          fitStr(hs, O1, us1), fitStr(hs, O2, us2));
  % X and N already carry a per-point resolvability filter; only the regime
  % condition remains
  fprintf('    band, max on cloud  d_k %s   d_k2 %s\n', ...
          fitStr(hs, X1, reg & isfinite(X1)), fitStr(hs, X2, reg & isfinite(X2)));
  fprintf('    band, nearest point d_k %s   d_k2 %s\n', ...
          fitStr(hs, N1, reg & isfinite(N1)), fitStr(hs, N2, reg & isfinite(N2)));
  % where does d_k2 stop improving?
  k = find(isfinite(O2), 1, 'last');
  [~, kmin] = min(O2);
  fprintf('  prescribed d_k2 is smallest at h = %.1e (err %.2e); at h = %.1e it is %.2e\n', ...
          hs(kmin), O2(kmin), hs(k), O2(k));
end

function s = fitStr(hs, e, use)
  if nnz(use) >= 2
    q = polyfit(log(hs(use)), log(e(use)), 1);
    s = sprintf('%6.3f (%d pts)', q(1), nnz(use));
  else
    s = '   n/a       ';
  end
end


%% ==== band construction ====================================================
% Window half-width 3*bw*dx about v: every captured point has |x-v| <= bw*dx,
% so the whole captured set is inside it.  Grid aligned to the global origin,
% so v is never a node.
function X = captureBandPoints(v, R, cen, a1, a2, dx, bw)
  Wl = 3*bw*dx;
  x1d = (floor((v(1)-Wl)/dx) : ceil((v(1)+Wl)/dx)) * dx;
  y1d = (floor((v(2)-Wl)/dx) : ceil((v(2)+Wl)/dx)) * dx;
  [xx, yy] = meshgrid(x1d, y1d);
  [cpx, cpy, dist, bdy] = cpArc(xx, yy, R, cen, a1, a2);
  tol = 1e-12*max(1, norm(v));
  cap = (abs(dist) <= bw*dx) & (bdy ~= 0) & ...
        (abs(cpx - v(1)) <= tol) & (abs(cpy - v(2)) <= tol);
  xs = xx(cap);  ys = yy(cap);
  if isempty(xs), X = []; return; end
  [~, ord] = sort((xs - v(1)).^2 + (ys - v(2)).^2);
  X = [xs(ord), ys(ord)];
end


%% ==== estimators ===========================================================
function [e1, e2] = probeErrors(cpf, x, v, t)
  [p0, bdy0] = cpf(x);
  if bdy0 == 0 || norm(p0 - v) > 1e-11*max(1, norm(v))
    e1 = NaN; e2 = NaN; return;
  end
  p1 = cpf(2*p0 - x);
  p2 = cpf(3*p0 - 2*x);
  d1 = p0 - p1;
  d2 = 1.5*p0 - 2*p1 + 0.5*p2;
  if norm(d1) == 0 || norm(d2) == 0, e1 = NaN; e2 = NaN; return; end
  e1 = norm(d1/norm(d1) - t);
  e2 = norm(d2/norm(d2) - t);
end

function [p, bdy] = arcCP(q, R, cen, a1, a2)
  [cx, cy, ~, bdy] = cpArc(q(1), q(2), R, cen, a1, a2);
  p = [cx cy];
end


%% ==== figure ===============================================================
function makeFigure(kap, R, S, hs, O1, O2, X1, X2, N1, N2, ...
                    v, t, n, cen, a1, a2, dxShow, W, bw)
  f = figure('Visible','off','Position',[100 100 1180 480]);

  % ---- left: the curve and its band, for context -------------------------
  subplot(1,2,1); hold on;
  x1d = (floor((v(1)-W)/dxShow) : ceil((v(1)+W)/dxShow)) * dxShow;
  y1d = (floor((v(2)-W)/dxShow) : ceil((v(2)+W)/dxShow)) * dxShow;
  [xx, yy] = meshgrid(x1d, y1d);
  [cpx, cpy, dist, bdy] = cpArc(xx, yy, R, cen, a1, a2);
  band = abs(dist) <= bw*dxShow;
  tol  = 1e-12*max(1, norm(v));
  cap  = band & (bdy ~= 0) & (abs(cpx-v(1)) <= tol) & (abs(cpy-v(2)) <= tol);

  th = linspace(a1, a2, 4000);
  P  = toLocal([cen(1)+R*cos(th(:)), cen(2)+R*sin(th(:))], v, t, n);
  Pb = toLocal([xx(band) yy(band)], v, t, n);
  Pc = toLocal([xx(cap)  yy(cap)],  v, t, n);
  plot(Pb(:,1), Pb(:,2), '.', 'Color',[0.75 0.75 0.75], 'MarkerSize',7);
  plot(P(:,1), P(:,2), '-', 'LineWidth',1.6, 'Color',[0 0.35 0.75]);
  plot(Pc(:,1), Pc(:,2), 'o', 'Color',[0.2 0.6 0.2], 'MarkerSize',5);
  % the prescribed ray a = b = h
  hh = logspace(-2.2, log10(0.25), 40);
  plot(hh, hh, '-', 'Color',[0.85 0.33 0.10], 'LineWidth',1.4);
  plot(0.1, 0.1, 'o', 'Color',[0.85 0.33 0.10], 'MarkerFaceColor',[0.85 0.33 0.10], ...
       'MarkerSize',6);
  plot(0, 0, 'kp', 'MarkerFaceColor','k', 'MarkerSize',13);
  axis equal; grid on; box on;
  xlim([-0.32 0.32]); ylim([-0.32 0.32]);
  xlabel('along t'); ylabel('along n  (inward)');
  title({sprintf('\\kappa = %g  (R = %g)', kap, R), ...
         sprintf('band drawn at dx = %g  (bw\\cdotdx = %.3g%% of R)', ...
                 dxShow, 100*bw*dxShow/R)}, 'FontSize',9);
  legend({'band','arc','cp(x) = v','prescribed ray b = a','x at h=0.1','vertex v'}, ...
         'Location','northwest','FontSize',7);

  % ---- right: error vs h, both point sets --------------------------------
  subplot(1,2,2); hold on;
  loglog(hs, O1, '-o',  'Color',[0 0.45 0.74],  'LineWidth',2.4, 'MarkerSize',6, ...
         'MarkerFaceColor',[0 0.45 0.74]);
  loglog(hs, O2, '-s',  'Color',[0.85 0.33 0.10],'LineWidth',2.4, 'MarkerSize',6, ...
         'MarkerFaceColor',[0.85 0.33 0.10]);
  loglog(hs, X1, '--^', 'Color',[0.30 0.70 0.95],'LineWidth',1.6, 'MarkerSize',5);
  loglog(hs, X2, '--v', 'Color',[0.95 0.65 0.30],'LineWidth',1.6, 'MarkerSize',5);
  loglog(hs, N2, ':d',  'Color',[0.55 0.35 0.25],'LineWidth',1.0, 'MarkerSize',4);
  loglog(hs, (kap/2)*hs,     'k:',  'LineWidth',1.1);
  loglog(hs, 2*kap^2*hs.^2,  'k-.', 'LineWidth',1.1);
  loglog(hs, 4*eps*S./hs,    'k--', 'LineWidth',1.4);
  set(gca,'XScale','log','YScale','log'); grid on; box on;
  xlim([min(hs)/2 max(hs)*2]);
  xticks(10.^(-8:1:-1));
  ylim([1e-13 10]);
  xlabel('h        (prescribed a = b = h;  band dx = h)');
  ylabel('|$\hat d - t$|','Interpreter','latex');
  title({'prescribed (solid) vs narrow band (dashed)', ...
         'd_{k2} turns up once 2\kappa^2h^2 drops below \epsilon R/h'}, 'FontSize',9);
  legend({'d_k    prescribed', 'd_{k2} prescribed', ...
          'd_k    band, max on cloud', 'd_{k2} band, max on cloud', ...
          'd_{k2} band, nearest pt', ...
          'theory (\kappa/2) h', 'theory 2\kappa^2 h^2', ...
          'roundoff  4\epsilon R / h'}, 'Location','eastoutside','FontSize',7);

  % figures go to ./figs beside this script (created on first run), not into
  % examples_2026 itself
  name   = sprintf('tangent2_combined_kappa_%g.png', kap);
  figdir = fullfile(fileparts(mfilename('fullpath')), 'figs');
  if ~exist(figdir, 'dir'), mkdir(figdir); end
  exportgraphics(f, fullfile(figdir, name), 'Resolution', 150); close(f);
  fprintf('  saved figs/%s\n', name);
end

function Q = toLocal(P, v, t, n)
  D = P - v;
  Q = [D*t(:), D*n(:)];
end
