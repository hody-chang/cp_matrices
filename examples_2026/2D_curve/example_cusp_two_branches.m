%% iCPM on y^2 = x^3 - x^4: two branches and two Cartesian grids
%
% Solve $u-u_{ss}=f$ on the intrinsic closed loop, with continuity of u and
% conservation of outward arclength flux at (0,0) and (1,0).  Top and bottom
% have independent grid unknowns, even where their embedding bands overlap.
% Only endpoint extension rows couple the branches; there is no whole-curve
% closest-point map, shared-node averaging, or uncut-system baseline.
%
% Compare endpoint angle rotation using d_{k2} and exact tangents.  The exact
% rotation is pi at the cusp and zero at the smooth right join.  Each branch
% has |kappa| ~ 3/(4 sqrt(x)) at the cusp: neither a positive uniform tubular
% radius nor the usual smooth-endpoint second-order argument applies there.
% d_{k2} names the estimator, not a promised convergence order at this cusp.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..', 'cp_matrices'));
addpath(fullfile(here, '..', 'rotations'));

%% Geometry and intrinsic orientation
% With 0 <= t <= pi/2, x=sin(t)^2 and y=+/-sin(t)^3*cos(t).
% Top runs cusp -> right join; bottom runs right join -> cusp in global s.
% The speed in t is smooth, including its zero at the cusp.  Integrating
% this speed avoids the singular graph derivative at x=1.
vertices = [0 0; 1 0];
speed = @(t) sin(t).*sqrt(4*cos(t).^2 + ...
    sin(t).^2.*(3*cos(t).^2-sin(t).^2).^2);
arc = ode113(@(t,s) speed(t), [0 pi/2], 0, ...
    odeset('RelTol', 1e-12, 'AbsTol', 1e-14));
ell = deval(arc, pi/2);
geo = cell(1,2);
for ib = 1:2
  sg = 3-2*ib;
  geo{ib}.cpf = @(x,y) cusp_cp(x,y,sg);
  geo{ib}.parfun = @(x,y) cusp_arclength(x,y,ib,arc,ell);
end
% Rows are cusp, right join; tangents point OUT of each branch.
geo{1}.tauout = [-1 0; 0 -1];
geo{2}.tauout = [-1 0; 0 1];

%% Manufactured periodic solution and analytic forcing
% The nonzero derivative at the cusp exercises flux transfer, rather than
% choosing an even solution that hides coupling errors at the singularity.
omega = pi/ell;
ufun = @(s) sin(omega*s + 0.7) + 0.5*cos(2*omega*s - 0.3);
ffun = @(s) (1+omega^2)*sin(omega*s + 0.7) + ...
    0.5*(1+4*omega^2)*cos(2*omega*s - 0.3);

%% Separate embedding grids and two-band discretization
hvals = logspace(log10(0.04), log10(0.0025), 20);
p = 3;
order = 2;
bw = 1.0002*sqrt(((p+1)/2)^2 + (order/2+(p+1)/2)^2);
old_banding_checks = getenv('ICPM2009BANDINGCHECKS');
cleanup_banding_checks = onCleanup(@() setenv('ICPM2009BANDINGCHECKS', old_banding_checks)); %#ok<NASGU>
setenv('ICPM2009BANDINGCHECKS', '1');
labels = {'d_{k2}', 'exact rotation'};
errors = nan(numel(hvals),2);
results = struct([]);
fprintf('Cusp loop: branch length = %.15g; total length = %.15g\n', ell, 2*ell);
fprintf('       h          d_{k2} error    rate       exact R error   rate\n');
for ih = 1:numel(hvals)
  h = hvals(ih);
  pad = (ceil(bw)+4)*h;
  ymax = 3*sqrt(3)/16;
  br = cell(1,2);
  for ib = 1:2
    % Separate full grids: top and bottom have different y extents and
    % different subcell offsets.  Every interpolant uses its TARGET grid.
    x1d = (-pad + (0.217+0.1*ib)*h):h:(1+pad);
    if ib == 1
      y1d = (-pad+0.211*h):h:(ymax+pad);
    else
      y1d = (-ymax-pad+0.373*h):h:pad;
    end
    [xx,yy] = meshgrid(x1d,y1d);
    br{ib} = setup_branch(x1d,y1d,xx,yy,h,p,order,bw,geo{ib},vertices);
    br{ib}.x1d = x1d; br{ib}.y1d = y1d;
  end

  %% Endpoint extension and stabilized elliptic solves
  % Reuse each grid/band for both rotations so only the tangent data change.
  for ivar = 1:2
    [out,ubr] = solve_rotation(br,vertices,p,ivar == 1,ffun);
    branch_error = zeros(1,2);
    for ib = 1:2
      % Parameter samples include both joins and cluster in x near the cusp.
      t = linspace(0,pi/2,max(801,ceil(8*ell/h)))';
      xq = sin(t).^2; yq = (3-2*ib)*sin(t).^3.*cos(t);
      xq([1 end]) = [0;1]; yq([1 end]) = 0;
      Eq = branch_interp(br{ib},xq,yq,p);
      s = cusp_arclength(xq,yq,ib,arc,ell);
      branch_error(ib) = max(abs(Eq*ubr{ib}-ufun(s)));
    end
    errors(ih,ivar) = max(branch_error);
    results(ih,ivar).method = labels{ivar};
    results(ih,ivar).h = h;
    results(ih,ivar).branch_error = branch_error;
    results(ih,ivar).errsurf = errors(ih,ivar);
    results(ih,ivar).theta = out.theta;
    results(ih,ivar).ncross = out.ncross;
    results(ih,ivar).rowsum = out.rowsum;
    results(ih,ivar).residual = out.residual;
    results(ih,ivar).ndof = [br{1}.nin br{2}.nin];
  end

  %% Surface L-infinity refinement result
  rate = [NaN NaN];
  if ih > 1
    rate = log(errors(ih-1,:)./errors(ih,:))/log(hvals(ih-1)/h);
  end
  fprintf('%12.5e  %12.5e  %7.4f   %12.5e  %7.4f\n', ...
      h,errors(ih,1),rate(1),errors(ih,2),rate(2));
end
% hvals, errors, and results remain in the workspace for inspection.
% MATLAB R2026a, 20 logarithmic levels: finest errors are 1.98010e-4
% (d_{k2}) and 1.04902e-3 (exact R).  Adjacent-level rates fluctuate;
% d_{k2} is not monotone over this range, whereas exact-R errors decrease.
% The cusp has unbounded curvature: assess the measured rates rather than
% assuming second order.  Exact tangents need not minimize discretization error.

%% Requested visualization
figdir = fullfile(fileparts(mfilename('fullpath')), '..', 'figs');
if ~exist(figdir, 'dir'), mkdir(figdir); end
fig = figure('Color', 'w', 'Position', [100 100 1100 460]);
colors = lines(2);
subplot(1,2,1); hold on; axis equal; box on;
tplot = linspace(0,pi/2,500)';
xplot = sin(tplot).^2;
yplot = sin(tplot).^3.*cos(tplot);
plot(xplot, yplot, 'Color', colors(1,:), 'LineWidth', 1.6, ...
    'DisplayName', 'Top branch');
plot(xplot, -yplot, 'Color', colors(2,:), 'LineWidth', 1.6, ...
    'DisplayName', 'Bottom branch');
plot(vertices(:,1), vertices(:,2), 'k.', 'MarkerSize', 15, ...
    'HandleVisibility', 'off');
text(0, -0.04, 'cusp', 'HorizontalAlignment', 'center');
text(1, -0.04, 'smooth join', 'HorizontalAlignment', 'center');
xlim([-0.15 1.15]); ylim([-0.45 0.45]);
title('y^2 = x^3 - x^4: two branches');
xlabel('x'); ylabel('y'); legend('Location', 'northwest');

subplot(1,2,2); hold on; box on;
set(gca, 'XScale', 'log', 'YScale', 'log');
% Fit every configured level; these are overall slopes, not claims of an
% asymptotic order at the cusp.  The markers retain nonmonotone behavior.
fitted_rates = zeros(1,2);
for ivar = 1:2
  fit = polyfit(log(hvals(:)), log(errors(:,ivar)), 1);
  fitted_rates(ivar) = fit(1);
  label = sprintf('%s, fitted slope = %.2f', labels{ivar}, fit(1));
  loglog(hvals, errors(:,ivar), 'o-', 'Color', colors(ivar,:), ...
      'LineWidth', 1.5, 'DisplayName', label);
end
loglog(hvals, errors(1,2)*(hvals/hvals(1)), 'k:', ...
    'DisplayName', 'O(h)');
loglog(hvals, errors(1,2)*(hvals/hvals(1)).^2, 'k--', ...
    'DisplayName', 'O(h^2)');
title('Surface L_\infty error'); xlabel('h'); ylabel('error');
legend('Location', 'southeast', 'FontSize', 9); grid on;
exportgraphics(fig, fullfile(figdir, 'cusp_two_branches_convergence.png'), ...
    'Resolution', 180);

function [cpx,cpy,dist,bdy] = cusp_cp(x,y,sg)
%CUSP_CP Global closest points on top (sg=1) or bottom (sg=-1).
% Inputs have a common shape.  Endpoint ids 1/2 mean cusp/right join.
% Use r=tan(t/2) in [0,1]: X=4*r^2/(1+r^2)^2 and
% Y=sg*8*r^3*(1-r^2)/(1+r^2)^4.  The unsquared stationary
% distance polynomial, after removing the endpoint factor 8*r, is
% (4*r^2-x*d^2)*(1-r^2)*d^4
%   +(8*r^3*(1-r^2)-sg*y*d^4)*r*(3-10*r^2+3*r^4), d=1+r^2.
% Its coefficients are p0+x*px+sg*y*py below, in descending powers.
% Compare all real interior stationary points and both endpoints.  This
% avoids both single-start minimization and artificial double roots from
% squaring the graph equation when a query lies on the x axis.
  p0 = [0 0 -4 0 -36 0 96 0 -96 0 36 0 4 0 0];
  px = [1 0 5 0 9 0 5 0 -5 0 -9 0 -5 0 -1];
  py = [0 -3 0 -2 0 19 0 36 0 19 0 -2 0 -3 0];
  shape = size(x); x = x(:); y = y(:);
  cpx = zeros(size(x)); cpy = cpx; bdy = cpx; dist = cpx;
  for k = 1:numel(x)
    r = roots(p0+x(k)*px+sg*y(k)*py);
    use = abs(imag(r)) <= 1e-10 & real(r) > 0 & real(r) < 1;
    r = [0;1;real(r(use))];
    z = 4*r.^2./(1+r.^2).^2;
    w = sg*8*r.^3.*(1-r.^2)./(1+r.^2).^4;
    d2 = (z-x(k)).^2+(w-y(k)).^2;
    [dmin,j] = min(d2);
    cpx(k) = z(j); cpy(k) = w(j); dist(k) = sqrt(dmin);
    if j <= 2, bdy(k) = j; end
  end
  cpx = reshape(cpx,shape); cpy = reshape(cpy,shape);
  dist = reshape(dist,shape); bdy = reshape(bdy,shape);
end

function s = cusp_arclength(x,y,ib,arc,ell)
%CUSP_ARCLENGTH Global oriented arclength of branch closest points (x,y).
% Top uses s=a(t), bottom uses s=2*ell-a(t).  Recover t from both
% coordinates to avoid cancellation in 1-x near the vertical tangent.
  t = atan2(x(:).^2,abs(y(:)));
  a = deval(arc,t).';
  if ib == 1, s = a; else, s = 2*ell-a; end
  s = reshape(s,size(x));
end

function E = branch_interp(br,x,y,p)
%BRANCH_INTERP Interpolate this branch's independent inner unknowns at x,y.
% Full-grid columns must map into this branch's inner band without omission.
  [ii,jj,v] = interp2_matrix(br.x1d,br.y1d,x,y,p);
  cols = br.inv_inner(jj);
  if any(cols == 0), error('interpolation stencil leaves the target inner band'); end
  E = sparse(ii,cols,v,numel(x),br.nin);
end

function [out,ubr] = solve_rotation(br,vertices,p,estimated,ffun)
%SOLVE_ROTATION Couple two branch grids by endpoint rotation and solve u-Mu=f.
% estimated selects d_{k2}; otherwise use analytic outward tangent limits.
  Eb = {br{1}.E,sparse(br{1}.nout,br{2}.nin); ...
        sparse(br{2}.nout,br{1}.nin),br{2}.E};
  ts = cell(1,2); tau = cell(1,2);
  for ib = 1:2
    ts{ib} = br{ib}.geo.parfun(br{ib}.cpxout,br{ib}.cpyout);
    tau{ib} = br{ib}.geo.tauout;
    if estimated
      for iv = 1:2
        tau{ib}(iv,:) = cp_tangent_dk2(br{ib},vertices(iv,:));
      end
    end
  end
  out.theta = zeros(2,2); out.ncross = zeros(2,2);
  for source = 1:2
    target = 3-source;
    for iv = 1:2
      rows = find(br{source}.vid == iv);
      if isempty(rows), error('no endpoint rows for branch %d, vertex %d',source,iv); end
      out.ncross(source,iv) = numel(rows);
      % Rodrigues' formula about the out-of-plane axis gives the rotation
      % as a (cos, sin) pair straight from the two tangents; it is applied
      % as built, and the angle below is recorded for reporting only.
      [cc,ss] = glue_rot2d(tau{source}(iv,:),tau{target}(iv,:));
      out.theta(source,iv) = atan2(ss,cc);
      q = rotate_about2d([br{source}.xout(rows),br{source}.yout(rows)], ...
                         vertices(iv,:),cc,ss);
      [cx,cy] = br{target}.geo.cpf(q(:,1),q(:,2));
      Eb{source,target}(rows,:) = branch_interp(br{target},cx,cy,p);
      Eb{source,source}(rows,:) = 0;
      % Routed unknowns represent target CP values, including in the forcing.
      ts{source}(rows) = br{target}.geo.parfun(cx,cy);
    end
  end
  E = [Eb{1,1} Eb{1,2};Eb{2,1} Eb{2,2}];
  L = blkdiag(br{1}.L,br{2}.L); R = blkdiag(br{1}.R,br{2}.R);
  out.rowsum = max(abs(full(sum(E,2))-1));
  if out.rowsum > 1e-10, error('extension does not preserve constants'); end
  M = lapsharp_unordered(L,E,R);
  rhs = [ffun(br{1}.R*ts{1});ffun(br{2}.R*ts{2})];
  A = speye(size(M,1))-M;
  u = A\rhs;
  out.residual = norm(A*u-rhs,inf)/max(1,norm(rhs,inf));
  if any(~isfinite(u)), error('non-finite elliptic solution'); end
  ubr = {u(1:br{1}.nin),u(br{1}.nin+(1:br{2}.nin))};
end

function s = setup_branch(x1d, y1d, xx, yy, h, p, order, bw, branch, vertices)
%SETUP_BRANCH Build checked initial, inner, and outer bands for one cusp branch.
% xx/yy are Cartesian grids; the returned maps index the branch inner unknowns.
  switch order
    case 2, sten = 5;
    otherwise, error('order %d not implemented', order);
  end
  cand = (1:numel(xx))';
  [cpx, cpy, dist, bdy] = branch.cpf(xx(:), yy(:));
  keep = abs(dist) <= bw*h;
  s.dx = h;
  s.band = cand(keep); s.xinit = xx(keep); s.yinit = yy(keep);
  s.cpxinit = cpx(keep); s.cpyinit = cpy(keep); s.bdyinit = bdy(keep);
  [Ei, Ej, Es] = interp2_matrix(x1d, y1d, s.cpxinit, s.cpyinit, p);
  s.innerband = unique(Ej); s.nin = numel(s.innerband);
  s.inv_inner = make_invbandmap(numel(xx), s.innerband);
  Einit = sparse(Ei, s.inv_inner(Ej), Es, numel(s.band), s.nin);
  Ltemp = laplacian_2d_matrix(x1d, y1d, order, s.innerband, s.band);
  if nnz(Ltemp) ~= sten*s.nin, error('the Laplacian stencil leaves the initial band'); end
  [~, jj] = find(Ltemp); outertemp = unique(jj);
  [tf, loc] = ismember(s.innerband, s.band(outertemp));
  if ~all(tf), error('the inner band is not contained in the outer band'); end
  s.outerband = s.band(outertemp); s.nout = numel(outertemp);
  s.L = Ltemp(:, outertemp); s.E = Einit(outertemp,:);
  s.R = sparse(1:s.nin, loc, 1, s.nin, s.nout);
  s.xout = s.xinit(outertemp); s.yout = s.yinit(outertemp);
  s.cpxout = s.cpxinit(outertemp); s.cpyout = s.cpyinit(outertemp);
  s.bdyout = s.bdyinit(outertemp); s.geo = branch;
  tol = 100*eps(max(1, max(abs(vertices(:)))));
  s.vid = zeros(s.nout,1);
  for iv = 1:2
    s.vid(hypot(s.cpxout - vertices(iv,1), s.cpyout - vertices(iv,2)) <= tol) = iv;
  end
  if any((s.bdyout ~= 0) & (s.vid == 0))
    error('an endpoint closest point does not match a curve vertex');
  end
end

function tau = cp_tangent_dk2(br, v)
%CP_TANGENT_DK2 Estimate an outward endpoint tangent from second-order CP differences.
% br is one branch band and v is one of its exact endpoint coordinates.
  tol = 100*eps(max(1, max(abs(v))));
  m = abs(br.cpxout - v(1)) <= tol & abs(br.cpyout - v(2)) <= tol;
  x = br.xout(m); y = br.yout(m);
  cx = br.cpxout(m); cy = br.cpyout(m);
  if isempty(x), error('no grid point has this endpoint as closest point'); end
  [b1x, b1y] = br.geo.cpf(2*cx - x, 2*cy - y);
  [b2x, b2y] = br.geo.cpf(3*cx - 2*x, 3*cy - 2*y);
  dvx = 1.5*cx - 2*b1x + 0.5*b2x;
  dvy = 1.5*cy - 2*b1y + 0.5*b2y;
  nrm = hypot(dvx, dvy);
  good = isfinite(nrm) & nrm > tol;
  if ~any(good), error('d_{k2} gave no usable endpoint tangent'); end
  tau = [mean(dvx(good)./nrm(good)), mean(dvy(good)./nrm(good))];
  tau = tau/norm(tau);
end

