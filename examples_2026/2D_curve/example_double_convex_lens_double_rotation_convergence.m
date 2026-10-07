function results = example_double_convex_lens_double_rotation_convergence(hvals, Rright_values, shift)
%EXAMPLE_DOUBLE_CONVEX_LENS_DOUBLE_ROTATION_CONVERGENCE Grid-seeded CP curvature glue.
%   RESULTS = EXAMPLE_DOUBLE_CONVEX_LENS_DOUBLE_ROTATION_CONVERGENCE(HVALS,
%   RRIGHT_VALUES, SHIFT) solves the same periodic u-u_ss=f lens problem as
%   example_double_convex_lens_curvature_convergence. Defaults are
%   HVALS=.02*2.^-(0:5), RRIGHT_VALUES=[1.9 1.95 2 2.05 2.1]/4,
%   SHIFT=[.317 .211]. Bands, endpoint rows and CP jets are shared by all arms:
%     rotation_exact : exact tangent rotation (control);
%     rotation_cp    : CP-estimated tangent rotation;
%     double_vertex  : second rotation about V, angle a*delta_k/2;
%     double_midpoint: second rotation about V+a*t/2, angle a*delta_k;
%     arclength      : exact circular continuation (control).
%   Here q1 is the first rotated grid node, t the target inward tangent,
%   a=(q1-V)*t', and delta_k=cross2(t,Ktarget-R*Ksource). A single constant
%   second angle cannot fit curvature: the angle must depend on each node.
%
%   Jets use endpoint-cap grid nodes to bootstrap reflected d_k2 tangents,
%   then CP probes at distances h,2h,3h along the estimated inward tangent.
%   This reuses the spatial CP jet estimator with zero z coordinates. These
%   are h-spaced CP evaluations, not exclusively existing Cartesian nodes.
%   No exact radius, center, tangent or arclength enters either CP rotation.
%   Analytic geometry is used only for the two controls and diagnostics.
%
%   The moving pivot corrects both bending and normal-offset transport:
%   q2-q1 = a^2*DeltaK/2 - a*(DeltaK.d)*t + O(h^3), d=q1-V-a*t.
%   The vertex pivot supplies only half the tangential term. We measure the
%   map defect as well as solution error; an improved map need not reduce
%   solution error or change its observed order. Outputs (.mat and per-radius
%   .png figures) are saved in examples_2026/figs/. Surface samples include
%   both junctions, and rates use the actual spacing ratios.

  if nargin < 1 || isempty(hvals), hvals = .02*2.^-(0:5); end
  if nargin < 2 || isempty(Rright_values), Rright_values = [1.9 1.95 2 2.05 2.1]/4; end
  if nargin < 3 || isempty(shift), shift = [.317 .211]; end
  hvals = hvals(:).'; Rright_values = Rright_values(:).';
  if numel(hvals) < 2 || any(~isfinite(hvals)) || any(hvals <= 0) || any(diff(hvals) >= 0)
    error('hvals must contain at least two finite positive decreasing spacings');
  end
  H = .4; Rleft = .5;
  if isempty(Rright_values) || any(~isfinite(Rright_values)) || any(Rright_values <= H)
    error('right radii must be finite and greater than H=0.4');
  end
  if numel(shift) ~= 2 || any(~isfinite(shift)), error('shift must have two finite entries'); end
  shift = reshape(shift,1,2);
  here = fileparts(mfilename('fullpath'));
  addpath(here);
  addpath(fullfile(here,'..','3D_curve'));
  addpath(fullfile(here,'..','..','cp_matrices'));
  addpath(fullfile(here,'..','..','surfaces'));
  old_checks = getenv('ICPM2009BANDINGCHECKS');
  cleanup_checks = onCleanup(@() setenv('ICPM2009BANDINGCHECKS',old_checks));
  setenv('ICPM2009BANDINGCHECKS','1');
  p = 3; order = 2; dim = 2;
  bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + (order/2+(p+1)/2)^2);
  modes = {'rotation_exact','rotation_cp','double_vertex','double_midpoint','arclength'};
  nh = numel(hvals); nm = numel(modes);
  figdir = fullfile(here,'..','figs');
  if ~exist(figdir,'dir'), mkdir(figdir); end
  for ir = 1:numel(Rright_values)
    geo = lens_geometry(H,Rleft,Rright_values(ir));
    omega = 2*pi/geo.total_length;
    ufun = @(s) sin(omega*s+.7) + .5*cos(2*omega*s-.3);
    ffun = @(s) (1+omega^2)*sin(omega*s+.7) + .5*(1+4*omega^2)*cos(2*omega*s-.3);
    r = struct('Rleft',Rleft,'Rright',Rright_values(ir),'delta_kappa',geo.delta_kappa, ...
        'h',hvals,'shift',shift,'modes',{modes},'error',nan(nh,nm), ...
        'map_defect',nan(nh,nm),'rowsum',nan(nh,nm),'residual',nan(nh,nm), ...
        'trace_jump',nan(nh,nm),'theta_max',zeros(nh,nm), ...
        'tangent_error',nan(nh,1),'curvature_error',nan(nh,1), ...
        'routed',zeros(nh,2,2),'jets',{cell(nh,1)},'angle_samples',{cell(nh,2,2)});
    fprintf('\nLens Rleft=%.6g, Rright=%.6g, |delta kappa|=%.6g, shift=[%.3f %.3f]\n', ...
        Rleft,r.Rright,r.delta_kappa,shift);
    fprintf('h          mode              surface_sup   rate    map_defect    theta_max    residual    rowsum\n');
    for ih = 1:nh
      h = hvals(ih); pad = (ceil(bw)+4)*h;
      x1d = (geo.bounds(1)-pad+shift(1)*h):h:(geo.bounds(2)+pad+shift(1)*h);
      y1d = (geo.bounds(3)-pad+shift(2)*h):h:(geo.bounds(4)+pad+shift(2)*h);
      [xx,yy] = meshgrid(x1d,y1d);
      br(1) = setup_branch(x1d,y1d,xx,yy,h,p,order,bw,geo.branches(1),geo.vertices);
      br(2) = setup_branch(x1d,y1d,xx,yy,h,p,order,bw,geo.branches(2),geo.vertices);
      jets = cell(2,2); te = 0; ke = 0;
      for ib = 1:2
        for iv = 1:2
          cap = br(ib).vid == iv;
          r.routed(ih,ib,iv) = nnz(cap);
          if ~any(cap), error('no endpoint rows for branch %d vertex %d',ib,iv); end
          nodes = [br(ib).xout(cap),br(ib).yout(cap),zeros(nnz(cap),1)];
          cpf = br(ib).geo.cpf;
          jets{ib,iv} = cp_curve_rotation_glue_3d('estimate', ...
              @(X) planar_cp(cpf,X),[geo.vertices(iv,:) 0],nodes,h);
          j = jets{ib,iv}; v = geo.vertices(iv,:);
          te = max(te,norm(j.tout(1:2)-br(ib).geo.tauout(iv,:)));
          exact_K = (br(ib).geo.cen-v)/br(ib).geo.R^2;
          ke = max(ke,norm(j.K(1:2)-exact_K));
        end
      end
      r.jets{ih} = jets; r.tangent_error(ih) = te; r.curvature_error(ih) = ke;
      for im = 1:nm
        out = solve_arm(geo,br,jets,x1d,y1d,p,modes{im},ufun,ffun);
        r.error(ih,im) = out.error; r.map_defect(ih,im) = out.map_defect;
        r.rowsum(ih,im) = out.rowsum; r.residual(ih,im) = out.residual;
        r.trace_jump(ih,im) = out.trace_jump; r.theta_max(ih,im) = out.theta_max;
        if strcmp(modes{im},'double_midpoint')
          r.angle_samples(ih,:,:) = reshape(out.angle_samples,[1 2 2]);
        end
        rate = NaN;
        if ih > 1, rate = log(r.error(ih-1,im)/out.error)/log(hvals(ih-1)/h); end
        fprintf('%.7g  %-17s %.6e  %5.2f  %.6e  %.3e  %.3e  %.3e\n', ...
            h,modes{im},out.error,rate,out.map_defect,out.theta_max,out.residual,out.rowsum);
      end
      fprintf('  cap rows=[%d %d;%d %d], tangent error=%.3e, curvature error=%.3e\n', ...
          reshape(squeeze(r.routed(ih,:,:)).',1,[]),te,ke);
    end
    r.rate = log(r.error(1:end-1,:)./r.error(2:end,:))./log(hvals(1:end-1)'./hvals(2:end)');
    r.map_rate = log(r.map_defect(1:end-1,:)./r.map_defect(2:end,:))./log(hvals(1:end-1)'./hvals(2:end)');
    r.fit = nan(1,nm); r.map_fit = nan(1,nm);
    tail = max(1,nh-2):nh;
    for im = 1:nm
      r.fit(im) = fitrate(hvals,r.error(:,im).');
      % Controls may reach roundoff in the intrinsic-map diagnostic.
      if all(r.map_defect(tail,im) > 1e-12*geo.total_length)
        q = polyfit(log(hvals(tail)),log(r.map_defect(tail,im).'),1); r.map_fit(im) = q(1);
      end
      fprintf('%-17s surface fit=%.4f, last-%d map fit=%.4f\n',modes{im},r.fit(im),numel(tail),r.map_fit(im));
    end
    if ir == 1, results = r; else, results(ir) = r; end
    stem = sprintf('double_convex_lens_double_rotation_Rright_%0.6g_shift_%0.6g_%0.6g',r.Rright,shift);
    plot_results(r,geo,fullfile(figdir,[stem '.png']));
  end
  save(fullfile(figdir,'double_convex_lens_double_rotation_convergence.mat'),'results');
end

function P = planar_cp(cpf,X)
% Lift the existing planar CP map to the spatial jet estimator's contract.
  [cx,cy] = cpf(X(:,1),X(:,2)); P = [cx,cy,zeros(size(cx))];
end

function geo = lens_geometry(H,Rleft,Rright)
  Vtop = [0 H]; Vbottom = [0 -H];
  cleft = sqrt(Rleft^2-H^2); cright = sqrt(Rright^2-H^2);
  left = make_branch(Rleft,[cleft 0],pi-asin(H/Rleft),-pi+asin(H/Rleft),0,true);
  right = make_branch(Rright,[-cright 0],-asin(H/Rright),asin(H/Rright),left.length,false);
  geo.vertices = [Vtop;Vbottom]; geo.branches = [left right];
  geo.total_length = left.length+right.length;
  geo.delta_kappa = abs(1/Rleft-1/Rright);
  geo.bounds = [cleft-Rleft,Rright-cright,-H,H];
end

function b = make_branch(R,cen,angle1,angle2,s0,starts_top)
  span = mod(angle2-angle1,2*pi);
  b.R = R; b.cen = cen; b.angle1 = angle1; b.s0 = s0; b.length = R*span;
  b.cpf = @(x,y) cpArc(x,y,R,cen,angle1,angle2);
  b.parfun = @(x,y) branch_arclength(x,y,b);
  b.pointfun = @(s) branch_points(s,b);
  tout_start = [sin(angle1),-cos(angle1)];
  tout_end = [-sin(angle1+span),cos(angle1+span)];
  if starts_top, b.tauout = [tout_start;tout_end]; else, b.tauout = [tout_end;tout_start]; end
end

function s = branch_arclength(x,y,b)
  theta = atan2(y-b.cen(2),x-b.cen(1));
  dtheta = mod(theta-b.angle1,2*pi);
  tol = 100*eps(max(1,max(abs([x(:);y(:);b.cen(:)]))));
  dtheta(abs(dtheta-2*pi) <= tol) = 0; dtheta(dtheta < tol) = 0;
  span = b.length/b.R; dtheta(abs(dtheta-span) <= tol) = span;
  if any(dtheta < -tol | dtheta > span+tol), error('closest point lies outside its branch'); end
  s = b.s0+b.R*min(max(dtheta,0),span);
end

function xy = branch_points(s,b)
  q = min(max(s,b.s0),b.s0+b.length); th = b.angle1+(q-b.s0)/b.R;
  xy = [b.cen(1)+b.R*cos(th),b.cen(2)+b.R*sin(th)];
end

function s = setup_branch(x1d,y1d,xx,yy,h,p,order,bw,branch,vertices)
% Same checked two-band construction as the original lens experiment.
  cand = (1:numel(xx))';
  [cpx,cpy,dist,bdy] = branch.cpf(xx(:),yy(:)); keep = abs(dist) <= bw*h;
  s.dx = h; s.band = cand(keep); s.xinit = xx(keep); s.yinit = yy(keep);
  s.cpxinit = cpx(keep); s.cpyinit = cpy(keep); s.bdyinit = bdy(keep);
  [Ei,Ej,Es] = interp2_matrix(x1d,y1d,s.cpxinit,s.cpyinit,p);
  s.innerband = unique(Ej); s.nin = numel(s.innerband);
  s.inv_inner = make_invbandmap(numel(xx),s.innerband);
  Einit = sparse(Ei,s.inv_inner(Ej),Es,numel(s.band),s.nin);
  Ltemp = laplacian_2d_matrix(x1d,y1d,order,s.innerband,s.band);
  if nnz(Ltemp) ~= 5*s.nin, error('the Laplacian stencil leaves the initial band'); end
  [~,jj] = find(Ltemp); outertemp = unique(jj);
  [tf,loc] = ismember(s.innerband,s.band(outertemp));
  if ~all(tf), error('the inner band is not contained in the outer band'); end
  s.nout = numel(outertemp); s.L = Ltemp(:,outertemp); s.E = Einit(outertemp,:);
  s.R = sparse(1:s.nin,loc,1,s.nin,s.nout);
  s.xout = s.xinit(outertemp); s.yout = s.yinit(outertemp);
  s.cpxout = s.cpxinit(outertemp); s.cpyout = s.cpyinit(outertemp); s.geo = branch;
  tol = 100*eps(max(1,max(abs(vertices(:))))); s.vid = zeros(s.nout,1);
  for iv = 1:2
    s.vid(hypot(s.cpxout-vertices(iv,1),s.cpyout-vertices(iv,2)) <= tol) = iv;
  end
  if any(s.bdyinit(outertemp) ~= 0 & s.vid == 0), error('an endpoint does not match a lens vertex'); end
end

function out = solve_arm(geo,br,jets,x1d,y1d,p,mode,ufun,ffun)
  Eb = {br(1).E,sparse(br(1).nout,br(2).nin);sparse(br(2).nout,br(1).nin),br(2).E};
  ts = {br(1).geo.parfun(br(1).cpxout,br(1).cpyout),br(2).geo.parfun(br(2).cpxout,br(2).cpyout)};
  out.map_defect = 0; out.theta_max = 0; out.angle_samples = cell(2,2);
  for source = 1:2
    target = 3-source; rows = find(br(source).vid ~= 0); vids = br(source).vid(rows);
    X = [br(source).xout(rows),br(source).yout(rows)]; qmap = zeros(size(X));
    for iv = 1:2
      use = vids == iv; v = geo.vertices(iv,:);
      oracle_q = map_by_arclength(X(use,:),v,br(source).geo,br(target).geo, ...
          br(source).geo.tauout(iv,:),-br(target).geo.tauout(iv,:));
      if strcmp(mode,'arclength')
        qmap(use,:) = oracle_q;
      elseif strcmp(mode,'rotation_exact')
        alpha = glue_angle(br(source).geo.tauout(iv,:),br(target).geo.tauout(iv,:));
        qmap(use,:) = rotate_about(X(use,:),v,alpha);
      else
        [qmap(use,:),a,beta,dk] = map_cp_nodes(X(use,:),v,jets{source,iv},jets{target,iv},mode);
        out.theta_max = max(out.theta_max,max(abs(beta)));
        out.angle_samples{source,iv} = struct('a',a,'theta',beta,'delta_k',dk);
      end
      [cx,cy] = br(target).geo.cpf(qmap(use,1),qmap(use,2));
      mapped_s = br(target).geo.parfun(cx,cy);
      oracle_s = br(target).geo.parfun(oracle_q(:,1),oracle_q(:,2));
      out.map_defect = max(out.map_defect,max(abs(mapped_s-oracle_s)));
    end
    [cx,cy] = br(target).geo.cpf(qmap(:,1),qmap(:,2));
    [Ei,Ej,Es] = interp2_matrix(x1d,y1d,cx,cy,p); jj = br(target).inv_inner(Ej);
    if any(jj == 0), error('a cross-branch interpolation stencil leaves the inner band'); end
    Eb{source,target} = sparse(rows(Ei),jj,Es,br(source).nout,br(target).nin);
    Eself = Eb{source,source}; Eself(rows,:) = 0; Eb{source,source} = Eself;
    ts{source}(rows) = br(target).geo.parfun(cx,cy);
  end
  Eblk = [Eb{1,1} Eb{1,2};Eb{2,1} Eb{2,2}];
  out.rowsum = max(abs(full(sum(Eblk,2))-1));
  if out.rowsum > 1e-10, error('extension row-sum defect is %g',out.rowsum); end
  M = lapsharp_unordered(blkdiag(br(1).L,br(2).L),Eblk,blkdiag(br(1).R,br(2).R));
  rhs = [ffun(br(1).R*ts{1});ffun(br(2).R*ts{2})]; A = speye(size(M,1))-M; u = A\rhs;
  if any(~isfinite(u)), error('the elliptic solve returned non-finite values'); end
  out.residual = norm(A*u-rhs,inf)/max(1,norm(rhs,inf));
  if out.residual > 1e-7, error('relative linear residual is %g',out.residual); end
  ubr = {u(1:br(1).nin),u(br(1).nin+(1:br(2).nin))};
  out.error = 0; traces = zeros(2,2);
  for ib = 1:2
    nq = max(400,ceil(4*br(ib).geo.length/br(ib).dx));
    s = linspace(br(ib).geo.s0,br(ib).geo.s0+br(ib).geo.length,nq+1)';
    xy = br(ib).geo.pointfun(s);
    values = surface_values(x1d,y1d,p,br(ib),ubr{ib},xy);
    out.error = max(out.error,max(abs(values-ufun(s))));
    traces(ib,:) = surface_values(x1d,y1d,p,br(ib),ubr{ib},geo.vertices).';
  end
  out.trace_jump = max(abs(traces(1,:)-traces(2,:)));
  if ~isfinite(out.error) || ~isfinite(out.map_defect), error('non-finite error diagnostic'); end
end

function [q,a,beta,dk] = map_cp_nodes(X,v,sJ,tJ,mode)
% Planar specialization of cp_curve_rotation_glue_3d's map, using its jets.
  alpha = glue_angle(sJ.tout(1:2),tJ.tout(1:2));
  q1 = rotate_about(X,v,alpha); t = -tJ.tout(1:2);
  RK = rotate_about(sJ.K(1:2),[0 0],alpha);
  DK = tJ.K(1:2)-RK; DK = DK-(DK*t.')*t;
  dk = t(1)*DK(2)-t(2)*DK(1); a = (q1-v)*t.';
  switch mode
    case 'rotation_cp', beta = zeros(size(a)); q = q1;
    case 'double_vertex', beta = .5*a*dk; q = rotate_about(q1,v,beta);
    case 'double_midpoint', beta = a*dk; q = rotate_about(q1,v+.5*a.*t,beta);
    otherwise, error('unknown CP rotation mode %s',mode);
  end
end

function angle = glue_angle(ts,tt)
  angle = atan2(-tt(2),-tt(1))-atan2(ts(2),ts(1));
  angle = atan2(sin(angle),cos(angle));
end

function q = rotate_about(x,v,angle)
% angle and v may have one row or one row per node.
  c = cos(angle); s = sin(angle); d = x-v;
  q = v+[c.*d(:,1)-s.*d(:,2),s.*d(:,1)+c.*d(:,2)];
end

function q = map_by_arclength(xq,v,source,target,tauout_source,tauin_target)
  rv = (v-source.cen)/source.R; rq = xq-source.cen; nrq = hypot(rq(:,1),rq(:,2));
  if any(nrq == 0), error('a source grid point equals the circle centre'); end
  rq = rq./nrq;
  angle = atan2(rv(1)*rq(:,2)-rv(2)*rq(:,1),rq*rv.');
  source_sign = sign(dot(tauout_source,[-rv(2),rv(1)])); xi = source.R*source_sign*angle;
  tol = 100*eps(max(1,max(abs([xq(:);v(:)]))));
  if any(xi < -tol), error('circular continuation produced a negative endpoint offset'); end
  xi(xi < 0) = 0; rvt = (v-target.cen)/target.R;
  target_sign = sign(dot(tauin_target,[-rvt(2),rvt(1)])); alpha = target_sign*xi/target.R;
  q = target.cen+target.R*[cos(alpha)*rvt(1)-sin(alpha)*rvt(2), ...
                          sin(alpha)*rvt(1)+cos(alpha)*rvt(2)];
end

function values = surface_values(x1d,y1d,p,br,u,xy)
  [Ei,Ej,Es] = interp2_matrix(x1d,y1d,xy(:,1),xy(:,2),p); jj = br.inv_inner(Ej);
  if any(jj == 0), error('a surface sample stencil leaves the inner band'); end
  values = sparse(Ei,jj,Es,size(xy,1),br.nin)*u;
  if any(~isfinite(values)), error('non-finite surface interpolation'); end
end

function rate = fitrate(h,e)
% Keep the original lens experiment's pre-floor fitting policy.
  [~,imin] = min(e); cutoff = imin;
  if imin ~= numel(e), cutoff = imin-1; end
  use = find(isfinite(e) & e > 0 & (1:numel(e)) <= max(cutoff,2));
  if numel(use) < 2, rate = NaN; else, q = polyfit(log(h(use)),log(e(use)),1); rate = q(1); end
end

function plot_results(r,geo,filename)
  fig = figure('Color','w','Position',[100 100 1200 850]);
  colors = lines(numel(r.modes));
  subplot(2,2,1); hold on; axis equal; box on;
  for ib = 1:2
    b = geo.branches(ib); xy = b.pointfun(linspace(b.s0,b.s0+b.length,500)');
    plot(xy(:,1),xy(:,2),'LineWidth',1.5);
  end
  plot(geo.vertices(:,1),geo.vertices(:,2),'k.','MarkerSize',15);
  title(sprintf('Lens: R_L=%.3g, R_R=%.3g, |Delta k|=%.3g',r.Rleft,r.Rright,r.delta_kappa));
  xlabel('x'); ylabel('y');
  subplot(2,2,2); hold on; box on;
  for im = 1:numel(r.modes)
    loglog(r.h,r.error(:,im),'o-','Color',colors(im,:), ...
        'DisplayName',sprintf('%s (%.2f)',strrep(r.modes{im},'_',' '),r.fit(im)));
  end
  loglog(r.h,r.error(1,4)*(r.h/r.h(1)).^2,'k:','DisplayName','O(h^2)');
  set(gca,'XScale','log','YScale','log'); title('Surface sup error (including junctions)');
  xlabel('h'); ylabel('error'); legend('Location','northwest');
  subplot(2,2,3); hold on; box on;
  for im = 1:4
    loglog(r.h,r.map_defect(:,im),'o-','Color',colors(im,:), ...
        'DisplayName',sprintf('%s (%.2f)',strrep(r.modes{im},'_',' '),r.map_fit(im)));
  end
  loglog(r.h,r.map_defect(1,4)*(r.h/r.h(1)).^3,'k:','DisplayName','O(h^3)');
  set(gca,'XScale','log','YScale','log'); title('CP correspondence arclength defect');
  xlabel('h'); ylabel('max |s_{mapped}-s_{oracle}|'); legend('Location','northwest');
  subplot(2,2,4); hold on; box on;
  for ib = 1:2
    for iv = 1:2
      sample = r.angle_samples{1,ib,iv}; [a,idx] = sort(sample.a);
      plot(a,sample.theta(idx),'o-', ...
          'DisplayName',sprintf('%d to %d, vertex %d, delta k=%.3g',ib,3-ib,iv,sample.delta_k));
    end
  end
  title(sprintf('Grid-derived midpoint second angles, h=%.4g',r.h(1)));
  xlabel('axial offset a'); ylabel('second angle (radians)'); legend('Location','best');
  exportgraphics(fig,filename,'Resolution',180);
  fprintf('saved %s\n',filename);
end
