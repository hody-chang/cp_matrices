%% Optimized Schwarz DDM on a cut ellipse: separate grids, rotation glue
%
% The surface intrinsic positive Helmholtz equation on a closed 1-manifold,
%
%     $$ (c - \Delta_S)\, u = f, \qquad c = 1, $$
%
% solved by the closest point method with the domain decomposed into N
% OVERLAPPING subdomains and iterated with Robin transmission conditions, after
%
%   A. Yazdani, R. D. Haynes, S. J. Ruuth, "Optimized Schwarz domain
%   decomposition algorithms for the closest point method on closed
%   manifolds", Numer. Algorithms 101 (2024) 33-64.
%
% Two departures from that paper, and one consequence of them.
%
% SEPARATE GRIDS.  Each subdomain carries its own Cartesian grid, with its own
% bounding box and its own subcell offset, so no node of one subdomain is a
% node of another.  The paper partitions a single global mesh with METIS and
% iterates in restricted-additive form on one global residual; that is not
% available here, so this script runs the direct iteration, their eq. (12) with
% the method selector of their eq. (13).  The transmission is then an honest
% interpolation between two independent discretizations rather than a
% shared-node identity.
%
% ROTATION GLUE.  A row clamped to an interface is routed across it by a rigid
% rotation about the interface point, after which the neighbour's closest point
% map is applied to the rotated node -- the technique the rest of this example
% family uses at a singular join.  The rotation angle is either estimated from
% closest-point differences (the second-order d_k2 estimator) or taken from the
% analytic tangents.
%
% TWO MANIFOLDS, chosen by the flag "flipped":
%
%   plain     the ellipse.  The arcs join smoothly at the cut points, the exact
%             glue rotation is the identity, and any rotation an estimator
%             produces is pure error.
%   flipped   the arc below the cut is reflected UP across the cut line, so the
%             two arcs meet at a genuine corner and the exact rotation is a
%             definite nonzero angle.
%
% The reflection is an isometry, so the two curves are the SAME 1-manifold:
% same length L, same arclength parameter, same subdomain lengths and overlaps,
% and therefore the SAME theoretical convergence factor.  Any difference in the
% observed rate between them is attributable to the embedding and the rotation
% glue, not to the decomposition.  That is the point of running both.
%
% INTERFACES AT THE CORNERS.  The disjoint subdomain boundaries are placed on
% the two cut points, so a corner is always an interface and never falls inside
% a subdomain where that subdomain's Laplacian would see a non-smooth curve.
%
% The decomposition therefore does NOT overlap.  With both interfaces at the
% cut points there is nowhere to overlap without a subdomain straddling a
% corner, and an arc that straddled one would need both embeddings at once,
% which a single closest-point map cannot provide.  Their Definition 2 allows
% this, requiring only delta_j >= 0, and the paper notes that alpha = 1 remains
% optimal even for a non-overlapping partitioning.  So each subdomain is one
% branch of the manifold and the two are coupled by the rotation glue alone.
%
% The partition is unequal and non-overlapping, so the closed form of their
% Corollary 4 does not apply and the contraction factor is taken as
% rho(M_dagger) from their eqs. (23)-(24).  At alpha = sqrt(c) the factor
% alpha^2 - c vanishes, M_dagger becomes two disjoint N-cycles, the overlaps
% telescope away and rho = exp(-sqrt(c)*L/N) for ANY partition and overlaps --
% the general-partition case of their Corollary 4, which is stated there only
% for equal-sized subdomains.  The script asserts this on the partition it
% uses.  All of this is SOLVER diagnostics: it governs how fast the iteration
% reaches the fixed point, not what the fixed point is.  The primary outputs
% of this example are the surface accuracy, its dependence on the curvature
% mismatch, the exact-versus-estimated rotation comparison, the interface
% trace discrepancy and the count of re-clamped rows.
%
% WHERE alpha DOES AND DOES NOT AFFECT THE ANSWER.  The transmission sets the
% clamped row to
%
%     (1 - alpha*s_i)*u_j(v) + [ u_k(P_i) - (1 - alpha*s_i)*u_k(v) ].
%
% If the two subdomains agree at the interface point, u_j(v) = u_k(v), the two
% (1 - alpha*s_i) terms cancel identically and what is left is u(P_i) -- exactly
% what the row stands for.  On the SMOOTH manifold they do agree to
% discretization order: the residue is the difference between two independent
% degree-p interpolations of the same function, O(dx^(p+1)), which the
% Laplacian's dx^-2 turns into O(dx^2).  There alpha controls the ITERATION
% rate and not the fixed point, the solution error is second order, and the
% sweep below confirms that the converged solution moves by less than the
% discretization error as alpha varies.
%
% At a CORNER the cancellation is incomplete.  A rigid rotation preserves
% Euclidean offsets rather than the target arc's arclength continuation, so
% P_i is not where the row's own arclength would put it, and the leftover
% rides on (1 - alpha*s_i).  The fixed point then does move with alpha, by
% more than the discretization error.
%
% WHAT THE ERROR LOOKS LIKE AT A SINGULAR INTERFACE.  Following the companion
% curvature-mismatch script and the draft it implements, the surface error
% behaves like
%
%     ||u_h - u||_inf = C_dk * h + C_2 * h^2 + o(h^2),
%
% with C_dk controlled by the rotated curvature-vector mismatch delta_kappa.
% On the plain ellipse delta_kappa is exactly zero, C_dk vanishes and the
% scheme is second order -- the smooth control, which also confirms that the
% decomposition and the separate-grid interpolation cost no order by
% themselves.  Over eleven levels at sqrt(2) spacing the control holds a
% level-to-level rate of 1.93 to 2.06 until round-off takes over below
% dx ~ 1e-3.
%
% On the reflected manifold the measurement is more delicate than a rate.  The
% error there is first order, but C_dk JITTERS as the interface moves relative
% to each subdomain's lattice: over the same eleven levels err/h wanders by a
% factor of two to six about its mean with no trend, so consecutive levels
% give rates anywhere from -2 to +4, and a fit over three or four levels
% reports whatever the jitter happened to do.  Only the trend over many levels
% is meaningful, and the statistic worth reading is mean(err/h), not the rate.
% Measured that way the ORDER is first whenever delta_kappa is nonzero and
% flat in delta_kappa, while the CONSTANT grows with the mismatch roughly
% linearly -- which is where the size of delta_kappa actually shows up.  The
% script reports both, with the spread of err/h beside the mean.
%
% THE RATE STUDY.  The cut height is swept over
% ycut_list = [-0.05 0 0.05 0.1 0.2 0.3 0.4], which moves the curvature at the
% interface over roughly an order of magnitude in delta_kappa and so moves the
% crossover h* across the grid range.  For each cut the script reports the
% interface curvature, the rotated curvature mismatch, the closest approach of
% the two branches away from the interfaces, the surface error at every h, the
% level-to-level rates, the fitted rate and the rates over the coarsest and
% finest pairs separately, the interface trace discrepancy, the count of
% re-clamped rows, and the angle error of the d_k2 estimator against the
% analytic rotation.
%
% THREE QUANTITIES share the letter kappa in this literature.  They are kept
% apart by name throughout:
%
%   rho_theory       the Schwarz contraction factor rho(M_dagger), a property
%                    of the PARTITION alone -- lengths and overlaps.  It does
%                    not depend on the cut height here, because the two arcs
%                    always make up the same closed manifold of length L.
%   kappa_interface  the geometric curvature of the ellipse at the cut point.
%   delta_kappa      the rotated curvature-vector mismatch ||c2 - R*c1|| there.
%
% What this family can and cannot decide: cutting and reflecting makes the join
% a MIRROR image, so the two branches carry equal curvature magnitudes and
% delta_kappa = |kappa1 + kappa2| = 2*kappa_interface identically, at every cut
% height, while on the plain ellipse delta_kappa is exactly zero.  The two
% predictors are therefore proportional here and the rate plot against one is
% the plot against the other with a relabelled axis.  The experiment shows that
% the rate degrades as curvature at the cut grows; it cannot decide which of
% the two quantities is the operative predictor.  Separating them needs a
% family in which delta_kappa varies independently of curvature magnitude,
% which is what example_curvature_mismatch_delta_kappa_convergence.m provides
% with two circular arcs of independently chosen radii.

% adjust as appropriate
addpath('../cp_matrices');
addpath('../surfaces');


%% Geometry, conventions, and the manufactured solution

cen = [0.5 0.5];
a = 1.4;      % semi-axis along x (the major one)
b = 0.6;      % semi-axis along y

dim = 2;
p = 3;        % interpolation degree
order = 2;    % Laplacian order
% The formula for bw is found in [Ruuth & Merriman 2008] and the 1.0002
% is a safety factor.
bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ((order/2+(p+1)/2)^2));

c = 1;                 % the Helmholtz shift; alpha = sqrt(c) is optimal
sqrt_c = sqrt(c);

% The cut heights.  Each sets the curvature the glue has to work across, and
% with it delta_kappa and the crossover h* between the two error terms.
ycut_list = [-0.05 0 0.05 0.1 0.2 0.3 0.4];
ycut_ref = 0.3;        % the cut the solver diagnostics and the pictures use

% Manufactured solution in the ellipse parameter t, and the analytic right-hand
% side of c*u - laplacian_S u = f.  The reflection preserves t, so u and f are
% the same functions of the parameter on both manifolds and at every cut.
ufun   = @(t) sin(t + 0.7) + 0.5*cos(2*t - 0.3);
utfun  = @(t) cos(t + 0.7) - sin(2*t - 0.3);
uttfun = @(t) -sin(t + 0.7) - 2*cos(2*t - 0.3);
gfun   = @(t) a^2*sin(t).^2 + b^2*cos(t).^2;
gtfun  = @(t) (a^2 - b^2)*sin(2*t);
lapfun = @(t) uttfun(t)./gfun(t) - utfun(t).*gtfun(t)./(2*gfun(t).^2);
ffun   = @(t) c*ufun(t) - lapfun(t);

% Subcell offsets, one pair per subdomain.  They differ so that no node of one
% subdomain coincides with a node of the other, which is what makes the
% transmission a genuine cross-grid interpolation.
offsets = [0.317 0.211; 0.417 0.373];

GEO = {'plain', 'flipped'};
SCH = {'d_k2', 'exact'};
MET = {'OPS', 'OAS'};
N   = 2;       % one subdomain per branch, interfaces at the two cut points

% sqrt(2) refinement, eleven levels: enough points to fit a rate even after
% the reach test below discards the coarsest ones at a tight cut.  Past
% dx ~ 7e-4 a degree-3 weight built from coordinates of size X carries
% ~eps*X/dx of rounding which the Laplacian's dx^-2 turns into ~eps*X/dx^3,
% so the smooth control bottoms out there; fitrate drops the levels at and
% past the minimum, and the singular cases stay well above that floor.
hvals = 0.02*2.^-(0:0.5:5);
% A band point has a unique closest point only while bw*dx stays under the
% smallest radius of curvature of the ellipse.
hvals = hvals(bw*hvals < 0.5*min(b^2/a, a^2/b));
nh = numel(hvals);
dx_hist = 0.01;        % the grid size the iteration figure is drawn at

[sigma, t_of_s] = arclength_map(a, b);
L = sigma(2*pi);

figdir = 'figs';   % .png output goes here
if ~exist(figdir, 'dir'), mkdir(figdir); end

fprintf('ellipse a = %g, b = %g, c = %g, total length L = %.10f\n', a, b, c, L);


%% Accuracy: surface error against the curvature mismatch at the interface
% The central experiment.  The plain ellipse is the smooth control: the join
% is artificial, the exact rotation is the identity, delta_kappa is zero and
% the scheme must recover the ordinary second-order CPM accuracy, which is
% what shows that the decomposition and the separate-grid interpolation cost
% no order by themselves.  The reflected manifold is the singular experiment:
% the branches meet at corners, no single smooth closest-point map runs
% through the junction, and the rotation carries extension rows across it.

ncut = numel(ycut_list);
nrow = numel(GEO)*numel(SCH)*numel(MET);
labels  = cell(1, nrow);
errcut  = nan(nrow, nh, ncut);
erroff  = nan(nrow, nh, ncut);     % error away from the interfaces
dcorner = nan(nrow, nh, ncut);     % where the worst point sits
gapcut  = nan(nrow, nh, ncut);     % interface trace discrepancy
ratefit = nan(nrow, ncut);
rateoff = nan(nrow, ncut);         % fitted rate away from the interfaces
nkeep   = nan(nrow, ncut);         % levels admissible under the reach test
Cfirst  = nan(nrow, ncut);         % mean of err/h, the first-order constant
Cspread = nan(nrow, ncut);         % max/min of err/h over the levels
ratec   = nan(nrow, ncut);         % rate over the two coarsest levels
ratef   = nan(nrow, ncut);         % rate over the two finest levels
kap_int = nan(numel(GEO), ncut);
dkap    = nan(numel(GEO), ncut);
brgap   = nan(numel(GEO), ncut);
ok_reach = true(numel(GEO), nh, ncut);   % is bw*dx under half the branch gap
cutgeom = struct('ycut', {}, 'sig1', {}, 'piece_up', {}, 'piece_lo', {}, ...
                 't1', {}, 't2', {}, 'rho_theory', {});
ithist  = struct('name', {}, 'e', {}, 'rho_theory', {});

for ic = 1:ncut
  ycut = ycut_list(ic);

  s0 = (ycut - cen(2))/b;
  if (abs(s0) > 0.95)
    error('cut at y = %g misses the ellipse or grazes it (sin t = %g)', ycut, s0);
  end
  if (abs(s0) < 0.05)
    error(['the flipped curve degenerates at y = %g: reflecting the lower ' ...
           'half across the centre line puts it on top of the upper half'], ycut);
  end
  t1 = asin(s0);
  t2 = pi - t1;
  sig1 = sigma(t1);
  piece_up = sigma(t2) - sig1;
  piece_lo = L - piece_up;

  [piece, ~] = corner_partition(sig1, piece_up, piece_lo);
  rho_theory = schwarz_rho(sqrt_c, c, piece, zeros(N,1));
  if (abs(rho_theory - exp(-sqrt_c*L/N))/exp(-sqrt_c*L/N) > 1e-12)
    error('the general-partition identity fails on the two-arc partition');
  end

  cutgeom(ic).ycut = ycut;          cutgeom(ic).sig1 = sig1;
  cutgeom(ic).piece_up = piece_up;  cutgeom(ic).piece_lo = piece_lo;
  cutgeom(ic).t1 = t1;              cutgeom(ic).t2 = t2;
  cutgeom(ic).rho_theory = rho_theory;

  fprintf('\n================ cut at y = %g (sin t = %.4f) ================\n', ...
          ycut, s0);

  for ig = 1:numel(GEO)
    flipped = strcmp(GEO{ig}, 'flipped');
    if flipped, yflip2 = -1; else, yflip2 = 1; end
    [kap_int(ig,ic), dkap(ig,ic)] = curvature_mismatch(t1, yflip2, a, b);
    brgap(ig,ic) = min_branch_gap(t1, t2, ycut, flipped, a, b, cen);
    fprintf('%-8s: kappa_interface = %.4f, delta_kappa = %.4f', ...
            GEO{ig}, kap_int(ig,ic), dkap(ig,ic));
    if isfinite(brgap(ig,ic))
      fprintf(', branch gap = %.4f', brgap(ig,ic));
      if (bw*hvals(1) > 0.5*brgap(ig,ic))
        fprintf('  [tight: bw*dx = %.4f at the coarsest grid]', bw*hvals(1));
      end
    end
    fprintf('\n');
    ehist_g = cell(1, numel(MET));

    for k = 1:nh
      dx = hvals(k);
      % The two branches of the reflected manifold approach each other away
      % from the interfaces.  Once bw*dx reaches half that distance the two
      % bands interpenetrate and the level says nothing about the order, so
      % it is computed but excluded from the fit.
      ok_reach(ig,k,ic) = (bw*dx < 0.5*brgap(ig,ic));
      sub  = build_subdomains(sig1, piece_up, piece_lo, dx, ...
                              a, b, cen, ycut, flipped, p, order, bw, ...
                              offsets, t_of_s);
      % both glues are built at every level so the estimator's angle error
      % against the analytic rotation can be reported alongside
      gl = struct();
      for isc = 1:numel(SCH)
        gl.(SCH{isc}) = make_glue(sub, SCH{isc}, a, b);
      end
      dth = max(abs(wrapangle(gl.d_k2.theta(:) - gl.exact.theta(:))));

      for isc = 1:numel(SCH)
        prep = schwarz_prepare(sub, gl.(SCH{isc}), p);
        ops = schwarz_operators(sub, prep, c, sqrt_c);
        fv0 = schwarz_forcing(sub, prep, []);
        fvf = schwarz_forcing(sub, prep, ffun);
        for im = 1:numel(MET)
          u1 = cellfun(@(s) ones(s.nin,1), sub, 'UniformOutput', false);
          [~, e_rate] = schwarz_solve(sub, prep, ops, fv0, u1, im, 200, 1e-13);
          u0 = cellfun(@(s) zeros(s.nin,1), sub, 'UniformOutput', false);
          uc = schwarz_solve(sub, prep, ops, fvf, u0, im, 200, 1e-13);

          ir = ((ig-1)*numel(SCH) + (isc-1))*numel(MET) + im;
          [errcut(ir,k,ic), erroff(ir,k,ic), dcorner(ir,k,ic)] = ...
              surface_error(sub, uc, a, b, cen, ufun, t_of_s, 0.05);
          gapcut(ir,k,ic) = interface_gap(sub, uc, a, b, cen);
          labels{ir} = sprintf('%s, %s, %s', GEO{ig}, SCH{isc}, MET{im});

          if (im == 1)
            if (k == 1)
              pw = NaN;
            else
              pw = log(errcut(ir,k-1,ic)/errcut(ir,k,ic)) / ...
                   log(hvals(k-1)/hvals(k));
            end
            fprintf('  %-6s dx %-8.4g rc %-3d err %10.3e', ...
                    SCH{isc}, dx, prep.nreclamp, errcut(ir,k,ic));
            if isnan(pw), fprintf('   rate   --'); else, fprintf('   rate %5.2f', pw); end
            fprintf('   off %9.3e  d %6.3f', erroff(ir,k,ic), dcorner(ir,k,ic));
            fprintf('   trace %8.2e', gapcut(ir,k,ic));
            if ~ok_reach(ig,k,ic), fprintf('  [bands overlap: excluded]'); end
            if (isc == 1), fprintf('   dtheta %8.2e', dth); end
            fprintf('\n');
          end

          if (strcmp(SCH{isc}, 'exact') && abs(dx - dx_hist) < 1e-14 && ...
              abs(ycut - ycut_ref) < 1e-14)
            ehist_g{im} = e_rate;
          end
        end
      end
    end

    if (abs(ycut - ycut_ref) < 1e-14)
      ithist(ig).name = GEO{ig};
      ithist(ig).e = ehist_g;
      ithist(ig).rho_theory = rho_theory;
    end
  end

  for ir = 1:nrow
    ig = 1 + floor((ir-1)/(numel(SCH)*numel(MET)));
    keep = squeeze(ok_reach(ig,:,ic));
    nkeep(ir,ic) = sum(keep);
    if (sum(keep) >= 2)
      ratefit(ir,ic) = fitrate(hvals(keep), errcut(ir,keep,ic));
      rateoff(ir,ic) = fitrate(hvals(keep), erroff(ir,keep,ic));
      kk = find(keep);
      ratec(ir,ic) = log(errcut(ir,kk(1),ic)/errcut(ir,kk(2),ic)) / ...
                     log(hvals(kk(1))/hvals(kk(2)));
      ratef(ir,ic) = log(errcut(ir,kk(end-1),ic)/errcut(ir,kk(end),ic)) / ...
                     log(hvals(kk(end-1))/hvals(kk(end)));
      % At a singular interface the error is first order with a constant that
      % jitters as the interface moves relative to the lattice, so err/h is
      % the statistic worth reporting and any single pairwise rate is not.
      Ch = errcut(ir,kk,ic)./hvals(kk);
      Cfirst(ir,ic) = mean(Ch);
      Cspread(ir,ic) = max(Ch)/min(Ch);
    end
  end
end


%% Requested validation: the rate against the curvature mismatch
% The fitted rate alone averages the transition away, so the rates over the
% coarsest and the finest pair of levels are reported beside it: the gap
% between them IS the crossover between C_2*h^2 and C_dk*h.

fprintf('\n\n================ rate vs curvature mismatch ================\n');
fprintf('%-26s', 'ycut');
fprintf('%8.2f', ycut_list);
fprintf('\n%-26s', 'kappa_interface');
fprintf('%8.3f', kap_int(1,:));
fprintf('\n%-26s', 'delta_kappa (flipped)');
fprintf('%8.3f', dkap(2,:));
fprintf('\n%-26s', 'delta_kappa (plain)');
fprintf('%8.3f', dkap(1,:));
fprintf('\n');
for ir = 1:nrow
  if ~contains(labels{ir}, 'OPS'), continue; end
  fprintf('%-26s', [labels{ir}(1:end-5) ' fit']);
  fprintf('%8.2f', ratefit(ir,:));
  fprintf('\n%-26s', [labels{ir}(1:end-5) ' coarse pair']);
  fprintf('%8.2f', ratec(ir,:));
  fprintf('\n%-26s', [labels{ir}(1:end-5) ' fine pair']);
  fprintf('%8.2f', ratef(ir,:));
  fprintf('\n%-26s', [labels{ir}(1:end-5) ' off-interface']);
  fprintf('%8.2f', rateoff(ir,:));
  fprintf('\n%-26s', [labels{ir}(1:end-5) ' levels used']);
  fprintf('%8d', nkeep(ir,:));
  fprintf('\n%-26s', [labels{ir}(1:end-5) ' mean err/h']);
  fprintf('%8.4f', Cfirst(ir,:));
  fprintf('\n%-26s', [labels{ir}(1:end-5) ' err/h spread']);
  fprintf('%8.2f', Cspread(ir,:));
  fprintf('\n');
end

% Is the first-order constant proportional to the curvature mismatch, as the
% companion curvature-mismatch script's draft predicts?  Fitted on the
% reflected manifold with the analytic rotation, which carries no
% tangent-estimation error.
ir_fl = ((2-1)*numel(SCH) + (2-1))*numel(MET) + 1;    % flipped, exact, OPS
xv = dkap(2,:).';
yv = Cfirst(ir_fl,:).';
ok = isfinite(xv) & isfinite(yv);
pf = polyfit(xv(ok), yv(ok), 1);
res = yv(ok) - polyval(pf, xv(ok));
r2 = 1 - sum(res.^2)/sum((yv(ok) - mean(yv(ok))).^2);
fprintf(['\nfirst-order constant against delta_kappa (flipped, exact):\n' ...
         '  C = %.4f*delta_kappa + %.4f,  R^2 = %.4f\n'], pf(1), pf(2), r2);


%% Solver diagnostics: the Schwarz contraction factor
% Secondary to the accuracy study above.  The contraction factor governs how
% fast the iteration reaches the fixed point, not what the fixed point is, and
% the fixed point is what the rest of this script measures.

icr = find(abs(ycut_list - ycut_ref) < 1e-14, 1);
if isempty(icr)
  error('the reference cut y = %g is not in ycut_list', ycut_ref);
end
cg = cutgeom(icr);

k_eq = schwarz_rho(sqrt_c, c, (L/N + 0.25)*ones(N,1), 0.25*ones(N,1));
fprintf('\n---- Schwarz contraction factor ----\n');
fprintf('equal overlapping partition %.15e vs exp(-sqrt(c)L/N) %.15e\n', ...
        k_eq, exp(-sqrt_c*L/N));
if (abs(k_eq - exp(-sqrt_c*L/N))/exp(-sqrt_c*L/N) > 1e-12)
  error('the equal-partition convergence factor is wrong');
end
fprintf('rho_theory on the two-arc partition = %.6f (cut independent)\n', ...
        cg.rho_theory);

alphas = unique([linspace(0.1, 4, 14) linspace(0.7, 1.4, 8) sqrt_c])';
na = numel(alphas);
dx_sweep = hvals(1);
sweep = struct('rho', cell(numel(GEO), numel(SCH), numel(MET)), ...
               'rho_theory', cell(numel(GEO), numel(SCH), numel(MET)));
[piece_r, ~] = corner_partition(cg.sig1, cg.piece_up, cg.piece_lo);
rho_a = arrayfun(@(al) schwarz_rho(al, c, piece_r, zeros(N,1)), alphas);

fprintf('\nRobin sweep at dx = %g, cut y = %g\n', dx_sweep, ycut_ref);
for ig = 1:numel(GEO)
  flipped = strcmp(GEO{ig}, 'flipped');
  sub = build_subdomains(cg.sig1, cg.piece_up, cg.piece_lo, dx_sweep, ...
                         a, b, cen, ycut_ref, flipped, p, order, bw, ...
                         offsets, t_of_s);
  for isc = 1:numel(SCH)
    prep = schwarz_prepare(sub, make_glue(sub, SCH{isc}, a, b), p);
    fv0 = schwarz_forcing(sub, prep, []);
    fvf = schwarz_forcing(sub, prep, ffun);
    u1 = cellfun(@(s) ones(s.nin,1), sub, 'UniformOutput', false);
    u0 = cellfun(@(s) zeros(s.nin,1), sub, 'UniformOutput', false);
    rho = nan(na, numel(MET));
    uc_all = cell(na, 1);
    for ia = 1:na
      ops = schwarz_operators(sub, prep, c, alphas(ia));
      for im = 1:numel(MET)
        [~, e_rate] = schwarz_solve(sub, prep, ops, fv0, u1, im, 300, 1e-13);
        rho(ia, im) = fit_rho(e_rate);
      end
      uc_all{ia} = schwarz_solve(sub, prep, ops, fvf, u0, 1, 300, 1e-13);
    end
    for im = 1:numel(MET)
      sweep(ig, isc, im).rho = rho(:,im);
      sweep(ig, isc, im).rho_theory = rho_a;
      [rmin, imin] = min(rho(:,im));
      fprintf('  %-8s %-6s %-4s: argmin alpha = %.3f (rho %.4f)\n', ...
              GEO{ig}, SCH{isc}, MET{im}, alphas(imin), rmin);
    end

    % the fixed point should barely move with alpha on the smooth manifold
    uref = uc_all{abs(alphas - sqrt_c) < 1e-14};
    dmax = 0;
    for ia = 1:na
      for j = 1:N
        dmax = max(dmax, norm(uc_all{ia}{j} - uref{j}, inf));
      end
    end
    fprintf('  %-8s %-6s     : max over alpha of |u(alpha) - u(sqrt c)| = %.3e\n', ...
            GEO{ig}, SCH{isc}, dmax);
  end
end


%% Requested visualization

part = struct('name', {}, 'cx', {}, 'cy', {}, 'nodes', {}, 'V', {}, 'iscorner', {});
dx_show = 0.03;   % coarse enough to resolve individual band nodes in the figure
TVr = [cen(1) + a*cos([cg.t1; cg.t2]), cen(2) + b*sin([cg.t1; cg.t2])];
for ig = 1:numel(GEO)
  flipped = strcmp(GEO{ig}, 'flipped');
  sub = build_subdomains(cg.sig1, cg.piece_up, cg.piece_lo, dx_show, ...
                         a, b, cen, ycut_ref, flipped, p, order, bw, ...
                         offsets, t_of_s);
  tq = linspace(0, 2*pi, 2001)';
  xyu = arcpoint([1 0], tq, a, b, cen);
  if flipped
    lowmask = (tq > cg.t2) & (tq < cg.t1 + 2*pi);
    xyu(lowmask,2) = 2*ycut_ref - xyu(lowmask,2);
  end
  nodes = cell(1, N);
  Vp = zeros(N, 2);
  isc = false(N, 1);
  for j = 1:N
    nodes{j} = [sub{j}.xin sub{j}.yin];
    Vp(j,:) = arcpoint([sub{j}.yflip sub{j}.yoff], sub{j}.tb, a, b, cen);
    isc(j) = any(hypot(Vp(j,1) - TVr(:,1), Vp(j,2) - TVr(:,2)) < 1e-10);
  end
  part(ig).name = GEO{ig};
  part(ig).cx = xyu(:,1);   part(ig).cy = xyu(:,2);
  part(ig).nodes = nodes;   part(ig).V = Vp;   part(ig).iscorner = isc;
end

plot_partition(figdir, part);
plot_iteration(figdir, ithist);
plot_alpha(figdir, sweep, alphas, GEO);
plot_accuracy(figdir, hvals, errcut(:,:,icr), ratefit(:,icr), labels);
plot_rate_curvature(figdir, dkap, ratefit, Cfirst, Cspread, GEO, SCH, MET);

fprintf('\nfigures written to %s\n', figdir);


%% Geometry of the manifold and its partition

function [sigma, t_of_s] = arclength_map(a, b)
%ARCLENGTH_MAP  the arclength of the ellipse and its inverse
%   sigma(t) is the arclength from t = 0, by composite Simpson on a uniform
%   parameter grid; t_of_s(s) inverts it for ANY real s by writing
%   s = m*L + r and returning 2*pi*m + t(r), which is monotone over all of R
%   and so delivers ta < tb with tb - ta < 2*pi for any a_j < b_j.

  nq = 2^14 + 1;
  tq = linspace(0, 2*pi, nq)';
  h = tq(2) - tq(1);
  sp = ellipse_speed(tq, a, b);
  % cumulative Simpson: whole panels on the odd nodes, then one half panel
  % (Simpson 3/8-free, using the three surrounding values) on the even ones
  sv = zeros(nq, 1);
  sv(3:2:end) = cumsum(h/3*(sp(1:2:end-2) + 4*sp(2:2:end-1) + sp(3:2:end)));
  sv(2:2:end-1) = sv(1:2:end-2) + ...
                  h/12*(5*sp(1:2:end-2) + 8*sp(2:2:end-1) - sp(3:2:end));
  L = sv(end);
  sigma = @(t) interp1(tq, sv, mod(t, 2*pi), 'pchip') + floor(t/(2*pi))*L;
  t_of_s = @(s) ext_t_of_s(s, L, tq, sv, a, b);
end


function t = ext_t_of_s(s, L, tq, sv, a, b)
%EXT_T_OF_S  the parameter at arclength s, for any real s
%   A pchip seed on the tabulated arclength followed by three Newton steps
%   with d(sigma)/dt = |gamma'(t)|.

  m = floor(s/L);
  r = s - m*L;
  t = interp1(sv, tq, r, 'pchip');
  for it = 1:3
    t = t - (interp1(tq, sv, t, 'pchip') - r)./ellipse_speed(t, a, b);
  end
  t = t + 2*pi*m;
end


function sp = ellipse_speed(t, a, b)
%ELLIPSE_SPEED  |gamma'(t)| for gamma(t) = cen + [a*cos t, b*sin t]

  sp = sqrt(a^2*sin(t).^2 + b^2*cos(t).^2);
end


function [piece, sigstart] = corner_partition(sig1, piece_up, piece_lo)
%CORNER_PARTITION  the two arcs the cut points bound, in cyclic order
%   Subdomain 1 is the upper arc and subdomain 2 the reflected lower one, so
%   the two interfaces are exactly the two cut points and each subdomain is
%   one branch of the manifold.  The decomposition is NON-OVERLAPPING: with
%   both interfaces at the cut points there is nowhere to overlap without a
%   subdomain straddling a corner, which a single closest-point map cannot
%   represent.  Their Definition 2 allows this, asking only delta_j >= 0, and
%   the convergence factor at alpha = sqrt(c) is unaffected because the
%   overlaps telescope away whatever their values -- here they are all zero,
%   so ell_j is just the arc length and kappa = exp(-sqrt(c)*L/2).

  piece = [piece_up; piece_lo];
  sigstart = [sig1; sig1 + piece_up];
end


function rho_theory = schwarz_rho(alpha, c, ell, dl)
%SCHWARZ_RHO  rho(M_dagger) for the optimized parallel Schwarz iteration
%   Yazdani, Haynes & Ruuth (2024), eqs. (23)-(24), for an arbitrary
%   partition of a closed 1-manifold.  ell(j) is the length of the jth
%   OVERLAPPING subdomain and dl(j) the overlap at its far end; both are
%   cyclic in j, so dl(j-1) is the overlap at its near end.  Returns the
%   spectral radius, which is the convergence factor.

  N = numel(ell);
  s = sqrt(c);
  M = zeros(2*N, 2*N);
  wrap = @(i) 1 + mod(i - 1, 2*N);
  for j = 1:N
    lj = ell(j);
    dp = dl(1 + mod(j-2, N));
    dj = dl(j);
    D  = (alpha-s)^2 - (alpha+s)^2*exp(2*s*lj);
    pd = ((alpha^2-c) - (alpha^2-c)*exp(2*s*(lj-dp)))/D * exp(s*dp);
    rd = ((alpha-s)^2 - (alpha+s)^2*exp(2*s*dp))/D * exp(s*(lj-dp));
    sd = ((alpha-s)^2 - (alpha+s)^2*exp(2*s*dj))/D * exp(s*(lj-dj));
    qd = ((alpha^2-c) - (alpha^2-c)*exp(2*s*(lj-dj)))/D * exp(s*dj);
    M(wrap(2*j-1), wrap(2*j-2)) = pd;
    M(wrap(2*j-1), wrap(2*j+1)) = rd;
    M(wrap(2*j),   wrap(2*j-2)) = sd;
    M(wrap(2*j),   wrap(2*j+1)) = qd;
  end
  rho_theory = max(abs(eig(M)));
end


function xy = arcpoint(piece, t, a, b, cen)
%ARCPOINT  the point of an arc at parameter t; piece is [yflip yoff]

  xy = [cen(1) + a*cos(t), piece(1)*(cen(2) + b*sin(t)) + piece(2)];
end


function t = arcparam(yflip, yoff, x, y, a, b, cen)
%ARCPARAM  the ellipse parameter of a point on an arc, reflected or not

  t = atan2((((y - yoff)/yflip) - cen(2))/b, (x - cen(1))/a);
end


function [cx, cy, dist, bdy] = cpflip(cpf, x, y, yc)
%CPFLIP  closest point on an arc reflected across the line y = yc
%   A reflection is an isometry and is its own inverse, so the closest point
%   on the reflected arc is the reflection of the closest point of the
%   reflected query.

  [cx, cy0, dist, bdy] = cpf(x, 2*yc - y);
  cy = 2*yc - cy0;
end


function th = wrapangle(th)
%WRAPANGLE  put an angle into (-pi, pi]

  th = angle(exp(1i*th));
end


%% Per-subdomain grid, band and operators

function cand = candidate_nodes(x1d, y1d, dx, rad, ta, tb, yflip, yoff, a, b, cen)
%CANDIDATE_NODES  grid nodes that could lie within the band of one arc
%   Walks the arc so that consecutive samples are at most dx/2 apart, snaps
%   each to the nearest node of this subdomain's lattice, and keeps every node
%   within rad cells of one of them.  Linear indices are into the ny-by-nx
%   meshgrid ordering the interpolation and Laplacian matrices use.

  nx = numel(x1d);
  ny = numel(y1d);
  mask = false(ny, nx);
  [oi, oj] = meshgrid(-rad:rad, -rad:rad);

  % |gamma'| <= max(a,b), so this many samples are dx/2 apart at worst
  ns = ceil(2*(tb - ta)*max(a,b)/dx) + 1;
  xy = arcpoint([yflip yoff], linspace(ta, tb, ns)', a, b, cen);
  is = round((xy(:,1) - x1d(1))/dx) + 1;
  js = round((xy(:,2) - y1d(1))/dx) + 1;

  for cc = 1:numel(oi)
    ii = is + oi(cc);
    jj = js + oj(cc);
    ok = (ii >= 1) & (ii <= nx) & (jj >= 1) & (jj <= ny);
    mask(sub2ind([ny nx], jj(ok), ii(ok))) = true;
  end
  cand = find(mask);
end


function sub = build_subdomains(sig1, piece_up, piece_lo, dx, ...
                                a, b, cen, ycut, flipped, p, order, bw, ...
                                offsets, t_of_s)
%BUILD_SUBDOMAINS  one band, one grid and one operator set per subdomain
%   Subdomain 1 is the upper arc and subdomain 2 the lower one, each running
%   exactly from one cut point to the other: the decomposition does not
%   overlap, so a subdomain is one branch of the manifold.
%   Each gets its OWN bounding box and its own subcell offset, so no node of
%   one subdomain is a node of another and the transmission below is a real
%   cross-grid interpolation.  Subdomains on the lower arc carry the
%   reflected embedding (yflip = -1, yoff = 2*ycut) when flipped is true.
%   Returns a cell array; subdomain fields are named as in the sibling
%   scripts, plus the interface bookkeeping bnd_a / bnd_b and nbr_a / nbr_b.

  [piece, sigstart] = corner_partition(sig1, piece_up, piece_lo);
  N = 2;
  switch (order)
    case 2
      sten = 5;
    case 4
      sten = 9;
    otherwise
      error('order %d not implemented', order);
  end

  sub = cell(N, 1);
  for j = 1:N
    sig_da = sigstart(j);
    sig_db = sigstart(j) + piece(j);
    sig_a  = sig_da;
    sig_b  = sig_db;
    ta = t_of_s(sig_a);
    tb = t_of_s(sig_b);

    if (j == 2 && flipped)
      yflip = -1;  yoff = 2*ycut;
    else
      yflip =  1;  yoff = 0;
    end

    base = @(x, y) cpEllipseArc(x, y, a, b, cen, ta, tb);
    if (yflip == 1)
      cpf = base;
    else
      cpf = @(x, y) cpflip(base, x, y, ycut);
    end

    % The box is exact, not sampled: x and y along the arc are a*cos t and
    % +-b*sin t, so both are stationary only at multiples of pi/2, and the
    % extremes are at the ends or at whichever of those the arc contains.
    tc = [ta; tb; (ceil(ta/(pi/2)):floor(tb/(pi/2)))'*(pi/2)];
    xy = arcpoint([yflip yoff], tc, a, b, cen);
    lo = min(xy, [], 1);
    hi = max(xy, [], 1);

    pad = (bw + order/2 + (p+1)/2 + 2)*dx;
    io = 1 + mod(j-1, size(offsets,1));
    x1d = ((lo(1) - pad + offsets(io,1)*dx) : dx : (hi(1) + pad))';
    y1d = ((lo(2) - pad + offsets(io,2)*dx) : dx : (hi(2) + pad))';
    nx = numel(x1d);
    ny = numel(y1d);

    % A meshgrid over the whole box cannot be built at the finest dx, and all
    % but a sliver of it is discarded by the banding anyway.  So walk the arc
    % instead, snap each sample to the nearest node, and keep every node
    % within rad cells of one of those.
    rad = ceil(bw) + 1;
    cand = candidate_nodes(x1d, y1d, dx, rad, ta, tb, yflip, yoff, a, b, cen);
    [jc, ic] = ind2sub([ny nx], cand);
    xc = x1d(ic);  yc = y1d(jc);
    [cpx, cpy, dist, bdy] = cpf(xc, yc);

    % the initial band: the nodes within bw*dx of the arc
    keep = abs(dist) <= bw*dx;
    band = cand(keep);
    if isempty(band)
      error('empty band at subdomain %d', j);
    end
    xinit = xc(keep);      yinit = yc(keep);
    cpxinit = cpx(keep);   cpyinit = cpy(keep);
    bdyinit = bdy(keep);

    % the inner band is the set of columns the closest point interpolation
    % touches
    [Ei, Ej, Es] = interp2_matrix(x1d, y1d, cpxinit, cpyinit, p);
    innerband = unique(Ej);
    nin = numel(innerband);
    inv_inner = make_invbandmap(nx*ny, innerband);
    Einit = sparse(Ei, inv_inner(Ej), Es, numel(band), nin);

    % the outer band is the set of columns the Laplacian on the inner band
    % touches
    Ltemp = laplacian_2d_matrix(x1d, y1d, order, innerband, band);
    if (nnz(Ltemp) ~= sten*nin)
      error('the Laplacian stencil of the inner band leaves the initial band');
    end
    [~, jj] = find(Ltemp);
    outertemp = unique(jj);
    [tf, loc] = ismember(innerband, band(outertemp));
    if ~all(tf)
      error('the inner band is not contained in the outer band');
    end
    nout = numel(outertemp);

    s.N = N;  s.j = j;  s.dx = dx;
    s.a = a;  s.b = b;  s.cen = cen;  s.bw = bw;  s.p = p;
    s.yflip = yflip;  s.yoff = yoff;
    s.ta = ta;  s.tb = tb;
    s.sig_a = sig_a;  s.sig_b = sig_b;
    s.sig_da = sig_da;  s.sig_db = sig_db;
    s.x1d = x1d;  s.y1d = y1d;
    s.innerband = innerband;
    s.inv_inner = inv_inner;
    s.nin = nin;  s.nout = nout;
    s.cpf = cpf;
    s.L = Ltemp(:, outertemp);
    s.E = Einit(outertemp, :);
    s.R = sparse(1:nin, loc, ones(nin,1), nin, nout);
    s.Loff = s.L - (s.R .* s.L);       % the part lapsharp applies to E
    s.xout = xinit(outertemp);
    s.yout = yinit(outertemp);
    s.cpxout = cpxinit(outertemp);
    s.cpyout = cpyinit(outertemp);
    s.bdyout = bdyinit(outertemp);
    s.xin = s.R*s.xout;
    s.yin = s.R*s.yout;

    % cpEllipseArc reports bdy = 1 at the ta end and bdy = 2 at the tb end;
    % the ta end faces subdomain j-1 and the tb end subdomain j+1, cyclically
    s.bnd_a = find(s.bdyout == 1);
    s.bnd_b = find(s.bdyout == 2);
    s.nbr_a = 1 + mod(j-2, N);
    s.nbr_b = 1 + mod(j, N);
    if (isempty(s.bnd_a) || isempty(s.bnd_b))
      error('subdomain %d has no clamped rows at one of its interfaces', j);
    end
    sub{j} = s;
    clear s
  end
end


%% The rotation glue at the interfaces

function glue = make_glue(sub, scheme, a, b)
%MAKE_GLUE  the angle each subdomain turns its clamped rows through
%   glue.theta(e, j) is the rotation subdomain j applies at its end e, with
%   e = 1 the ta end and e = 2 the tb end.  The angle is a property of the
%   CURVE at the interface point v, and needs the one-sided tangents of the
%   two arcs meeting there.  Because the decomposition is non-overlapping and
%   both interfaces are the cut points, BOTH subdomains end at v and each
%   supplies its own tangent directly -- the same construction the sibling
%   ellipse_cut scripts use at a cut point.
%
%     'exact'  the analytic tangents, with the y component negated on a
%              reflected piece
%     'd_k2'   the second-order closest-point estimator of cp_tangent
%
%   On a smooth join the far tangent is the negative of the near one and the
%   angle is zero; at a corner it is the turning angle of the curve.

  N = sub{1}.N;
  glue.theta = zeros(2, N);
  glue.scheme = scheme;

  for j = 1:N
    sj = sub{j};
    for e = 1:2
      if (e == 1)
        dirn = -1;  tv = sj.ta;  k = sj.nbr_a;
      else
        dirn = +1;  tv = sj.tb;  k = sj.nbr_b;
      end
      sk = sub{k};
      v = arcpoint([sj.yflip sj.yoff], tv, a, b, sj.cen);

      if strcmp(scheme, 'exact')
        % dirn is the direction of travel out of subdomain j at v, so the far
        % arc's own outward tangent there points the other way.
        tau_j =  dirn * embedded_tangent(tv, sj.yflip, a, b);
        tau_k = -dirn * embedded_tangent(tv, sk.yflip, a, b);
      else
        tau_j = cp_tangent(sj, v, 2);
        tau_k = cp_tangent(sk, v, 2);
      end

      % A row of j sitting past v lies roughly along tau_j from it, and the
      % glue has to put it where the far arc continues, which is -tau_k.
      glue.theta(e, j) = wrapangle(atan2(-tau_k(2), -tau_k(1)) - ...
                                   atan2( tau_j(2),  tau_j(1)));
    end
  end
end


function tau = embedded_tangent(t, yflip, a, b)
%EMBEDDED_TANGENT  the unit tangent gamma'(t)/|gamma'(t)| in the embedding
%   yflip = -1 reflects the arc across a horizontal line, which negates the
%   y component of every tangent.

  g = [-a*sin(t), yflip*b*cos(t)];
  tau = g/norm(g);
end



function tau = cp_tangent(s, v, scheme)
%CP_TANGENT  outward unit tangent at an arc endpoint, from cp differences
%   Uses the outer-band nodes whose closest point on the arc is the endpoint
%   v.  With cpbar = cp(2*cp(x)-x) and cp2bar = cp(3*cp(x)-2*x),
%
%     scheme 1:  d_k  = cp - cpbar                          (first order)
%     scheme 2:  d_k2 = 1.5*cp - 2*cpbar + 0.5*cp2bar       (second order)
%
%   The difference points from inside the arc towards v, so the result is the
%   tangent pointing OUT of the arc at v.

  tol = 100*eps(max(1, max(abs(v))));
  m = (abs(s.cpxout - v(1)) <= tol) & (abs(s.cpyout - v(2)) <= tol);
  x = s.xout(m);   y = s.yout(m);
  cx = s.cpxout(m); cy = s.cpyout(m);
  if isempty(x)
    error('no grid points have this interface point as their closest point');
  end

  [b1x, b1y] = s.cpf(2*cx - x, 2*cy - y);
  if (scheme == 1)
    dvx = cx - b1x;
    dvy = cy - b1y;
  else
    [b2x, b2y] = s.cpf(3*cx - 2*x, 3*cy - 2*y);
    dvx = 1.5*cx - 2*b1x + 0.5*b2x;
    dvy = 1.5*cy - 2*b1y + 0.5*b2y;
  end

  nrm = hypot(dvx, dvy);
  ok = isfinite(nrm) & (nrm > tol);
  if ~any(ok)
    error('every cp difference at this interface was too small to use');
  end
  avg = [mean(dvx(ok)./nrm(ok)) mean(dvy(ok)./nrm(ok))];
  tau = avg/norm(avg);
end


%% The Schwarz subproblems and the iteration

function prep = schwarz_prepare(sub, glue, p)
%SCHWARZ_PREPARE  everything about the transmission that does not depend on
%   the Robin parameter
%   For each clamped row i of subdomain j at interface point v, the node is
%   rotated about v through the glue angle and the NEIGHBOUR's closest point
%   map is applied to the rotated node, giving the point P_i the row stands
%   for.  Two interpolation matrices on the neighbour's grid are built, at
%   P_i and at v; prep.svec{j}(i) is the arclength from v to P_i, which is
%   the offset the Robin condition differentiates over, and prep.tP{j}(i) is
%   the parameter of P_i, where the row's forcing must be evaluated.
%   Cp{j,k} and Cv{j,k} accumulate over both ends, because with N = 2 the two
%   ends of a subdomain share the same neighbour.

  N = sub{1}.N;
  prep.Cp = cell(N, N);
  prep.Cv = cell(N, N);
  prep.svec = cell(N, 1);
  prep.tP = cell(N, 1);
  prep.nreclamp = 0;

  for j = 1:N
    sj = sub{j};
    prep.svec{j} = zeros(sj.nout, 1);
    prep.tP{j} = zeros(sj.nout, 1);
    Ip = cell(2,1);  Jp = cell(2,1);  Sp = cell(2,1);
    Iv = cell(2,1);  Jv = cell(2,1);  Sv = cell(2,1);
    kk = zeros(2,1);

    for e = 1:2
      if (e == 1)
        rows = sj.bnd_a;  k = sj.nbr_a;  tv = sj.ta;
      else
        rows = sj.bnd_b;  k = sj.nbr_b;  tv = sj.tb;
      end
      kk(e) = k;
      sk = sub{k};
      th = glue.theta(e, j);

      % every row at this end is clamped to the same interface point
      v = [sj.cpxout(rows(1)), sj.cpyout(rows(1))];
      ddx = sj.xout(rows) - v(1);
      ddy = sj.yout(rows) - v(2);
      xr = v(1) + cos(th)*ddx - sin(th)*ddy;
      yr = v(2) + sin(th)*ddx + cos(th)*ddy;

      % A wrong glue can push a row back across the normal line, and its
      % closest point on the far side is then clamped to the interface again.
      % That is the method doing what it does with a wrong glue, not a
      % failure: it is counted, not rejected.
      [cpxr, cpyr, ~, bdyr] = sk.cpf(xr, yr);
      prep.nreclamp = prep.nreclamp + sum(bdyr ~= 0);

      % the parameter of P_i, brought onto the same branch as tv, and the
      % arclength from v to P_i by Simpson over that short interval
      tP = arcparam(sk.yflip, sk.yoff, cpxr, cpyr, sj.a, sj.b, sj.cen);
      tP = tv + wrapangle(tP - tv);
      dt = tP - tv;
      si = abs(dt)/6 .* (ellipse_speed(tv, sj.a, sj.b) + ...
                         4*ellipse_speed(tv + dt/2, sj.a, sj.b) + ...
                         ellipse_speed(tP, sj.a, sj.b));
      if any(si > 2*sj.bw*sj.dx)
        error('the rotation sent a row %g past the interface, beyond the band', ...
              max(si));
      end
      prep.svec{j}(rows) = si;
      prep.tP{j}(rows) = tP;

      [Ip{e}, Jp{e}, Sp{e}] = trip(sk, rows, cpxr, cpyr, p);
      [Iv{e}, Jv{e}, Sv{e}] = trip(sk, rows, ...
                                   repmat(v(1), numel(rows), 1), ...
                                   repmat(v(2), numel(rows), 1), p);
    end

    for e = 1:2
      k = kk(e);
      if isempty(prep.Cp{j,k})
        prep.Cp{j,k} = sparse(sj.nout, sub{k}.nin);
        prep.Cv{j,k} = sparse(sj.nout, sub{k}.nin);
      end
      prep.Cp{j,k} = prep.Cp{j,k} + ...
                     sparse(Ip{e}, Jp{e}, Sp{e}, sj.nout, sub{k}.nin);
      prep.Cv{j,k} = prep.Cv{j,k} + ...
                     sparse(Iv{e}, Jv{e}, Sv{e}, sj.nout, sub{k}.nin);
    end
  end
end


function [I, J, S] = trip(sk, rows, xq, yq, p)
%TRIP  triplets of the degree-p interpolation of the neighbour's unknowns
%   The stencil is built on the NEIGHBOUR'S grid and mapped through the
%   NEIGHBOUR'S inner-band numbering; the row index is the querying
%   subdomain's outer-band row.

  [Ei, Ej, Es] = interp2_matrix(sk.x1d, sk.y1d, xq, yq, p);
  J = sk.inv_inner(Ej);
  if any(J == 0)
    error('a transmission stencil leaves the neighbour inner band');
  end
  I = rows(Ei);
  S = Es;
end


function ops = schwarz_operators(sub, prep, c, alpha)
%SCHWARZ_OPERATORS  the local operators and their factorizations
%   Everything that depends on the Robin parameter but not on the forcing or
%   the algorithm, so one build serves the rate run, the accuracy run and both
%   Schwarz algorithms.  The Robin condition enters to first order as the
%   scalar (1 - alpha*s_i) on each clamped extension row, and c enters the
%   local operator as c*I - M so that it stays consistent with a forcing built
%   from the same c.

  N = numel(sub);
  ops.A = cell(N,1);
  ops.rs = cell(N,1);
  ops.rsd = cell(N,1);
  for j = 1:N
    s = sub{j};
    bnd = [s.bnd_a; s.bnd_b];
    ops.rs{j} = ones(s.nout, 1);
    ops.rs{j}(bnd) = 1 - alpha*prep.svec{j}(bnd);
    ops.rsd{j} = zeros(s.nout, 1);
    ops.rsd{j}(bnd) = ops.rs{j}(bnd);
    Ealpha = spdiags(ops.rs{j}, 0, s.nout, s.nout) * s.E;
    ops.A{j} = lu_prep(c*speye(s.nin) - lapsharp_unordered(s.L, Ealpha, s.R));
  end
end


function fv = schwarz_forcing(sub, prep, ffun)
%SCHWARZ_FORCING  the right-hand side over each subdomain's inner band
%   A clamped row stands for u where the glue sent it, so it is asked to
%   satisfy the equation at P_i and not at the interface point.  ffun = []
%   gives the zero forcing used for the rate measurement.

  N = numel(sub);
  fv = cell(N,1);
  for j = 1:N
    s = sub{j};
    if isempty(ffun)
      fv{j} = zeros(s.nin, 1);
    else
      bnd = [s.bnd_a; s.bnd_b];
      ts = arcparam(s.yflip, s.yoff, s.cpxout, s.cpyout, s.a, s.b, s.cen);
      ts(bnd) = prep.tP{j}(bnd);
      fv{j} = ffun(s.R * ts);
    end
  end
end


function [u, e_hist] = schwarz_solve(sub, prep, ops, fv, u0, method, maxit, tol)
%SCHWARZ_SOLVE  the optimized Schwarz iteration, their eqs. (12)-(13)
%   method = 1 is the parallel algorithm (every subdomain reads iterate n) and
%   method = 2 the alternating one (subdomain j reads iterate n+1 from the
%   subdomains already updated in this sweep and n from the rest).  With a
%   zero forcing the exact fixed point is 0 and the iterate IS the error, so
%   e_hist is then the iterate norm and otherwise the increment norm.

  N = numel(sub);
  u = u0;
  iszero = all(cellfun(@(v) ~any(v), fv));

  e_hist = zeros(0, 1);
  for it = 1:maxit
    uprev = u;
    for j = 1:N
      s = sub{j};
      d = zeros(s.nout, 1);
      for k = 1:N
        if ~isempty(prep.Cp{j,k})
          if (method == 1)
            uk = uprev{k};
          else
            uk = u{k};
          end
          d = d + prep.Cp{j,k}*uk - ops.rsd{j}.*(prep.Cv{j,k}*uk);
        end
      end
      u{j} = lu_solve(ops.A{j}, fv{j} + s.Loff*d);
    end

    en = 0;
    for j = 1:N
      if iszero
        en = max(en, norm(u{j}, inf));
      else
        en = max(en, norm(u{j} - uprev{j}, inf));
      end
    end
    e_hist(end+1, 1) = en; %#ok<AGROW>
    if (en < tol)
      break;
    end
  end
  if any(~isfinite(e_hist))
    error('the Schwarz iteration produced non-finite values');
  end
end


function f = lu_prep(A)
%LU_PREP  a reusable sparse factorization of A, valid in MATLAB and Octave

  [f.L, f.U, f.P, f.Q, f.D] = lu(A);
end


function x = lu_solve(f, rhs)
%LU_SOLVE  solve A*x = rhs from lu_prep, where A = D\(P'*L*U*Q')

  x = f.Q * (f.U \ (f.L \ (f.P * (f.D \ rhs))));
end


%% Requested validation

function [eLinf, eoff, dmin] = surface_error(sub, u, a, b, cen, ufun, t_of_s, skip)
%SURFACE_ERROR  L_inf error of the global solution on the manifold
%   The global solution is assembled the way the paper does it: each local
%   solution is restricted to its own DISJOINT piece, so the overlaps are not
%   double counted and every point of the curve is covered exactly once.
%   Samples are midpoints of a uniform partition of the piece in ARCLENGTH, so
%   none sits on an interface.
%
%   Returns three things, because at a corner the error is not spread evenly:
%
%     eLinf  the L_inf error over the whole manifold;
%     eoff   the L_inf error over the samples further than skip*(piece length)
%            from either end of their piece, which is the error away from the
%            interfaces;
%     dmin   the arclength distance from the worst point to the nearest
%            interface, which says whether the maximum is corner-localized.
%
%   The split matters because a rigid rotation at a corner leaves a narrow
%   defect there whose sampled height depends on how the grid happens to fall
%   relative to the interface.  The global L_inf then moves erratically under
%   refinement while the error away from the corner converges cleanly, and
%   reporting only the first would hide which of the two is happening.

  eLinf = 0;
  eoff = 0;
  dmin = inf;
  for j = 1:numel(sub)
    s = sub{j};
    ell = s.sig_db - s.sig_da;
    nq = max(400, ceil(4*ell/s.dx));
    sq = s.sig_da + ((0:nq-1)' + 0.5)*ell/nq;
    tq = t_of_s(sq);
    xy = arcpoint([s.yflip s.yoff], tq, a, b, cen);

    [~, Ej, Es] = interp2_matrix(s.x1d, s.y1d, xy(:,1), xy(:,2), s.p);
    ns = (s.p + 1)^2;
    JJ = reshape(s.inv_inner(Ej), nq, ns);
    SS = reshape(Es, nq, ns);
    good = all(JJ > 0, 2);
    if ~all(good)
      warning('%d surface samples fell outside an inner band', nq - sum(good));
    end

    ng = sum(good);
    ii = repmat((1:ng)', 1, ns);
    Eq = sparse(ii(:), JJ(good,:), SS(good,:), ng, s.nin);
    e = abs(Eq*u{j} - ufun(tq(good)));
    sg = sq(good);

    [em, im] = max(e);
    if (em > eLinf)
      eLinf = em;
      dmin = min(sg(im) - s.sig_da, s.sig_db - sg(im));
    end
    far = (sg - s.sig_da > skip*ell) & (s.sig_db - sg > skip*ell);
    if any(far)
      eoff = max(eoff, max(e(far)));
    end
  end
end


function rho = fit_rho(e)
%FIT_RHO  observed contraction factor of the error iteration
%   The slope of log(e_n) against n, exponentiated, over the window where the
%   decay is already geometric and round-off has not yet caught up.

  e = e(:);
  n = numel(e);
  if (n < 3)
    rho = NaN;
    return;
  end
  use = (e <= 0.1*e(1)) & (e >= 1e4*min(e)) & (e > 0);
  if (sum(use) < 3)
    rho = NaN;
    return;
  end
  idx = (1:n)';
  cf = polyfit(idx(use), log(e(use)), 1);
  rho = exp(cf(1));
end


function r = fitrate(dx, e)
%FITRATE  slope of log(e) against log(dx), over the levels before the floor
%   The level where e bottoms out is the one where round-off has caught up
%   with the truncation error, so it and everything past it say nothing about
%   the order and are dropped.

  n = numel(e);
  [~, imin] = min(e);
  if (imin < n)
    imin = imin - 1;
  end
  use = isfinite(e) & (e > 0) & ((1:n) <= max(imin, 2));
  if (sum(use) < 2)
    r = NaN;
    return;
  end
  cf = polyfit(log(dx(use)), log(e(use)), 1);
  r = cf(1);
end


%% Requested visualization helpers

function plot_partition(figdir, part)
%PLOT_PARTITION  Partition of the manifold into Schwarz subdomains
%   Visualizes the two geometries (plain and flipped) side-by-side, with
%   each subdomain's inner-band nodes in distinct colours. Interface points
%   (vertices) are marked, distinguishing corners from smooth interfaces.
%
%   part is a 1x2 struct array indexed by geometry (ig=1..2):
%     part(ig).name: char, 'plain' or 'flipped'
%     part(ig).cx, cy: column vectors, polyline of the whole manifold
%     part(ig).nodes: 1xN cell array; nodes{j} is Kx2 array of
%                     subdomain j's inner-band coordinates
%     part(ig).V: Nx2 interface point coordinates
%     part(ig).iscorner: Mx1 logical, true for corners
%
%   Outputs a two-panel figure with grid, box, and legend on each.

  figure('Position', [100 100 1100 480]);

  colors = [0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19; 0.49 0.18 0.56];

  for ig = 1:2
    subplot(1, 2, ig);
    hold on;

    % The manifold curve in black
    plot(part(ig).cx, part(ig).cy, 'k-', 'LineWidth', 1.5, ...
         'DisplayName', 'manifold');

    % Each subdomain's nodes in its own colour
    for j = 1:numel(part(ig).nodes)
      if ~isempty(part(ig).nodes{j})
        plot(part(ig).nodes{j}(:,1), part(ig).nodes{j}(:,2), '.', ...
             'Color', colors(j,:), 'MarkerSize', 4, ...
             'DisplayName', sprintf('subdomain %d', j));
      end
    end

    % Interface points: corners as filled black squares, others as open circles
    corners = part(ig).iscorner;
    smooth = ~corners;

    if any(corners)
      plot(part(ig).V(corners,1), part(ig).V(corners,2), 'ks', ...
           'MarkerFaceColor', 'k', 'MarkerSize', 8, ...
           'DisplayName', 'corner interface');
    end
    if any(smooth)
      plot(part(ig).V(smooth,1), part(ig).V(smooth,2), 'ko', ...
           'MarkerSize', 7, 'LineWidth', 1.0, ...
           'DisplayName', 'smooth interface');
    end

    axis equal;
    grid on; box on;
    xlabel('x'); ylabel('y');
    title(part(ig).name, 'FontSize', 10);
    legend('Location', 'best', 'FontSize', 8);
  end

  print(gcf, fullfile(figdir, 'ellipse_schwarz_ddm_partition'), '-dpng', '-r150');
end


function plot_iteration(figdir, ithist)
%PLOT_ITERATION  Iteration convergence: error norm per iterate
%   Plots error norms from two methods (OPS and OAS) on a semilogy scale
%   against iteration count, for two geometries side-by-side. Includes
%   a dashed reference line showing OPS theoretical convergence.
%
%   ithist is a 1x2 struct array indexed by geometry (ig=1..2):
%     ithist(ig).name: char
%     ithist(ig).e: 1x2 cell; e{im} is a column vector of error norms
%                   for method im (1=OPS, 2=OAS)
%     ithist(ig).rho_theory: scalar, the OPS theoretical convergence factor
%
%   Outputs a two-panel semilogy figure with grid, box, and legend.

  figure('Position', [100 100 1100 480]);

  colors = [0.85 0.33 0.10; 0.00 0.45 0.74];
  method_names = {'OPS', 'OAS'};

  for ig = 1:2
    subplot(1, 2, ig);
    hold on;

    % Plot error curves for both methods
    for im = 1:2
      n_iter = length(ithist(ig).e{im});
      semilogy(1:n_iter, ithist(ig).e{im}, '-o', ...
               'Color', colors(im,:), 'LineWidth', 1.5, 'MarkerSize', 5, ...
               'MarkerFaceColor', colors(im,:), ...
               'DisplayName', method_names{im});
    end

    % OPS theoretical reference line: e_0 * kappa^n
    n_iter = length(ithist(ig).e{1});
    e0 = ithist(ig).e{1}(1);
    kappa = ithist(ig).rho_theory;
    semilogy(1:n_iter, e0*kappa.^(0:(n_iter-1)), 'k--', ...
             'LineWidth', 1.2, ...
             'DisplayName', 'OPS \rho_{theory}^n');

    % subplot created the axes linear and hold was already on, so semilogy
    % above did not switch the scale; set it explicitly
    set(gca, 'YScale', 'log');
    grid on; box on;
    xlabel('n, iteration', 'FontSize', 9);
    ylabel('|e^n|_\infty', 'FontSize', 9);
    title(ithist(ig).name, 'FontSize', 10);
    legend('Location', 'best', 'FontSize', 8);
  end

  print(gcf, fullfile(figdir, 'ellipse_schwarz_ddm_iteration'), '-dpng', '-r150');
end


function plot_alpha(figdir, sweep, alphas, GEO)
%PLOT_ALPHA  observed contraction factor against the Robin parameter
%   One panel per geometry.  In each, the four scheme x method combinations
%   are drawn with colour by method (OPS / OAS) and line style by scheme
%   (dashed d_k2, solid exact), against the OPS theoretical factor.  The
%   theory curve is identical for both methods and both schemes and is only
%   meaningful for OPS; the paper gives no closed form for OAS.
%
%   sweep is a 2x2x2 struct array indexed sweep(ig, isc, im) with fields
%     .rho    column vector over alphas, observed factor, may contain NaN
%     .rho_theory  column vector over alphas, OPS theoretical factor
%   alphas is the column of swept Robin parameters and GEO names the two
%   geometries.

  figure('Position', [100 100 1200 480]);

  colors_method = [0.85 0.33 0.10; 0.00 0.45 0.74];   % OPS, OAS
  scheme_styles = {'--', '-'};                        % d_k2, exact
  scheme_names  = {'d_k2', 'exact'};
  method_names  = {'OPS', 'OAS'};

  for ig = 1:2
    subplot(1, 2, ig);
    hold on;

    for isc = 1:2
      for im = 1:2
        rho = sweep(ig, isc, im).rho;
        ok = isfinite(rho);
        if any(ok)
          plot(alphas(ok), rho(ok), [scheme_styles{isc} 'o'], ...
               'Color', colors_method(im,:), 'LineWidth', 1.5, ...
               'MarkerSize', 5, 'MarkerFaceColor', colors_method(im,:), ...
               'DisplayName', sprintf('%s, %s', scheme_names{isc}, method_names{im}));
        end
      end
    end

    kappa_theory = sweep(ig, 1, 1).rho_theory;
    ok = isfinite(kappa_theory);
    if any(ok)
      plot(alphas(ok), kappa_theory(ok), 'k--', 'LineWidth', 1.2, ...
           'DisplayName', 'OPS \rho(\alpha), theory (none exists for OAS)');
    end

    plot([1 1], [0 1], 'k:', 'LineWidth', 1.0, 'DisplayName', '\alpha = 1');

    grid on; box on;
    xlabel('\alpha', 'FontSize', 9);
    ylabel('\rho', 'FontSize', 9);
    title(GEO{ig}, 'FontSize', 10);
    xlim([0 max(alphas)]);
    ylim([0 1]);
    legend('Location', 'best', 'FontSize', 7);
  end

  print(gcf, fullfile(figdir, 'ellipse_schwarz_ddm_alpha'), '-dpng', '-r150');
end


function plot_accuracy(figdir, hvals, errtab, ratetab, labels)
%PLOT_ACCURACY  Convergence of solution error vs grid spacing
%   loglog plot of L_inf surface error against dx for eight method
%   combinations (indexed by geometry, scheme, method). Each curve shows
%   one row of errtab with its fitted convergence rate in the legend.
%   Rows that are entirely NaN are skipped. O(dx) and O(dx^2) slope guides
%   are added via slopeguides().
%
%   hvals: 1xK vector of grid spacings (descending)
%   errtab: 8xK matrix of L_inf surface errors
%   ratetab: 8x1 vector of fitted convergence rates
%   labels: 1x8 cell of char labels in (geometry, scheme, method) order

  figure('Position', [100 100 900 700]);

  markers = {'o', 's', 'd', '^', 'v', '>'};
  colors = [0 0 0; 0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19; ...
            0.49 0.18 0.56; 0.30 0.75 0.93];

  hold on;

  for i = 1:size(errtab, 1)
    e = errtab(i, :);

    % Skip rows that are entirely NaN
    if all(~isfinite(e))
      continue;
    end

    % Plot only finite values
    valid = isfinite(e) & (e > 0);
    if any(valid)
      marker_idx = mod(i-1, length(markers)) + 1;
      color_idx = mod(i-1, size(colors, 1)) + 1;

      loglog(hvals(valid), e(valid), ['-' markers{marker_idx}], ...
             'Color', colors(color_idx,:), 'LineWidth', 1.5, ...
             'MarkerSize', 6, 'MarkerFaceColor', colors(color_idx,:), ...
             'DisplayName', sprintf('%s, rate %.2f', labels{i}, ratetab(i)));
    end
  end

  % Add slope guides
  base = 3*max(errtab(:, 1));
  slopeguides(hvals, base);

  set(gca, 'XScale', 'log', 'YScale', 'log');
  grid on; box on;
  xlabel('dx', 'FontSize', 9);
  ylabel('L_\infty error', 'FontSize', 9);
  legend('Location', 'best', 'FontSize', 8);

  print(gcf, fullfile(figdir, 'ellipse_schwarz_ddm_accuracy'), '-dpng', '-r150');
end


function slopeguides(hvals, base)
%SLOPEGUIDES  Reference lines for O(dx) and O(dx^2) convergence
%   Draws two dashed reference curves at the specified base level,
%   scaled linearly and quadratically by the ratio of grid spacings.
%
%   hvals: vector of grid spacings
%   base: baseline error level for scaling

  plot(hvals, base*(hvals/hvals(1)), 'k:', 'LineWidth', 1.2, ...
       'DisplayName', 'O(dx)');
  plot(hvals, base*(hvals/hvals(1)).^2, 'k--', 'LineWidth', 1.2, ...
       'DisplayName', 'O(dx^2)');
end


function [kap, dk] = curvature_mismatch(t, yflip2, a, b)
%CURVATURE_MISMATCH  interface curvature and the rotated curvature mismatch
%   At the cut parameter t on the ellipse, returns
%
%     kap  = ||d^2 gamma / ds^2||, the geometric curvature there, which is the
%            same for both branches because they are the same ellipse;
%     dk   = ||c2 - R*c1||, the curvature-vector mismatch the exact tangent
%            rotation R leaves behind, which the companion curvature-mismatch
%            script identifies as the quantity controlling the angle-rotation
%            error: O(h) when it is nonzero, O(h^2) when it vanishes.
%
%   yflip2 is the embedding of the SECOND branch: +1 leaves it alone (the
%   plain ellipse, a smooth join) and -1 reflects it across a horizontal line
%   (the cut-and-reflect manifold, a corner).  c1 is branch one's curvature
%   vector and c2 branch two's in its own embedding; R is the rotation taking
%   branch one's outward tangent onto the negative of branch two's, the same
%   angle make_glue applies.
%
%   On the plain ellipse R is the identity and c2 = c1, so dk = 0 exactly.  On
%   the reflected manifold the join is a mirror image, the two branches carry
%   equal curvature magnitudes, and dk = |kappa1 + kappa2| = 2*kap identically.

  d1 = [-a*sin(t), b*cos(t)];
  d2 = [-a*cos(t), -b*sin(t)];
  s2 = d1*d1.';
  cv = (d2*s2 - d1*(d1*d2.'))/s2^2;      % d^2 gamma / ds^2
  kap = norm(cv);

  T = d1/sqrt(s2);
  Mref = [1 0; 0 yflip2];
  tau1 = -T;                              % branch one's outward tangent at t
  tau2 = (Mref*T.').';                    % branch two's, in its embedding
  th = wrapangle(atan2(-tau2(2), -tau2(1)) - atan2(tau1(2), tau1(1)));
  R = [cos(th) -sin(th); sin(th) cos(th)];
  dk = norm(Mref*cv.' - R*cv.');
end


function plot_rate_curvature(figdir, dkap, ratefit, Cfirst, Cspread, GEO, SCH, MET)
%PLOT_RATE_CURVATURE  how the singular interface degrades the solution
%   Left panel: the fitted convergence rate against the rotated curvature
%   mismatch.  Right panel: the first-order constant mean(err/h) against the
%   same, with the spread of err/h over the refinement drawn as a bar.
%
%   The two panels answer different questions.  The RATE is what the rotation
%   costs in order: second on the smooth control, first once delta_kappa is
%   nonzero, and flat in delta_kappa thereafter.  The CONSTANT is what it
%   costs in accuracy, and that is where the size of the mismatch shows up.
%   The spread bars matter: at a singular interface the constant jitters as
%   the interface moves relative to the lattice, so no single pair of levels
%   determines the rate and only the trend over many levels is meaningful.
%
%   dkap is numel(GEO) x (number of cuts); ratefit, Cfirst and Cspread are
%   (numel(GEO)*numel(SCH)*numel(MET)) x (number of cuts) in the driver's row
%   order.  Only the parallel algorithm is drawn, since both algorithms reach
%   the same fixed point and so have identical surface errors.

  figure('Position', [100 100 1150 470]);
  colors = [0.85 0.33 0.10; 0.00 0.45 0.74];   % plain, flipped
  marks = {'o', 's'};
  nS = numel(SCH);
  nM = numel(MET);

  subplot(1, 2, 1);
  hold on;
  for ig = 1:numel(GEO)
    for isc = 1:nS
      ir = ((ig-1)*nS + (isc-1))*nM + 1;
      plot(dkap(ig,:), ratefit(ir,:), ['-' marks{isc}], 'Color', colors(ig,:), ...
           'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor', colors(ig,:), ...
           'DisplayName', sprintf('%s, %s', GEO{ig}, SCH{isc}));
    end
  end
  xl = xlim;
  plot(xl, [1 1], 'k:',  'LineWidth', 1.2, 'DisplayName', 'rate 1');
  plot(xl, [2 2], 'k--', 'LineWidth', 1.2, 'DisplayName', 'rate 2');
  xlim(xl);  ylim([0 2.6]);
  grid on; box on;
  xlabel('\delta\kappa = ||c_2 - Rc_1||', 'FontSize', 10);
  ylabel('fitted L_\infty rate', 'FontSize', 10);
  title('order: second when \delta\kappa = 0, first otherwise', 'FontSize', 10);
  legend('Location', 'southwest', 'FontSize', 8);

  subplot(1, 2, 2);
  hold on;
  ig = 2;                       % the reflected manifold; the plain one has C ~ 0
  for isc = 1:nS
    ir = ((ig-1)*nS + (isc-1))*nM + 1;
    x = dkap(ig,:);
    y = Cfirst(ir,:);
    ok = isfinite(x) & isfinite(y);
    % the jitter of err/h over the refinement, drawn about the mean
    for q = find(ok)
      lo = y(q)*2/(1 + Cspread(ir,q));
      hi = lo*Cspread(ir,q);
      plot([x(q) x(q)], [lo hi], '-', 'Color', colors(ig,:), 'LineWidth', 0.8, ...
           'HandleVisibility', 'off');
    end
    plot(x(ok), y(ok), ['-' marks{isc}], 'Color', colors(ig,:), ...
         'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor', colors(ig,:), ...
         'DisplayName', sprintf('%s, %s', GEO{ig}, SCH{isc}));
    if (isc == nS)
      pf = polyfit(x(ok), y(ok), 1);
      xs = linspace(0, max(x(ok)), 50);
      plot(xs, polyval(pf, xs), 'k--', 'LineWidth', 1.2, ...
           'DisplayName', sprintf('fit %.3f\\delta\\kappa + %.3f', pf(1), pf(2)));
    end
  end
  grid on; box on;
  xlabel('\delta\kappa = ||c_2 - Rc_1||', 'FontSize', 10);
  ylabel('mean err/h over the refinement', 'FontSize', 10);
  title('accuracy: the constant grows with the mismatch', 'FontSize', 10);
  legend('Location', 'northwest', 'FontSize', 8);

  print(gcf, fullfile(figdir, 'ellipse_schwarz_ddm_rate_curvature'), ...
        '-dpng', '-r150');
end


function val = interp_at(s, u, pts)
%INTERP_AT  one subdomain's interpolant evaluated at points of the curve
%   The same degree-p interpolation the operator itself uses, restricted to
%   this subdomain's inner band.  Errors if a stencil leaves that band.

  n = size(pts, 1);
  [~, Ej, Es] = interp2_matrix(s.x1d, s.y1d, pts(:,1), pts(:,2), s.p);
  ns = (s.p + 1)^2;
  JJ = reshape(s.inv_inner(Ej), n, ns);
  SS = reshape(Es, n, ns);
  if any(JJ(:) == 0)
    error('an interpolation stencil at an interface leaves the inner band');
  end
  % u(JJ) takes the orientation of u when both are vectors, so reshape it
  % back to n-by-ns before the row sum; with n = 1 it would otherwise
  % broadcast against SS into an ns-by-ns array
  val = sum(SS.*reshape(u(JJ), n, ns), 2);
end


function g = interface_gap(sub, u, a, b, cen)
%INTERFACE_GAP  how far apart the two branches put u at the interfaces
%   At each interface point both adjacent subdomains carry a value for u
%   there, and the glue only makes them agree to the order of the scheme.
%   Returns the largest discrepancy over the interfaces, which is the trace
%   consistency of the coupling.

  g = 0;
  for j = 1:numel(sub)
    sj = sub{j};
    sk = sub{sj.nbr_b};
    v = arcpoint([sj.yflip sj.yoff], sj.tb, a, b, cen);
    vk = arcpoint([sk.yflip sk.yoff], sk.ta, a, b, cen);
    g = max(g, abs(interp_at(sj, u{j}, v) - interp_at(sk, u{sj.nbr_b}, vk)));
  end
end


function d = min_branch_gap(t1, t2, ycut, flipped, a, b, cen)
%MIN_BRANCH_GAP  closest approach of the two branches away from the interfaces
%   The reflected manifold can come close to touching itself well away from
%   the cut points, and once bw*dx exceeds half that distance the two bands
%   interpenetrate.  Each subdomain still has its own arc and its own closest
%   point map, so the discretization stays well posed, but the geometry is
%   tight and the transmission is worth distrusting there.  Returns inf on the
%   plain ellipse, where the two arcs are parts of one convex curve.

  if ~flipped
    d = inf;
    return;
  end
  nq = 2000;
  tu = linspace(t1, t2, nq)';
  tl = linspace(t2, t1 + 2*pi, nq)';
  up = arcpoint([1 0], tu, a, b, cen);
  lo = arcpoint([-1 2*ycut], tl, a, b, cen);
  m = max(2, round(nq/50));           % drop the shared endpoints
  up = up(m:end-m,:);
  lo = lo(m:end-m,:);
  d = inf;
  for i = 1:size(up,1)
    d = min(d, min(hypot(lo(:,1) - up(i,1), lo(:,2) - up(i,2))));
  end
end
