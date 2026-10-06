%% A cap folded into an ellipsoid: does the glued solve still converge?  (vGMM)
%
% The three-dimensional companion to
% example_ellipse_cut_branch_consistency_vgmm.m.  There a plane curve was
% cut at a point and one arc reflected, so the two halves met at a corner.
% Here a surface is cut along a CURVE and one piece reflected, so the two
% pieces meet along a whole circle of corner.  Everything else -- the
% operator, the glue, the manufactured solution, the four cross-branch
% measures -- is carried over unchanged, which is the point: the question
% is what the extra dimension does to the junction, not what a different
% method would do.
%
% THE SURFACE.  The ellipsoid of revolution about the z axis,
%
%     gamma(phi, th) = cen + [a*sin(phi)*cos(th), a*sin(phi)*sin(th), ...
%                             c*cos(phi)],        phi in [0, pi],
%
% cut by the plane z = zcut.  Branch 1 is the bowl below the cut,
% phi in [phi_c, pi]; branch 2 is the cap above it, phi in [0, phi_c].
%
%   flipped = false   the two pieces as they stand.  They join smoothly,
%                     the exact glue rotation is the identity, and the
%                     curvature mismatch delta_kappa below is exactly zero.
%
%   flipped = true    the cap is reflected DOWN across the cut plane,
%                     (x,y,z) -> (x, y, 2*zcut - z).  It becomes a dimple
%                     inside the bowl, the two pieces meet along the cut
%                     circle in a genuine crease, and delta_kappa is a
%                     definite nonzero number.  This is the default.
%
% A reflection is an isometry, so the flipped surface carries the same
% metric as the ellipsoid in the same (phi, th): the manufactured solution
% and its Laplace-Beltrami are the same functions of (phi, th) on both, and
% the errors are directly comparable.
%
% THE GLUE.  A band node whose closest point on its own branch is a point V
% of the cut circle is rotated about V and interpolated on the other
% branch, exactly as in two dimensions.  In three dimensions the rotation
% axis is the tangent to the cut circle at V, i.e. the azimuthal direction
% e_th.  Two facts make this the honest lift of the planar construction.
% First, V is the closest point of the node on a circle, so the node lies
% in V's own meridian half plane and its offset from V has no e_th
% component; a rotation about e_th therefore keeps it in that half plane.
% Second, the surface is one of revolution, so the closest point of the
% rotated node on the other branch lies in the same half plane.  The whole
% construction is the planar one applied meridian by meridian, and the
% angle is the planar angle in the (rho, z) coordinates of that half plane.
%
% THE RIGHT-HAND SIDE AT A JUNCTION ROW.  A rotated row's unknown does not
% stand for u at that row's own closest point -- that point is on the cut
% circle -- but for u at the rotated target on the other branch, which is
% up to bw*dx of arclength away.  Its forcing is therefore f at the ROTATED
% TARGET.  Using f at the row's own clamped closest point instead, as the
% planar script does, leaves a residual f(target) - f(V) = O(dx) on every
% one of those rows; with O(1) of them in the plane that is invisible, with
% O(1/dx) of them around a circle it is not.  It is the one place where
% this script deliberately differs from its planar companion.
%
% That choice needs a caveat, because it is exactly the convention
% [MMC 2026b] singles out as the dangerous one: evaluating the forcing on
% the OPPOSITE side without correcting a trace jump can leave
% rho_(f,h) = O(1) rather than O(dx), which would promote h*rho_(f,h) from
% O(dx^2) to O(dx) and put it on a level with the junction term itself.
% It is safe here only because [f] = 0 -- u and f are the same functions of
% (phi, th) on both branches, so the two one-sided traces at the cut circle
% are the same number and there is no jump to correct.  Anything that
% breaks that symmetry, a forcing written per branch rather than per
% (phi, th) most obviously, would need f at the SOURCE-side edge point
% instead, or an explicit trace correction.
%
% DELTA_KAPPA.  The quantity the junction analysis says governs the rate is
% the mismatch of the two branches' vector-valued second fundamental forms
% once the glue rotation has been applied,
%
%     delta_kappa = || II_1 - R II_2 R^T ||_F.
%
% Here both branches are pieces of the same ellipsoid met at the same
% phi_c, so they share the meridian and parallel curvatures
%
%     kappa_m = a*c/G^(3/2),   kappa_p = c/(a*sqrt(G)),
%     G(phi)  = a^2*cos(phi)^2 + c^2*sin(phi)^2,
%
% and the mismatch collapses to |n_1 - R n_2| * sqrt(kappa_m^2+kappa_p^2)
% with n the inward unit normals.  It is computed that way below rather
% than assumed.  Two consequences are worth having in mind while reading
% the tables.  On the unflipped cut R is the identity and n_2 = n_1, so
% delta_kappa vanishes identically and the glued solve should recover the
% second-order rate of the uncut one.  On the flipped cut R carries n_2 to
% -n_1 -- the composition of the reflection with a rotation that matches
% the tangents is a reflection of the meridian plane fixing the tangent --
% so |n_1 - R n_2| = 2 and
%
%     delta_kappa = 2*sqrt(kappa_m^2 + kappa_p^2)       (flipped)
%
% at every cut height, varying only through where on the meridian the cut
% falls.  Cutting near the equator gives the large value, near the pole the
% small one, and that spread is what the cut list is chosen to span.
%
% HOW THAT LINES UP WITH [MMC 2026b].  Its surface bound is
%
%     ||Q_h u_h - u||_inf <= C{h^q + h^(p-1) + h(D + J_2) + h^2
%                              + eps_R + h*rho_(f,h)},
%
% which for q = 2, p = 3 is C{h^2 + h(D + J_2) + eps_R} when the forcing
% convention is source-consistent.  Two of those constants are worth having
% in mind while reading the tables, and neither is delta_kappa itself.
%
%   D    is the sup over the edge of |b_2^perp - b_1^perp|, the mismatch of
%        only the MIXED and TRANSVERSE parts (b_sr, b_rr) of the second
%        fundamental form in the matched edge frame.  Here the meridian and
%        the parallel are the principal directions, so b_sr = 0 on both
%        branches and D = 2*kappa_m -- the along-edge curvature kappa_p,
%        which delta_kappa does carry, drops out of D because the
%        transmission conditions already match U_ss.  delta_kappa is the
%        "full matched-frame second fundamental form difference" that paper
%        calls a valid but less sharp replacement for D; on this family it
%        runs 18 to 34 per cent above D and is monotone in the same
%        direction, so ranking the cuts by either gives the same order.
%
%   J_2  is the intrinsic defect [U_rr] = (k_2 - k_1) U_r - [f], with k_i
%        the geodesic curvature of the cut circle inside branch i.  It
%        vanishes here, and both of its terms vanish separately: the two
%        branches are mirror images across the cut plane so k_2 = k_1, and
%        u and f are the same functions of (phi, th) on both so [f] = 0.
%        That is a different route to J_2 = 0 than the two-sphere example
%        of that paper, where the two terms are individually nonzero and
%        cancel.  So on the flipped surface the whole junction defect here
%        is extrinsic, and the bound reads C{h^2 + h*D + eps_R}.
%
% WHY SO MANY GRID SIZES.  The surface error of the flipped solve is not a
% smooth function of dx.  Its size is set by the net, signed junction
% residual summed around the cut circle, and that sum is a near
% cancellation between the two branches -- each contributes a systematic
% total of one sign and the other the opposite sign -- so what survives
% depends on where the cut circle happens to fall between grid planes.
% Neighbouring dx can differ by a factor of several, and level-to-level
% rates are meaningless.  What is meaningful is the least-squares slope
% through many levels, which is what the "fitted" line of each table
% reports and the only rate quoted anywhere below.  In the plane the same
% cancellation exists but the junction is two points rather than a circle,
% the sums are over O(1) rows instead of O(1/dx), and the noise never
% surfaces.
%
% Run headlessly, from this directory:
%   matlab -batch "example_ellipsoid_flipped_cap_branch_consistency_vgmm"
%
%
% References
%
% # [vGMM 2013]  Ingrid von Glehn, Thomas Marz, and Colin B. Macdonald.
%   An embedded method-of-lines approach to solving partial
%   differential equations on surfaces.  2013.
% # [MMC 2026]  Thomas Marz, Colin B. Macdonald, and Yujia Chen.
%   Consistency and stability of closest point iterations.  Draft.
% # [MMC 2026b]  Closest point rotation across a shared surface edge: local
%   proofs and conditional convergence in R^3.  Draft,
%   output/pdf/cpm_surface_edge_analysis.pdf.  Theorem 6.4 is the bound
%   quoted above; Assumption 6.3 is the discrete stability it is
%   conditional on, and Proposition 7.1 is what the uncut baseline below
%   measures.
% # [Ruuth & Merriman 2008]  Steven J. Ruuth and Barry Merriman.  A simple
%   embedding method for solving partial differential equations on
%   surfaces.  2008.  (the bandwidth formula)

% Resolve dependencies relative to this example.
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', '..', 'cp_matrices'));
addpath(fullfile(here, '..', '..', 'surfaces'));
addpath(fullfile(here, '..', 'surfaces'));
addpath(fullfile(here, '..', 'rotations'));


%% Parameters

cen = [0 0 0];
a = 1.0;     % equatorial semi-axis
c = 0.65;    % polar semi-axis

% The manifold.  false: the plain ellipsoid, where the two pieces join
% smoothly, the exact glue rotation is the identity and delta_kappa is
% zero.  true: the cap is reflected down across the cut plane, so the
% surface has a crease along the whole cut circle.
flipped = true;
if flipped
  geoname = 'flipped';
else
  geoname = 'ellipsoid';
end

% Cut heights, written as sin(t_c) = (zcut - cen(3))/c so that they are read
% off the meridian rather than the z axis.  On this family the crease angle
% and delta_kappa move together: both grow as the cut slides from the pole
% towards the equator, so a cut list is a list of delta_kappa.
%
% Two ends of the family are excluded.  zcut = cen(3) is excluded for the
% same reason the planar script excludes the half cut: reflecting the cap
% across the equatorial plane lands it exactly on the bowl, the two branches
% coincide, and no closest point is defined.  And a cut far enough towards
% the equator that the crease angle passes roughly ninety degrees is
% excluded too, for a reason measured rather than assumed: there the
% grid-alignment scatter described at the top swallows the signal and the
% surface error stops converging at all.  Runs at s0 = 0.5 and 0.25 (crease
% 97 and 137 degrees, delta_kappa 3.50 and 4.61) gave fitted slopes of 0.62
% and 0.68 with errors that bounce by a factor of ten between neighbouring
% levels -- worth knowing, and worth staying below.  These four cuts sit in
% the range where the scheme is in its asymptotic regime.
s0_list = [0.6 0.7 0.8 0.9];
zcut_list = cen(3) + c*s0_list;

dim = 3;    % dimension
p = 3;      % interpolation degree
order = 2;  % Laplacian order
% [Ruuth & Merriman 2008], with the 1.0002 a safety factor.  The two
% tangential directions of a surface each contribute an interpolation
% half-stencil and the normal one an interpolation half-stencil plus a
% Laplacian half-stencil, which is what the composition E*L reads.
bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ((order/2+(p+1)/2)^2));

% A band node has a unique closest point on its branch only while bw*dx
% stays under the smallest radius of curvature, which on an oblate
% ellipsoid is c^2/a at the equator.  0.65 rather than the planar script's
% 0.5 because three-dimensional bands are expensive and the coarse levels
% are needed for the span of the rate fit.
hvals = 0.065*0.82.^(0:9);
hvals = hvals(bw*hvals < 0.65*min(c^2/a, a^2/c));
nh = length(hvals);
if (nh < 4)
  error('too few usable grid sizes for a = %g, c = %g', a, c);
end

% The matrix-free solve; M is never assembled.  E*L in three dimensions has
% some hundreds of entries per row, so forming it is neither necessary nor
% affordable at the finer levels; E and L separately are 64 and 7 per row.
% maxit is the number of restart cycles, so the cap is maxit*restart
% iterations; the finest level of the widest cut is the one that needs them.
itertol = 1e-10;
gmres_restart = 40;
gmres_maxit = 400;

% Which cut gets the picture of the error on the surface itself, and on what
% grid.  That figure costs a solve of its own, so it is drawn for the cut
% heights named here rather than for all of them: set z_3d to a z from
% zcut_list, to several of them, or to [] for none.  The solve is
% display-only -- fine enough that the error field has the shape it will
% keep, coarse enough to be cheap, and independent of hvals -- so none of
% the numbers in the tables come from it, and trimming s0_list and hvals is
% a legitimate way to regenerate just this picture.
z_3d = 0.390;
dx_show = 0.0241;

% caught here rather than after the study has run
ztol = 1e-6*max(1, c);
for iz = 1:numel(z_3d)
  if ~any(abs(zcut_list - z_3d(iz)) <= ztol)
    error('z_3d = %g is not one of the cut heights; zcut_list is %s', ...
          z_3d(iz), mat2str(zcut_list, 4));
  end
end

figdir = fullfile(here, '..', 'figs');   % .png output goes here

if flipped
  exactlabel = 'exact R';
  baselabel = 'uncut vGMM, crease ignored';
else
  exactlabel = 'exact R = I';
  baselabel = 'exact vGMM (uncut)';
end
varlabels = {'rotation from d_k', 'rotation from d_{k2}', exactlabel};
nvar = length(varlabels);
varlabels4 = [{baselabel} varlabels];


%% Manufactured solution, in the ellipsoid parameters (phi, th)
%
%   u = cos(phi) + 0.5*cos(2*phi) + B*sin(phi)^2*cos(2*th),
%
% which in Cartesian coordinates is a polynomial in z plus
% B*((x-cen(1))^2 - (y-cen(2))^2)/a^2, so it is smooth at both poles where
% the (phi, th) chart is not.  On a surface of revolution with meridian
% (rho(phi), z(phi)) and G = rho'^2 + z'^2 the metric is G dphi^2 +
% rho^2 dth^2, and for w(phi)*cos(m*th),
%
%   laplacian_S = w''/G + (cot(phi)/G - G'/(2 G^2))*w' - m^2*w/rho^2.
%
% Both cot(phi)*w' below are written already divided through by sin(phi):
% u's phi-derivatives carry a factor of sin(phi), and m^2*w/rho^2 is the
% constant 4*B/a^2, so nothing here is singular at a pole.
B = 0.4;
ufun = @(ph,th) cos(ph) + 0.5*cos(2*ph) + B*sin(ph).^2.*cos(2*th);
ffun = @(ph,th) ufun(ph,th) - lapS(ph,th,a,c,B);


%% One cut height at a time

results = struct('zcut', {}, 'dkappa', {}, 'dkmean', {}, 'kappa', {}, ...
                 'phic', {}, 'thetaex', {}, 'dx', {}, 'gap', {}, ...
                 'vtx', {}, 'branch', {}, 'allgap', {}, 'nrot', {}, ...
                 'nshare', {}, 'thetamax', {}, 'rate', {}, 'ratev', {}, ...
                 'err', {}, 'rateerr', {});

for ci = 1:length(zcut_list)
  zc = zcut_list(ci);
  s0 = (zc - cen(3))/c;
  if (abs(s0) > 0.95)
    error('cut at z = %g misses the ellipsoid or grazes it (sin t = %g)', zc, s0);
  end
  if (flipped && abs(s0) < 0.05)
    error(['the flipped surface degenerates at z = %g: reflecting the cap ' ...
           'across the equatorial plane puts it on top of the bowl'], zc);
  end

  % the meridian ellipse is (rho, z) = (a*cos t, cen(3) + c*sin t), so the
  % arc parameter t and the polar angle phi are related by t = pi/2 - phi.
  % Branch 1 (the bowl) is t in [-pi/2, tc], branch 2 (the cap) t in
  % [tc, pi/2]; the ends t = -+pi/2 are the poles, where the surface of
  % revolution closes up and nothing is clamped.
  tc = asin(s0);
  phic = pi/2 - tc;
  rcut = a*cos(tc);
  [kappam, kappap] = principal_curvatures(phic, a, c);

  geo = cell(2,1);
  geo{1} = make_cap(-pi/2, tc, false, zc);
  geo{2} = make_cap(tc, pi/2, flipped, zc);
  geowhole = make_union(geo);

  %% Exact endpoint tangents, the exact glue rotation, and delta_kappa
  % Everything here lives in the (rho, z) coordinates of a meridian half
  % plane.  unit_tangent(t, sgn) with sgn = +1 at an arc's upper end and -1
  % at its lower end gives the tangent pointing OUT of that arc, away from
  % its own points; reflecting branch 2 negates the z component of its
  % tangent, which is what opens the crease.
  tauex = [unit_tangent(tc,  1, a, c); ...
           unit_tangent(tc, -1, a, c)];
  if flipped
    tauex(2,2) = -tauex(2,2);
  end
  % the glue is stored as the (cos, sin) pair Rodrigues' formula gives
  % from the tangents, never as an angle; thetaex is derived from it for
  % the report and for the curvature mismatch below
  csex = glue_cs(tauex(1,:), tauex(2,:));
  thetaex = cs_angle(csex);
  % thetaex(2) is branch 2's rotation, the one that carries its frame onto
  % branch 1's continuation, which is the R the mismatch is measured with
  [dkappa, dkmean] = curvature_mismatch(phic, csex(2,:), a, c, flipped);

  err = nan(nvar+1, nh);    % surface L_inf error, baseline then the three
  gap = nan(nvar, nh);      % max |e_A - e_B| at the shared rotated nodes
  vtx = nan(nvar, nh);      % max |u_A - u_B| on the cut circle
  branch = nan(nvar, nh);   % max(|e_A|, |e_B|) at the same nodes
  allgap = nan(nvar, nh);   % the same gap over the whole overlap
  nrot = nan(1, nh);        % how many shared rotated nodes there are
  nshare = nan(1, nh);      % how many shared nodes there are
  dcrease = nan(1, nh);     % where the exact-rotation solve's worst point is
  thetamax = nan(2, nh);    % max |theta - theta_exact|, for reference

  fprintf(['\n==== vGMM, %s, cut at z = %g: sin t = %.4f, phi_c = %.4f, ' ...
           'r_cut = %.4f ====\n'], geoname, zc, s0, phic, rcut);
  fprintf(['cut-circle curvatures: kappa_m = %.4f, kappa_p = %.4f;  ' ...
           'exact glue angle %.4f rad (%.1f deg)\n'], ...
          kappam, kappap, thetaex(1), thetaex(1)*180/pi);
  fprintf(['delta_kappa = ||II_1 - R II_2|| = %.4f   ' ...
           '(mean-curvature-vector mismatch %.4f)\n'], dkappa, dkmean);

  for k = 1:nh
    dx = hvals(k);

    %% Construct a grid, and the band and operators of the two branches
    [x1d, y1d, z1d, cand] = make_grid(geo, a, c, cen, phic, dx, bw);
    [xc, yc, zc3] = grid_points(x1d, y1d, z1d, cand);
    br = cell(2,1);
    for j = 1:2
      br{j} = setup_band(x1d, y1d, z1d, dx, p, order, bw, cand, ...
                         xc, yc, zc3, a, c, cen, geo{j});
      br{j}.rim = rim_rows(br{j}, a, c, cen, rcut, zc);
    end

    %% Baseline: vGMM on the whole surface, one band, no cut
    swhole = setup_band(x1d, y1d, z1d, dx, p, order, bw, cand, ...
                        xc, yc, zc3, a, c, cen, geowhole);
    clear xc yc zc3
    outw = vgmm_solve({swhole}, [], x1d, y1d, z1d, p, dim, a, c, cen, ...
                      ufun, ffun, itertol, gmres_restart, gmres_maxit);
    err(1,k) = outw.errsurf;
    clear swhole outw

    %% The nodes the two branches have in common
    sh = shared_points(br, geo, a, c, cen, csex);
    nrot(k) = sum(sh.rot);
    nshare(k) = length(sh.i1);
    if (nrot(k) == 0)
      error(['at dx = %g no node clamped to the cut circle by one branch ' ...
             'is carried by the other; there is nothing to compare'], dx);
    end

    %% The glue rotation each scheme produces
    % thetamax is the error in the angle, |theta - theta_exact|, not
    % |theta| itself: on the flipped surface the exact angle is not zero.
    cs = zeros(2, 2, nvar);      % (branch, [cos sin], variant)
    cs(:,:,nvar) = csex;         % the last variant is the exact rotation
    for scheme = 1:2
      tau = [cp_tangent(br{1}, cen, scheme); cp_tangent(br{2}, cen, scheme)];
      cs(:,:,scheme) = glue_cs(tau(1,:), tau(2,:));
      % the angle of the relative rotation R_exact' * R, which is in
      % (-pi, pi] by construction: no difference of two angles and no
      % rewrap.  See ../rotations/rot2d_err.m.
      thetamax(scheme,k) = ...
          max(abs(rot2d_err(cs(:,1,scheme), cs(:,2,scheme), ...
                            csex(:,1), csex(:,2))));
    end

    %% Solve, once per variant, and compare the branches node by node
    for iv = 1:nvar
      out = vgmm_solve(br, cs(:,:,iv), x1d, y1d, z1d, p, dim, a, c, cen, ...
                       ufun, ffun, itertol, gmres_restart, gmres_maxit);
      if (out.rowsum > 1e-10)
        error('extension matrix rows do not sum to one (%g)', out.rowsum);
      end
      err(iv+1,k) = out.errsurf;
      if (iv == nvar)
        dcrease(k) = abs(out.phierr - phic) * ...
                     sqrt(a^2*cos(out.phierr)^2 + c^2*sin(out.phierr)^2);
      end
      % each branch against the value its own unknown stands for; on the
      % unflipped surface those are the same number and eA - eB is just
      % u_A - u_B
      eA = out.ubr{1}(sh.i1) - ufun(sh.phA, sh.thA);
      eB = out.ubr{2}(sh.i2) - ufun(sh.phB, sh.thB);

      gap(iv,k) = norm(eA(sh.rot) - eB(sh.rot), inf);
      branch(iv,k) = max(norm(eA(sh.rot), inf), norm(eB(sh.rot), inf));
      allgap(iv,k) = norm(eA - eB, inf);

      % the same comparison on the manifold: each branch's interpolant on
      % the cut circle, where both of them stand for u(phi_c, th)
      V = rim_samples(rcut, zc, cen, dx);
      uAv = interp_at(x1d, y1d, z1d, p, br{1}, out.ubr{1}, V);
      uBv = interp_at(x1d, y1d, z1d, p, br{2}, out.ubr{2}, V);
      vtx(iv,k) = norm(uAv - uBv, inf);
      clear out
    end

    % dcrease is where the exact-rotation solve's worst point sits, as an
    % arclength from the cut circle in multiples of dx: the test of whether
    % the crease is what hurts it
    fprintf(['dx = %-8.4g n %-8d shared %-6d rot %-6d both %-5d ' ...
             'dth %8.2e %8.2e   err %9.3e (%.1f dx from the crease)   ' ...
             'gap %9.3e %9.3e %9.3e\n'], ...
            dx, br{1}.n + br{2}.n, nshare(k), nrot(k), sh.nboth, ...
            thetamax(1,k), thetamax(2,k), err(nvar+1,k), dcrease(k)/dx, ...
            gap(:,k));
    clear br sh
  end

  %% Convergence table
  % Only the fitted, least-squares slope through all the levels is a rate;
  % see the note at the top about why the level-to-level ratios are not.
  rate = nan(1, nvar);
  ratev = nan(1, nvar);
  fprintf('\n%s\n', repmat('-', 1, 78));

  rateerr = nan(1, nvar+1);
  fprintf('\nerr    = surface L_inf error of u\n      dx    ');
  for iv = 1:nvar+1
    fprintf('%-22s', varlabels4{iv});
  end
  fprintf('\n');
  for k = 1:nh
    fprintf('%10.4g  ', hvals(k));
    for iv = 1:nvar+1
      fprintf('%12.3e', err(iv,k));
      if (k == 1)
        fprintf('   --  ');
      else
        fprintf(' %6.2f', ...
                log(err(iv,k-1)/err(iv,k))/log(hvals(k-1)/hvals(k)));
      end
      fprintf('   ');
    end
    fprintf('\n');
  end
  fprintf('fitted      ');
  for iv = 1:nvar+1
    [rateerr(iv), nuse] = fitrate(hvals, err(iv,:));
    fprintf('%12.2f (%dl)  ', rateerr(iv), nuse);
  end
  fprintf('\n');
  for imetric = 1:4
    switch imetric
      case 1
        cols = gap;
        name = 'gap    = max |e_A - e_B| at the shared rotated nodes';
      case 2
        cols = vtx;
        name = 'vtx    = max |u_A - u_B| on the cut circle';
      case 3
        cols = branch;
        name = 'branch = max(|e_A|, |e_B|) at the same nodes';
      case 4
        cols = allgap;
        name = 'allgap = max |e_A - e_B| over the whole overlap';
    end
    fprintf('\n%s\n      dx    ', name);
    for iv = 1:nvar
      fprintf('%-22s', varlabels{iv});
    end
    fprintf('\n');
    for k = 1:nh
      fprintf('%10.4g  ', hvals(k));
      for iv = 1:nvar
        fprintf('%12.3e', cols(iv,k));
        if (k == 1)
          fprintf('   --  ');
        else
          fprintf(' %6.2f', ...
                  log(cols(iv,k-1)/cols(iv,k))/log(hvals(k-1)/hvals(k)));
        end
        fprintf('   ');
      end
      fprintf('\n');
    end
    fprintf('fitted      ');
    for iv = 1:nvar
      [r, nuse] = fitrate(hvals, cols(iv,:));
      if (imetric == 1)
        rate(iv) = r;
      elseif (imetric == 2)
        ratev(iv) = r;
      end
      fprintf('%12.2f (%dl)  ', r, nuse);
    end
    fprintf('\n');
  end
  fprintf('\nfitted rate of the angle error |theta - theta_exact|:');
  fprintf(' d_k %5.2f, d_k2 %5.2f\n', ...
          fitrate(hvals, thetamax(1,:)), fitrate(hvals, thetamax(2,:)));

  results(ci) = struct('zcut', zc, 'dkappa', dkappa, 'dkmean', dkmean, ...
                       'kappa', [kappam kappap], 'phic', phic, ...
                       'thetaex', thetaex(1), 'dx', hvals, 'gap', gap, ...
                       'vtx', vtx, 'branch', branch, 'allgap', allgap, ...
                       'nrot', nrot, 'nshare', nshare, ...
                       'thetamax', thetamax, 'rate', rate, ...
                       'ratev', ratev, 'err', err, 'rateerr', rateerr);

  %% Plot: the meridian of this cut, and the surface error against dx
  if ~exist(figdir, 'dir')
    mkdir(figdir);
  end
  figure(2*ci - 1); clf;
  set(gcf, 'Position', [100 100 1150 480]);
  axes('Position', [0.055 0.13 0.34 0.74]);
  meridianpanel(a, c, cen, zc, tc, phic, flipped, thetaex, geoname);
  axes('Position', [0.56 0.13 0.40 0.74]);
  plotconv(hvals, err, rateerr, varlabels4, 'surface L_\infty error of u');
  title({sprintf('vGMM:  u - \\Delta_S u = f on the cut surface (%s)', geoname), ...
         sprintf(['cut at z = %g, \\kappa_m = %.3f, \\kappa_p = %.3f, ' ...
                  '\\delta\\kappa = %.3f'], zc, kappam, kappap, dkappa)}, ...
        'FontSize', 9);
  legend('Location', 'southeast', 'FontSize', 7);
  outfile = fullfile(figdir, ...
                     sprintf('%s_vgmm_cut_z_%g_error.png', geoname, zc));
  exportgraphics(gcf, outfile, 'Resolution', 150);
  fprintf('saved %s\n', outfile);

  %% Plot: the error on the surface itself, for the cuts z_3d asks for
  % Where the error actually sits is not something the convergence table can
  % say, so this is the picture that answers it: the manifold in space, each
  % point coloured by |u_h - u| there.  A wedge of azimuth is left out
  % because the flipped cap is a dimple INSIDE the bowl and a closed render
  % would hide it entirely; through the missing wedge both sheets and the
  % crease between them are visible at once.  Both branches are drawn on one
  % colour scale, so the two sheets are directly comparable.
  if any(abs(z_3d - zc) <= ztol)
    figure(2*ci); clf;
    set(gcf, 'Position', [100 100 1150 500]);
    error_surface_figure(a, c, cen, phic, zc, tc, dx_show, p, order, bw, ...
                         dim, csex, geo, ufun, ffun, itertol, ...
                         gmres_restart, gmres_maxit, geoname, dkappa);
    outfile = fullfile(figdir, ...
                       sprintf('%s_vgmm_cut_z_%g_errorsurface.png', ...
                               geoname, zc));
    exportgraphics(gcf, outfile, 'Resolution', 150);
    fprintf('saved %s\n', outfile);
  end
end


%% Summary across the cut heights, against delta_kappa

fprintf('\n\nSummary: fitted rate of the surface error, against delta_kappa\n');
fprintf('    zcut   phi_c   kappa_m  kappa_p   theta   delta_kappa ');
for iv = 1:nvar+1
  fprintf('  %-14s', varlabels4{iv});
end
fprintf('\n');
for ci = 1:length(results)
  fprintf('%8.3g %7.4f %8.4f %8.4f %8.4f %11.4f ', ...
          results(ci).zcut, results(ci).phic, results(ci).kappa(1), ...
          results(ci).kappa(2), results(ci).thetaex, results(ci).dkappa);
  for iv = 1:nvar+1
    fprintf('  %6.2f %7.2e', results(ci).rateerr(iv), results(ci).err(iv,end));
  end
  fprintf('\n');
end

sumname = {'gap (u, the grid nodes)', 'vtx (u, on the cut circle)'};
sumfield = {'gap', 'vtx'};
sumrate = {'rate', 'ratev'};
for imetric = 1:2
  fprintf('\nSummary: fitted rate and %s at dx = %g, against delta_kappa\n', ...
          sumname{imetric}, hvals(end));
  fprintf('    zcut  delta_kappa ');
  for iv = 1:nvar
    fprintf('  %-18s', varlabels{iv});
  end
  fprintf('\n');
  for ci = 1:length(results)
    fprintf('%8.3g %11.4f ', results(ci).zcut, results(ci).dkappa);
    e = results(ci).(sumfield{imetric});
    r = results(ci).(sumrate{imetric});
    for iv = 1:nvar
      fprintf('  %5.2f  %9.3e', r(iv), e(iv,end));
    end
    fprintf('\n');
  end
end


%% One more figure: all the cuts together, ordered by delta_kappa

if (length(results) > 1)
  figure(2*length(results) + 1); clf;
  set(gcf, 'Position', [100 100 1150 500]);
  cmap = lines(length(results));

  % left: the meridian of every cut, and the profile each one produces
  axes('Position', [0.06 0.14 0.36 0.75]);
  hold on;
  tt = linspace(-pi/2, pi/2, 800)';
  hl = plot(a*cos(tt), cen(3) + c*sin(tt), '-', ...
            'Color', [0.7 0.7 0.7], 'LineWidth', 1.0);
  leg = {'the uncut meridian'};
  for ci = 1:length(results)
    zc = results(ci).zcut;
    tcc = pi/2 - results(ci).phic;
    tlo = linspace(-pi/2, tcc, 500)';
    thi = linspace(tcc, pi/2, 500)';
    zhi = cen(3) + c*sin(thi);
    if flipped
      zhi = 2*zc - zhi;
    end
    hl(end+1) = plot([a*cos(tlo); a*cos(thi)], ...
                     [cen(3) + c*sin(tlo); zhi], '-', ...
                     'Color', cmap(ci,:), 'LineWidth', 1.4);
    plot(a*cos(tcc), zc, 'o', 'Color', cmap(ci,:), ...
         'MarkerFaceColor', cmap(ci,:), 'MarkerSize', 7, ...
         'HandleVisibility', 'off');
    leg{end+1} = sprintf('z = %.3g, \\delta\\kappa = %.2f', ...
                         zc, results(ci).dkappa);
  end
  axis equal; grid on; box on;
  xlabel('\rho'); ylabel('z');
  title(sprintf('the cuts, in a meridian half plane (%s)', geoname), ...
        'FontSize', 9);
  legend(hl, leg, 'Location', 'southoutside', 'FontSize', 7, 'NumColumns', 2);

  % right: the exact-rotation surface error at every cut
  axes('Position', [0.55 0.16 0.41 0.72]);
  hold on;
  leg = {};
  for ci = 1:length(results)
    plot(hvals, results(ci).err(nvar+1,:), '-s', 'Color', cmap(ci,:), ...
         'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerFaceColor', cmap(ci,:));
    leg{end+1} = sprintf('\\delta\\kappa = %.2f, rate %.2f', ...
                         results(ci).dkappa, results(ci).rateerr(nvar+1));
  end
  base = 3*max(arrayfun(@(r) r.err(nvar+1,1), results));
  plot(hvals, base*(hvals/hvals(1)), 'k:', 'LineWidth', 1.2);
  leg{end+1} = 'O(dx)';
  plot(hvals, base*(hvals/hvals(1)).^2, 'k--', 'LineWidth', 1.2);
  leg{end+1} = 'O(dx^2)';
  set(gca, 'XScale', 'log', 'YScale', 'log');
  xticks([0.005 0.01 0.02 0.05 0.1]);
  grid on; box on;
  xlabel('dx'); ylabel('surface L_\infty error of u');
  title([exactlabel ': the surface error at every cut height'], 'FontSize', 9);
  legend(leg, 'Location', 'southeast', 'FontSize', 7);

  outfile = fullfile(figdir, sprintf('%s_vgmm_cut_summary.png', geoname));
  exportgraphics(gcf, outfile, 'Resolution', 150);
  fprintf('\nsaved %s\n\n', outfile);
end


%% ----------------------------------------------------------------------
%% local functions
%% ----------------------------------------------------------------------

function g = make_cap(t1, t2, flip, zcut)
%MAKE_CAP  one embedded piece of the surface of revolution
%   The piece is the meridian arc (rho, z) = (a*cos t, cen(3) + c*sin t)
%   for t in [t1, t2], spun about the z axis, embedded either as it stands
%   (flip = false) or reflected across the plane z = zcut (flip = true).
%   The ends t = -+pi/2 are the poles, where the surface closes up; only an
%   end at the cut carries a boundary.
%
%     g.t1, g.t2   the meridian arc
%     g.flip       whether this piece is the reflected one
%     g.zcut       the cut plane
%     g.pieces     the pieces this geometry is made of: itself, here

  g.t1 = t1;
  g.t2 = t2;
  g.flip = flip;
  g.zcut = zcut;
  g.pieces = {};
  g.pieces{1} = g;
end


function g = make_union(geo)
%MAKE_UNION  the whole surface the two branches make between them
%   For the uncut baseline.  On the unflipped surface this is just the
%   ellipsoid; on the flipped one it is the creased surface, and the
%   baseline is then the closest point method applied to the crease as if it
%   were not there -- the other thing worth knowing.  A closed surface, so
%   it has no boundary even where it has a crease.

  g.t1 = geo{1}.t1;
  g.t2 = geo{2}.t2;
  g.flip = false;
  g.zcut = geo{1}.zcut;
  g.pieces = {geo{1}.pieces{1}, geo{2}.pieces{1}};
end


function [cpx, cpy, cpz, dist, bdy] = cpcap(x, y, z, a, c, cen, g)
%CPCAP  closest point on a geometry, and the distance to it
%   For a single piece: a surface of revolution about z, so pass to the
%   meridian half plane (rho, z) with rho >= 0, use the planar arc closest
%   point there, and spin the answer back out along the query's own azimuth.
%   A reflection is an isometry and its own inverse, so a reflected piece is
%   handled by reflecting the query, solving, and reflecting the answer.
%
%   For a union of two pieces, whichever piece is nearer wins; the seam
%   between them is the cut circle, where the two agree, and bdy is zero
%   throughout because the union is closed.
%
%   On the axis rho = 0 the azimuth is undefined; any is taken, which is
%   harmless because the meridian closest point of an axis query inside the
%   band is the pole itself and has rho = 0 too.

  if (numel(g.pieces) > 1)
    [c1x, c1y, c1z, d1] = cpcap(x, y, z, a, c, cen, g.pieces{1});
    [c2x, c2y, c2z, d2] = cpcap(x, y, z, a, c, cen, g.pieces{2});
    take2 = d2 < d1;
    cpx = c1x;  cpx(take2) = c2x(take2);
    cpy = c1y;  cpy(take2) = c2y(take2);
    cpz = c1z;  cpz(take2) = c2z(take2);
    dist = min(d1, d2);
    bdy = zeros(size(x));
    return;
  end

  if g.flip
    z = 2*g.zcut - z;
  end
  X = x - cen(1);
  Y = y - cen(2);
  rho = hypot(X, Y);
  [crho, cz, dist, bdy] = cpEllipseArc(rho, z, a, c, [0 cen(3)], g.t1, g.t2);
  small = rho < 1e-14;
  ur = X./rho;  ur(small) = 1;
  us = Y./rho;  us(small) = 0;
  cpx = cen(1) + crho.*ur;
  cpy = cen(2) + crho.*us;
  cpz = cz;
  if g.flip
    cpz = 2*g.zcut - cpz;
  end
end


function [ph, th] = capparam(x, y, z, a, c, cen, g)
%CAPPARAM  the ellipsoid parameters of a point of a geometry
%   Undo the reflection first, if there was one, so that (phi, th) always
%   name the point of the ORIGINAL ellipsoid this one came from.  That is
%   what makes u and f the same functions on both surfaces.
%
%   On a union, which piece a point belongs to is settled by which one it
%   lies on.  On the cut circle it lies on both and they return the same
%   parameters, so the tie does not matter.

  if (numel(g.pieces) > 1)
    [c1x, c1y, c1z] = cpcap(x, y, z, a, c, cen, g.pieces{1});
    [c2x, c2y, c2z] = cpcap(x, y, z, a, c, cen, g.pieces{2});
    on2 = hypot(hypot(x - c2x, y - c2y), z - c2z) < ...
          hypot(hypot(x - c1x, y - c1y), z - c1z);
    [ph, th] = capparam(x, y, z, a, c, cen, g.pieces{1});
    [ph2, th2] = capparam(x, y, z, a, c, cen, g.pieces{2});
    ph(on2) = ph2(on2);
    th(on2) = th2(on2);
    return;
  end

  if g.flip
    z = 2*g.zcut - z;
  end
  rho = hypot(x - cen(1), y - cen(2));
  ph = atan2(rho/a, (z - cen(3))/c);
  th = atan2(y - cen(2), x - cen(1));
end


function L = lapS(ph, th, a, c, B)
%LAPS  laplacian_S of the manufactured solution, analytic
%   See the header of the manufactured-solution section.  cotAp and cotwp
%   are cot(phi) times the phi-derivative, already divided through by the
%   sin(phi) that derivative carries, so nothing is singular at a pole.

  G = a^2*cos(ph).^2 + c^2*sin(ph).^2;
  Gp = (c^2 - a^2)*sin(2*ph);
  Ap = -sin(ph) - sin(2*ph);
  App = -cos(ph) - 2*cos(2*ph);
  cotAp = -cos(ph).*(1 + 2*cos(ph));
  wp = sin(2*ph);
  wpp = 2*cos(2*ph);
  cotwp = 2*cos(ph).^2;
  L = (App + cotAp)./G - Gp./(2*G.^2).*Ap ...
      + B*cos(2*th).*((wpp + cotwp)./G - Gp./(2*G.^2).*wp - 4/a^2);
end


function [km, kp] = principal_curvatures(ph, a, c)
%PRINCIPAL_CURVATURES  meridian and parallel curvatures of the ellipsoid
%   Both with respect to the inward normal, so both positive on a convex
%   surface.  G = rho'^2 + z'^2 in the polar angle phi.

  G = a^2*cos(ph).^2 + c^2*sin(ph).^2;
  km = a*c./G.^1.5;
  kp = c./(a*sqrt(G));
end


function tau = unit_tangent(t, sgn, a, c)
%UNIT_TANGENT  sgn * gamma'(t)/|gamma'(t)| on the meridian ellipse
%   In the (rho, z) coordinates of a meridian half plane, where the
%   meridian is (a*cos t, c*sin t) up to the centre.  sgn = +1 at an arc's
%   upper end and -1 at its lower end gives the tangent pointing OUT of
%   that arc.

  g = [-a*sin(t) c*cos(t)];
  tau = sgn*g/norm(g);
end


function cs = glue_cs(tauA, tauB)
%GLUE_CS  the rotation each branch needs, from the two outward tangents
%   tauA and tauB are the outward unit tangents of the two branches at the
%   cut circle, in the (rho, z) coordinates of a meridian half plane --
%   outward meaning away from that branch's own surface.  A row of branch A
%   sitting past the cut lies roughly in the direction tauA from it, and
%   the glue has to put it where branch B continues, which is the direction
%   -tauB.  So the rotation is the one carrying tauA onto -tauB, and the
%   other branch's is the mirror of that.  On a smooth join tauB = -tauA
%   and both rotations are the identity.
%
%   cs(k,1) and cs(k,2) are the cosine and sine of branch k's rotation.
%   The rotation about the cut circle's tangent is planar in the meridian
%   half plane, so this is the same two-dimensional construction the
%   ellipse_cut scripts use, and glue_rot2d builds it the same way: by
%   Rodrigues' formula about the out-of-plane axis, the cosine from a dot
%   product of the tangents and the sine from their cross product, with no
%   angle formed and no transcendental evaluated.  See
%   ../2D_curve/example_ellipse_cut_rotation_construction.m for what that
%   buys over going through atan2.

  cs = zeros(2, 2);
  [cs(1,1), cs(1,2)] = glue_rot2d(tauA, tauB);
  [cs(2,1), cs(2,2)] = glue_rot2d(tauB, tauA);
end


function th = cs_angle(cs)
%CS_ANGLE  the angles of the rotations in cs, for printing only
%   Nothing the operator applies is computed from these.

  th = atan2(cs(:,2), cs(:,1));
end


function [dk, dkmean] = curvature_mismatch(phic, cs2, a, c, flipped)
%CURVATURE_MISMATCH  ||II_1 - R II_2 R^T|| at the cut circle
%   cs2 = [cos sin] of BRANCH 2's glue rotation, the one that carries its
%   frame onto branch 1's continuation.  It is taken as the pair glue_cs
%   produced rather than as an angle, so that the R assembled below is the
%   same matrix the operator applies, bit for bit.
%
%   Built, not assumed.  Work at azimuth zero, where a meridian half plane
%   is the (x, z) plane and the cut circle's tangent -- the rotation axis --
%   is e_y.  Both branches are pieces of the same ellipsoid met at the same
%   phi_c, so they share kappa_m and kappa_p; what differs is the inward
%   normal, which the reflection turns over, and the rotation R that the
%   glue then applies.  Since the meridian and the parallel are the
%   principal directions of both branches and R carries one frame to the
%   other, the mismatch is |n_1 - R n_2| times the norm of the shared
%   curvature pair.
%
%   dk is the Frobenius norm of the second-fundamental-form mismatch and
%   dkmean the mismatch of the mean curvature VECTORS, |H_1 - R H_2|; both
%   vanish exactly on the unflipped cut.

  [km, kp] = principal_curvatures(phic, a, c);

  % inward unit normal of the ellipsoid at phi_c, in the meridian plane
  nin = -[sin(phic)/a, cos(phic)/c];
  nin = nin/norm(nin);
  n1 = nin;
  n2 = nin;
  if flipped
    n2(2) = -n2(2);      % the reflection turns the cap's normal over
  end

  R = [cs2(1) -cs2(2); cs2(2) cs2(1)];
  d = norm(n1 - (R*n2.').');

  dk = d*sqrt(km^2 + kp^2);
  dkmean = d*(km + kp);
end


function [x1d, y1d, z1d, cand] = make_grid(geo, a, c, cen, phic, dx, bw)
%MAKE_GRID  a tight grid around the surface, and the nodes near it
%   The box is the bounding box of the two pieces actually drawn, which for
%   the flipped surface is a good deal shorter in z than the ellipsoid's.
%   The offsets by odd fractions of dx matter for the same reason they do
%   in the plane: a grid symmetric about the cut plane would give the two
%   branches mirror-image tangent estimates whose errors cancel in the glue
%   rotation and flatter the estimator.

  rad = ceil(bw) + 2;      % see candidate_nodes
  pad = (rad + 2)*dx;

  zc = geo{2}.zcut;
  if geo{2}.flip
    zlo = min(cen(3) - c, 2*zc - (cen(3) + c));
    zhi = zc;
  else
    zlo = cen(3) - c;
    zhi = cen(3) + c;
  end
  x1d = ((cen(1) - a - pad - 0.317*dx) : dx : (cen(1) + a + pad))';
  y1d = ((cen(2) - a - pad - 0.211*dx) : dx : (cen(2) + a + pad))';
  z1d = ((zlo - pad - 0.133*dx) : dx : (zhi + pad))';

  S = [surface_samples(a, c, cen, phic, pi, dx, geo{1}); ...
       surface_samples(a, c, cen, 0, phic, dx, geo{2})];
  cand = candidate_nodes(x1d, y1d, z1d, dx, rad, S);
end


function S = surface_samples(a, c, cen, p1, p2, dx, g)
%SURFACE_SAMPLES  points of one piece, no more than about dx/2 apart
%   Walked in the polar angle phi, and at each phi around the parallel with
%   a count set by that parallel's own circumference, so the poles are not
%   oversampled.  Used only to find the grid nodes near the surface.

  nph = ceil(2*(p2 - p1)*max(a,c)/dx) + 1;
  ph = linspace(p1, p2, nph)';
  out = cell(nph, 1);
  for i = 1:nph
    r = a*sin(ph(i));
    z = cen(3) + c*cos(ph(i));
    if g.flip
      z = 2*g.zcut - z;
    end
    nth = max(1, ceil(4*pi*r/dx));
    th = (0:nth-1)'*2*pi/nth;
    out{i} = [cen(1) + r*cos(th), cen(2) + r*sin(th), z*ones(nth,1)];
  end
  S = cell2mat(out);
end


function cand = candidate_nodes(x1d, y1d, z1d, dx, rad, S)
%CANDIDATE_NODES  grid nodes that could be within bw*dx of the surface
%   A meshgrid over the whole box cannot be built at the finest dx and all
%   but a shell of it would be discarded by the banding anyway.  So snap the
%   surface samples to the nearest grid node and grow that seed set by rad
%   cells.  The growth is done one axis at a time, taking unique linear
%   indices after each, which keeps the working set the size of the band
%   rather than (2*rad+1)^3 times it.
%
%   rad must cover bw plus the distance from a closest point to the nearest
%   snapped seed, which is at most half a sample spacing plus half a cell
%   per axis; ceil(bw) + 2 is comfortable.  Too small a rad would silently
%   thin the band, and setup_band's stencil checks are what would catch it.

  nx = length(x1d);
  ny = length(y1d);
  nz = length(z1d);
  i = round((S(:,1) - x1d(1))/dx) + 1;
  j = round((S(:,2) - y1d(1))/dx) + 1;
  k = round((S(:,3) - z1d(1))/dx) + 1;

  off = (-rad:rad);
  m = numel(off);
  for ax = 1:3
    switch ax
      case 1
        i = reshape(i + off, [], 1);
        j = repmat(j, m, 1);  k = repmat(k, m, 1);
        ok = (i >= 1) & (i <= nx);
      case 2
        j = reshape(j + off, [], 1);
        i = repmat(i, m, 1);  k = repmat(k, m, 1);
        ok = (j >= 1) & (j <= ny);
      case 3
        k = reshape(k + off, [], 1);
        i = repmat(i, m, 1);  j = repmat(j, m, 1);
        ok = (k >= 1) & (k <= nz);
    end
    i = i(ok);  j = j(ok);  k = k(ok);
    [cand, ia] = unique(sub2ind([ny nx nz], j, i, k));
    i = i(ia);  j = j(ia);  k = k(ia);
  end
end


function [x, y, z] = grid_points(x1d, y1d, z1d, cand)
%GRID_POINTS  coordinates of a list of meshgrid linear indices

  [j, i, k] = ind2sub([length(y1d) length(x1d) length(z1d)], cand);
  x = x1d(i);
  y = y1d(j);
  z = z1d(k);
end


function s = setup_band(x1d, y1d, z1d, dx, p, order, bw, cand, ...
                        xc, yc, zc, a, c, cen, g)
%SETUP_BAND  the band and the operators for one piece of surface
%   One band, not two: vGMM applies L to the band values and extends the
%   result, so E and L are both square on the same set of nodes and no
%   restriction operator is needed.  Every node within bw*dx of the piece
%   carries an unknown.
%
%   s.lfull marks the rows of L whose whole stencil landed in the band.
%   Rows at the outer edge lose part of theirs, which is harmless as long as
%   nothing reads them: in E*L a row of L is read only if E has a nonzero in
%   that column, and bw is chosen so those columns are interior.  That is
%   checked here for E and again in vgmm_solve for the cross-cut blocks.

  [cpx, cpy, cpz, dist] = cpcap(xc, yc, zc, a, c, cen, g);

  keep = dist <= bw*dx;
  s.band = cand(keep);
  s.n = length(s.band);
  s.geo = g;
  s.dx = dx;
  s.ellip = [a c];
  s.x = xc(keep);    s.y = yc(keep);    s.z = zc(keep);
  s.cpx = cpx(keep); s.cpy = cpy(keep); s.cpz = cpz(keep);
  [s.ph, s.th] = capparam(s.cpx, s.cpy, s.cpz, a, c, cen, g);
  s.invband = make_invbandmap(length(x1d)*length(y1d)*length(z1d), s.band);

  % the closest point extension, square on the band
  [Ei, Ej, Es] = interp3_matrix(x1d, y1d, z1d, s.cpx, s.cpy, s.cpz, p);
  jj = s.invband(Ej);
  if any(jj == 0)
    error('a closest point interpolation stencil leaves the band');
  end
  s.E = sparse(Ei, jj, Es, s.n, s.n);

  % the Laplacian, square on the same band; entries outside it are dropped
  % by laplacian_3d_matrix, which is what lfull records
  s.L = laplacian_3d_matrix(x1d, y1d, z1d, order, s.band, s.band);
  s.lfull = (full(sum(s.L ~= 0, 2)) == 2*3*(order/2) + 1);
  if ~all(s.lfull(unique(jj)))
    error('the Laplacian stencil of a row the extension reads leaves the band');
  end
end


function m = rim_rows(s, a, c, cen, rcut, zcut)
%RIM_ROWS  which band nodes are clamped to the cut circle
%   A node with m true is one whose closest point on this piece is a point
%   of the cut circle: those are the ones the glue rotates and extends
%   through the other branch.  Tested on the meridian coordinates of the
%   closest point, with the reflection undone first, because cpEllipseArc
%   returns the clamped end of the arc exactly.  The poles, the other ends
%   of the two meridian arcs, are not boundaries -- the surface of
%   revolution closes up there -- and are correctly not matched here.

  if isempty(s.cpx)
    m = false(0,1);
    return;
  end
  rho = hypot(s.cpx - cen(1), s.cpy - cen(2));
  z = s.cpz;
  if s.geo.flip
    z = 2*s.geo.zcut - z;
  end
  tol = 1e4*eps(max(1, max(a, c)));
  m = (abs(rho - rcut) <= tol) & (abs(z - zcut) <= tol);
end


function V = rim_samples(rcut, zcut, cen, dx)
%RIM_SAMPLES  points spread around the cut circle, about dx apart

  nth = max(16, ceil(2*pi*rcut/dx));
  th = ((0:nth-1)' + 0.5)*2*pi/nth;
  V = [cen(1) + rcut*cos(th), cen(2) + rcut*sin(th), zcut*ones(nth,1)];
end


function [xr, yr, zr] = rotate_about_rim(s, rows, cen, cth, sth)
%ROTATE_ABOUT_RIM  spin clamped nodes about the cut circle's tangent
%   Row i of rows sits at X_i with closest point V_i on the cut circle.
%   Because V_i is the closest point of X_i on a circle, X_i - V_i has no
%   component along the circle's tangent e_th, so the rotation about that
%   tangent through the angle whose cosine is cth and sine sth is a planar
%   rotation of (d_rho, d_z) in X_i's own meridian half plane -- the planar
%   construction, meridian by meridian.  The e_th component is carried
%   through anyway so the formula does not quietly depend on it vanishing.
%
%   cth and sth come from glue_cs, which never forms the angle; taking the
%   pair rather than an angle is what keeps this routine from having to
%   undo an atan2 with a cos and a sin.

  x0 = s.cpx(rows);  y0 = s.cpy(rows);  z0 = s.cpz(rows);
  th0 = atan2(y0 - cen(2), x0 - cen(1));
  ct = cos(th0);  st = sin(th0);

  ddx = s.x(rows) - x0;
  ddy = s.y(rows) - y0;
  ddz = s.z(rows) - z0;
  dr = ddx.*ct + ddy.*st;      % along e_rho
  dt = -ddx.*st + ddy.*ct;     % along e_th, zero up to rounding

  nr = cth*dr - sth*ddz;
  nz = sth*dr + cth*ddz;

  xr = x0 + nr.*ct - dt.*st;
  yr = y0 + nr.*st + dt.*ct;
  zr = z0 + nz;
end


function tau = cp_tangent(s, cen, scheme)
%CP_TANGENT  outward unit tangent at the cut circle, from cp differences
%   Uses the band nodes whose closest point on this piece is on the cut
%   circle -- the same nodes the glue then rotates.  With
%   cpbar = cp(2*cp - x) and cp2bar = cp(3*cp - 2*x),
%
%     scheme 1:  d_k  = cp - cpbar                          (first order)
%     scheme 2:  d_k2 = 1.5*cp - 2*cpbar + 0.5*cp2bar       (second order)
%
%   Each difference is resolved in its own node's meridian frame
%   (e_rho, e_z) before averaging.  By rotational symmetry that is the frame
%   the glue angle is measured in, and averaging the raw vectors instead
%   would cancel them around the circle.  The component along e_th is
%   discarded; it is zero for the exact tangent and only noise here.

  rows = find(s.rim);
  if isempty(rows)
    error('no grid points have the cut circle as their closest point');
  end
  x = s.x(rows);    y = s.y(rows);    z = s.z(rows);
  cx = s.cpx(rows); cy = s.cpy(rows); cz = s.cpz(rows);

  [b1x, b1y, b1z] = cpcap(2*cx - x, 2*cy - y, 2*cz - z, ...
                          s.ellip(1), s.ellip(2), cen, s.geo);
  if (scheme == 1)
    dvx = cx - b1x;  dvy = cy - b1y;  dvz = cz - b1z;
  else
    [b2x, b2y, b2z] = cpcap(3*cx - 2*x, 3*cy - 2*y, 3*cz - 2*z, ...
                            s.ellip(1), s.ellip(2), cen, s.geo);
    dvx = 1.5*cx - 2*b1x + 0.5*b2x;
    dvy = 1.5*cy - 2*b1y + 0.5*b2y;
    dvz = 1.5*cz - 2*b1z + 0.5*b2z;
  end

  th0 = atan2(cy - cen(2), cx - cen(1));
  dr = dvx.*cos(th0) + dvy.*sin(th0);
  dz = dvz;

  nrm = hypot(dr, dz);
  tol = 1e4*eps(max(1, max(abs([cx; cz]))));
  ok = isfinite(nrm) & (nrm > tol);
  if ~any(ok)
    error('every cp difference on the cut circle was too small to use');
  end
  avg = [mean(dr(ok)./nrm(ok)) mean(dz(ok)./nrm(ok))];
  tau = avg/norm(avg);
end


function sh = shared_points(br, geo, a, c, cen, csex)
%SHARED_POINTS  the grid nodes carrying an unknown in BOTH branches
%   With one band per branch the unknowns are the band itself, stored as
%   global linear indices, so the overlap is their intersection.  sh.rot
%   marks the nodes at least one branch clamps to the cut circle, which is
%   where the glue rotation acts.
%
%   The point of this function is sh.phA/thA and sh.phB/thB: the parameters
%   each branch's unknown at a shared node stands for.  For a branch that is
%   not clamping the node that is just its own closest point.  For the
%   branch that IS clamping it, the glue does not extend by the value on the
%   cut circle: it rotates the node by the EXACT angle and interpolates on
%   the other branch, so the unknown stands for u there.  On the unflipped
%   surface the exact angle is zero and the two agree, so the difference of
%   the unknowns is itself the error; at a crease they do not, the raw
%   difference is O(dx) whatever the scheme does, and what has to be
%   compared is each branch's own error.

  [~, i1, i2] = intersect(br{1}.band, br{2}.band);
  sh.i1 = i1;
  sh.i2 = i2;

  rot = {br{1}.rim(i1), br{2}.rim(i2)};
  sh.rot = rot{1} | rot{2};
  sh.nboth = sum(rot{1} & rot{2});

  ii = {i1, i2};
  ph = cell(2,1);
  th = cell(2,1);
  for k = 1:2
    o = 3 - k;
    ph{k} = br{k}.ph(ii{k});
    th{k} = br{k}.th(ii{k});
    m = find(rot{k});
    if ~isempty(m)
      [xr, yr, zr] = rotate_about_rim(br{k}, ii{k}(m), cen, ...
                                      csex(k,1), csex(k,2));
      [qx, qy, qz] = cpcap(xr, yr, zr, a, c, cen, geo{o});
      [ph{k}(m), th{k}(m)] = capparam(qx, qy, qz, a, c, cen, geo{o});
    end
  end
  sh.phA = ph{1};  sh.thA = th{1};
  sh.phB = ph{2};  sh.thB = th{2};
end


function out = vgmm_solve(br, cs, x1d, y1d, z1d, p, dim, a, c, cen, ...
                          ufun, ffun, itertol, restart, maxit)
%VGMM_SOLVE  assemble and solve u - laplacian_S u = f the vGMM way
%   br is one branch (the uncut surface, the baseline) or two (the glued
%   pieces).  With two branches, the band nodes clamped to the cut circle
%   are rotated about it by the rotation whose cosine and sine are cs(k,1)
%   and cs(k,2) -- k the branch -- and then extended by
%   interpolating on the other branch, which puts their weights in the
%   off-diagonal block E_{k,o} of the extension.
%
%   The operator is the embedded method-of-lines one of [vGMM 2013],
%
%       M = E*L - gamma*(I - E),      gamma = 2*dim/dx^2,
%
%   and the glue needs no further handling.  L is block diagonal, so
%   (E*L)_{k,o} = E_{k,o}*L_o extends the OTHER branch's L u across the cut,
%   and the penalty row u_i - (E u)_i at a rotated node of branch k is the
%   statement that branch k's value there agrees with branch o's
%   interpolant at the rotated point.
%
%   The forcing of a rotated row is f at the ROTATED TARGET, not at the
%   row's own closest point; see the header.  M is applied matrix-free --
%   E*L in three dimensions has some hundreds of entries per row and is not
%   worth forming -- under restarted GMRES with Jacobi preconditioning.
%
%   Returns the solution split by branch in out.ubr, which is what the
%   branches are compared through, and the surface L_inf error in
%   out.errsurf.

  nb = numel(br);
  dx = br{1}.dx;
  out.nclamp = 0;

  Eb = cell(nb, nb);
  for i = 1:nb
    for j = 1:nb
      if (i == j)
        Eb{i,j} = br{i}.E;
      else
        Eb{i,j} = sparse(br{i}.n, br{j}.n);
      end
    end
  end

  % rhs first, so the rotated rows can overwrite their own entries below
  rhs = zeros(sum(cellfun(@(s) s.n, br(:)')), 1);
  off = 0;
  for k = 1:nb
    rhs(off + (1:br{k}.n)) = ffun(br{k}.ph, br{k}.th);
    off = off + br{k}.n;
  end
  offs = [0 cumsum(cellfun(@(s) s.n, br(:)'))];

  for k = 1:(nb - 1)*2      % nothing to route when there is one branch
    o = 3 - k;
    rows = find(br{k}.rim);
    if isempty(rows)
      error('no rows to route across the cut on branch %d', k);
    end
    [xr, yr, zr] = rotate_about_rim(br{k}, rows, cen, cs(k,1), cs(k,2));

    % A wrong rotation can push a row back across the normal plane, and its
    % closest point on the other branch is then clamped to the cut circle
    % again.  That is the method doing what it does with a wrong angle, not
    % a failure: it is counted, not rejected.
    [qx, qy, qz, ~, qbdy] = cpcap(xr, yr, zr, a, c, cen, br{o}.geo);
    if br{o}.geo.flip
      out.nclamp = out.nclamp + sum(qbdy == 1);
    else
      out.nclamp = out.nclamp + sum(qbdy == 2);
    end

    [Ei, Ej, Es] = interp3_matrix(x1d, y1d, z1d, qx, qy, qz, p);
    jj = br{o}.invband(Ej);
    if any(jj == 0)
      error('a cross-cut interpolation stencil leaves the other band');
    end
    % E*L reads L on those columns, so they must be full-stencil rows
    if ~all(br{o}.lfull(unique(jj)))
      error('a cross-cut stencil reads a row whose Laplacian leaves the band');
    end
    Eb{k,o} = sparse(rows(Ei), jj, Es, br{k}.n, br{o}.n);
    % these rows are extended through the other branch now, not this one
    Ekk = Eb{k,k};
    Ekk(rows,:) = 0;
    Eb{k,k} = Ekk;

    % and their unknowns stand for u at the rotated target, so that is
    % where their forcing is evaluated
    [phq, thq] = capparam(qx, qy, qz, a, c, cen, br{o}.geo);
    rhs(offs(k) + rows) = ffun(phq, thq);
  end

  if (nb == 2)
    Eblk = [Eb{1,1} Eb{1,2}; Eb{2,1} Eb{2,2}];
    Lblk = blkdiag(br{1}.L, br{2}.L);
  else
    Eblk = Eb{1,1};
    Lblk = br{1}.L;
  end
  clear Eb

  % the interpolation weights of every row must still sum to one
  out.rowsum = max(abs(full(sum(Eblk, 2)) - 1));

  %% The vGMM operator, then the elliptic solve
  %   (I - M) u = f  with  M = E*L - gamma*(I - E),  i.e.
  %   (I - M) v = (1 + gamma)*v - E*(L*v + gamma*v).
  n = size(Eblk, 1);
  gamma = 2*dim/dx^2;
  afun = @(v) (1 + gamma)*v - Eblk*(Lblk*v + gamma*v);
  % the diagonal of I - M, for Jacobi: diag(E*L)_i = sum_j E_ij L_ji
  dg = (1 + gamma) - full(sum(Eblk.*(Lblk.'), 2)) - gamma*full(diag(Eblk));
  dg(abs(dg) < eps) = 1;

  [u, flag, relres] = gmres(afun, rhs, restart, itertol, maxit, @(v) v./dg);
  res = norm(afun(u) - rhs)/norm(rhs);
  if (flag ~= 0 && res > 1e-7)
    warning('vgmm_solve:gmres', ...
            'GMRES flag %d, relres %g, true residual %g', flag, relres, res);
  end
  if any(~isfinite(u))
    error('the elliptic solve returned non-finite values');
  end

  %% Split by branch, and the surface error alongside
  out.ubr = cell(nb,1);
  out.errsurf = 0;
  out.ndrop = 0;
  out.phierr = NaN;
  off = 0;
  for k = 1:nb
    out.ubr{k} = u(off + (1:br{k}.n));
    [es, nd, phm] = surface_error(x1d, y1d, z1d, p, br{k}, out.ubr{k}, ...
                                  a, c, cen, ufun);
    if (es > out.errsurf)
      out.errsurf = es;
      out.phierr = phm;
    end
    out.ndrop = out.ndrop + nd;
    off = off + br{k}.n;
  end
  out.unknowns = n;
end


function [eLinf, ndrop, phmax] = surface_error(x1d, y1d, z1d, p, s, u, ...
                                               a, c, cen, ufun)
%SURFACE_ERROR  L_inf error of the interpolant on the surface
%   phmax is the polar angle where the maximum sits, which is how the
%   caller tells whether the crease is what hurts the solve.
%   Midpoints of a uniform partition of the piece's polar-angle interval,
%   and around each parallel a count set by its own circumference, so no
%   sample sits exactly on the cut circle or at a pole.  A sample whose
%   interpolation stencil is not entirely inside the band is dropped and
%   counted.  One piece for a branch, both for the uncut baseline.

  eLinf = 0;
  ndrop = 0;
  phmax = NaN;
  ns = (p+1)^3;
  for ip = 1:numel(s.geo.pieces)
    g = s.geo.pieces{ip};
    p1 = pi/2 - g.t2;
    p2 = pi/2 - g.t1;
    nph = max(60, ceil(2*(p2 - p1)*max(a,c)/s.dx));
    ph = p1 + ((0:nph-1)' + 0.5)*(p2 - p1)/nph;

    for i = 1:nph
      r = a*sin(ph(i));
      nth = max(8, ceil(2*2*pi*r/s.dx));
      th = ((0:nth-1)' + 0.5)*2*pi/nth;
      X = cen(1) + r*cos(th);
      Y = cen(2) + r*sin(th);
      Z = cen(3) + c*cos(ph(i));
      if g.flip
        Z = 2*g.zcut - Z;
      end

      [~, Ej, Es] = interp3_matrix(x1d, y1d, z1d, X, Y, Z*ones(nth,1), p);
      JJ = reshape(s.invband(Ej), nth, ns);
      SS = reshape(Es, nth, ns);
      good = all(JJ > 0, 2);
      ndrop = ndrop + nth - sum(good);
      if ~any(good)
        continue;
      end
      val = sum(SS(good,:).*u(JJ(good,:)), 2);
      e = norm(val - ufun(ph(i)*ones(sum(good),1), th(good)), inf);
      if (e > eLinf)
        eLinf = e;
        phmax = ph(i);
      end
    end
  end
end


function val = interp_at(x1d, y1d, z1d, p, s, u, pts)
%INTERP_AT  one branch's interpolant evaluated at points of the surface
%   The same degree-p interpolation the operator itself uses, restricted to
%   this branch's band.  Used on the cut circle, where both branches have a
%   value and both of them stand for u there.

  n = size(pts, 1);
  [~, Ej, Es] = interp3_matrix(x1d, y1d, z1d, pts(:,1), pts(:,2), pts(:,3), p);
  ns = (p+1)^3;
  JJ = reshape(s.invband(Ej), n, ns);
  SS = reshape(Es, n, ns);
  if any(JJ(:) == 0)
    error('an interpolation stencil on the cut circle leaves the band');
  end
  val = sum(SS.*u(JJ), 2);
end


function meridianpanel(a, c, cen, zcut, tc, phic, flipped, thetaex, geoname)
%MERIDIANPANEL  the cut in a meridian half plane
%   The whole glue construction lives in one of these, so this is the
%   picture that shows it: the two arcs as they are embedded, the cut point
%   where they meet, each arc's outward tangent there, and the angle between
%   the first and the reverse of the second, which is the glue rotation.

  hold on;
  tlo = linspace(-pi/2, tc, 600)';
  thi = linspace(tc, pi/2, 600)';
  zhi = cen(3) + c*sin(thi);
  if flipped
    zhi = 2*zcut - zhi;
  end
  h1 = plot(a*cos(tlo), cen(3) + c*sin(tlo), '-', ...
            'Color', [0.00 0.45 0.74], 'LineWidth', 1.8);
  h2 = plot(a*cos(thi), zhi, '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.8);
  h3 = plot(a*[-0.15 1.15], zcut*[1 1], '--', ...
            'Color', [0.4 0.4 0.4], 'LineWidth', 0.8);

  V = [a*cos(tc), zcut];
  L = 0.22*max(a,c);
  tau = [unit_tangent(tc, 1, a, c); unit_tangent(tc, -1, a, c)];
  if flipped
    tau(2,2) = -tau(2,2);
  end
  h4 = quiver(V(1), V(2), L*tau(1,1), L*tau(1,2), 0, ...
              'Color', [0.00 0.45 0.74], 'LineWidth', 1.4, 'MaxHeadSize', 0.6);
  quiver(V(1), V(2), L*tau(2,1), L*tau(2,2), 0, ...
         'Color', [0.85 0.33 0.10], 'LineWidth', 1.4, 'MaxHeadSize', 0.6);
  h5 = plot(V(1), V(2), 'kp', 'MarkerFaceColor', 'y', 'MarkerSize', 12);

  axis equal; grid on; box on;
  xlim([-0.1*a 1.25*a]);
  xlabel('\rho'); ylabel('z');
  title({sprintf('%s:  a = %g, c = %g, cut at z = %g', geoname, a, c, zcut), ...
         sprintf('\\phi_c = %.4f, glue angle %.4f rad (%.1f deg)', ...
                 phic, thetaex(1), thetaex(1)*180/pi)}, 'FontSize', 9);
  legend([h1 h2 h3 h4 h5], ...
         {'branch 1 (bowl)', 'branch 2 (cap)', 'the cut plane', ...
          'outward tangents', 'the cut circle'}, ...
         'Location', 'southwest', 'FontSize', 7);
end


function error_surface_figure(a, c, cen, phic, zcut, tc, dx, p, order, bw, ...
                              dim, csex, geo, ufun, ffun, itertol, ...
                              restart, maxit, geoname, dkappa)
%ERROR_SURFACE_FIGURE  the manifold in space, coloured by |u_h - u|
%   Solves once more at the display grid with the EXACT rotation -- the
%   variant whose error is the scheme's own rather than an estimator's --
%   and paints each branch with the error of its interpolant there.
%
%   Two panels, the same data and the same colour scale.  The left one is
%   the whole manifold, from outside and a little above, with a wedge of
%   azimuth removed so that the dimple and the crease are not hidden inside
%   the bowl.  The right one is a close-up on the crease, which at the
%   milder cuts is shallow enough that nothing of it can be read at the
%   scale of the whole surface.

  %% the display solve
  [x1d, y1d, z1d, cand] = make_grid(geo, a, c, cen, phic, dx, bw);
  [xc, yc, zc] = grid_points(x1d, y1d, z1d, cand);
  rcut = a*cos(tc);
  br = cell(2,1);
  for j = 1:2
    br{j} = setup_band(x1d, y1d, z1d, dx, p, order, bw, cand, ...
                       xc, yc, zc, a, c, cen, geo{j});
    br{j}.rim = rim_rows(br{j}, a, c, cen, rcut, zcut);
  end
  clear xc yc zc
  out = vgmm_solve(br, csex, x1d, y1d, z1d, p, dim, a, c, cen, ...
                   ufun, ffun, itertol, restart, maxit);

  %% the error field on each branch, over all but a wedge of azimuth
  % gap0 is where the wedge starts and gapw how wide it is; the surf grids
  % are left open across it, which is what makes the cutaway.
  gap0 = -0.70*pi;
  gapw = 0.55*pi;
  S = cell(2,1);
  cmax = 0;
  for j = 1:2
    S{j} = error_field(x1d, y1d, z1d, p, br{j}, out.ubr{j}, a, c, cen, ...
                       dx, ufun, gap0 + gapw, 2*pi - gapw);
    cmax = max(cmax, max(S{j}.E(:)));
  end

  %% the two panels
  thr = (gap0 + gapw) + linspace(0, 2*pi - gapw, 600)';
  for ip = 1:2
    if (ip == 1)
      axes('Position', [0.05 0.09 0.40 0.78]);
    else
      axes('Position', [0.55 0.09 0.36 0.78]);
    end
    hold on;
    for j = 1:2
      surf(S{j}.X, S{j}.Y, S{j}.Z, S{j}.E, 'EdgeColor', 'none');
    end
    % the crease itself, drawn over the same wedge as the surfaces
    plot3(cen(1) + rcut*cos(thr), cen(2) + rcut*sin(thr), ...
          zcut*ones(size(thr)), 'k-', 'LineWidth', 2.0);
    shading interp;
    colormap(gca, parula);
    set(gca, 'CLim', [0 cmax]);
    axis equal; box on; grid on;
    xlabel('x'); ylabel('y'); zlabel('z');
    if (ip == 1)
      view([-35 16]);
      title({sprintf('|u_h - u| on the %s surface, cut at z = %g', ...
                     geoname, zcut), ...
             sprintf(['\\delta\\kappa = %.3f, crease %.1f deg, ' ...
                      'display grid dx = %.4g'], ...
                     dkappa, thetaex(1)*180/pi, dx)}, 'FontSize', 9);
    else
      % a cube about one point of the crease circle, the one at azimuth
      % zero.  A cube, not a flat box: with axis equal anything else
      % squashes the very thing the panel is for, which is the angle the two
      % sheets make where they meet.  Its size follows the depth of the
      % dimple, bounded at both ends so that the panel stays a close-up for
      % a deep dimple and still holds something for a shallow one.
      w = min(0.45*a, max(0.12*a, 2.5*(c - abs(zcut - cen(3)))));
      view([-14 10]);
      xlim(cen(1) + rcut + [-w w]);
      ylim(cen(2) + [-w w]);
      zlim(zcut + [-w w]);
      title('close-up on the crease, at azimuth zero', 'FontSize', 9);
    end
  end

  cb = colorbar('Position', [0.955 0.20 0.012 0.55]);
  cb.Label.String = '|u_h - u|';
  cb.Label.FontSize = 8;
  set(cb, 'FontSize', 7);
end


function S = error_field(x1d, y1d, z1d, p, s, u, a, c, cen, dx, ufun, ...
                         th0, thspan)
%ERROR_FIELD  |u_h - u| on one branch, as a surf-ready grid
%   A (phi, th) mesh over this branch's own polar-angle range and over the
%   azimuth window [th0, th0+thspan], carried into space by the branch's own
%   embedding, with the interpolant of u evaluated at every point.  A sample
%   whose interpolation stencil is not entirely in the band is left as NaN,
%   which surf renders as a hole rather than as a wrong colour.

  g = s.geo.pieces{1};
  p1 = pi/2 - g.t2;
  p2 = pi/2 - g.t1;
  nph = max(60, ceil(1.2*(p2 - p1)*max(a,c)/dx));
  nth = max(90, ceil(1.2*thspan*a/dx));
  ph = linspace(p1, p2, nph)';
  th = linspace(th0, th0 + thspan, nth);

  r = a*sin(ph);
  S.X = cen(1) + r*cos(th);
  S.Y = cen(2) + r*sin(th);
  zc = cen(3) + c*cos(ph);
  if g.flip
    zc = 2*g.zcut - zc;
  end
  S.Z = repmat(zc, 1, nth);

  n = numel(S.X);
  [~, Ej, Es] = interp3_matrix(x1d, y1d, z1d, S.X(:), S.Y(:), S.Z(:), p);
  ns = (p+1)^3;
  JJ = reshape(s.invband(Ej), n, ns);
  SS = reshape(Es, n, ns);
  good = all(JJ > 0, 2);
  JJ(~good,:) = 1;
  val = sum(SS.*u(JJ), 2);
  val(~good) = NaN;

  uex = ufun(repmat(ph, 1, nth), repmat(th, nph, 1));
  S.E = reshape(abs(val - uex(:)), nph, nth);
end


function plotconv(hvals, err, rate, labels, ylab)
%PLOTCONV  one family of error curves against dx, with slope guides
%   Black for the uncut baseline, then one colour per glue scheme.  The
%   legend quotes the fitted slope through all the levels, which for the
%   flipped surface is the only rate the data support; see the header.

  hold on;
  markers = {'o', 's', 'd', '^'};
  colors = [0 0 0; 0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19];
  n = size(err, 1);
  off = 4 - n;
  for iv = 1:n
    if (iv == 1 && off == 0)
      lw = 3.0;  ms = 10;
    else
      lw = 1.5;  ms = 6;
    end
    plot(hvals, err(iv,:), ['-' markers{iv+off}], ...
         'Color', colors(iv+off,:), 'LineWidth', lw, 'MarkerSize', ms, ...
         'MarkerFaceColor', colors(iv+off,:), ...
         'DisplayName', sprintf('%s, rate %.2f', labels{iv}, rate(iv)));
  end
  slopeguides(hvals, 3*max(err(:,1)));
  set(gca, 'XScale', 'log', 'YScale', 'log');
  xticks([0.005 0.01 0.02 0.05 0.1]);
  grid on; box on;
  xlabel('dx'); ylabel(ylab);
end


function slopeguides(hvals, base)
%SLOPEGUIDES  the O(dx) and O(dx^2) reference lines

  plot(hvals, base*(hvals/hvals(1)), 'k:', 'LineWidth', 1.2, ...
       'DisplayName', 'O(dx)');
  plot(hvals, base*(hvals/hvals(1)).^2, 'k--', 'LineWidth', 1.2, ...
       'DisplayName', 'O(dx^2)');
end


function [r, nuse] = fitrate(dx, e)
%FITRATE  slope of log(e) against log(dx), over every usable level
%   Unlike the planar companion this keeps all the levels rather than
%   stopping at the one where the column bottoms out.  Nothing here is
%   anywhere near a round-off floor; what a non-monotone column means on the
%   flipped surface is the grid-alignment scatter described in the header,
%   and the whole point of the fit is to average over it.

  use = isfinite(e) & (e > 0);
  nuse = sum(use);
  if (nuse < 2)
    r = NaN;
    return;
  end
  cf = polyfit(log(dx(use)), log(e(use)), 1);
  r = cf(1);
end
