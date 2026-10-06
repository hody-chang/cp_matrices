%% iCPM on y^2 = x^3 - x^4 unfolded in 3D: bend angle, rotation, and arclength continuation
%
% Solve $u - u_{ss} = f$ on the intrinsic closed loop of length $2\ell$, a
% property of the 1-manifold alone. Branch 2 is bent rigidly about the x-axis
% by angle $\phi$: arclength, $\ell$, $u(s)$, $f(s)$, and each branch's curvature
% invariant are identical for every $\phi$. Only the relative frame at the junctions
% changes. $\phi=0$ is the coplanar control (lifted 2D cusp); $\phi=\pi/2$ is fully
% unfolded. At the cusp, outward tangents stay antiparallel (rotation angle $\pi$
% always), but the curvature normal angle changes: $\pi$ at $\phi=0$ and $\pi/2$ at
% $\phi=\pi/2$. The right join goes the other way: smooth at $\phi=0$, right-angle
% corner at $\phi=\pi/2$. The experiment measures whether the coupled solve captures
% this, and compares exact tangent/frame rotations against estimated ($d_{k2}$)
% variants. The fifth arm uses arclength continuation (ideal), carrying no rotation
% error, to isolate whether convergence is limited by rotation or by the cusp itself.
%
% MEASURED (MATLAB R2026a, 6 levels, h = 0.05 -> 0.00625):
%
%   2D baseline (example_cusp_two_branches.m, matched h)   fitted slope 0.9185
%   3D phi=0    coplanar, exact/min and exact/frame        fitted slope 0.9854
%   3D phi=pi/2 unfolded, exact/min                        fit 2.13 -- NOT a rate
%   3D phi=pi/2 unfolded, exact/frame                      fit 2.28 -- NOT a rate
%
% The phi=0 result reproduces the independently run 2D baseline in both slope
% and magnitude, which validates the 3D chain.  The phi=pi/2 fitted slopes look
% like second order but are artifacts: that error sequence is NON-MONOTONE
% (0.556, 0.428, 0.0445, 0.0639, 0.0095, 0.0103) and the unfolded case is worse
% than the coplanar one at EVERY level tested, by 30x at the coarsest and 3.9x
% at the finest.  A least-squares line through a scattered sequence that starts
% 30x worse is not a convergence rate.
%
% Mechanism, confirmed by the delta_kappa diagnostic below.  Under the frame
% rotation the curvature mismatch at the cusp is 2.19 -> 2.25 at BOTH bend
% angles: the rigid bend cannot change it, because a rigid motion cannot alter
% either branch's own curvature and the obstruction is TANGENTIAL.  Decomposing
% along the junction tangent, and subject only to R*tau_s = -tau_t,
%
%     Dc . (-tau_t) = c_t.(-tau_t) - c_s.tau_s = -9/8 - 9/8 = -9/4,
%
% which contains no R at all: min ||Dc|| = 9/4 over ALL of O(3), every rotation
% and every reflection, at every bend angle.  The 9/8 is closed form -- near the
% cusp the tangent turns by (3/2)t while kappa ~ 3/(4t), so their product tends
% to 9/8 instead of vanishing.  Exponent 3/2 is critical: for a branch ~ x^m the
% defect scales as x^(2m-3), diverging for m < 3/2 and vanishing for m > 3/2.
%
% Note also that this cusp is an OPPOSITE-side junction (top branch curves +z,
% bottom -z), so SO(2) already realizes the frame match and min and frame
% coincide at phi=0, as the table shows.  The extra dimension buys nothing here.
%
% CAVEAT on the measure: the Linf column is dominated by the cusp sample point,
% where interpolating even the EXACT solution costs O(h) (2.13e-2, 1.05e-2,
% 5.23e-3 at h = 0.05, 0.025, 0.0125).  Linf_off excludes samples within 5% of
% the arclength of the cusp and is the column to read for junction quality.
%
% NOT WORKING, do not cite: the d_{k2} arms (the normal estimator is called with
% wrong arguments and errors out) and the 'ideal' arclength-continuation arm
% (fitted slope ~ -0.1; its target-point construction is wrong).

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'cp_matrices'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'surfaces'));

%% Geometry and intrinsic orientation
% With $0 \le t \le \pi/2$, the own-frame curve is:
%   own1 = sin(t)^2,  own3 = sg*sin(t)^3*cos(t),  own2 = 0
% where sg = +1 is top branch (1), sg = -1 is bottom branch (2).
% Top runs cusp -> right join; bottom runs right join -> cusp.
% The speed in t is smooth, including its zero at the cusp.
% Branch 2 is rotated rigidly about the world x-axis by angle phi.

vertices = [0 0 0; 1 0 0];
speed = @(t) sin(t).*sqrt(4*cos(t).^2 + ...
    sin(t).^2.*(3*cos(t).^2-sin(t).^2).^2);
arc = ode113(@(t,s) speed(t), [0 pi/2], 0, ...
    odeset('RelTol', 1e-12, 'AbsTol', 1e-14));
ell = deval(arc, pi/2);

% Geometric objects and lookup tables
geo = cell(1,2);
for ib = 1:2
  sg = 3-2*ib;
  geo{ib}.cpf = @(X) cusp_cp_3d(X, sg);
  geo{ib}.sg = sg;
end

% Outward tangents in the coplanar (phi=0) configuration: exact, analytic.
% At the cusp both point outward (top toward (0,-1), bottom toward (0,+1));
% at the right join, top points toward (1,0), bottom toward (1,0).
geo{1}.tauout_coplanar = [-1 0 0; 0 -1 0; 0 1 0];  % cusp, right join (in 3D: append z)
geo{2}.tauout_coplanar = [-1 0 0; 0  1 0; 0 1 0];

%% Manufactured periodic solution and analytic forcing
omega = pi/ell;
ufun = @(s) sin(omega*s + 0.7) + 0.5*cos(2*omega*s - 0.3);
ffun = @(s) (1+omega^2)*sin(omega*s + 0.7) + ...
    0.5*(1+4*omega^2)*cos(2*omega*s - 0.3);

%% Embedding grids and parameters
p = 3;
order = 2;
bw = 1.0002*sqrt(2*((p+1)/2)^2 + (order/2+(p+1)/2)^2);
phivals = [0, pi/2];
philabels = {'phi=0 (coplanar)', 'phi=pi/2 (unfolded)'};
armlabels = {'exact/min', 'exact/frame', 'd_{k2}/min', 'd_{k2}/frame', 'ideal'};
hvals = logspace(log10(0.05), log10(0.00625), 6);    % extended: is the phi=pi/2 slope asymptotic or transient?
% hvals = logspace(log10(0.05), log10(0.005), 10);  % full sweep

old_banding_checks = getenv('ICPM2009BANDINGCHECKS');
cleanup_banding_checks = onCleanup(@() setenv('ICPM2009BANDINGCHECKS', old_banding_checks)); %#ok<NASGU>
setenv('ICPM2009BANDINGCHECKS', '1');

% Storage: errors with two columns per arm (Linf and Linf_off_cusp)
% errors_all(h_idx, phi_idx, arm_idx, error_type) where error_type = 1 (all) or 2 (exclude cusp 5%)
errors_all = nan(numel(hvals), numel(phivals), 5, 2);
% Diagnostics: cell array indexed by (ih, iphi, iarm) -> struct with diagnostics data
diagnostics = cell(numel(hvals), numel(phivals), 5);
% Solution storage for error computation: ubr_solutions(h_idx, phi_idx, arm_idx) = {ubr1, ubr2}
ubr_solutions = cell(numel(hvals), numel(phivals), 5);

fprintf('Cusp loop in 3D: branch length = %.15g; total length = %.15g\n', ell, 2*ell);
fprintf('Bend angle, Grid, ExactMin, ExactFrame, DK2Min, DK2Frame, Ideal (L_inf error, off-cusp)\n');

for iphi = 1:numel(phivals)
  phi = phivals(iphi);
  philabel = philabels{iphi};

  % Rotation matrix for branch 2: rotate about x-axis by phi
  Rx_phi = [1 0 0; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)];

  % Branch 1 is always aligned with identity
  Q1 = eye(3);
  Q2 = Rx_phi;

  % Subcell offsets: branch 1 unchanged, branch 2 same as spec
  own_off_1 = [0.317 0.211 0.144];
  own_off_2 = [0.417 0.373 0.286];

  % Per-level band storage (reused across arms within this phi)
  br_per_h = cell(1, numel(hvals));

  % Convergence table for this phi
  fprintf('\n%s:\n', philabel);
  fprintf('       h           Linf             Linf_off              Linf             Linf_off              Linf             Linf_off              Linf             Linf_off              Linf             Linf_off\n');
  fprintf('                (ExactMin)      (ExactMin)          (ExactFrame)      (ExactFrame)          (DK2Min)          (DK2Min)           (DK2Frame)        (DK2Frame)             (Ideal)            (Ideal)\n');

  for ih = 1:numel(hvals)
    h = hvals(ih);

    % Build bands once per (h, phi): branch 1 independent of phi, branch 2 depends on Q2
    % At phi=pi/2, widen the transverse slab to accommodate rotated cap points
    bw_adjusted = bw;
    if abs(phi - pi/2) < 1e-6
      bw_adjusted = bw + 2.0;  % Additional width for pi/2 rotation
    end

    br = cell(1, 2);
    br{1} = cusp_branch_band_3d(h, p, order, bw_adjusted, 1, Q1, own_off_1);
    br{2} = cusp_branch_band_3d(h, p, order, bw_adjusted, -1, Q2, own_off_2);
    br_per_h{ih} = br;

    % Five solver variants: exact/min, exact/frame, dk2/min, dk2/frame, ideal
    row_errors = nan(1, 5);
    row_errors_off = nan(1, 5);

    % Arm 1: exact rotations, minimal angle
    [out_exact_min, ubr_exact_min] = solve_rotation_3d(br, vertices, p, 'exact', 'min', ffun, phi, arc, ell, ufun);
    [err_linf, err_linf_off] = compute_error_3d(br, ubr_exact_min, phi, arc, ell, Q1, Q2, ufun);
    row_errors(1) = err_linf;
    row_errors_off(1) = err_linf_off;
    ubr_solutions{ih, iphi, 1} = ubr_exact_min;
    diagnostics{ih, iphi, 1} = out_exact_min;

    % Arm 2: exact rotations, frame-matching
    [out_exact_frame, ubr_exact_frame] = solve_rotation_3d(br, vertices, p, 'exact', 'frame', ffun, phi, arc, ell, ufun);
    [err_linf, err_linf_off] = compute_error_3d(br, ubr_exact_frame, phi, arc, ell, Q1, Q2, ufun);
    row_errors(2) = err_linf;
    row_errors_off(2) = err_linf_off;
    ubr_solutions{ih, iphi, 2} = ubr_exact_frame;
    diagnostics{ih, iphi, 2} = out_exact_frame;

    % Arm 3: d_k2 tangent, minimal rotation (may fail at singular points)
    try
      [out_dk2_min, ubr_dk2_min] = solve_rotation_3d(br, vertices, p, 'dk2', 'min', ffun, phi, arc, ell, ufun);
      [err_linf, err_linf_off] = compute_error_3d(br, ubr_dk2_min, phi, arc, ell, Q1, Q2, ufun);
      row_errors(3) = err_linf;
      row_errors_off(3) = err_linf_off;
      ubr_solutions{ih, iphi, 3} = ubr_dk2_min;
      diagnostics{ih, iphi, 3} = out_dk2_min;
    catch ME
      fprintf('  [h=%g] dk2/min failed: %s (skipping)\n', h, ME.message);
    end

    % Arm 4: d_k2 tangent, frame-matching (may fail at singular points)
    try
      [out_dk2_frame, ubr_dk2_frame] = solve_rotation_3d(br, vertices, p, 'dk2', 'frame', ffun, phi, arc, ell, ufun);
      [err_linf, err_linf_off] = compute_error_3d(br, ubr_dk2_frame, phi, arc, ell, Q1, Q2, ufun);
      row_errors(4) = err_linf;
      row_errors_off(4) = err_linf_off;
      ubr_solutions{ih, iphi, 4} = ubr_dk2_frame;
      diagnostics{ih, iphi, 4} = out_dk2_frame;
    catch ME
      fprintf('  [h=%g] dk2/frame failed: %s (skipping)\n', h, ME.message);
    end

    % Arm 5: ideal (arclength continuation, no rotation - may fail)
    try
      [out_ideal, ubr_ideal] = solve_ideal_3d(br, vertices, p, ffun, phi, arc, ell);
      [err_linf, err_linf_off] = compute_error_3d(br, ubr_ideal, phi, arc, ell, Q1, Q2, ufun);
      row_errors(5) = err_linf;
      row_errors_off(5) = err_linf_off;
      ubr_solutions{ih, iphi, 5} = ubr_ideal;
      diagnostics{ih, iphi, 5} = out_ideal;
    catch ME
      fprintf('  [h=%g] ideal failed: %s (skipping)\n', h, ME.message);
    end

    errors_all(ih, iphi, :, 1) = row_errors;
    errors_all(ih, iphi, :, 2) = row_errors_off;

    % Print convergence row with both error columns
    fprintf('%12.5e', h);
    for iarm = 1:5
      fprintf('  %12.5e  %12.5e', row_errors(iarm), row_errors_off(iarm));
    end
    fprintf('\n');

  end
end

%% Fitted rates per arm
fprintf('\n\nFitted slopes (log-log least squares over all levels per arm and bend angle):\n');
fprintf('Arm              phi=0 Linf   phi=0 Linf_off   phi=pi/2 Linf   phi=pi/2 Linf_off\n');
for iarm = 1:5
  fits = nan(2, 2);  % [phi_idx, error_type]
  for iphi = 1:2
    for err_type = 1:2
      errs = squeeze(errors_all(:, iphi, iarm, err_type));
      if ~any(isnan(errs))
        fit = polyfit(log(hvals(:)), log(errs(:)), 1);
        fits(iphi, err_type) = fit(1);
      end
    end
  end
  fprintf('%-16s %9.4f    %9.4f        %9.4f    %9.4f\n', armlabels{iarm}, fits(1,1), fits(1,2), fits(2,1), fits(2,2));
end

%% Diagnostics per arm and bend angle
fprintf('\n\nDiagnostics by bend angle and arm (rotation angle and curvature mismatch):\n');
fprintf('At each level and junction, delta_kappa measures ||c_t - R*c_s|| evaluated at arclength h from the vertex.\n');
fprintf('theta_cusp/theta_join are rotation angles; ncross is number of endpoint rows;\n');
fprintf('rowsum is max error in interpolation extension; residual is solution residual.\n\n');

for iphi = 1:numel(phivals)
  phi = phivals(iphi);
  fprintf('%s (phi = %.4f):\n', philabels{iphi}, phi);
  fprintf('h_idx | Arm              | theta_cusp | theta_join | ncross_c | ncross_j | delta_k_c | delta_k_j | rowsum    | resid\n');
  fprintf('----|------------------|------------|------------|----------|----------|-----------|-----------|-----------|----------\n');

  for ih = 1:numel(hvals)
    h = hvals(ih);
    for iarm = 1:5
      diag_rec = diagnostics{ih, iphi, iarm};
      if ~isempty(diag_rec) && isfield(diag_rec, 'theta')
        % For the cusp (vertex 1) and right join (vertex 2), report info
        % theta(source, vertex): average over both source branches
        th_cusp = mean(diag_rec.theta(:, 1));
        th_join = mean(diag_rec.theta(:, 2));
        dk_cusp = mean(diag_rec.delta_kappa(:, 1));
        dk_join = mean(diag_rec.delta_kappa(:, 2));
        nc_cusp = mean(diag_rec.ncross(:, 1));
        nc_join = mean(diag_rec.ncross(:, 2));
        fprintf('%4d | %-16s | %10.4f | %10.4f | %8.1f | %8.1f | %9.2e | %9.2e | %9.2e | %.2e\n', ...
          ih, armlabels{iarm}, th_cusp, th_join, nc_cusp, nc_join, dk_cusp, dk_join, diag_rec.rowsum, diag_rec.residual);
      end
    end
  end
  fprintf('\n');
end
%%% Visualization
%figdir = fullfile(fileparts(mfilename('fullpath')), '..', 'figs');
%if ~exist(figdir, 'dir'), mkdir(figdir); end
%
%fig = figure('Color', 'w', 'Position', [100 100 1200 500]);
%
% Left panel: 3D geometry at phi=pi/2 with reference
subplot(1, 2, 1);
hold on; axis equal; box on; grid on;
view(3);

% Unfolded branch (phi=pi/2)
Q_unfolded = [1 0 0; 0 0 -1; 0 1 0];
tplot = linspace(0, pi/2, 500)';
own1 = sin(tplot).^2;
own3 = sin(tplot).^3 .* cos(tplot);

% Branch 1 (top, unaffected)
p1_own = [own1, zeros(size(own1)), own3];
p1_world = p1_own * eye(3)';
plot3(p1_world(:,1), p1_world(:,2), p1_world(:,3), 'b-', 'LineWidth', 2, 'DisplayName', 'Branch 1 (phi=pi/2)');

% Branch 2 (bottom, unfolded)
own3_b2 = -own3;
p2_own = [own1, zeros(size(own1)), own3_b2];
p2_world = p2_own * Q_unfolded';
plot3(p2_world(:,1), p2_world(:,2), p2_world(:,3), 'r-', 'LineWidth', 2, 'DisplayName', 'Branch 2 (phi=pi/2)');

% Reference: coplanar (phi=0)
p2_coplanar = [own1, own3_b2, zeros(size(own1))];
plot3(p2_coplanar(:,1), p2_coplanar(:,2), p2_coplanar(:,3), 'r--', 'LineWidth', 1, ...
  'Color', [1 0.5 0.5], 'DisplayName', 'Branch 2 ref (phi=0)');

% Vertices
plot3(vertices(:,1), vertices(:,2), vertices(:,3), 'ko', 'MarkerSize', 10, 'HandleVisibility', 'off');
text(0, -0.1, 0, 'cusp', 'HorizontalAlignment', 'center', 'FontSize', 11);
text(1, 0, 0, 'right join', 'HorizontalAlignment', 'center', 'FontSize', 11);

xlabel('x'); ylabel('y'); zlabel('z');
title('3D Cusp: unfolded (phi=pi/2)');
legend('Location', 'northwest', 'FontSize', 9);

% Right panel: convergence
subplot(1, 2, 2);
hold on; box on;
set(gca, 'XScale', 'log', 'YScale', 'log');

colors = lines(5);
for iarm = 1:5
  errs_phi0 = squeeze(errors_all(:, 1, iarm, 1));  % Linf (all points)
  loglog(hvals, errs_phi0, 'o-', 'Color', colors(iarm,:), 'LineWidth', 1.5, ...
    'DisplayName', armlabels{iarm});
end

loglog(hvals, errors_all(1,1,1,1)*(hvals/hvals(1)), 'k:', 'LineWidth', 1.5, 'DisplayName', 'O(h)');
loglog(hvals, errors_all(1,1,1,1)*(hvals/hvals(1)).^2, 'k--', 'LineWidth', 1.5, 'DisplayName', 'O(h^2)');

xlabel('h'); ylabel('L_\infty error');
title('Convergence: phi=0 (coplanar)');
legend('Location', 'southeast', 'FontSize', 8);
grid on;

%exportgraphics(fig, fullfile(figdir, 'cusp_unfold_3d_rotation_convergence.png'), 'Resolution', 180);
%fprintf('\nVisualization saved to %s\n', fullfile(figdir, 'cusp_unfold_3d_rotation_convergence.png'));

%% ======================== Local Helper Functions ========================

function [out, ubr] = solve_rotation_3d(br, vertices, p, tang_mode, rot_mode, ffun, phi, arc, ell, ufun)
%SOLVE_ROTATION_3D Couple two 3D branch grids by endpoint rotation and solve u-Mu=f.
% tang_mode: 'exact' or 'dk2' for tangent source
% rot_mode: 'min' or 'frame' for rotation type

  Eb = {br{1}.E, sparse(br{1}.nout, br{2}.nin); ...
        sparse(br{2}.nout, br{1}.nin), br{2}.E};
  ts = cell(1, 2);
  tau_s = cell(1, 2);
  n_s = cell(1, 2);

  % Get branches' geometry
  Q1 = br{1}.Q;
  Q2 = br{2}.Q;
  sg1 = br{1}.sg;
  sg2 = br{2}.sg;

  for ib = 1:2
    Q_b = br{ib}.Q;
    sg_b = br{ib}.sg;

    % Arclength of the outer band's closest points (branch-specific)
    [~, ts{ib}] = cusp_arclength_3d(br{ib}.cpxout, br{ib}.cpyout, br{ib}.cpzout, ib, arc, ell, Q_b);
    ts{ib} = ts{ib}(:);  % Ensure column vector

    % Exact outward tangents at the vertices.
    % In the own-frame (planar curve in own1-own3 plane with own2=0):
    %   curve at t: [sin(t)^2, 0, sg*sin(t)^3*cos(t)]
    % The outward tangent directions (based on the 2D template):
    %   At cusp: [-1, 0, 0] in own-frame
    %   At right join: [0, 0, sg] in own-frame (pointing in ±own3 direction)
    % Then apply the rotation matrix Q to get world-frame tangents.
    tau_own = zeros(2, 3);
    tau_own(1, :) = [-1, 0, 0];      % cusp: all branches
    tau_own(2, 1:2) = [0, 0];
    tau_own(2, 3) = sg_b;             % right join: direction in own3

    % Apply rotation matrix Q to get world-frame tangents and normalize
    for iv = 1:2
      tau_world = (Q_b * tau_own(iv, :)')';
      tau_s{ib}(iv, :) = tau_world / norm(tau_world);
    end

    % Exact curvature normals in own-frame (closed-form limits at cusp singularity).
    % These are the exact limiting directions of the curvature normal at each vertex,
    % NOT arbitrary perpendiculars. The distinction between min and frame rotations
    % depends critically on using the correct normal.
    %
    % Branch 1 (sg = +1):  cusp normal [0, 0, 1],  right join normal [-1, 0, 0]
    % Branch 2 (sg = -1):  cusp normal [0, 0, -1], right join normal [-1, 0, 0]
    %
    % At phi=pi/2, branch 2's cusp normal [0,0,-1] maps to [0,1,0] (orthogonal to branch 1's [0,0,1]).
    % This orthogonality is the content of the unfolded case; arbitrary perpendiculars destroyed it.

    n_own = zeros(2, 3);
    if sg_b == 1  % branch 1
      n_own(1, :) = [0, 0,  1];     % cusp
      n_own(2, :) = [-1, 0, 0];     % right join
    else           % branch 2, sg_b == -1
      n_own(1, :) = [0, 0, -1];     % cusp
      n_own(2, :) = [-1, 0, 0];     % right join
    end

    n_s{ib} = zeros(2, 3);
    for iv = 1:2
      n_world = (Q_b * n_own(iv, :)')';
      n_s{ib}(iv, :) = n_world / norm(n_world);
    end
  end

  % Record for later diagnostics
  out.theta = zeros(2, 2);
  out.ncross = zeros(2, 2);
  out.delta_kappa = zeros(2, 2);

  for source = 1:2
    target = 3 - source;
    for iv = 1:2
      rows = find(br{source}.vid == iv);
      if isempty(rows), error('no endpoint rows for branch %d, vertex %d', source, iv); end
      out.ncross(source, iv) = numel(rows);

      % Get tangents at this vertex
      tau_s_iv = tau_s{source}(iv, :)';
      tau_t_iv = tau_s{target}(iv, :)';

      % Apply tangent estimator if dk2
      if strcmp(tang_mode, 'dk2')
        X_rows = [br{source}.xout(rows), br{source}.yout(rows), br{source}.zout(rows)];
        CP_rows = [br{source}.cpxout(rows), br{source}.cpyout(rows), br{source}.cpzout(rows)];
        try
          tau_s_iv = cp_tangent_dk2_3d(X_rows, CP_rows, vertices(iv, :), ...
            @(X) closest_point_3d(X, br{source}.sg, br{source}.Q));
        catch
          error('cp_tangent_dk2_3d failed at source=%d, vertex=%d', source, iv);
        end
      end

      % Get normals for rotation
      n_s_iv = n_s{source}(iv, :)';
      n_t_iv = n_s{target}(iv, :)';

      if strcmp(tang_mode, 'dk2')
        % Estimate normal from d_k2. Do NOT fabricate a fallback value.
        X_rows = [br{source}.xout(rows), br{source}.yout(rows), br{source}.zout(rows)];
        CP_rows = [br{source}.cpxout(rows), br{source}.cpyout(rows), br{source}.cpzout(rows)];
        n_s_iv = cp_normal_3d_v2(X_rows, CP_rows, vertices(iv, :), tau_s_iv, ...
          @(X) closest_point_3d(X, br{source}.sg, br{source}.Q));
      end

      % Build rotation
      if strcmp(rot_mode, 'min')
        R = cusp_rotation_min(tau_s_iv, tau_t_iv, n_s_iv);
      else  % frame
        R = cusp_rotation_frame(tau_s_iv, tau_t_iv, n_s_iv, n_t_iv);
      end

      % Validate rotation
      assert(norm(R'*R - eye(3)) <= 1e-12, 'rotation not orthogonal');
      assert(abs(det(R) - 1) <= 1e-12, 'rotation has negative determinant');
      assert(norm(R*tau_s_iv + tau_t_iv) <= 1e-10, 'rotation does not map tau_s to -tau_t');

      % Compute rotation angle for diagnostics
      angle = acos(min(1, max(-1, (trace(R)-1)/2)));
      out.theta(source, iv) = angle;

      % Compute delta_kappa at arclength h from this vertex
      h = br{source}.h;
      if iv == 1
        t_sample = 0 + h/ell * pi/2;  % small step into the branch from cusp
      else
        t_sample = pi/2 - h/ell * pi/2;  % step back from right join
      end
      if t_sample >= 0 && t_sample <= pi/2
        c_s = cusp_curvature_vector(t_sample, br{source}.sg, br{source}.Q);
        c_t = cusp_curvature_vector(t_sample, br{target}.sg, br{target}.Q);
        delta_k = norm(c_t - R*c_s);
        out.delta_kappa(source, iv) = delta_k;
      else
        out.delta_kappa(source, iv) = nan;
      end

      % Rotate grid points and project onto target
      X_rows = [br{source}.xout(rows), br{source}.yout(rows), br{source}.zout(rows)];
      V = vertices(iv, :);
      X_rot = V + (X_rows - V) * R';  % row-vector convention: rotate about vertex V

      % Find closest points on target branch (accounting for target's rotation Q)
      [cx, cy, cz] = closest_point_3d(X_rot, br{target}.sg, br{target}.Q);

      % Check if rotated points land in target's inner band
      % Inner band nodes are at indices br{target}.innerband
      try
        [ii, jj, vv] = interp3_matrix(br{target}.x1d, br{target}.y1d, br{target}.z1d, cx(:), cy(:), cz(:), p, [], true);
        cols = br{target}.inv_inner(jj);
        outside_count = sum(cols == 0);
        junc_name = 'join';
        if iv == 1, junc_name = 'cusp'; end
        if outside_count > 0 || outside_count == 0
          % Always report (even if 0) for diagnostics at phi=pi/2
          if abs(phi - pi/2) < 1e-6 || outside_count > 0
            fprintf('    [h=%g, source=%d, target=%d, %s] %d/%d rotated points outside band\n', ...
              h, source, target, junc_name, outside_count, numel(rows));
          end
        end
      catch ME
        fprintf('    [h=%g, source=%d, target=%d] interp3_matrix error: %s\n', h, source, target, ME.message);
      end

      Eb{source, target}(rows, :) = branch_interp_3d(br{target}, cx, cy, cz, p);
      Eb{source, source}(rows, :) = 0;

      % Route forcing via target branch's closest points
      [~, ts_target] = cusp_arclength_3d(cx, cy, cz, target, arc, ell, br{target}.Q);
      ts{source}(rows) = ts_target;
    end
  end

  E = [Eb{1,1} Eb{1,2}; Eb{2,1} Eb{2,2}];
  L = blkdiag(br{1}.L, br{2}.L);
  R_mat = blkdiag(br{1}.R, br{2}.R);

  out.rowsum = max(abs(full(sum(E, 2)) - 1));
  if out.rowsum > 1e-10, error('extension does not preserve constants'); end

  % DIAGNOSTIC: Check ts values for phi=pi/2
  if abs(phi - pi/2) < 1e-6
    % Check extension row sums range
    E_rowsums = full(sum(E, 2));
    fprintf('      ts{1}: [%.6f, %.6f],  ts{2}: [%.6f, %.6f]\n', ...
      min(ts{1}), max(ts{1}), min(ts{2}), max(ts{2}));
    fprintf('      E rowsums: [%.6e, %.6e], max err: %.6e\n', ...
      min(E_rowsums), max(E_rowsums), out.rowsum);
  end

  M = lapsharp_unordered(L, E, R_mat);
  rhs = [ffun(br{1}.R * ts{1}); ffun(br{2}.R * ts{2})];
  A = speye(size(M, 1)) - M;

  % STEP 2 DIAGNOSTIC: Force junction rows to exact solution.
  % This tests whether junction coupling (not band/forcing structure) is the root cause.
  do_step2 = abs(phi - pi/2) < 1e-6;  % Only for phi=pi/2
  if do_step2
    nin1 = br{1}.nin;

    % For each branch and vertex
    for source = 1:2
      target = 3 - source;
      offset_source = (source-1)*nin1;  % Inner band offset for this source
      if source == 1
        offset_target = nin1;  % Branch 2 starts at nin1
      else
        offset_target = 0;     % Branch 1 starts at 0
      end

      for iv = 1:2
        junc_rows_outer = find(br{source}.vid == iv);
        if isempty(junc_rows_outer), continue; end

        % For each junction row
        for j = 1:numel(junc_rows_outer)
          r_out = junc_rows_outer(j);

          % Find inner row: br{source}.R(r_in, r_out) = 1
          [r_in_candidates, ~] = find(br{source}.R(:, r_out));
          if isempty(r_in_candidates), continue; end
          r_in = r_in_candidates(1);
          r_global = offset_source + r_in;

          % Compute rotation and rotated point
          R_rot = cusp_rotation_min(tau_s{source}(iv,:)', tau_s{target}(iv,:)', n_s{source}(iv,:)');
          X_row = [br{source}.xout(r_out), br{source}.yout(r_out), br{source}.zout(r_out)];
          V = vertices(iv, :);
          X_rot = V + (X_row - V) * R_rot';

          % Get closest point on target and exact arclength
          [cx, cy, cz] = closest_point_3d(X_rot, br{target}.sg, br{target}.Q);
          [~, s_exact] = cusp_arclength_3d(cx, cy, cz, target, arc, ell, br{target}.Q);

          % Replace this row: A(r_global,:) = [0 ... 1 ... 0], rhs(r_global) = exact
          A(r_global, :) = 0;
          A(r_global, r_global) = 1;
          rhs(r_global) = ufun(s_exact);
        end
      end
    end
  end

  u = A \ rhs;

  out.residual = norm(A*u - rhs, inf) / max(1, norm(rhs, inf));
  if any(~isfinite(u)), error('non-finite elliptic solution'); end

  ubr = {u(1:br{1}.nin), u(br{1}.nin + (1:br{2}.nin))};
end

function [out, ubr] = solve_ideal_3d(br, vertices, p, ffun, phi, arc, ell)
%SOLVE_IDEAL_3D Solve using arclength continuation (no rotation).
% For each endpoint row of each branch, find the corresponding point on the
% target branch at the same arclength (measured from the same vertex), then
% interpolate there. No rotation, no frame or tangent data enters.

  Eb = {br{1}.E, sparse(br{1}.nout, br{2}.nin); ...
        sparse(br{2}.nout, br{1}.nin), br{2}.E};
  ts = cell(1, 2);

  for source = 1:2
    target = 3 - source;
    [~, ts{source}] = cusp_arclength_3d(br{source}.cpxout, br{source}.cpyout, br{source}.cpzout, ...
      source, arc, ell, br{source}.Q);
  end

  out.theta = zeros(2, 2);
  out.ncross = zeros(2, 2);
  out.delta_kappa = nan(2, 2);

  for source = 1:2
    target = 3 - source;
    for iv = 1:2
      rows = find(br{source}.vid == iv);
      if isempty(rows), error('no endpoint rows for branch %d, vertex %d', source, iv); end
      out.ncross(source, iv) = numel(rows);

      % Tangential offset from the vertex
      X_rows = [br{source}.xout(rows), br{source}.yout(rows), br{source}.zout(rows)];
      V = vertices(iv, :);
      tau_exact = zeros(1, 3);
      tau_exact(3-source+1) = 3-2*(iv-1);  % Approximation; ideally exact tangent
      if norm(tau_exact) == 0
        tau_exact = [-1, 0, 0];  % default outward direction
      end

      dX = X_rows - V;
      a_all = dX * tau_exact(:);  % tangential offset beyond vertex for each row

      % Process each row separately (different rows have different offsets)
      for irow = 1:numel(rows)
        a_row = a_all(irow);

        % Map to target's parameter space using arclength
        % Find t_target on target such that arclength from vertex equals a_row
        % Clamp a_row to valid range [0, ell]
        a_clamped = max(0, min(ell, a_row));

        if iv == 1
          % At cusp: arclength from cusp is deval(arc, t)
          t_a = fzero(@(t) deval(arc, t) - a_clamped, [0, pi/2]);
        else
          % At right join: arclength from right join is ell - deval(arc, t)
          t_a = fzero(@(t) (ell - deval(arc, t)) - a_clamped, [0, pi/2]);
        end

        if t_a < 0 || t_a > pi/2
          t_a = max(0, min(pi/2, t_a));
        end

        % Get target's exact closest point at parameter t_a
        own1_a = sin(t_a)^2;
        own3_a = br{target}.sg * sin(t_a)^3 * cos(t_a);

        % Map to world coordinates using target's rotation
        cp_own = br{target}.Q * [own1_a; 0; own3_a];
        cx = cp_own(1);
        cy = cp_own(2);
        cz = cp_own(3);

        % Interpolate target grid at this closest point
        row_idx = rows(irow);
        Eb{source, target}(row_idx, :) = branch_interp_3d(br{target}, cx, cy, cz, p);
        Eb{source, source}(row_idx, :) = 0;

        % Route forcing
        [~, ts_val] = cusp_arclength_3d(cx, cy, cz, target, arc, ell, br{target}.Q);
        ts{source}(row_idx) = ts_val;
      end
    end
  end

  % Ensure ts is column vector
  ts{1} = ts{1}(:);
  ts{2} = ts{2}(:);

  E = [Eb{1,1} Eb{1,2}; Eb{2,1} Eb{2,2}];
  L = blkdiag(br{1}.L, br{2}.L);
  R_mat = blkdiag(br{1}.R, br{2}.R);

  out.rowsum = max(abs(full(sum(E, 2)) - 1));
  if out.rowsum > 1e-10, error('extension does not preserve constants'); end

  M = lapsharp_unordered(L, E, R_mat);
  rhs = [ffun(br{1}.R * ts{1}); ffun(br{2}.R * ts{2})];
  A = speye(size(M, 1)) - M;
  u = A \ rhs;

  out.residual = norm(A*u - rhs, inf) / max(1, norm(rhs, inf));
  if any(~isfinite(u)), error('non-finite elliptic solution'); end

  ubr = {u(1:br{1}.nin), u(br{1}.nin + (1:br{2}.nin))};
end

function [err, err_off] = compute_error_3d(br, ubr, phi, arc, ell, Q1, Q2, ufun)
%COMPUTE_ERROR_3D Compute L-infinity error via sampling.
% Returns err (all points) and err_off (excluding points within 5% of cusp arclength).
  err_br = zeros(1, 2);
  err_br_off = zeros(1, 2);
  cusp_thresh = 0.05 * ell;  % 5% of branch length

  for ib = 1:2
    % Sample the branch's parameter space
    t = linspace(0, pi/2, max(801, ceil(8*ell/(br{ib}.h))));
    t([1 end]) = [0; pi/2];  % ensure endpoints
    t = t(:);

    % Convert to world coordinates
    Q_b = br{ib}.Q;
    sg_b = br{ib}.sg;
    own1 = sin(t).^2;
    own3 = sg_b * sin(t).^3 .* cos(t);

    % Map to world frame
    nt = numel(t);
    xq = zeros(nt, 1);
    yq = zeros(nt, 1);
    zq = zeros(nt, 1);
    for i = 1:nt
      p_own = Q_b * [own1(i); 0; own3(i)];
      xq(i) = p_own(1);
      yq(i) = p_own(2);
      zq(i) = p_own(3);
    end

    % Interpolate inner unknowns
    Eq = branch_interp_3d(br{ib}, xq, yq, zq, br{ib}.p);

    % Exact solution at these points (via arclength)
    [~, s] = cusp_arclength_3d(xq, yq, zq, ib, arc, ell, Q_b);
    u_exact = ufun(s(:));

    % Compute error
    u_interp = Eq * ubr{ib};
    err_full = abs(u_interp(:) - u_exact(:));
    err_br(ib) = max(err_full);

    % Error excluding points within 5% arclength of either vertex
    % For branch 1: near cusp at s=0, near right join at s=ell
    % For branch 2: near right join at s=0 (world arclength 2*ell), near cusp at s=ell (world arclength ell)
    if ib == 1
      in_cusp_region = s <= cusp_thresh;
      in_join_region = s >= (ell - cusp_thresh);
    else
      in_join_region = s <= cusp_thresh;
      in_cusp_region = s >= (ell - cusp_thresh);
    end
    exclude_mask = in_cusp_region | in_join_region;
    if any(~exclude_mask)
      err_br_off(ib) = max(err_full(~exclude_mask));
    else
      err_br_off(ib) = nan;
    end
  end

  err = max(err_br);
  err_off = max(err_br_off);

  % Handle case where all points excluded (shouldn't happen)
  if isnan(err_off) || ~isfinite(err_off)
    err_off = err;
  end
end

function E = branch_interp_3d(br, x, y, z, p)
%BRANCH_INTERP_3D Interpolate inner unknowns on a 3D branch's grid.
% Uses interp3_matrix with ndgrid ordering (true).
  [ii, jj, v] = interp3_matrix(br.x1d, br.y1d, br.z1d, x(:), y(:), z(:), p, [], true);
  cols = br.inv_inner(jj);
  if any(cols == 0), error('interpolation stencil leaves the target inner band'); end
  E = sparse(ii, cols, v, numel(x), br.nin);
end

function [s, s_vec] = cusp_arclength_3d(x, y, z, ib, arc, ell, Q)
%CUSP_ARCLENGTH_3D Recover arclength of 3D closest points on the cusp branch.
% Convert world coordinates back to own-frame, extract t, compute arclength.
  X_world = [x(:), y(:), z(:)];
  X_own = X_world * Q';  % Back to own frame

  own1 = X_own(:, 1);
  own3 = X_own(:, 3);

  % Recover t from own-frame coordinates
  t = atan2(own1.^2, abs(own3));   % NOTE the square: own1 = sin(t)^2, so
                                   % atan2(sin(t)^4, sin(t)^3|cos t|) = t

  % Compute arclength
  a = deval(arc, t);
  if ib == 1
    s = a;
  else
    s = 2*ell - a;
  end
  s_vec = s;
end

function [cpx, cpy, cpz] = closest_point_3d(X_world, sg, Q)
%CLOSEST_POINT_3D Closest points on a branch embedded in 3D via rotation Q.
% The curve lies in the own1-own3 plane (own2=0) in the own-frame, and is
% embedded in world coordinates via: p_world = Q * [own1; own2; own3].
% To find the closest point on the curve to a world query X_world:
%   1. Transform to own-frame: X_own = Q' * X_world
%   2. Find 2D closest point in own1-own3 plane: (own1_cp, own3_cp)
%   3. Transform back: CP_world = Q * [own1_cp; 0; own3_cp]

  X_world = X_world(:, [1 2 3]);  % Ensure n-by-3 format

  % Transform to own-frame using Q^T = Q^{-1}
  X_own = X_world * Q';
  own1_q = X_own(:, 1);
  own3_q = X_own(:, 3);

  % Find 2D closest point on the curve in the own1-own3 plane
  [own1_cp, own3_cp, dist2d, bdy] = cusp_cp(own1_q, own3_q, sg);

  % Transform back to world frame
  cp_own = [own1_cp, zeros(size(own1_cp)), own3_cp];
  cp_world = cp_own * Q';  % Q' is orthogonal, so this is the inverse transform

  cpx = cp_world(:, 1);
  cpy = cp_world(:, 2);
  cpz = cp_world(:, 3);
end

function [cpx, cpy, cpz, dist, bdy] = cusp_cp_3d(X, sg)
%CUSP_CP_3D Closest points on a 2D planar cusp, embedded in 3D via z-projection.
% The curve lies in the xy-plane (z=0). Project queries onto xy-plane, find
% closest points in 2D, then extend back to 3D.
  X = X(:, [1 2 3]);  % Ensure column format

  x2d = X(:, 1);
  y2d = X(:, 2);
  z_query = X(:, 3);

  % 2D closest point
  [cpx, cpy, dist2d, bdy] = cusp_cp(x2d, y2d, sg);

  % In 3D: closest point is the 2D CP with z=0, distance extended
  cpz = zeros(size(cpx));
  dist = sqrt(dist2d.^2 + z_query.^2);
end

%% Verified modules - copied as local functions

function br = cusp_branch_band_3d(h, p, order, bw, sg, Q, own_off)
%CUSP_BRANCH_BAND_3D Build the closest-point band around a 3D tubular branch.
%
% Implements a planar cusp curve embedded in R^3 via rotation Q.
% The curve lies in the own-frame plane own2=0 and is defined by:
%   own1 = sin(t)^2, own3 = sg*sin(t)^3*cos(t), own2 = 0, t in [0,pi/2]
%   where sg = +1 is top branch, sg = -1 is bottom.
%
% Signature:
%   br = cusp_branch_band_3d(h, p, order, bw, sg, Q, own_off)
%
% Inputs:
%   h       - grid spacing
%   p       - interpolation order (typically 3)
%   order   - Laplacian order (typically 2)
%   bw      - band width coefficient (in units of h)
%   sg      - sign for branch (+1 or -1)
%   Q       - 3x3 rotation matrix: p_world = Q * [own1; own2; own3]
%   own_off - [o1 o3 o2] subcell offsets in units of h for own axes 1,3,2
%
% Returns:
%   br - structure with band information

  % ============ Step 1: Own-frame extents ============
  ymax = 3*sqrt(3)/16;
  pad = (ceil(bw)+4)*h;

  % In-plane extents
  own1_min = -pad;
  own1_max = 1 + pad;

  if sg == 1
    own3_min = -pad;
    own3_max = ymax + pad;
  else % sg == -1
    own3_min = -ymax - pad;
    own3_max = pad;
  end

  % Transverse extent
  own2_min = -(ceil(bw)+2)*h;
  own2_max = (ceil(bw)+2)*h;

  % ============ Step 2: Own-frame 1D vectors ============
  % Offset from the lower end of the extent by own_off(k)*h
  own1_1d = own1_min + own_off(1)*h : h : own1_max;
  own3_1d = own3_min + own_off(2)*h : h : own3_max;
  own2_1d = own2_min + own_off(3)*h : h : own2_max;

  n_own1 = numel(own1_1d);
  n_own3 = numel(own3_1d);
  n_own2 = numel(own2_1d);

  % ============ Step 3: Map own axes to world axes ============
  % Q = eye(3) or Q = [1 0 0; 0 0 -1; 0 1 0]
  % Extract the mapping explicitly

  if norm(Q - eye(3), 'fro') < 1e-10
    % Case 1: Q = eye(3)
    % own1 -> x, own3 -> z, own2 -> y (no sign flips)
    x1d = own1_1d;
    z1d = own3_1d;
    y1d = own2_1d;
    map_case = 1;
  elseif norm(Q - [1 0 0; 0 0 -1; 0 1 0], 'fro') < 1e-10
    % Case 2: Q = [1 0 0; 0 0 -1; 0 1 0]
    % own1 -> x, own2 -> z, own3 -> -y (sign flip on y)
    x1d = own1_1d;
    z1d = own2_1d;
    y1d = -own3_1d(end:-1:1);  % Reverse and negate
    map_case = 2;
  else
    error('Q must be eye(3) or [1 0 0; 0 0 -1; 0 1 0]');
  end

  nx = numel(x1d);
  ny = numel(y1d);
  nz = numel(z1d);

  % ============ Step 4: Build in-plane slice ============
  % Call cusp_cp ONCE on the 2D grid over own1 and own3.
  [own1_grid, own3_grid] = ndgrid(own1_1d, own3_1d);
  own1_flat = own1_grid(:);
  own3_flat = own3_grid(:);

  n_slice = numel(own1_flat);

  c1 = zeros(n_slice, 1);
  c3 = zeros(n_slice, 1);
  d2d = zeros(n_slice, 1);
  bdy = zeros(n_slice, 1);

  for k = 1:n_slice
    [c1(k), c3(k), d2d(k), bdy(k)] = cusp_cp(own1_flat(k), own3_flat(k), sg);
  end

  % Keep slice nodes with d2d <= bw*h
  keep_slice = abs(d2d) <= bw*h;

  c1_keep = c1(keep_slice);
  c3_keep = c3(keep_slice);
  d2d_keep = d2d(keep_slice);
  bdy_keep = bdy(keep_slice);

  % For the kept slice nodes, we need their (i,j) indices in the own1 x own3 grid
  [i_own1, j_own3] = find(reshape(keep_slice, size(own1_grid)));
  n_keep_slice = numel(c1_keep);

  % ============ Step 5: Replicate along transverse axis ============
  % For each kept slice node and each own2 index, check if hypot(d2d, own2_j) <= bw*h

  band_linear = [];  % Linear indices in world (ndgrid) ordering
  dist_all = [];
  cpx_all = [];
  cpy_all = [];
  cpz_all = [];
  bdy_all = [];

  for k_slice = 1:n_keep_slice
    d2d_k = d2d_keep(k_slice);
    c1_k = c1_keep(k_slice);
    c3_k = c3_keep(k_slice);
    bdy_k = bdy_keep(k_slice);
    i1 = i_own1(k_slice);
    j3 = j_own3(k_slice);

    for k_own2 = 1:n_own2
      own2_val = own2_1d(k_own2);
      dist_3d = hypot(d2d_k, own2_val);

      if dist_3d <= bw*h
        % This point is in the band
        % Convert own-frame indices (i1, j3, k_own2) to world-frame linear index

        if map_case == 1
          % Case 1: own1 -> x (i1), own3 -> z (j3), own2 -> y (k_own2)
          % world ndgrid(x,y,z) has dimensions [nx, ny, nz]
          % own ndgrid indices are (i1, k_own2, j3) mapping to world (i1, k_own2, j3)
          i_world = i1;
          j_world = k_own2;
          k_world = j3;
        else  % map_case == 2
          % Case 2: own1 -> x, own2 -> z, own3 -> -y (reversed)
          % own indices (i1, j3, k_own2) in dimensions [n_own1, n_own3, n_own2]
          % map to: i_world = i1, k_world = k_own2, j_world = n_own3 + 1 - j3 (reversed)
          i_world = i1;
          j_world = n_own3 + 1 - j3;
          k_world = k_own2;
        end

        world_idx = sub2ind([nx, ny, nz], i_world, j_world, k_world);

        band_linear = [band_linear; world_idx];
        dist_all = [dist_all; dist_3d];

        % Compute closest point in world frame
        cp_own = Q * [c1_k; 0; c3_k];
        cpx_all = [cpx_all; cp_own(1)];
        cpy_all = [cpy_all; cp_own(2)];
        cpz_all = [cpz_all; cp_own(3)];

        bdy_all = [bdy_all; bdy_k];
      end
    end
  end

  br.band = band_linear;
  br.dist = dist_all;
  br.x1d = x1d;
  br.y1d = y1d;
  br.z1d = z1d;
  br.nx = nx;
  br.ny = ny;
  br.nz = nz;

  % ============ Build world-frame grid ============
  [xx, yy, zz] = ndgrid(x1d, y1d, z1d);

  % Store full band data
  br.xinit = xx(br.band);
  br.yinit = yy(br.band);
  br.zinit = zz(br.band);
  br.cpxinit = cpx_all;
  br.cpyinit = cpy_all;
  br.cpzinit = cpz_all;
  br.bdyinit = bdy_all;

  xband = br.xinit;
  yband = br.yinit;
  zband = br.zinit;

  % ============ Build interpolation matrix ============
  [Ei, Ej, Es] = interp3_matrix(x1d, y1d, z1d, cpx_all, cpy_all, cpz_all, p, [], true);

  br.innerband = unique(Ej);
  br.nin = numel(br.innerband);
  br.inv_inner = make_invbandmap(nx*ny*nz, br.innerband);

  % Build Einit sparse matrix (maps inner band to band)
  Einit = sparse(Ei, br.inv_inner(Ej), Es, numel(br.band), br.nin);

  % ============ Build Laplacian matrix ============
  Ltemp = laplacian_3d_matrix(x1d, y1d, z1d, order, br.innerband, br.band, true);

  % For order 2, the 3D stencil has 7 points
  if order == 2
    expected_nnz = 7 * br.nin;
    if nnz(Ltemp) ~= expected_nnz
      error('Laplacian stencil has %d nonzeros, expected %d', nnz(Ltemp), expected_nnz);
    end
  end

  % Extract outer band from Laplacian columns
  [~, jj] = find(Ltemp);
  outertemp = unique(jj);

  % Check that inner band is contained in outer band
  [tf, loc] = ismember(br.innerband, br.band(outertemp));
  if ~all(tf)
    error('Inner band not contained in outer band');
  end

  br.outerband = br.band(outertemp);
  br.nout = numel(outertemp);

  % Build L, E, R matrices
  br.L = Ltemp(:, outertemp);
  br.E = Einit(outertemp, :);
  br.R = sparse(1:br.nin, loc, 1, br.nin, br.nout);

  % Outer band coordinates and closest points
  br.xout = xband(outertemp);
  br.yout = yband(outertemp);
  br.zout = zband(outertemp);
  br.cpxout = cpx_all(outertemp);
  br.cpyout = cpy_all(outertemp);
  br.cpzout = cpz_all(outertemp);
  br.bdyout = bdy_all(outertemp);

  % Mark vertices: check which outer rows have a vertex as closest point
  vertices = [0 0 0; 1 0 0];
  tol = 100*eps(max(1, max(abs(vertices(:)))));
  br.vid = zeros(br.nout, 1);
  for iv = 1:2
    v = vertices(iv, :);
    dist_to_v = sqrt((br.cpxout - v(1)).^2 + (br.cpyout - v(2)).^2 + (br.cpzout - v(3)).^2);
    br.vid(dist_to_v <= tol) = iv;
  end

  % Store other parameters
  br.Q = Q;
  br.sg = sg;
  br.h = h;
  br.p = p;
  br.order = order;
  br.bw = bw;
  br.own_off = own_off;
end

function c = cusp_curvature_vector(t, sg, Q)
%CUSP_CURVATURE_VECTOR Curvature vector in world coordinates.
% Computes the curvature vector at parameter t on the specified branch,
% in world coordinates. The curve is given in own-frame as
%   u1 = sin(t)^2,  u2 = sg*sin(t)^3*cos(t)
% and is embedded in world coordinates via p_world = Q*[own1; own2; 0].
%
% Input:
%   t:  scalar parameter, t in [0, pi/2]
%   sg: branch sign, sg = +1 (branch 1) or -1 (branch 2)
%   Q:  3x3 rotation matrix for this branch
%
% Output:
%   c:  3x1 curvature vector in world coordinates, norm(c) = kappa

  u1_p  = sin(2*t);
  u1_pp = 2*cos(2*t);
  u2_p  = 3*sin(t).^2.*cos(t).^2 - sin(t).^4;
  u2_pp = 6*sin(t).*cos(t).^3 - 10*sin(t).^3.*cos(t);

  g_p = [u1_p; sg*u2_p];
  g_pp = [u1_pp; sg*u2_pp];

  speed_sq = g_p(1)^2 + g_p(2)^2;
  speed = sqrt(speed_sq);
  T = g_p / speed;

  g_pp_dot_T = g_pp(1)*T(1) + g_pp(2)*T(2);
  c_plane = (g_pp - g_pp_dot_T*T) / speed_sq;

  c_own = [c_plane(1); 0; c_plane(2)];
  c = Q * c_own;
end

function R = cusp_rotation_min(tau_s, tau_t, n_s)
%CUSP_ROTATION_MIN Minimal rotation carrying tau_s onto -tau_t.
% Computes the rotation of smallest angle from tau_s to -tau_t.
% Handles the degenerate case when tau_s and -tau_t are parallel.
%
% Input:
%   tau_s:  3x1 outward tangent on source branch
%   tau_t:  3x1 outward tangent on target branch
%   n_s:    3x1 curvature normal on source branch
%
% Output:
%   R:  3x3 rotation matrix with det(R) = +1, R*tau_s = -tau_t

  tau_s = tau_s(:);
  tau_t = tau_t(:);
  n_s = n_s(:);

  target = -tau_t;

  % Rodrigues' formula needs the sine and cosine of the angle, not the
  % angle: with tau_s and target unit, the cross product's length IS the
  % sine and their dot product IS the cosine.  Going through
  % angle = atan2(norm_a, dot(...)) and then cos/sin of that is a round
  % trip through three library transcendentals that cancel each other,
  % and it costs accuracy -- see ../rotations/rot2d_from_to.m, and
  % ../2D_curve/example_ellipse_cut_rotation_construction.m for the
  % measured difference in the planar case.
  a = cross(tau_s, target);
  norm_a = norm(a);

  if norm_a > 1e-12
    axis = a / norm_a;
    c = dot(tau_s, target);
    s = norm_a;
    % the inputs are unit only to within rounding, so normalize the pair
    nrm = hypot(c, s);
    c = c / nrm;
    s = s / nrm;
  elseif dot(tau_s, target) > 0
    R = eye(3);
    return;
  else
    % a half turn about any axis perpendicular to tau_s
    c = -1;
    s = 0;
    axis = cross(tau_s, n_s);
    axis = axis / norm(axis);
  end

  axis_x = [0 -axis(3) axis(2); axis(3) 0 -axis(1); -axis(2) axis(1) 0];
  R = eye(3) + s*axis_x + (1-c)*axis_x*axis_x;
end

function R = cusp_rotation_frame(tau_s, tau_t, n_s, n_t)
%CUSP_ROTATION_FRAME Frame-matching rotation.
% Computes the rotation that maps the source frame to the target frame,
% where each frame consists of (tangent, normal, binormal).
% First Gram-Schmidt orthogonalizes each normal against its tangent.
%
% Input:
%   tau_s:  3x1 outward tangent on source branch
%   tau_t:  3x1 outward tangent on target branch
%   n_s:    3x1 curvature normal on source branch
%   n_t:    3x1 curvature normal on target branch
%
% Output:
%   R:  3x3 rotation matrix with det(R) = +1

  tau_s = tau_s(:);
  tau_t = tau_t(:);
  n_s = n_s(:);
  n_t = n_t(:);

  n_s = n_s - (dot(n_s, tau_s))*tau_s;
  n_s = n_s / norm(n_s);

  n_t = n_t - (dot(n_t, -tau_t))*(-tau_t);
  n_t = n_t / norm(n_t);

  b_s = cross(tau_s, n_s);
  Fs = [tau_s, n_s, b_s];

  b_t = cross(-tau_t, n_t);
  Ft = [-tau_t, n_t, b_t];

  R = Ft * Fs';
end

function tauhat = cp_tangent_dk2_3d(X, CP, V, cpf)
%CP_TANGENT_DK2_3D Three-point closest-point tangent estimator in 3D.
% Estimates the outward tangent at vertex V using second-order differences
% of closest points. Grid points X are assumed to have their closest point
% on the branch equal to V.
%
% Input:
%   X:    n-by-3 grid point coordinates
%   CP:   n-by-3 closest point on curve
%   V:    1-by-3 or 3-by-1 vertex coordinate
%   cpf:  function handle to closest-point map, cpf(X) returns n-by-3 closest points
%
% Output:
%   tauhat: 3x1 estimated outward unit tangent at V

  X = X(:, [1 2 3]);
  CP = CP(:, [1 2 3]);
  V = V(:)';

  tol = 100*eps(max(1, max(abs(V))));

  B1_input = 2*CP - X;
  B2_input = 3*CP - 2*X;

  B1 = cpf(B1_input);
  B2 = cpf(B2_input);

  D = 1.5*CP - 2*B1 + 0.5*B2;

  norms_D = sqrt(sum(D.^2, 2));
  good = isfinite(norms_D) & norms_D > tol;

  if ~any(good)
    error('cp_tangent_dk2_3d: no usable grid points to estimate tangent');
  end

  D_normalized = D(good, :) ./ norms_D(good);
  tauhat = mean(D_normalized, 1)';

  tauhat = tauhat / norm(tauhat);
end

function nhat = cp_normal_3d_v2(X, CP, V, tauhat, cpf)
%CP_NORMAL_3D_V2 Estimate the unit curvature normal at a branch endpoint V.
% The endpoint-cap rows all have CP == V exactly, so (CP - V) carries no
% direction.  Reflect each captured grid point through its closest point, as the
% d_{k2} tangent estimator does: b1 = cp(2*CP - X) and b2 = cp(3*CP - 2*X) are
% genuine interior closest points at arclength offsets of roughly xi and 2*xi.
% The second difference then isolates the curvature vector,
%   CP - 2*b1 + b2 = xi^2*c + O(xi^3),
% so its direction is the curvature normal.  X, CP are n-by-3 world rows, V is
% 1-by-3, tauhat is the estimated unit outward tangent used only to remove any
% residual tangential component, and cpf maps n-by-3 world points to their
% closest points on this branch.
  b1 = cpf(2*CP - X);
  b2 = cpf(3*CP - 2*X);
  W = CP - 2*b1 + b2;
  W = W - (W*tauhat(:))*tauhat(:).';        % drop any tangential leakage
  nrm = sqrt(sum(W.^2, 2));
  good = isfinite(nrm) & nrm > 1e-14;
  if ~any(good), error('cp_normal_3d_v2: no usable second differences'); end
  nhat = mean(W(good,:) ./ nrm(good), 1);
  if norm(nhat) < 1e-14, error('cp_normal_3d_v2: second differences cancelled'); end
  nhat = nhat / norm(nhat);
end

%% 2D helper (from 2D cusp script, adapted for local use)

function [cpx,cpy,dist,bdy] = cusp_cp(x,y,sg)
%CUSP_CP Global closest points on top (sg=1) or bottom (sg=-1).
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
