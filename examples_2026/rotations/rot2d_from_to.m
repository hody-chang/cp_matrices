function [c, s, R, theta] = rot2d_from_to(u, v, method)
%ROT2D_FROM_TO  the plane rotation carrying one direction onto another
%   [c, s, R, theta] = rot2d_from_to(u, v) returns the rotation that takes
%   the direction u onto the direction v.  u and v are N-by-2 arrays of
%   (approximately) unit vectors, one row per rotation, and the outputs are
%
%     c, s    N-by-1 columns with cos and sin of the rotation angle,
%             normalized so that c^2 + s^2 = 1 to within one rounding;
%     R       2-by-2-by-N rotations, R(:,:,k) = [c -s; s c], each of which
%             satisfies R*u' = v' when u and v are unit;
%     theta   N-by-1 angles in (-pi, pi], for REPORTING ONLY -- nothing in
%             the rotation itself is computed from theta.
%
%   CONSTRUCTION.  Rodrigues' rotation formula about the unit axis k is
%
%     R = I + sin(theta) K + (1 - cos(theta)) K^2,   K = skew(k),
%
%   and in the plane the axis is the out-of-plane direction e_z, whose
%   restriction to the plane is K = [0 -1; 1 0] with K^2 = -I.  The formula
%   therefore collapses to R = cos(theta) I + sin(theta) K = [c -s; s c],
%   and the cosine and sine come straight out of the two directions:
%
%     c = u . v,     s = (u x v) . e_z = u1 v2 - u2 v1,
%
%   followed by one division by hypot(c, s) to undo whatever the inputs were
%   off unit by.  This is the same construction angle3D.m uses in three
%   dimensions, specialized to the plane, so the 2D and 3D rotations agree.
%
%   WHY NOT atan2.  The angle-first alternative is
%
%     theta = wrap(atan2(v(2), v(1)) - atan2(u(2), u(1)));
%     R     = [cos(theta) -sin(theta); sin(theta) cos(theta)];
%
%   which reaches the same matrix through six library transcendental calls
%   (two atan2, the complex exponential and the atan2 inside the wrap, then
%   cos and sin) where the construction above uses none.  The consequences:
%
%     - it is a round trip.  cos and sin have to undo what atan2 just did,
%       and each of the six calls carries its own rounding.  Against a
%       rotation known to full precision, ||R - R_exact|| comes out two to
%       seven times larger for the angle-first path at every angle sampled
%       over a full turn, four times larger at the median; worst case over
%       that sweep, 1.2e-15 against 2.8e-16.  Both are a few units in the
%       last place, so this is a reason to prefer the construction and not
%       a defect being repaired: see
%       ../2D_curve/example_ellipse_cut_rotation_construction.m, which also
%       shows the difference does not reach the computed solution.
%     - it needs a branch cut.  The difference of two atan2 values lands in
%       (-2pi, 2pi] and has to be wrapped back, so the stored datum is
%       discontinuous at theta = +-pi even though the rotation it stands for
%       is not.  A (c, s) pair has no such seam, which matters as soon as
%       rotations are compared, averaged or interpolated; rot2d_err is what
%       that buys.
%     - it is not reproducible.  IEEE 754 pins down the accuracy of
%       multiply, add, divide and hypot but says nothing about atan2, sin or
%       cos, so the angle-first matrix can differ in its last bits between
%       platforms, libm versions and MATLAB releases.  Dot, cross and hypot
%       cannot.
%
%   Pass method = 'atan2' for the angle-first path instead; it exists so the
%   experiment above can run the two side by side, and nothing else uses it.
%   method = 'rodrigues' is the default.
%
%   See also GLUE_ROT2D, ROTATE_ABOUT2D, ROT2D_ERR, ANGLE3D.

  if (nargin < 3) || isempty(method)
    method = 'rodrigues';
  end

  if (size(u, 2) ~= 2) || (size(v, 2) ~= 2)
    error('rot2d_from_to: directions must be N-by-2');
  end
  if (size(u, 1) ~= size(v, 1))
    error('rot2d_from_to: direction arrays must have the same number of rows');
  end

  switch lower(method)
    case 'rodrigues'
      % Rodrigues about e_z: the cosine is the dot product and the sine the
      % out-of-plane component of the cross product.  No transcendental is
      % evaluated, and no angle exists to be wrapped.
      c = u(:,1).*v(:,1) + u(:,2).*v(:,2);
      s = u(:,1).*v(:,2) - u(:,2).*v(:,1);
      nrm = hypot(c, s);
      c = c ./ nrm;
      s = s ./ nrm;

    case 'atan2'
      % the angle-first path, kept only for the comparison experiment
      th = atan2(v(:,2), v(:,1)) - atan2(u(:,2), u(:,1));
      th = angle(exp(1i*th));           % rewrap to (-pi, pi]
      c = cos(th);
      s = sin(th);

    otherwise
      error('rot2d_from_to: unknown method ''%s''', method);
  end

  if (nargout > 2)
    n = numel(c);
    R = zeros(2, 2, n);
    R(1,1,:) = c;  R(1,2,:) = -s;
    R(2,1,:) = s;  R(2,2,:) =  c;
  end

  if (nargout > 3)
    % reporting only: every rotation above is already complete without it
    theta = atan2(s, c);
  end
end
