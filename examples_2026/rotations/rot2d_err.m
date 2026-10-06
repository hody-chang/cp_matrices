function [dth, dR] = rot2d_err(c, s, cref, sref)
%ROT2D_ERR  error of a plane rotation against a reference, from (cos, sin)
%   [dth, dR] = rot2d_err(c, s, cref, sref) compares the rotation with
%   cosine c and sine s against the reference rotation (cref, sref).  All
%   four arguments are arrays of the same size, or scalars, and both
%   outputs have that size.
%
%     dth   the signed angle of the relative rotation Rref' * R, in
%           (-pi, pi];
%     dR    ||R - Rref||_2, which for plane rotations is exactly
%           2*|sin(dth/2)|.
%
%   Both come from the relative rotation
%
%     Rref' * R = [crel -srel; srel crel],
%     crel = c*cref + s*sref,   srel = s*cref - c*sref,
%
%   which is already in (-pi, pi] as an angle, so there is nothing to
%   rewrap.  That is the point of taking (cos, sin) pairs rather than
%   angles: the obvious alternative, wrap(theta - thetaref), is a
%   difference of two angles that each had to be produced by an atan2 and
%   then pushed back across a branch cut, and it loses digits exactly where
%   the comparison matters, when theta and thetaref both sit near +-pi.
%
%   dR is formed by the half-angle identity rather than from dth, in the
%   branch that avoids cancellation:
%
%     |sin(dth/2)| = |srel| / sqrt(2*(1 + crel))     when crel >= 0,
%                  = sqrt((1 - crel)/2)              when crel <  0.
%
%   A good estimate has crel within a rounding of 1, where sqrt((1-crel)/2)
%   would deliver only half the digits of the input.
%
%   See also GLUE_ROT2D.

  crel = c.*cref + s.*sref;
  srel = s.*cref - c.*sref;

  dth = atan2(srel, crel);

  if (nargout > 1)
    near = crel >= 0;
    dR = zeros(size(crel));
    dR(near)  = 2*abs(srel(near)) ./ sqrt(2*(1 + crel(near)));
    dR(~near) = 2*sqrt((1 - crel(~near))/2);
  end
end
