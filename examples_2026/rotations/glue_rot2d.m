function [c, s, R, theta] = glue_rot2d(tausrc, tautgt, method)
%GLUE_ROT2D  the glue rotation at a cut point, from the two outward tangents
%   [c, s, R, theta] = glue_rot2d(tausrc, tautgt) returns the rotation that
%   carries the source branch's outward tangent onto the INWARD tangent of
%   the target branch, which is -tautgt.  Leaving the source arc at a cut
%   point means entering the target arc, so outward-onto-inward is the
%   pairing a rigid continuation needs.  On a smooth join tautgt = -tausrc
%   and the rotation is the identity.
%
%   tausrc and tautgt are N-by-2 arrays of (approximately) unit tangents,
%   one row per cut point, and the outputs are as in rot2d_from_to:
%
%     c, s    N-by-1 cosine and sine of the rotation, with c^2 + s^2 = 1 to
%             within one rounding;
%     R       2-by-2-by-N rotations, R(:,:,k) = [c -s; s c];
%     theta   N-by-1 angles in (-pi, pi], for REPORTING ONLY.
%
%   Apply the rotation about a cut point v with rotate_about2d, or inline as
%
%     xr = v(1) + c*(xq - v(1)) - s*(yq - v(2));
%     yr = v(2) + s*(xq - v(1)) + c*(yq - v(2));
%
%   The rotation is built by Rodrigues' formula about the out-of-plane axis,
%   so the cosine is a dot product of the two tangents and the sine their
%   cross product: no angle is formed and no transcendental is evaluated.
%   rot2d_from_to documents the construction and why it is preferred over
%   going through atan2.  method is forwarded there, and 'rodrigues' is the
%   default.
%
%   Swapping the two arguments gives the other branch's glue at the same cut
%   point.  Its cosine is the same dot product and its sine the negated
%   cross product, so the two branches' rotations are exact transposes of
%   one another -- an identity the angle-first path only satisfies to within
%   rounding.
%
%   See also ROT2D_FROM_TO, ROTATE_ABOUT2D, ROT2D_ERR, ANGLE3D.

  if (nargin < 3)
    method = '';
  end

  if (size(tautgt, 2) ~= 2)
    error('glue_rot2d: tangents must be N-by-2');
  end

  % the source's outward tangent goes onto the target's inward one
  switch nargout
    case {0, 1}
      c = rot2d_from_to(tausrc, -tautgt, method);
    case 2
      [c, s] = rot2d_from_to(tausrc, -tautgt, method);
    case 3
      [c, s, R] = rot2d_from_to(tausrc, -tautgt, method);
    otherwise
      [c, s, R, theta] = rot2d_from_to(tausrc, -tautgt, method);
  end
end
