function q = rotate_about2d(x, v, c, s)
%ROTATE_ABOUT2D  rotate points about a centre, from a (cos, sin) pair
%   q = rotate_about2d(x, v, c, s) turns the rows of the N-by-2 array x
%   about the point v = [vx vy] through the rotation whose cosine and sine
%   are c and s.  c and s are scalars, or N-by-1 columns for a per-row
%   rotation; v is 1-by-2, or N-by-2 for a per-row centre.
%
%   The pair (c, s) comes from glue_rot2d, which builds it from the two
%   outward tangents without ever forming the angle.  Taking (c, s) rather
%   than an angle is the point: an angle argument would have to be turned
%   back into a cosine and a sine here, which is the round trip glue_rot2d
%   exists to avoid, and it would be done again at every call site.
%
%   See also GLUE_ROT2D.

  d = x - v;
  q = [c.*d(:,1) - s.*d(:,2), s.*d(:,1) + c.*d(:,2)] + v;
end
