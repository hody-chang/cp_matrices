Code by Tom Maerz, computes CP and signed distance for general polygons

This code also does tangents and various other things, these have been
(temporarily?) to "other" to keep them out of the namespace (this
directory will likely be added to matlab's path).

Issues: the signed distance used to depend on the orientation of the
curve, and was sometimes incorrectly zero.  Both came from the way the
sign was deduced at the nearest vertex; helper_sDistPoly now takes the
sign from a point-in-polygon test instead.
