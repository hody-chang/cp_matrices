function [pass, str] = test_cpTrimmedCurve2d()
  str = 'cpTrimmedCurve2d tests on a circle with a circular cut';

  c = 0;
  pass = [];

  curve_leaf = csgLeaf(@(x, y) cpCircle(x, y, 1, [0 0]), 2);
  trim_leaf = csgLeaf(@(x, y) cpCircle(x, y, 0.5, [0 1]), 2);

  seam_y = (1 + 1^2 - 0.5^2) / 2;
  seam_x = sqrt(1 - seam_y^2);

  [cpx, cpy, dist, bdy] = cpTrimmedCurve2d(0, -2, curve_leaf, trim_leaf);
  c = c + 1;
  pass(c) = assertAlmostEqual([cpx cpy dist double(bdy)], [0 -1 1 0], 1e-8);

  [cpx, cpy, dist, bdy] = cpTrimmedCurve2d(0, 1.2, curve_leaf, trim_leaf);
  c = c + 1;
  pass(c) = assertAlmostEqual([abs(cpx) cpy dist double(bdy)], ...
                              [seam_x seam_y sqrt(seam_x^2 + (1.2 - seam_y)^2) 1], 1e-8);
