function [pass, str] = test_cpCSG_circle_boolean_ops()
  str = 'cpCSG boolean operations on two overlapping circles';

  c = 0;
  pass = [];

  R = 1;
  shift = 0.75;
  seam_y = sqrt(R^2 - shift^2);
  tol = 1e-8;
  cen1 = [-shift 0];
  cen2 = [ shift 0];

  left = csgLeaf(@(x, y) cpCircle(x, y, R, cen1), 2);
  right = csgLeaf(@(x, y) cpCircle(x, y, R, cen2), 2);

  union_tree = csgUnion(left, right);
  intersection_tree = csgIntersection(left, right);
  difference_tree = csgDifference(left, right);

  [cpx, cpy, dist] = cpCSG(0, 0, union_tree);
  c = c + 1;
  pass(c) = assertAlmostEqual(cpx, 0, tol);
  c = c + 1;
  pass(c) = assertAlmostEqual(abs(cpy), seam_y, tol);
  c = c + 1;
  pass(c) = assertAlmostEqual(dist, seam_y, tol);

  [cpx, cpy, dist] = cpCSG(0, 1, intersection_tree);
  c = c + 1;
  pass(c) = assertAlmostEqual(cpx, 0, tol);
  c = c + 1;
  pass(c) = assertAlmostEqual(cpy, seam_y, tol);
  c = c + 1;
  pass(c) = assertAlmostEqual(dist, 1 - seam_y, tol);

  [cpx, cpy, dist] = cpCSG(0, 0, intersection_tree);
  c = c + 1;
  pass(c) = assertAlmostEqual(abs(cpx), R - shift, tol);
  c = c + 1;
  pass(c) = assertAlmostEqual(cpy, 0, tol);
  c = c + 1;
  pass(c) = assertAlmostEqual(dist, R - shift, tol);

  [cpx, cpy, dist] = cpCSG(0, 0, difference_tree);
  c = c + 1;
  pass(c) = assertAlmostEqual([cpx cpy dist], [shift - R, 0, R - shift]);

  [cpx, cpy, dist] = cpCSG(-3, 0, union_tree);
  c = c + 1;
  pass(c) = assertAlmostEqual([cpx cpy dist], [-(shift + R), 0, 3 - (shift + R)]);

  [cpx, cpy, dist] = cpCSG(-3, 0, difference_tree);
  c = c + 1;
  pass(c) = assertAlmostEqual([cpx cpy dist], [-(shift + R), 0, 3 - (shift + R)]);
