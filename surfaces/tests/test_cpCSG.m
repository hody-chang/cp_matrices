function [pass, str] = test_cpCSG()
  str = 'cpCSG tests on unions of disjoint circles';

  c = 0;
  pass = [];

  R = 1;
  cen1 = [-2 0];
  cen2 = [2 0];
  cen3 = [0 3];

  leaf1 = csgLeaf(@(x, y) cpCircle(x, y, R, cen1), 2);
  leaf2 = csgLeaf(@(x, y) cpCircle(x, y, R, cen2), 2);
  leaf3 = csgLeaf(@(x, y) cpCircle(x, y, 0.5, cen3), 2);

  tree = csgUnion(leaf1, leaf2);
  tree_recursive = csgUnion(tree, leaf3);

  [cpx, cpy, dist] = cpCSG(-4, 0, tree);
  c = c + 1;
  pass(c) = assertAlmostEqual([cpx cpy dist], [-3 0 1]);

  [cpx, cpy, dist] = cpCSG(4, 0, tree);
  c = c + 1;
  pass(c) = assertAlmostEqual([cpx cpy dist], [3 0 1]);

  [cpx, cpy] = cpCSG([-4 4], [0 0], tree);
  c = c + 1;
  pass(c) = assertAlmostEqual(cpx, [-3 3]);
  c = c + 1;
  pass(c) = assertAlmostEqual(cpy, [0 0]);

  [cpx, cpy, dist] = cpCSG(0, 4, tree_recursive);
  c = c + 1;
  pass(c) = assertAlmostEqual([cpx cpy dist], [0 3.5 0.5]);
