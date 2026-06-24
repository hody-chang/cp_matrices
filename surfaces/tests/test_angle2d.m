function [pass, str] = test_angle2d()
  str = ['angle2d: cp/cpbar tangent estimate at arc endpoint'];

  pass = [];
  c = 0;

  cpf = @(x,y) cpArc(x, y, 1, [], -pi/2, pi/2);
  singularPoint = [0 1];

  x = [-0.010 -0.008 -0.006 -0.004 -0.002 0];
  y = ones(size(x));

  [theta, tangent, info] = angle2d(x, y, cpf, singularPoint);

  c = c + 1;
  pass(c) = assertAlmostEqual(tangent, [-1 0], 1e-2);

  c = c + 1;
  pass(c) = assertAlmostEqual(theta, pi, 1e-2);

  c = c + 1;
  pass(c) = (info.numZero > 0);

  c = c + 1;
  pass(c) = (info.numValid > 0);

  [theta0, tangent0, info0] = angle2d(0, 1, cpf, singularPoint);

  c = c + 1;
  pass(c) = (info0.numValid == 0);

  c = c + 1;
  pass(c) = all(isnan([theta0 tangent0]));
