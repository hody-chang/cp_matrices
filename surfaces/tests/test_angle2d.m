function [pass, str] = test_angle2d()
  str = ['angle2d: cp/cpbar tangent estimate at arc endpoint'];

  pass = [];
  c = 0;

  cpf = @(x,y) cpArc(x, y, 1, [], -pi/2, pi/2);
  singularPoint = [0 1];

  x = [-0.010 -0.008 -0.006 -0.004 -0.002 0];
  y = ones(size(x));

  % angle2d returns the ROTATION, not an angle: R takes [1; 0] onto the
  % estimated tangent, built straight from the tangent's components by
  % Rodrigues' formula about the out-of-plane axis.  The angle is in
  % info.theta, for reporting.
  [R, tangent, info] = angle2d(x, y, cpf, singularPoint);

  c = c + 1;
  pass(c) = assertAlmostEqual(tangent, [-1 0], 1e-2);

  % the exact tangent here is [-1 0], so the exact rotation is -I
  c = c + 1;
  pass(c) = assertAlmostEqual(R, [-1 0; 0 -1], 1e-2);

  c = c + 1;
  pass(c) = assertAlmostEqual(R*[1; 0], tangent(:), 100*eps);

  % a proper rotation: orthogonal with determinant +1
  c = c + 1;
  pass(c) = assertAlmostEqual(R.'*R, eye(2), 10*eps);

  c = c + 1;
  pass(c) = assertAlmostEqual(det(R), 1, 10*eps);

  % The angle is tested through its absolute value on purpose.  The exact
  % tangent lies on the branch cut of atan2, so info.theta is near +pi or
  % near -pi according to the sign of a tangent component that the
  % estimator only pins down to its own order.  R is continuous there and
  % the test above is the one that has a two-sided tolerance; this is the
  % reason angle2d returns R and not an angle.
  c = c + 1;
  pass(c) = assertAlmostEqual(abs(info.theta), pi, 1e-2);

  c = c + 1;
  pass(c) = (info.numZero > 0);

  c = c + 1;
  pass(c) = (info.numValid > 0);

  [R0, tangent0, info0] = angle2d(0, 1, cpf, singularPoint);

  c = c + 1;
  pass(c) = (info0.numValid == 0);

  c = c + 1;
  pass(c) = all(isnan([R0(:).' tangent0 info0.theta]));
