function [pass, str] = test_angle3D()
  str = ['angle3D: cp/cpbar conormal estimate at surface branch'];

  pass = [];
  c = 0;

  cpf = @test_angle3D_halfPlaneCp;
  singularPoint = [0 0 0];

  x = zeros(1,4);
  y = [-0.4 -0.2 -0.1 0];
  z = [0.02 -0.01 0 0];

  [theta, conormal, info] = angle3D(x, y, z, cpf, singularPoint);

  c = c + 1;
  pass(c) = assertAlmostEqual(conormal, [0 -1 0], 1e-12);

  c = c + 1;
  pass(c) = isnan(theta);

  c = c + 1;
  pass(c) = (info.numCandidates == 4);

  c = c + 1;
  pass(c) = (info.numValid == 3);

  c = c + 1;
  pass(c) = (info.numZero > 0);

  axis = [1 0 0];
  targetDirection = [0 0 -1];
  [theta2, conormal2, info2] = angle3D(x, y, z, cpf, singularPoint, axis, targetDirection);

  c = c + 1;
  pass(c) = assertAlmostEqual(conormal2, [0 -1 0], 1e-12);

  c = c + 1;
  pass(c) = assertAlmostEqual(theta2, pi/2, 1e-12);

  c = c + 1;
  pass(c) = assertAlmostEqual(info2.axisUnit, [1 0 0], 1e-12);

  noBdyCpf = @test_angle3D_noBdyHalfPlaneCp;
  [theta3, conormal3, info3] = angle3D([0 0], [-0.4 0], [0 0], noBdyCpf);

  c = c + 1;
  pass(c) = assertAlmostEqual(conormal3, [0 -1 0], 1e-12);

  c = c + 1;
  pass(c) = isnan(theta3);

  c = c + 1;
  pass(c) = (info3.numCandidates == 2);

  c = c + 1;
  pass(c) = ~info3.hasBdy;


function [cpx, cpy, cpz, dist, bdy] = test_angle3D_halfPlaneCp(x, y, z)

  cpx = x;
  cpy = max(y, 0);
  cpz = zeros(size(z));
  dist = sqrt((y - cpy).^2 + z.^2);
  bdy = (y < 0);


function [cpx, cpy, cpz, dist] = test_angle3D_noBdyHalfPlaneCp(x, y, z)

  cpx = x;
  cpy = max(y, 0);
  cpz = zeros(size(z));
  dist = sqrt((y - cpy).^2 + z.^2);
