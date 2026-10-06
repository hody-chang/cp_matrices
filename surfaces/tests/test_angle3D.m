function [pass, str] = test_angle3D()
  str = ['angle3D: cp/cpbar conormal estimate at surface branch'];

  pass = [];
  c = 0;

  cpf = @test_angle3D_halfPlaneCp;
  singularPoint = [0 0 0];

  x = zeros(1,4);
  y = [-0.4 -0.2 -0.1 0];
  z = [0.02 -0.01 0 0];

  % angle3D returns the ROTATION, not an angle.  Without an axis and a
  % target direction there is no rotation to build, so R is NaN(3).
  [R, conormal, info] = angle3D(x, y, z, cpf, singularPoint);

  c = c + 1;
  pass(c) = assertAlmostEqual(conormal, [0 -1 0], 1e-12);

  c = c + 1;
  pass(c) = all(isnan(R(:)));

  c = c + 1;
  pass(c) = isnan(info.theta);

  c = c + 1;
  pass(c) = (info.numCandidates == 4);

  c = c + 1;
  pass(c) = (info.numValid == 3);

  c = c + 1;
  pass(c) = (info.numZero > 0);

  % With an axis and a target, R is Rodrigues' rotation about that axis
  % carrying the conormal onto the target.  Here the axis is e_x, the
  % conormal -e_y and the target -e_z, so the turn is a quarter about e_x
  % and R is exactly [1 0 0; 0 0 -1; 0 1 0].
  axis = [1 0 0];
  targetDirection = [0 0 -1];
  [R2, conormal2, info2] = angle3D(x, y, z, cpf, singularPoint, axis, ...
                                   targetDirection);

  c = c + 1;
  pass(c) = assertAlmostEqual(conormal2, [0 -1 0], 1e-12);

  c = c + 1;
  pass(c) = assertAlmostEqual(R2, [1 0 0; 0 0 -1; 0 1 0], 1e-12);

  % the defining property: R carries the conormal onto the target
  c = c + 1;
  pass(c) = assertAlmostEqual(R2*conormal2(:), targetDirection(:), 1e-12);

  % a proper rotation, and one that fixes its own axis
  c = c + 1;
  pass(c) = assertAlmostEqual(R2.'*R2, eye(3), 1e-14);

  c = c + 1;
  pass(c) = assertAlmostEqual(det(R2), 1, 1e-14);

  c = c + 1;
  pass(c) = assertAlmostEqual(R2*axis(:), axis(:), 1e-14);

  c = c + 1;
  pass(c) = assertAlmostEqual(info2.theta, pi/2, 1e-12);

  c = c + 1;
  pass(c) = assertAlmostEqual(info2.axisUnit, [1 0 0], 1e-12);

  % the other branch's rotation is the inverse: swapping the source and
  % target directions about the same axis transposes R, exactly, because
  % Rodrigues takes the cosine from a dot product and the sine from a
  % cross product and only the latter changes sign
  [Rback, ~, infoback] = angle3D(x, y, z, cpf, singularPoint, axis, ...
                                 -targetDirection);

  c = c + 1;
  pass(c) = assertAlmostEqual(Rback, R2.', 0);

  c = c + 1;
  pass(c) = assertAlmostEqual(infoback.theta, -pi/2, 1e-12);

  noBdyCpf = @test_angle3D_noBdyHalfPlaneCp;
  [R3, conormal3, info3] = angle3D([0 0], [-0.4 0], [0 0], noBdyCpf);

  c = c + 1;
  pass(c) = assertAlmostEqual(conormal3, [0 -1 0], 1e-12);

  c = c + 1;
  pass(c) = all(isnan(R3(:)));

  c = c + 1;
  pass(c) = isnan(info3.theta);

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
