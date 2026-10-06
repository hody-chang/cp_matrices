function [pass, str] = test_cpPolygon_cpSquare_compare()
  str = ['cpPolygon test: compare to cpSquare'];

  pass = [];
  c = 0;

  poly = [-1    -1
           1    -1
           1     1
          -1     1];

  % delete the origin, cp is ambiguous there
  dx = 1;
  xx = [-3:dx:-dx  dx:dx:3];
  yy = [-2:dx:-dx  dx:dx:2];
  [x,y] = meshgrid(xx, yy);

  [cpx1, cpy1, sd1] = cpPolygon(x, y, poly);
  [cpx2, cpy2, sd2] = cpSquare(x, y);

  c = c + 1;
  pass(c) = assertAlmostEqual(cpx1, cpx2);
  c = c + 1;
  pass(c) = assertAlmostEqual(cpy1, cpy2);

  c = c + 1;
  pass(c) = assertAlmostEqual(sd1, sd2);


  % (-2,-1) lies along the line through the bottom edge: the sign of the
  % signed distance used to come out as zero here.
  [cpxa, cpya, sda] = cpPolygon(-2, -1, poly);
  [cpxb, cpyb, sdb] = cpSquare(-2, -1);
  c = c + 1;
  pass(c) = assertAlmostEqual(sda, 1);
  c = c + 1;
  pass(c) = assertAlmostEqual(sdb, 1);

  % same square, vertices listed the other way round: the signed
  % distance must not care about the orientation
  [cpx3, cpy3, sd3] = cpPolygon(x, y, flipud(poly));
  c = c + 1;
  pass(c) = assertAlmostEqual(cpx3, cpx2);
  c = c + 1;
  pass(c) = assertAlmostEqual(cpy3, cpy2);
  c = c + 1;
  pass(c) = assertAlmostEqual(sd3, sd2);
