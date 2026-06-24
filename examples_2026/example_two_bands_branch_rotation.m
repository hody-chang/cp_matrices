%% Heat equation on a circle cut into two glued branches
% Prototype iCPM branch gluing by endpoint angle rotation/unfolding.
%
% Branch A is the left semicircle and branch B is the right semicircle.
% Endpoint extension rows are routed through the opposite branch after
% rotating the embedding point about the shared endpoint.


%% Using cp_matrices

% Include the cp_matrices folder (edit as appropriate)
addpath('../cp_matrices');

% add functions for finding the closest points
addpath('../surfaces');


global ICPM2009BANDINGCHECKS

% this is a bit dangerous: will break other less tightly banded
% codes, turn it off later
ICPM2009BANDINGCHECKS = 1;
cleanupICPM2009BANDINGCHECKS = onCleanup(@() reset_icpm2009bandingchecks());


%%
% 2D example on a circle
% Construct a grid in the embedding space

makePlots = true;


dx = 0.025/2;   % grid size

% make vectors of x, y, positions of the grid
x1d = (-2:dx:2)';
y1d = (-1.4:dx:1.4)';
dy = dx;

nx = length(x1d);
ny = length(y1d);


%% Find closest points on the two branches

[xx yy] = meshgrid(x1d, y1d);

RADIUS = 1;
cen = [0 0];

% A is the left semicircle, B is the right semicircle.
angleA1 = pi/2;   angleA2 = -pi/2;
angleB1 = -pi/2;  angleB2 = pi/2;

[cpxA, cpyA, distA, bdyA] = cpArc(xx, yy, RADIUS, cen, angleA1, angleA2);
[cpxB, cpyB, distB, bdyB] = cpArc(xx, yy, RADIUS, cen, angleB1, angleB2);


%% Banding parameters

dim = 2;  % dimension
p = 3;    % interpolation order
order = 2;  % Laplacian order: bw will need to increase if changed
fd_stenrad = order/2;  % Finite difference stencil radius
bw = 1.0002*sqrt((dim-1)*((p+1)/2)^2 + ((fd_stenrad+(p+1)/2)^2));


%% Branch A two-band iCPM matrices

band_initA = find(abs(distA) <= bw*dx);
cpxg_initA = cpxA(band_initA); cpyg_initA = cpyA(band_initA);
xg_initA = xx(band_initA); yg_initA = yy(band_initA);
bdyg_initA = bdyA(band_initA);

disp('Constructing branch A interpolation matrix');
EtempA = interp2_matrix(x1d, y1d, cpxg_initA, cpyg_initA, p);
[iA,jA,SA] = find(EtempA);
innerbandA = unique(jA);

LtempA = laplacian_2d_matrix(x1d, y1d, order, innerbandA, band_initA);
[iA,jA,SA] = find(LtempA);
outerbandtempA = unique(jA);
outerbandA = band_initA(outerbandtempA);

cpxgoutA = cpxg_initA(outerbandtempA); cpygoutA = cpyg_initA(outerbandtempA);
xgoutA = xg_initA(outerbandtempA); ygoutA = yg_initA(outerbandtempA);
bdygoutA = bdyg_initA(outerbandtempA);

LA = LtempA(:, outerbandtempA);
EAA = EtempA(outerbandtempA, innerbandA);
clear LtempA EtempA outerbandtempA

innerInOuterA = zeros(size(innerbandA));
RA = sparse([],[],[],length(innerbandA),length(outerbandA),length(innerbandA));
for i=1:length(innerbandA)
  I = find(outerbandA == innerbandA(i));
  innerInOuterA(i) = I;
  RA(i,I) = 1;
end

cpxginA = RA*cpxgoutA;  cpyginA = RA*cpygoutA;
xginA = RA*xgoutA;  yginA = RA*ygoutA;


%% Branch B two-band iCPM matrices

band_initB = find(abs(distB) <= bw*dx);
cpxg_initB = cpxB(band_initB); cpyg_initB = cpyB(band_initB);
xg_initB = xx(band_initB); yg_initB = yy(band_initB);
bdyg_initB = bdyB(band_initB);

disp('Constructing branch B interpolation matrix');
EtempB = interp2_matrix(x1d, y1d, cpxg_initB, cpyg_initB, p);
[iB,jB,SB] = find(EtempB);
innerbandB = unique(jB);

LtempB = laplacian_2d_matrix(x1d, y1d, order, innerbandB, band_initB);
[iB,jB,SB] = find(LtempB);
outerbandtempB = unique(jB);
outerbandB = band_initB(outerbandtempB);

cpxgoutB = cpxg_initB(outerbandtempB); cpygoutB = cpyg_initB(outerbandtempB);
xgoutB = xg_initB(outerbandtempB); ygoutB = yg_initB(outerbandtempB);
bdygoutB = bdyg_initB(outerbandtempB);

LB = LtempB(:, outerbandtempB);
EBB = EtempB(outerbandtempB, innerbandB);
clear LtempB EtempB outerbandtempB

innerInOuterB = zeros(size(innerbandB));
RB = sparse([],[],[],length(innerbandB),length(outerbandB),length(innerbandB));
for i=1:length(innerbandB)
  I = find(outerbandB == innerbandB(i));
  innerInOuterB(i) = I;
  RB(i,I) = 1;
end

cpxginB = RB*cpxgoutB;  cpyginB = RB*cpygoutB;
xginB = RB*xgoutB;  yginB = RB*ygoutB;


%% Endpoint angle rotations for branch gluing

% Endpoint ids follow cpArc: bdy=1 is angle1, bdy=2 is angle2.
% The smooth cut-circle prototype has zero tangent mismatch, but these
% stored angles are still used below in the row routing.
thetaAtoB = zeros(2,1);
thetaBtoA = zeros(2,1);

tangentA = [-sin(angleA1) cos(angleA1); ...
            -sin(angleA2) cos(angleA2)];
tangentB = [-sin(angleB1) cos(angleB1); ...
            -sin(angleB2) cos(angleB2)];

thetaAtoB(1) = angle(exp(1i*(atan2(tangentB(2,2),tangentB(2,1)) - ...
                             atan2(tangentA(1,2),tangentA(1,1)))));
thetaAtoB(2) = angle(exp(1i*(atan2(tangentB(1,2),tangentB(1,1)) - ...
                             atan2(tangentA(2,2),tangentA(2,1)))));
thetaBtoA(1) = angle(exp(1i*(atan2(tangentA(2,2),tangentA(2,1)) - ...
                             atan2(tangentB(1,2),tangentB(1,1)))));
thetaBtoA(2) = angle(exp(1i*(atan2(tangentA(1,2),tangentA(1,1)) - ...
                             atan2(tangentB(2,2),tangentB(2,1)))));

endpointA1 = cen + RADIUS*[cos(angleA1) sin(angleA1)];
endpointA2 = cen + RADIUS*[cos(angleA2) sin(angleA2)];
endpointB1 = cen + RADIUS*[cos(angleB1) sin(angleB1)];
endpointB2 = cen + RADIUS*[cos(angleB2) sin(angleB2)];
endpointTol = 100*eps(max(1,RADIUS));

% cpArc labels off-arc closest points with bdy=1 or bdy=2, but exact
% endpoint-angle points can still have bdy=0.  Keep this local endpoint id
% for routing and theta lookup.
endpointIdA = bdygoutA;
endpointIdA(hypot(cpxgoutA - endpointA1(1), cpygoutA - endpointA1(2)) <= endpointTol) = 1;
endpointIdA(hypot(cpxgoutA - endpointA2(1), cpygoutA - endpointA2(2)) <= endpointTol) = 2;

endpointIdB = bdygoutB;
endpointIdB(hypot(cpxgoutB - endpointB1(1), cpygoutB - endpointB1(2)) <= endpointTol) = 1;
endpointIdB(hypot(cpxgoutB - endpointB2(1), cpygoutB - endpointB2(2)) <= endpointTol) = 2;

crossRowsA = find(endpointIdA ~= 0);
crossRowsB = find(endpointIdB ~= 0);

EAB = sparse(length(outerbandA), length(innerbandB));
EBA = sparse(length(outerbandB), length(innerbandA));

if (~isempty(crossRowsA))
  th = thetaAtoB(endpointIdA(crossRowsA));
  x0 = cpxgoutA(crossRowsA);  y0 = cpygoutA(crossRowsA);
  dx0 = xgoutA(crossRowsA) - x0;
  dy0 = ygoutA(crossRowsA) - y0;
  xr = x0 + cos(th).*dx0 - sin(th).*dy0;
  yr = y0 + sin(th).*dx0 + cos(th).*dy0;
  [cpxAtoB, cpyAtoB] = cpArc(xr, yr, RADIUS, cen, angleB1, angleB2);
  EAB(crossRowsA,:) = interp2_matrix(x1d, y1d, cpxAtoB, cpyAtoB, p, innerbandB);
  EAA(crossRowsA,:) = 0;
end

if (~isempty(crossRowsB))
  th = thetaBtoA(endpointIdB(crossRowsB));
  x0 = cpxgoutB(crossRowsB);  y0 = cpygoutB(crossRowsB);
  dx0 = xgoutB(crossRowsB) - x0;
  dy0 = ygoutB(crossRowsB) - y0;
  xr = x0 + cos(th).*dx0 - sin(th).*dy0;
  yr = y0 + sin(th).*dx0 + cos(th).*dy0;
  [cpxBtoA, cpyBtoA] = cpArc(xr, yr, RADIUS, cen, angleA1, angleA2);
  EBA(crossRowsB,:) = interp2_matrix(x1d, y1d, cpxBtoA, cpyBtoA, p, innerbandA);
  EBB(crossRowsB,:) = 0;
end

fprintf('Cross-branch extension rows A->B: %d of %d\n', ...
        length(crossRowsA), length(outerbandA));
fprintf('Cross-branch extension rows B->A: %d of %d\n', ...
        length(crossRowsB), length(outerbandB));
fprintf('Endpoint rotation angles A->B: [%g %g]\n', thetaAtoB(1), thetaAtoB(2));
fprintf('Endpoint rotation angles B->A: [%g %g]\n', thetaBtoA(1), thetaBtoA(2));


%% Diagonal splitting for the combined branch operator

% Cross-branch extension rows live in the off-diagonal blocks of Eblock.
Lblock = blkdiag(LA, LB);
Eblock = [EAA EAB; EBA EBB];
Rblock = blkdiag(RA, RB);
M = lapsharp_unordered(Lblock, Eblock, Rblock);

% after building matrices, don't need this set
ICPM2009BANDINGCHECKS = 0;


%% Construct interpolation matrices for diagnostics

nplot = 500;
[xpA,ypA,thplotA] = paramArc(nplot, RADIUS, cen, angleA1, angleA2);
[xpB,ypB,thplotB] = paramArc(nplot, RADIUS, cen, angleB1, angleB2);

EplotA = interp2_matrix(x1d, y1d, xpA, ypA, p, innerbandA);
EplotB = interp2_matrix(x1d, y1d, xpB, ypB, p, innerbandB);


%% Function u in the embedding space, initial conditions

[thgA, rgA] = cart2pol(cpxginA,cpyginA);
[thgB, rgB] = cart2pol(cpxginB,cpyginB);

u0A = cos(thgA);
u0B = cos(thgB);
u = [u0A; u0B];

uexactfn = @(t,th) exp(-t)*cos(th);
arcplot0A = EplotA*u0A;
arcplot0B = EplotB*u0B;


%% Shared vertex evaluation and averaging

sx = [endpointA1(1); endpointA2(1)];
sy = [endpointA1(2); endpointA2(2)];

EvertexA = interp2_matrix(x1d, y1d, sx, sy, p, innerbandA);
EvertexB = interp2_matrix(x1d, y1d, sx, sy, p, innerbandB);

endpointNodeTol = dx*1e-8;
xInnerA = xx(innerbandA);  yInnerA = yy(innerbandA);
xInnerB = xx(innerbandB);  yInnerB = yy(innerbandB);
vertexWriteIdxA = zeros(length(sx),1);
vertexWriteIdxB = zeros(length(sx),1);
vertexWriteExactA = true(length(sx),1);
vertexWriteExactB = true(length(sx),1);

for k = 1:length(sx)
  I = find((abs(xInnerA - sx(k)) <= endpointNodeTol) & ...
           (abs(yInnerA - sy(k)) <= endpointNodeTol));
  if (isempty(I))
    [dummy, I] = min(hypot(xInnerA - sx(k), yInnerA - sy(k)));
    vertexWriteExactA(k) = false;
  end
  vertexWriteIdxA(k) = I(1);

  I = find((abs(xInnerB - sx(k)) <= endpointNodeTol) & ...
           (abs(yInnerB - sy(k)) <= endpointNodeTol));
  if (isempty(I))
    [dummy, I] = min(hypot(xInnerB - sx(k), yInnerB - sy(k)));
    vertexWriteExactB(k) = false;
  end
  vertexWriteIdxB(k) = I(1);
end

if (any(~vertexWriteExactA) || any(~vertexWriteExactB))
  fprintf('Vertex averaging fallback used: nearest innerband DOF for missing endpoint node.\n');
end


%% Time-stepping for the heat equation

Tf = 1.00;
dt = dx/10;
numtimesteps = ceil(Tf/dt);
dt = Tf / numtimesteps;

I = speye(size(M));
A = I - dt*M;

for kt = 1:numtimesteps
  u = A \ u;
  uA = u(1:length(innerbandA));
  uB = u(length(innerbandA)+1:end);

  vertexValuesA = EvertexA*uA;
  vertexValuesB = EvertexB*uB;
  vertexAvgValues = 0.5*(vertexValuesA + vertexValuesB);
  uA(vertexWriteIdxA) = vertexAvgValues;
  uB(vertexWriteIdxB) = vertexAvgValues;
  u = [uA; uB];

  t = kt*dt;

  % plotting
  if (makePlots && ((kt < 5) || (mod(kt,200) == 0) || (kt == numtimesteps)))

    % plot in the embedded domain: shows the computational bands
    figure(1);
    plot2d_compdomain([uA; uB], [xginA; xginB], [yginA; yginB], dx, dy, 1);
    hold on;
    plot(xpA, ypA, 'k-', 'linewidth', 2);
    plot(xpB, ypB, 'k--', 'linewidth', 2);
    title( ['embedded domain: soln at time ' num2str(t) ...
            ', timestep #' num2str(kt)] );

    % plot value on circle branches
    figure(2); clf;
    arcplotA = EplotA*uA;
    arcplotB = EplotB*uB;
    hA = plot(thplotA, arcplotA, 'b-');
    hold on;
    hB = plot(thplotB, arcplotB, 'c-');
    hExact = plot(thplotA, uexactfn(t,thplotA), 'r--');
    plot(thplotB, uexactfn(t,thplotB), 'r--');
    hInitial = plot(thplotA, arcplot0A, 'g-.');
    plot(thplotB, arcplot0B, 'g-.');
    title( ['soln at time ' num2str(t) ', on circle branches'] );
    xlabel('theta'); ylabel('u');
    legend([hA hB hExact hInitial], ...
           'branch A iCPM', 'branch B iCPM', 'exact answer', ...
           'initial condition ', 'Location', 'SouthEast');

    pause(0);
  end
end

t = numtimesteps*dt;
uA = u(1:length(innerbandA));
uB = u(length(innerbandA)+1:end);

arcplotA = EplotA*uA;
arcplotB = EplotB*uB;

errorA = max(abs(uexactfn(t,thplotA) - arcplotA));
errorB = max(abs(uexactfn(t,thplotB) - arcplotB));
vertexValuesA = EvertexA*uA;
vertexValuesB = EvertexB*uB;
vertexDiffFinal = abs(vertexValuesA - vertexValuesB);

fprintf('Max error on branch A at t=%g: %g\n', t, errorA);
fprintf('Max error on branch B at t=%g: %g\n', t, errorB);
fprintf('Final branch vertex differences [top bottom]: [%g %g]\n', ...
        vertexDiffFinal(1), vertexDiffFinal(2));


function reset_icpm2009bandingchecks()
  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = 0;
end
