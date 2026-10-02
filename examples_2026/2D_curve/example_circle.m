%% Heat equation on a circle cut into two glued branches
% Prototype iCPM branch gluing by endpoint angle rotation/unfolding.
%
% Branch A is the left semicircle and branch B is the right semicircle.
% Endpoint extension rows are routed through the opposite branch after
% rotating the embedding point about the shared endpoint.


%% Using cp_matrices

% Resolve dependencies relative to this example.
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', '..', 'cp_matrices'));
addpath(fullfile(here, '..', '..', 'surfaces'));


global ICPM2009BANDINGCHECKS

% this is a bit dangerous: will break other less tightly banded
% codes, turn it off later
ICPM2009BANDINGCHECKS = 1;
cleanup_bandingchecks = onCleanup(@() reset_icpm2009bandingchecks());


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

inner_in_outerA = zeros(size(innerbandA));
RA = sparse([],[],[],length(innerbandA),length(outerbandA),length(innerbandA));
for i=1:length(innerbandA)
  I = find(outerbandA == innerbandA(i));
  inner_in_outerA(i) = I;
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

inner_in_outerB = zeros(size(innerbandB));
RB = sparse([],[],[],length(innerbandB),length(outerbandB),length(innerbandB));
for i=1:length(innerbandB)
  I = find(outerbandB == innerbandB(i));
  inner_in_outerB(i) = I;
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
endpttol = 100*eps(max(1,RADIUS));

% cpArc labels off-arc closest points with bdy=1 or bdy=2, but exact
% endpoint-angle points can still have bdy=0.  Keep this local endpoint id
% for routing and theta lookup.
endptidA = bdygoutA;
endptidA(hypot(cpxgoutA - endpointA1(1), cpygoutA - endpointA1(2)) <= endpttol) = 1;
endptidA(hypot(cpxgoutA - endpointA2(1), cpygoutA - endpointA2(2)) <= endpttol) = 2;

endptidB = bdygoutB;
endptidB(hypot(cpxgoutB - endpointB1(1), cpygoutB - endpointB1(2)) <= endpttol) = 1;
endptidB(hypot(cpxgoutB - endpointB2(1), cpygoutB - endpointB2(2)) <= endpttol) = 2;

crossrowsA = find(endptidA ~= 0);
crossrowsB = find(endptidB ~= 0);

EAB = sparse(length(outerbandA), length(innerbandB));
EBA = sparse(length(outerbandB), length(innerbandA));

if (~isempty(crossrowsA))
  th = thetaAtoB(endptidA(crossrowsA));
  x0 = cpxgoutA(crossrowsA);  y0 = cpygoutA(crossrowsA);
  dx0 = xgoutA(crossrowsA) - x0;
  dy0 = ygoutA(crossrowsA) - y0;
  xr = x0 + cos(th).*dx0 - sin(th).*dy0;
  yr = y0 + sin(th).*dx0 + cos(th).*dy0;
  [cpxAtoB, cpyAtoB] = cpArc(xr, yr, RADIUS, cen, angleB1, angleB2);
  EAB(crossrowsA,:) = interp2_matrix(x1d, y1d, cpxAtoB, cpyAtoB, p, innerbandB);
  EAA(crossrowsA,:) = 0;
end

if (~isempty(crossrowsB))
  th = thetaBtoA(endptidB(crossrowsB));
  x0 = cpxgoutB(crossrowsB);  y0 = cpygoutB(crossrowsB);
  dx0 = xgoutB(crossrowsB) - x0;
  dy0 = ygoutB(crossrowsB) - y0;
  xr = x0 + cos(th).*dx0 - sin(th).*dy0;
  yr = y0 + sin(th).*dx0 + cos(th).*dy0;
  [cpxBtoA, cpyBtoA] = cpArc(xr, yr, RADIUS, cen, angleA1, angleA2);
  EBA(crossrowsB,:) = interp2_matrix(x1d, y1d, cpxBtoA, cpyBtoA, p, innerbandA);
  EBB(crossrowsB,:) = 0;
end

fprintf('Cross-branch extension rows A->B: %d of %d\n', ...
        length(crossrowsA), length(outerbandA));
fprintf('Cross-branch extension rows B->A: %d of %d\n', ...
        length(crossrowsB), length(outerbandB));
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

endpt_nodetol = dx*1e-8;
xinnerA = xx(innerbandA);  yinnerA = yy(innerbandA);
xinnerB = xx(innerbandB);  yinnerB = yy(innerbandB);
vertex_writeidxA = zeros(length(sx),1);
vertex_writeidxB = zeros(length(sx),1);
vertex_write_exactA = true(length(sx),1);
vertex_write_exactB = true(length(sx),1);

for k = 1:length(sx)
  I = find((abs(xinnerA - sx(k)) <= endpt_nodetol) & ...
           (abs(yinnerA - sy(k)) <= endpt_nodetol));
  if (isempty(I))
    [~, I] = min(hypot(xinnerA - sx(k), yinnerA - sy(k)));
    vertex_write_exactA(k) = false;
  end
  vertex_writeidxA(k) = I(1);

  I = find((abs(xinnerB - sx(k)) <= endpt_nodetol) & ...
           (abs(yinnerB - sy(k)) <= endpt_nodetol));
  if (isempty(I))
    [~, I] = min(hypot(xinnerB - sx(k), yinnerB - sy(k)));
    vertex_write_exactB(k) = false;
  end
  vertex_writeidxB(k) = I(1);
end

if (any(~vertex_write_exactA) || any(~vertex_write_exactB))
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

  vertex_valsA = EvertexA*uA;
  vertex_valsB = EvertexB*uB;
  vertex_avg = 0.5*(vertex_valsA + vertex_valsB);
  uA(vertex_writeidxA) = vertex_avg;
  uB(vertex_writeidxB) = vertex_avg;
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
    hexact = plot(thplotA, uexactfn(t,thplotA), 'r--');
    plot(thplotB, uexactfn(t,thplotB), 'r--');
    hinit = plot(thplotA, arcplot0A, 'g-.');
    plot(thplotB, arcplot0B, 'g-.');
    title( ['soln at time ' num2str(t) ', on circle branches'] );
    xlabel('theta'); ylabel('u');
    legend([hA hB hexact hinit], ...
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
vertex_valsA = EvertexA*uA;
vertex_valsB = EvertexB*uB;
vertex_diff_final = abs(vertex_valsA - vertex_valsB);

fprintf('Max error on branch A at t=%g: %g\n', t, errorA);
fprintf('Max error on branch B at t=%g: %g\n', t, errorB);
fprintf('Final branch vertex differences [top bottom]: [%g %g]\n', ...
        vertex_diff_final(1), vertex_diff_final(2));


function reset_icpm2009bandingchecks()
  global ICPM2009BANDINGCHECKS
  ICPM2009BANDINGCHECKS = 0;
end
