function d = helper_sDistPoly(x, y, poly)
%helper function, use cpPolygon instead
%
%   Signed distance to the polygon: positive inside, negative
%   outside.  The result does not depend on whether the vertices in
%   'poly' are listed CW or CCW.

d = Inf*ones(size(x));

m = size(poly,1);
polyc = [poly; poly(1,:)];

% tangents
T = diff(polyc);
h = sqrt(T(:,1).^2 + T(:,2).^2);
T = T./(h * ones(1,2));

for k=1:m
    hx = x(:)-polyc(k,1);
    hy = y(:)-polyc(k,2);
    hn = sqrt(hx.^2 + hy.^2);

    alpha = hx .* T(k,1) + hy .* T(k,2);
    dda = abs(-hx .* T(k,2) + hy .* T(k,1));

    % if the perpendicular foot lands inside edge k, that perpendicular
    % distance is the candidate, otherwise use the distance to the start
    % vertex polyc(k,:).  (The other end of edge k is the start vertex of
    % edge k+1, so every vertex gets its turn.)
    onedge = (0 <= alpha) & (alpha <= h(k));
    cand = hn;
    cand(onedge) = dda(onedge);

    d(:) = min(d(:), cand);
end

% The sign comes from a point-in-polygon test.  It used to be read off
% the local geometry at the nearest vertex, by expressing the offset
% from that vertex in the basis of the two adjacent edge normals and
% taking the sign of the first coefficient.  That is wrong in two ways:
% the coefficient is exactly zero whenever the point happens to lie
% along the second normal (giving a signed distance of 0 for points
% nowhere near the polygon), and the normals flip with the orientation
% of the vertex list (giving the wrong sign for a CW polygon).
s = -ones(size(x));
s(inpolygon(x, y, poly(:,1), poly(:,2))) = 1;

d = s.*d;
