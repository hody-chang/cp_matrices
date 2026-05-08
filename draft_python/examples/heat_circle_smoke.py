"""One small heat-equation smoke example using the draft CPM package."""

from __future__ import annotations

import numpy as np

from cpmatrices import circle_closest_point, interp2_matrix, laplacian_2d_matrix, rm_bandwidth


def main():
    dx = 0.1
    x1d = np.arange(-2.0, 2.0 + 0.5 * dx, dx)
    y1d = x1d.copy()
    xx, yy = np.meshgrid(x1d, y1d, indexing="xy")

    cpx, cpy, dist = circle_closest_point(xx, yy)
    degree = 3
    order = 2
    band = np.flatnonzero(np.abs(dist.ravel(order="F")) <= rm_bandwidth(2, degree, order / 2) * dx)

    xg = xx.ravel(order="F")[band]
    yg = yy.ravel(order="F")[band]
    cpx_band = cpx.ravel(order="F")[band]
    cpy_band = cpy.ravel(order="F")[band]

    theta = np.arctan2(yg, xg)
    u = np.cos(theta)

    e_mat = interp2_matrix(x1d, y1d, cpx_band, cpy_band, degree=degree, band=band)
    l_mat = laplacian_2d_matrix(x1d, y1d, order=order, band1=band, band2=band)

    dt = 0.2 * dx**2
    u_new = e_mat @ (u + dt * (l_mat @ u))
    print(f"band size: {band.size}")
    print(f"E shape: {e_mat.shape}, nnz: {e_mat.nnz}")
    print(f"L shape: {l_mat.shape}, nnz: {l_mat.nnz}")
    print(f"one-step max |u|: {np.max(np.abs(u_new)):.6f}")


if __name__ == "__main__":
    main()
