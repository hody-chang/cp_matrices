import numpy as np

from cpmatrices.laplacian import (
    laplacian_2d_matrix,
    laplacian_3d_matrix,
    second_derivative_matrices_nd,
)


def test_laplacian_2d_center_row_meshgrid_order():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    band = np.arange(9)

    l_mat = laplacian_2d_matrix(x, y, order=2, band1=band, band2=band)
    row = l_mat.toarray()[4]
    expected = np.zeros(9)
    expected[[1, 3, 4, 5, 7]] = [1.0, 1.0, -4.0, 1.0, 1.0]

    np.testing.assert_allclose(row, expected)


def test_laplacian_2d_equals_sum_of_second_derivatives():
    x = np.arange(-1.0, 1.1, 0.5)
    y = np.arange(-1.0, 1.1, 0.5)
    band = np.arange(x.size * y.size)

    l_mat = laplacian_2d_matrix(x, y, order=2, band1=band, band2=band)
    dxx, dyy = second_derivative_matrices_nd([x, y], order=2, band1=band, band2=band, ordering="meshgrid")

    np.testing.assert_allclose(l_mat.toarray(), (dxx + dyy).toarray())


def test_laplacian_3d_center_row_meshgrid_order():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0, 2.0])
    band = np.arange(27)

    l_mat = laplacian_3d_matrix(x, y, z, order=2, band1=band, band2=band)
    row = l_mat.toarray()[13]
    expected = np.zeros(27)
    expected[[4, 10, 12, 13, 14, 16, 22]] = [1.0, 1.0, 1.0, -6.0, 1.0, 1.0, 1.0]

    np.testing.assert_allclose(row, expected)
