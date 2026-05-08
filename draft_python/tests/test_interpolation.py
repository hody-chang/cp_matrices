import numpy as np

from cpmatrices.interpolation import (
    find_grid_interp_base_points,
    interp2_matrix,
    lagrange_weights_1d,
)


def test_lagrange_weights_are_one_hot_at_grid_nodes():
    weights = lagrange_weights_1d(
        base_points=np.zeros(4),
        points=0.25 * np.arange(4),
        dx=0.25,
        stencil_size=5,
    )

    np.testing.assert_allclose(weights[:, :4], np.eye(4))
    np.testing.assert_allclose(weights[:, 4], np.zeros(4))


def test_find_grid_interp_base_points_matches_asymmetric_stencil():
    base, coords = find_grid_interp_base_points([np.array([0.35])], degree=3, lower_left=[0.0], dx=[0.1])

    np.testing.assert_array_equal(base[0], np.array([2]))
    np.testing.assert_allclose(coords[0], np.array([0.2]))


def test_interp2_degree_one_matches_bilinear_weights_meshgrid_order():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])

    e_mat = interp2_matrix(x, y, np.array([0.25]), np.array([0.5]), degree=1)

    np.testing.assert_allclose(e_mat.toarray(), np.array([[0.375, 0.375, 0.125, 0.125]]))


def test_interp2_banded_restricts_columns():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    band = np.array([0, 1, 2, 3])

    full = interp2_matrix(x, y, np.array([0.25]), np.array([0.5]), degree=1)
    banded = interp2_matrix(x, y, np.array([0.25]), np.array([0.5]), degree=1, band=band)

    np.testing.assert_allclose(banded.toarray(), full.toarray())


def test_interp2_multiple_rows_do_not_mix_stencils():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])

    e_mat = interp2_matrix(
        x,
        y,
        np.array([0.25, 0.75]),
        np.array([0.5, 0.5]),
        degree=1,
    )

    expected = np.array(
        [
            [0.375, 0.375, 0.125, 0.125],
            [0.125, 0.125, 0.375, 0.375],
        ]
    )
    np.testing.assert_allclose(e_mat.toarray(), expected)
