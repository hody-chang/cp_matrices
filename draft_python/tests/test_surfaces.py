import numpy as np

from cpmatrices.surfaces import circle_closest_point, sphere_closest_point


def test_circle_closest_point_signed_distance():
    cpx, cpy, dist = circle_closest_point(np.array([2.0, 0.0]), np.array([0.0, 0.0]))

    np.testing.assert_allclose(cpx, np.array([1.0, 1.0]))
    np.testing.assert_allclose(cpy, np.array([0.0, 0.0]))
    np.testing.assert_allclose(dist, np.array([1.0, -1.0]))


def test_sphere_closest_point_signed_distance():
    cpx, cpy, cpz, dist = sphere_closest_point(
        np.array([0.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([0.0, 0.0]),
    )

    np.testing.assert_allclose(cpx, np.array([1.0, 0.0]))
    np.testing.assert_allclose(cpy, np.array([0.0, 1.0]))
    np.testing.assert_allclose(cpz, np.array([0.0, 0.0]))
    np.testing.assert_allclose(dist, np.array([-1.0, 1.0]))
