import numpy as np

from cpmatrices.bands import make_invbandmap, rm_bandwidth


def test_make_invbandmap_uses_minus_one_outside_band():
    inv = make_invbandmap(7, np.array([1, 4, 6]))

    np.testing.assert_array_equal(inv, np.array([-1, 0, -1, -1, 1, -1, 2]))


def test_rm_bandwidth_matches_reference_formula():
    expected = 1.0001 * np.sqrt((2 - 1) * ((3 + 1) / 2) ** 2 + (1 + (3 + 1) / 2) ** 2)

    assert rm_bandwidth(2, 3, 1) == expected
