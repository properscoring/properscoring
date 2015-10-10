import unittest

import numpy as np
from numpy.testing import assert_allclose

from properscoring._utils import argsort_indices, move_axis_to_end


class TestArgsortIndices(unittest.TestCase):
    def test_argsort_indices(self):
        x = np.random.randn(5, 6, 7)
        for axis in [0, 1, 2, -1]:
            expected = np.sort(x, axis=axis)
            idx = argsort_indices(x, axis=axis)
            assert_allclose(expected, x[idx])


class TestMoveAxis(unittest.TestCase):
    def test_move_axis_to_end(self):
        x = np.random.randn(5, 6, 7)
        for axis, expected in [(0, (6, 7, 5)),
                               (1, (5, 7, 6)),
                               (2, (5, 6, 7)),
                               (-1, (5, 6, 7))]:
            actual = move_axis_to_end(x, axis=axis).shape
            self.assertEqual(actual, expected)
