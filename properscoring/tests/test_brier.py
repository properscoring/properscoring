import functools
import unittest

import numpy as np
from scipy import stats, special
from numpy.testing import assert_allclose

from properscoring import brier_score, threshold_decomposition


class TestBrierScore(unittest.TestCase):
    def test_validation(self):
        failures = [(-1, 0),
                    (0, 2)]
        for failure in failures:
            with self.assertRaises(ValueError):
                brier_score(*failure)
        # this should not raise an exception:
        brier_score(np.nan, np.nan)


class TestThresholdDecomposition(unittest.TestCase):
    def test_examples(self):
        examples = [
            (0, 0, [0], [0]),
            (0, 0, [-1, 0, 1], [0, 0, 0]),
            (0, [-1, 1], [-2, 0, 2], [0, 0.25, 0]),
            (0, 0, [0], [0]),
            ([0, np.nan], [0, 0], [0], [[0], [np.nan]]),
            (np.nan, [-1, 1], [0, 1], [np.nan, np.nan]),
            (0, [-1, 1, np.nan], [-2, 0, 2], [0, 0.25, 0]),
            (0, [0, 0, 0, 1], [0], [0.0625]),
        ]
        for observations, forecasts, thresholds, expected in examples:
            assert_allclose(
                threshold_decomposition(observations, forecasts, thresholds),
                expected)
