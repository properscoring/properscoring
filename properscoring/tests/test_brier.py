import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose

from properscoring import brier_score, threshold_brier_score, crps_ensemble
from properscoring._brier import (_threshold_brier_score_vectorized,
                                  _threshold_brier_score_core)


class TestBrierScore(unittest.TestCase):
    def test_validation(self):
        failures = [(-1, 0),
                    (0, 2)]
        for failure in failures:
            with self.assertRaises(ValueError):
                brier_score(*failure)
        # this should not raise an exception:
        brier_score(np.nan, np.nan)


class TestThresholdBrierScore(unittest.TestCase):
    def test_examples(self):
        examples = [
            (0, 0, 0, 0),
            (0, 0, [0], [0]),
            (0, 0, [-1, 0, 1], [0, 0, 0]),
            (0, [-1, 1], [-2, 0, 2], [0, 0.25, 0]),
            ([0, np.nan], [0, 0], [0], [[0], [np.nan]]),
            (np.nan, [-1, 1], [0, 1], [np.nan, np.nan]),
            (0, [-1, 1, np.nan], [-2, 0, 2], [0, 0.25, 0]),
            (0, [0, 0, 0, 1], [0], [0.0625]),
        ]
        for observations, forecasts, thresholds, expected in examples:
            assert_allclose(
                threshold_brier_score(observations, forecasts, thresholds),
                expected)

    def test_crps_consistency(self):
        # verify that we can integrate the brier scores to calculate CRPS
        obs = np.random.RandomState(123).rand(100)
        forecasts = np.random.RandomState(456).rand(100, 100)
        thresholds = np.linspace(0, 1, num=10000)

        td = threshold_brier_score(obs, forecasts, thresholds)
        actual = td.sum(1) * (thresholds[1] - thresholds[0])
        desired = crps_ensemble(obs, forecasts)
        assert_allclose(actual, desired, atol=1e-4)

    def test_alt_implemenation_consistency(self):
        obs = np.random.RandomState(123).randn(100)
        forecasts = np.random.RandomState(456).randn(100, 100)
        thresholds = np.linspace(-2, 2, num=10)

        actual = threshold_brier_score(obs, forecasts, thresholds)
        desired = _threshold_brier_score_vectorized(obs, forecasts, thresholds)
        assert_allclose(actual, desired, atol=1e-10)

        obs[np.random.RandomState(231).rand(100) < 0.2] = np.nan
        forecasts[np.random.RandomState(231).rand(100, 100) < 0.2] = np.nan
        forecasts[:, ::8] = np.nan
        forecasts[::8, :] = np.nan

        actual = threshold_brier_score(obs, forecasts, thresholds)
        desired = _threshold_brier_score_vectorized(obs, forecasts, thresholds)
        assert_allclose(actual, desired, atol=1e-10)

    def test_errors(self):
        with self.assertRaisesRegexp(ValueError, 'must be scalar or 1-dim'):
            threshold_brier_score(1, [0, 1, 2], [[1]])
        with self.assertRaisesRegexp(ValueError, 'must be sorted'):
            threshold_brier_score(1, [0, 1, 2], [1, 0.5])
        with self.assertRaisesRegexp(ValueError, 'must have matching shapes'):
            threshold_brier_score([1, 2], [0, 1, 2], [0.5])

    def test_numba_is_used(self):
        try:
            import numba
            has_numba = True
        except ImportError:
            has_numba = False

        using_vectorized = (_threshold_brier_score_core is
                            _threshold_brier_score_vectorized)
        self.assertEqual(using_vectorized, not has_numba)
