import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose

from properscoring import brier_score, threshold_decomposition, crps_ensemble

from properscoring.tests.utils import suppress_warnings


def threshold_decomposition_alt(observations, forecasts, thresholds):
    observations = np.asarray(observations)
    thresholds = np.asarray(thresholds)
    forecasts = np.asarray(forecasts)

    def exceedances(x):
        # NaN safe calculation of threshold exceedances
        # add an extra dimension to `x` and broadcast `thresholds` so that it
        # varies along that new dimension
        with suppress_warnings('invalid value encountered in greater'):
            exceeds = (x[..., np.newaxis] >
                       thresholds.reshape((1,) * x.ndim + (-1,))
                       ).astype(float)
        if x.ndim == 0 and np.isnan(x):
            exceeds[:] = np.nan
        else:
            exceeds[np.where(np.isnan(x))] = np.nan
        return exceeds

    binary_obs = exceedances(observations)
    if observations.shape == forecasts.shape:
        prob_forecast = exceedances(forecasts)
    elif observations.shape == forecasts.shape[:-1]:
        # axis=-2 should be the 'realization' axis, after swapping that axes
        # to the end of forecasts and inserting one extra axis
        with suppress_warnings('Mean of empty slice'):
            prob_forecast = np.nanmean(exceedances(forecasts), axis=-2)
    else:
        raise AssertionError
    return brier_score(binary_obs, prob_forecast)


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

    def test_crps_consistency(self):
        # verify that we can integrate the brier scores to calculate CRPS
        obs = np.random.RandomState(123).rand(100)
        forecasts = np.random.RandomState(456).rand(100, 100)
        thresholds = np.linspace(0, 1, num=10000)

        td = threshold_decomposition(obs, forecasts, thresholds)
        actual = td.sum(1) * (thresholds[1] - thresholds[0])
        desired = crps_ensemble(obs, forecasts)
        assert_allclose(actual, desired, atol=1e-4)

    def test_alt_implemenation_consistency(self):
        obs = np.random.RandomState(123).randn(100)
        forecasts = np.random.RandomState(456).randn(100, 100)
        thresholds = np.linspace(-2, 2, num=10)

        actual = threshold_decomposition(obs, forecasts, thresholds)
        desired = threshold_decomposition(obs, forecasts, thresholds)
        assert_allclose(actual, desired, atol=1e-10)

        obs[np.random.RandomState(231).rand(100) < 0.2] = np.nan
        forecasts[np.random.RandomState(231).rand(100, 100) < 0.2] = np.nan
        forecasts[:, ::8] = np.nan
        forecasts[::8, :] = np.nan

        actual = threshold_decomposition(obs, forecasts, thresholds)
        desired = threshold_decomposition(obs, forecasts, thresholds)
        assert_allclose(actual, desired, atol=1e-10)

    def test_errors(self):
        with self.assertRaisesRegexp(ValueError, 'must be 1D and sorted'):
            threshold_decomposition(1, [0, 1, 2], [[1]])
        with self.assertRaisesRegexp(ValueError, 'must be 1D and sorted'):
            threshold_decomposition(1, [0, 1, 2], [1, 0.5])
        with self.assertRaisesRegexp(ValueError, 'must have matching shapes'):
            threshold_decomposition([1, 2], [0, 1, 2], [0.5])
