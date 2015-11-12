import functools
import unittest
import warnings

import numpy as np
from scipy import stats, special
from numpy.testing import assert_allclose

from properscoring import crps_ensemble, crps_quadrature, crps_gaussian
from properscoring._crps import (_crps_ensemble_vectorized,
                                 _crps_ensemble_core)


class TestDistributionBasedCRPS(unittest.TestCase):

    def setUp(self):
        np.random.seed(1983)
        shape = (2, 3)
        self.mu = np.random.normal(size=shape)
        self.sig = np.square(np.random.normal(size=shape))
        self.obs = np.random.normal(loc=self.mu, scale=self.sig, size=shape)

        n = 1000
        q = np.linspace(0. + 0.5 / n, 1. - 0.5 / n, n)
        # convert to the corresponding normal deviates
        normppf = special.ndtri
        z = normppf(q)

        forecasts = z.reshape(-1, 1, 1) * self.sig + self.mu
        self.expected = crps_ensemble(self.obs, forecasts, axis=0)

    def test_crps_quadrature_consistent(self):

        def normcdf(*args, **kwdargs):
            return stats.norm(*args, **kwdargs).cdf
        dists = np.vectorize(normcdf)(loc=self.mu, scale=self.sig)
        crps = crps_quadrature(self.obs, dists,
                        xmin=self.mu - 5 * self.sig,
                        xmax=self.mu + 5 * self.sig)
        np.testing.assert_allclose(crps, self.expected, rtol=1e-4)

    def test_pdf_derived_weights(self):
        # One way of evaluating the CRPS given a pdf is to simply evaluate
        # the pdf at a set of points (fcsts) and set weights=pdf(fcsts).
        # This tests that that method works.
        def normpdf(*args, **kwdargs):
            return stats.norm(*args, **kwdargs).pdf
        pdfs = np.vectorize(normpdf)(loc=self.mu, scale=self.sig)

        fcsts = np.linspace(-4., 4., 500)
        fcsts = (self.mu[..., np.newaxis] + self.sig[..., np.newaxis]
                 * fcsts[np.newaxis, np.newaxis, :])

        weights = np.empty_like(fcsts)
        for i, j in np.ndindex(pdfs.shape):
            weights[i, j] = pdfs[i, j](fcsts[i, j])

        actual = crps_ensemble(self.obs, fcsts, weights)
        np.testing.assert_allclose(actual, self.expected, rtol=1e-4)

    def test_crps_quadrature_fails(self):
        def normcdf(*args, **kwdargs):
            return stats.norm(*args, **kwdargs).cdf
        cdfs = np.vectorize(normcdf)(loc=self.mu, scale=self.sig)
        valid_call = functools.partial(crps_quadrature,
                                       self.obs, cdfs,
                                       xmin=self.mu - 5 * self.sig,
                                       xmax=self.mu + 5 * self.sig)
        # this should fail because we have redefined the xmin/xmax
        # bounds to unreasonable values.  In order for the crps_quadrature
        # function to work it needs xmin/xmax values that bound the
        # range of the corresponding distribution.
        self.assertRaises(ValueError, lambda: valid_call(xmin=self.mu))
        self.assertRaises(ValueError, lambda: valid_call(xmax=self.mu))

    def test_crps_gaussian_consistent(self):
        actual = crps_gaussian(self.obs, self.mu, self.sig)
        np.testing.assert_allclose(actual, self.expected, rtol=1e-4)

    def test_crps_gaussian_broadcast(self):
        expected = crps_gaussian(np.array([0, 1, 2]), mu=0, sig=1)
        actual = crps_gaussian([0, 1, 2], mu=[0], sig=1)
        np.testing.assert_allclose(actual, expected)

    def test_grad(self):
        from scipy import optimize
        f = lambda z: crps_gaussian(self.obs[0, 0], z[0], z[1], grad=False)
        g = lambda z: crps_gaussian(self.obs[0, 0], z[0], z[1], grad=True)[1]
        x0 = np.array([self.mu.reshape(-1),
                       self.sig.reshape(-1)]).T
        for x in x0:
            self.assertLessEqual(optimize.check_grad(f, g, x), 1e-6)


class TestCRPS(unittest.TestCase):
    def setUp(self):
        self.obs = np.random.randn(10)
        self.forecasts = np.random.randn(10, 5)

    def test_validation(self):
        failures = [([0, 1], 0),
                     (0, 0, [0, 1],
                     (0, 1, 1))]
        for failure in failures:
            with self.assertRaises(ValueError):
                crps_ensemble(*failure)

    def test_basic_consistency(self):
        expected = np.array([crps_ensemble(o, f) for o, f
                             in zip(self.obs, self.forecasts)])
        assert_allclose(
            crps_ensemble(self.obs, self.forecasts),
            expected)
        assert_allclose(
            crps_ensemble(self.obs, self.forecasts.T, axis=0),
            expected)
        assert_allclose(crps_ensemble(self.obs, self.obs), np.zeros(10))

    def test_crps_toy_examples(self):
        examples = [
            (0, 0, 0.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, [-1], 1.0),
            (0, [0], 0.0),
            (0, [1], 1.0),
            (0, [0, 0], 0.0),
            (0, [0, 1], 0.25),
            (0, [1, 0], 0.25),
            (0, [1, 1], 1.0),
            (2, [0, 1], 1.25),
            (0, [-1, 1], 0.5),
            (0, [0, 0, 1], 1.0 / 9),
            (1, [0, 0, 1], 4.0 / 9),
            (0, [-1, 0, 0, 1], 1.0 / 8),
        ]
        for x, ensemble, expected in examples:
            self.assertAlmostEqual(
                crps_ensemble(x, ensemble), expected)
            self.assertAlmostEqual(
                _crps_ensemble_vectorized(x, ensemble), expected)

    def test_high_dimensional_consistency(self):
        obs = np.random.randn(10, 20)
        forecasts = np.random.randn(10, 20, 5)
        assert_allclose(crps_ensemble(obs, forecasts),
                        _crps_ensemble_vectorized(obs, forecasts))

    def test_issorted(self):
        vec = np.random.random((10,))
        x = np.random.random()
        vec_sorted = np.sort(vec)
        self.assertEqual(
            crps_ensemble(x, vec),
            crps_ensemble(x, vec_sorted, issorted=True))
        self.assertEqual(
            crps_ensemble(x, vec_sorted, issorted=False),
            crps_ensemble(x, vec_sorted, issorted=True))

    def test_weight_normalization(self):
        x = np.random.random()
        vec = np.random.random((10,))
        expected = crps_ensemble(x, vec)
        for weights in [np.ones_like(vec), 0.1 * np.ones_like(vec)]:
            actual = crps_ensemble(x, vec, weights)
            self.assertAlmostEqual(expected, actual)
        with self.assertRaises(ValueError):
            # mismatched dimensions
            crps_ensemble(x, vec, np.ones(5))

    def test_crps_weight_examples(self):
        examples = [
            # Simplest test.
            (1., [0, 2], [0.5, 0.5], 0.5),
            # Out-of-order analogues.
            (1., [2, 0], [0.5, 0.5], 0.5),
            # Test non-equal weighting.
            (1., [0, 2], [0.8, 0.2], 0.64 + 0.04),
            # Test non-equal weighting + non-equal distances.
            (1.5, [0, 2], [0.8, 0.2], 0.64 * 1.5 + 0.04 * 0.5),
            # Test distances > 1.
            (1., [0, 3], [0.5, 0.5], 0.75),
            # Test distances > 1.
            (1., [-1, 3], [0.5, 0.5], 1),
            # Test weight = 0.
            (1., [0, 2], [1, 0], 1),
            # Test 3 analogues, observation aligned.
            (1., [0, 1, 2], [1./3, 1./3, 1./3], 2./9),
            # Test 3 analogues, observation not aligned.
            (1.5, [0, 1, 2], [1./3, 1./3, 1./3],
             1./9 + 4./9 * 0.5 + 1./9 * 0.5),
            # Test 3 analogues, observation below range.
             (-1., [0, 1, 2], [1./3, 1./3, 1./3],  1 + 1./9 + 4./9),
            # Test 3 analogues, observation above range.
            (2.5, [0, 1, 2], [1./3, 1./3, 1./3], 4./9 + 1./9 + 0.5 * 1),
            # Test 4 analogues, observation aligned.
            (1., [0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25], 3./8),
            # Test 4 analogues, observation not aligned.
            (1.5, [0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25],
             1./16 + 0.5 * 4./16 + 0.5 * 4./16 + 1./16),
        ]
        for x, ensemble, weights, expected in examples:
            self.assertAlmostEqual(
                crps_ensemble(x, ensemble, weights), expected)

    def test_crps_toy_examples_nan(self):
        examples = [
            (np.nan, 0),
            (0, np.nan),
            (0, [np.nan, np.nan]),
            (0, [1], [np.nan]),
            (0, [np.nan], [1]),
            (np.nan, [1], [1]),
        ]
        for args in examples:
            self.assertTrue(
                np.isnan(crps_ensemble(*args)))

    def test_crps_toy_examples_skipna(self):
        self.assertEqual(crps_ensemble(0, [np.nan, 1]), 1)
        self.assertEqual(crps_ensemble(0, [1, np.nan]), 1)
        self.assertEqual(crps_ensemble(1, [np.nan, 0]), 1)
        self.assertEqual(crps_ensemble(1, [0, np.nan]), 1)

    def test_nan_observations_consistency(self):
        rs = np.random.RandomState(123)
        self.obs[rs.rand(*self.obs.shape) > 0.5] = np.nan
        assert_allclose(
            crps_ensemble(self.obs, self.forecasts),
            _crps_ensemble_vectorized(self.obs, self.forecasts))

    def test_nan_forecasts_consistency(self):
        rs = np.random.RandomState(123)
        # make some forecasts entirely missing
        self.forecasts[rs.rand(*self.obs.shape) > 0.5] = np.nan
        assert_allclose(
            crps_ensemble(self.obs, self.forecasts),
            _crps_ensemble_vectorized(self.obs, self.forecasts))
        # forecasts shaped like obs
        forecasts = self.forecasts[:, 0]
        assert_allclose(
            crps_ensemble(self.obs, forecasts),
            _crps_ensemble_vectorized(self.obs, forecasts))

    def test_crps_nans(self):
        vec = np.random.random((10,))
        vec_with_nans = np.r_[vec, [np.nan] * 3]
        weights = np.random.rand(10)
        weights_with_nans = np.r_[weights, np.random.rand(3)]
        x = np.random.random()
        self.assertEqual(
            crps_ensemble(x, vec),
            crps_ensemble(x, vec_with_nans))
        self.assertAlmostEqual(
            crps_ensemble(x, vec, weights),
            crps_ensemble(x, vec_with_nans, weights_with_nans))

        self.assertTrue(np.isnan(crps_ensemble(np.nan, vec)))
        self.assertTrue(np.isnan(crps_ensemble(np.nan, vec_with_nans)))

    def test_crps_beyond_bounds(self):
        vec = np.random.random(size=(100,))
        self.assertAlmostEqual(
            crps_ensemble(-0.1, vec),
            0.1 + crps_ensemble(0, vec))
        self.assertAlmostEqual(
            crps_ensemble(+1.1, vec),
            0.1 + crps_ensemble(1, vec))

    def test_crps_degenerate_ensemble(self):
        x = np.random.random()
        vec = x * np.ones((10,))
        for delta in [-np.pi, 0.0, +np.pi]:
            computed = crps_ensemble(x + delta, vec)
            expected = np.abs(delta * 1.0 ** 2)
            self.assertAlmostEqual(computed, expected)

    def test_numba_is_used(self):
        try:
            import numba
            has_numba = True
        except ImportError:
            has_numba = False

        using_vectorized = _crps_ensemble_core is _crps_ensemble_vectorized
        self.assertEqual(using_vectorized, not has_numba)
