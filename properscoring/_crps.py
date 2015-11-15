import numpy as np

from scipy import special, integrate, stats

from ._utils import move_axis_to_end, argsort_indices, suppress_warnings


# The normalization constant for the univariate standard Gaussian pdf
_normconst = 1.0 / np.sqrt(2.0 * np.pi)


def _normpdf(x):
    """Probability density function of a univariate standard Gaussian
    distribution with zero mean and unit variance.
    """
    return _normconst * np.exp(-(x * x) / 2.0)


# Cumulative distribution function of a univariate standard Gaussian
# distribution with zero mean and unit variance.
_normcdf = special.ndtr


def crps_gaussian(x, mu, sig, grad=False):
    """
    Computes the CRPS of observations x relative to normally distributed
    forecasts with mean, mu, and standard deviation, sig.

    CRPS(N(mu, sig^2); x)

    Formula taken from Equation (5):

    Calibrated Probablistic Forecasting Using Ensemble Model Output
    Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
    Westveld, Goldman. Monthly Weather Review 2004

    http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1

    Parameters
    ----------
    x : scalar or np.ndarray
        The observation or set of observations.
    mu : scalar or np.ndarray
        The mean of the forecast normal distribution
    sig : scalar or np.ndarray
        The standard deviation of the forecast distribution
    grad : boolean
        If True the gradient of the CRPS w.r.t. mu and sig
        is returned along with the CRPS.

    Returns
    -------
    crps : scalar or np.ndarray or tuple of
        The CRPS of each observation x relative to mu and sig.
        The shape of the output array is determined by numpy
        broadcasting rules.
    crps_grad : np.ndarray (optional)
        If grad=True the gradient of the crps is returned as
        a numpy array [grad_wrt_mu, grad_wrt_sig].  The
        same broadcasting rules apply.
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    sig = np.asarray(sig)
    # standadized x
    sx = (x - mu) / sig
    # some precomputations to speed up the gradient
    pdf = _normpdf(sx)
    cdf = _normcdf(sx)
    pi_inv = 1. / np.sqrt(np.pi)
    # the actual crps
    crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
    if grad:
        dmu = 1 - 2 * cdf
        dsig = 2 * pdf - pi_inv
        return crps, np.array([dmu, dsig])
    else:
        return crps


def _discover_bounds(cdf, tol=1e-7):
    """
    Uses scipy's general continuous distribution methods
    which compute the ppf from the cdf, then use the ppf
    to find the lower and upper limits of the distribution.
    """
    class DistFromCDF(stats.distributions.rv_continuous):
        def cdf(self, x):
            return cdf(x)
    dist = DistFromCDF()
    # the ppf is the inverse cdf
    lower = dist.ppf(tol)
    upper = dist.ppf(1. - tol)
    return lower, upper


def _crps_cdf_single(x, cdf_or_dist, xmin=None, xmax=None, tol=1e-6):
    """
    See crps_cdf for docs.
    """
    # TODO: this function is pretty slow.  Look for clever ways to speed it up.

    # allow for directly passing in scipy.stats distribution objects.
    cdf = getattr(cdf_or_dist, 'cdf', cdf_or_dist)
    assert callable(cdf)

    # if bounds aren't given, discover them
    if xmin is None or xmax is None:
        # Note that infinite values for xmin and xmax are valid, but
        # it slows down the resulting quadrature significantly.
        xmin, xmax = _discover_bounds(cdf)

    # make sure the bounds haven't clipped the cdf.
    if (tol is not None) and (cdf(xmin) >= tol) or (cdf(xmax) <= (1. - tol)):
        raise ValueError('CDF does not meet tolerance requirements at %s '
                         'extreme(s)! Consider using function defaults '
                         'or using infinities at the bounds. '
                         % ('lower' if cdf(xmin) >= tol else 'upper'))

    # CRPS = int_-inf^inf (F(y) - H(x))**2 dy
    #      = int_-inf^x F(y)**2 dy + int_x^inf (1 - F(y))**2 dy
    def lhs(y):
        # left hand side of CRPS integral
        return np.square(cdf(y))
    # use quadrature to integrate the lhs
    lhs_int, lhs_tol = integrate.quad(lhs, xmin, x)
    # make sure the resulting CRPS will be with tolerance
    if (tol is not None) and (lhs_tol >= 0.5 * tol):
        raise ValueError('Lower integral did not evaluate to within tolerance! '
                         'Tolerance achieved: %f , Value of integral: %f \n'
                         'Consider setting the lower bound to -np.inf.' %
                         (lhs_tol, lhs_int))

    def rhs(y):
        # right hand side of CRPS integral
        return np.square(1. - cdf(y))
    rhs_int, rhs_tol = integrate.quad(rhs, x, xmax)
    # make sure the resulting CRPS will be with tolerance
    if (tol is not None) and (rhs_tol >= 0.5 * tol):
        raise ValueError('Upper integral did not evaluate to within tolerance! \n'
                         'Tolerance achieved: %f , Value of integral: %f \n'
                         'Consider setting the upper bound to np.inf or if '
                         'you already have, set warn_level to `ignore`.' %
                         (rhs_tol, rhs_int))

    return lhs_int + rhs_int

_crps_cdf = np.vectorize(_crps_cdf_single)


def crps_quadrature(x, cdf_or_dist, xmin=None, xmax=None, tol=1e-6):
    """
    Compute the continuously ranked probability score (CPRS) for a given
    forecast distribution (cdf) and observation (x) using numerical quadrature.

    This implementation allows the computation of CRPS for arbitrary forecast
    distributions. If gaussianity can be assumed ``crps_gaussian`` is faster.

    Parameters
    ----------
    x : np.ndarray
        Observations associated with the forecast distribution cdf_or_dist
    cdf_or_dist : callable or scipy.stats.distribution
        Function which returns the the cumulative density of the
        forecast distribution at value x.  This can also be an object with
        a callable cdf() method such as a scipy.stats.distribution object.
    xmin : np.ndarray or scalar
        The lower bounds for integration, this is required to perform
        quadrature.
    xmax : np.ndarray or scalar
        The upper bounds for integration, this is required to perform
        quadrature.
    tol : float , optional
        The desired accuracy of the CRPS, larger values will speed
        up integration. If tol is set to None, bounds errors or integration
        tolerance errors will be ignored.

    Returns
    -------
    crps : np.ndarray
        The continuously ranked probability score of an observation x
        given forecast distribution.
    """
    return _crps_cdf(x, cdf_or_dist, xmin, xmax, tol)


def _crps_ensemble_vectorized(observations, forecasts, weights=1):
    """
    An alternative but simpler implementation of CRPS for testing purposes

    This implementation is based on the identity:

    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    Hence it has runtime O(n^2) instead of O(n log(n)) where n is the number of
    ensemble members.

    Reference
    ---------
    Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
        prediction, and estimation, 2005. University of Washington Department of
        Statistics Technical Report no. 463R.
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    """
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)
    weights = np.asarray(weights)
    if weights.ndim > 0:
        weights = np.where(~np.isnan(forecasts), weights, np.nan)
        weights = weights / np.nanmean(weights, axis=-1, keepdims=True)

    if observations.ndim == forecasts.ndim - 1:
        # sum over the last axis
        assert observations.shape == forecasts.shape[:-1]
        observations = observations[..., np.newaxis]
        with suppress_warnings('Mean of empty slice'):
            score = np.nanmean(weights * abs(forecasts - observations), -1)
        # insert new axes along last and second to last forecast dimensions so
        # forecasts_diff expands with the array broadcasting
        forecasts_diff = (np.expand_dims(forecasts, -1) -
                          np.expand_dims(forecasts, -2))
        weights_matrix = (np.expand_dims(weights, -1) *
                          np.expand_dims(weights, -2))
        with suppress_warnings('Mean of empty slice'):
            score += -0.5 * np.nanmean(weights_matrix * abs(forecasts_diff),
                                       axis=(-2, -1))
        return score
    elif observations.ndim == forecasts.ndim:
        # there is no 'realization' axis to sum over (this is a deterministic
        # forecast)
        return abs(observations - forecasts)


try:
    from ._gufuncs import _crps_ensemble_gufunc as _crps_ensemble_core
except ImportError:
    _crps_ensemble_core = _crps_ensemble_vectorized


def crps_ensemble(observations, forecasts, weights=None, issorted=False,
                  axis=-1):
    """
    Calculate the continuous ranked probability score (CRPS) for a set of
    explicit forecast realizations.

    The CRPS compares the empirical distribution of an ensemble forecast
    to a scalar observation. Smaller scores indicate better skill.

    CRPS is defined for one-dimensional random variables with a probability
    density $p(x)$,

    .. math::
        CRPS(F, x) = \int_z (F(z) - H(z - x))^2 dz

    where $F(x) = \int_{z \leq x} p(z) dz$ is the cumulative distribution
    function (CDF) of the forecast distribution $F$ and $H(x)$ denotes the
    Heaviside step function, where $x$ is a point estimate of the true
    observation (observational error is neglected).

    This function calculates CRPS efficiently using the empirical CDF:
    http://en.wikipedia.org/wiki/Empirical_distribution_function

    The Numba accelerated version of this function requires time
    O(N * E * log(E)) and space O(N * E) where N is the number of observations
    and E is the size of the forecast ensemble.

    The non-Numba accelerated version much slower for large ensembles: it
    requires both time and space O(N * E ** 2).

    Parameters
    ----------
    observations : float or array_like
        Observations float or array. Missing values (NaN) are given scores of
        NaN.
    forecasts : float or array_like
        Array of forecasts ensemble members, of the same shape as observations
        except for the axis along which CRPS is calculated (which should be the
        axis corresponding to the ensemble). If forecasts has the same shape as
        observations, the forecasts are treated as deterministic. Missing
        values (NaN) are ignored.
    weights : array_like, optional
        If provided, the CRPS is calculated exactly with the assigned
        probability weights to each forecast. Weights should be positive, but
        do not need to be normalized. By default, each forecast is weighted
        equally.
    issorted : bool, optional
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
    axis : int, optional
        Axis in forecasts and weights which corresponds to different ensemble
        members, along which to calculate CRPS.

    Returns
    -------
    out : np.ndarray
        CRPS for each ensemble forecast against the observations.

    References
    ----------
    Jochen Broecker. Chapter 7 in Forecast Verification: A Practitioner's Guide
        in Atmospheric Science. John Wiley & Sons, Ltd, Chichester, UK, 2nd
        edition, 2012.
    Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
        prediction, and estimation, 2005. University of Washington Department of
        Statistics Technical Report no. 463R.
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    Wilks D.S. (1995) Chapter 8 in _Statistical Methods in the
        Atmospheric Sciences_. Academic Press.
    """
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)
    if axis != -1:
        forecasts = move_axis_to_end(forecasts, axis)

    if weights is not None:
        weights = move_axis_to_end(weights, axis)
        if weights.shape != forecasts.shape:
            raise ValueError('forecasts and weights must have the same shape')

    if observations.shape not in [forecasts.shape, forecasts.shape[:-1]]:
        raise ValueError('observations and forecasts must have matching '
                         'shapes or matching shapes except along `axis=%s`'
                         % axis)

    if observations.shape == forecasts.shape:
        if weights is not None:
            raise ValueError('cannot supply weights unless you also supply '
                             'an ensemble forecast')
        return abs(observations - forecasts)

    if not issorted:
        if weights is None:
            forecasts = np.sort(forecasts, axis=-1)
        else:
            idx = argsort_indices(forecasts, axis=-1)
            forecasts = forecasts[idx]
            weights = weights[idx]

    if weights is None:
        weights = np.ones_like(forecasts)

    return _crps_ensemble_core(observations, forecasts, weights)
