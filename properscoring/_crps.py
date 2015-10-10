import numpy as np
import pandas as pd

from numba import guvectorize
from scipy import special, integrate, stats

from ._utils import move_axis_to_end, argsort_indices


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
    Computes the CRPS of an observation x relative to a normal
    distribution with mean, mu, and standard deviation, sig.

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


def _discover_bounds(cdf_or_dist, tol=1e-7):
    """
    Uses scipy's general continuous distribution methods
    which compute the ppf from the cdf, then use the ppf
    to find the lower and upper limits of the distribution.
    """
    class DistFromCDF(stats.distributions.rv_continuous):
        def cdf(self, x):
            return cdf_or_dist(x)
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


def crps_cdf(x, cdf_or_dist, xmin=None, xmax=None, tol=1e-6):
    """
    Compute the continuously ranked probability score (CPRS)
    for a given forecast distribution (cdf) and observation (x).
    This implementation allows the computation of CRPS for arbitrary
    forecast distributions.  If gaussianity can be assumed the
    gaussian_crps function is faster.

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


@guvectorize(["void(float64[:], float64[:], float64[:], float64[:])"],
             "(),(n),(n)->()", nopython=True)
def _crps_gufunc(observation, forecasts, weights, result):
    # beware: forecasts are assumed sorted in NumPy's sort order

    # we index the 0th element to get the scalar value from this 0d array:
    # http://numba.pydata.org/numba-doc/0.18.2/user/vectorize.html#the-guvectorize-decorator
    obs = observation[0]

    if np.isnan(obs):
        result[0] = np.nan
        return

    total_weight = 0.0
    for n, weight in enumerate(weights):
        if np.isnan(forecasts[n]):
            # NumPy sorts NaN to the end
            break
        if not weight >= 0:
            # this catches NaN weights
            result[0] = np.nan
            return
        total_weight += weight

    obs_cdf = 0
    forecast_cdf = 0
    prev_forecast = 0
    integral = 0

    for n, forecast in enumerate(forecasts):
        if np.isnan(forecast):
            # NumPy sorts NaN to the end
            if n == 0:
                integral = np.nan
            # reset for the sake of the conditional below
            forecast = prev_forecast
            break

        if obs_cdf == 0 and obs < forecast:
            integral += (obs - prev_forecast) * forecast_cdf ** 2
            integral += (forecast - obs) * (forecast_cdf - 1) ** 2
            obs_cdf = 1
        else:
            integral += ((forecast - prev_forecast)
                         * (forecast_cdf - obs_cdf) ** 2)

        forecast_cdf += weights[n] / total_weight
        prev_forecast = forecast

    if obs_cdf == 0:
        # forecast can be undefined here if the loop body is never executed
        # (because forecasts have size 0), but don't worry about that because
        # we want to raise an error in that case, anyways
        integral += obs - forecast

    result[0] = integral


def crps_ensemble(observations, forecasts, weights=None, issorted=False,
                  axis=-1):
    """
    Calculate the continuous ranked probability score (CRPS)

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

    This function approximates CRPS efficiently using the standard definition
    of the empirical CDF:
    http://en.wikipedia.org/wiki/Empirical_distribution_function

    The runtime of this function is O(N * E * log(E)) where N is the number of
    observations and E is the size of the forecast ensemble.

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
        https://drive.google.com/a/climate.com/file/d/0B8AfRcot4nsIYmc3alpTeTZpLWc
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

    return _crps_gufunc(observations, forecasts, weights)
