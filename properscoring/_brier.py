import numpy as np
from numba import jit, guvectorize

from ._utils import move_axis_to_end


@jit(nopython=True)
def _check_valid_binary_obs(values):
    for val in values:
        # val != val checks for NaN
        if not (val == 0 or val == 1 or val != val):
            raise ValueError('observations can only contain 0, 1, or NaN')


def brier_score(observations, forecasts):
    """
    Calculate the Brier score (BS)

    The Brier score (BS) scores binary forecasts $k \in \{0, 1\}$,

    ..math:
        BS(p, k) = (p_1 - k)^2,

    where $p_1$ is the forecast probability of $k=1$.

    Parameters
    ----------
    observations, forecasts : array_like
        Broadcast compatible arrays of forecasts (probabilities) and
        observations (0, 1 or NaN).

    Returns
    -------
    out : np.ndarray
        Brier score for each forecast/observation.

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
    """
    machine_eps = np.finfo(float).eps
    forecasts = np.asarray(forecasts)
    if (forecasts < 0.0).any() or (forecasts > (1.0 + machine_eps)).any():
        raise ValueError('forecasts must not be outside of the unit interval '
                         '[0, 1]')
    observations = np.asarray(observations)
    _check_valid_binary_obs(observations.ravel(order='K'))
    return (forecasts - observations) ** 2


@guvectorize(["void(float64[:], float64[:], float64[:], float64[:])"],
             "(),(n),(m)->(m)", nopython=True)
def _threshold_decomp_gufunc(observation, forecasts, thresholds, result):
    # both forecasts and thresholds are assumed sorted in NumPy's sort order
    obs = observation[0]

    n_thresholds = len(thresholds)
    n_forecasts = len(forecasts)
    while np.isnan(forecasts[n_forecasts - 1]) and n_forecasts > 0:
        n_forecasts -= 1

    if np.isnan(obs) or n_forecasts == 0:
        result[:] = np.nan
        return

    inv_n_forecasts = 1.0 / n_forecasts

    i = 0
    j = 0
    while i < n_forecasts and j < n_thresholds:
        forecast = forecasts[i]
        threshold = thresholds[j]

        if forecast <= threshold:
            i += 1
        else:
            probability = i * inv_n_forecasts
            binary_obs = obs <= threshold
            result[j] = (probability - binary_obs) ** 2
            j += 1

    for k in range(j, n_thresholds):
        threshold = thresholds[k]
        binary_obs = obs <= threshold
        # probability is always 1, so we can skip the square
        result[k] = 1 - binary_obs


def threshold_decomposition(observations, forecasts, thresholds,
                            issorted=False, axis=-1):
    """
    Threshold decomposition of the continuous ranked probability score (CRPS)

    This function calculates the Brier score for exceedance at each given
    threshold (the values z in the equation below). The resulting Brier scores
    can thus be summed along the last axis to calculate CRPS, as

    .. math::
        CRPS(F, x) = \int_z BS(F(z), H(z - x)) dz

    where $F(x) = \int_{z \leq x} p(z) dz$ is the cumulative distribution
    function (CDF) of the forecast distribution $F$, $x$ is a point estimate of
    the true observation (observational error is neglected), $BS$ denotes the
    Brier score and $H(x)$ denotes the Heaviside step function, which we define
    here as equal to 1 for x >= 0 and 0 otherwise.

    It is more efficient to calculate CRPS directly, but this threshold
    decomposition itself provides a useful summary of model quality as a
    function of measurement values.

    Parameters
    ----------
    observations : float or np.ndarray
        Observations float or array. Missing values (NaN) are given scores of
        NaN.
    forecasts : float or np.ndarray
        Array of forecasts ensemble members, of the same shape as observations
        except for the axis along which the threshold decomposition is
        calculated (which should be the axis corresponding to the ensemble). If
        forecasts has the same shape as observations, the forecasts are treated
        as deterministic. Missing values (NaN) are ignored.
    thresholds : array_like
        Threshold values at which to calculate the exceedence Brier score which
        contributes to CRPS.
    issorted : bool, optional
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
    axis : int, optional
        Axis in forecasts which corresponds to different ensemble members,
        along which to calculate the threshold decomposition.

    Returns
    -------
    out : np.ndarray
        Threshold decomposition for each ensemble forecast against the
        observations. The threshold decomposition will have the same shape as
        observations, except for an additional final dimension, which
        corresponds to the different threshold values.

    References
    ----------
    Gneiting, T. and Ranjan, R. Comparing density forecasts using threshold-
       and quantile-weighted scoring rules. J. Bus. Econ. Stat. 29, 411-422
       (2011). http://www.stat.washington.edu/research/reports/2008/tr533.pdf

    See also
    --------
    crps, brier_score
    """
    observations = np.asarray(observations)
    thresholds = np.asarray(thresholds)
    forecasts = np.asarray(forecasts)
    if axis != -1:
        forecasts = move_axis_to_end(forecasts, axis)

    if forecasts.shape == observations.shape:
        forecasts = forecasts[..., np.newaxis]

    if observations.shape != forecasts.shape[:-1]:
        raise ValueError('observations and forecasts must have matching '
                         'shapes or matching shapes except along `axis=%s`'
                         % axis)

    if thresholds.ndim > 1 or not (np.sort(thresholds) == thresholds).all():
        raise ValueError('thresholds must be 1D and sorted')
    thresholds = thresholds.reshape((1,) * observations.ndim + (-1,))

    if not issorted:
        forecasts = np.sort(forecasts, axis=-1)

    return _threshold_decomp_gufunc(observations, forecasts, thresholds)
