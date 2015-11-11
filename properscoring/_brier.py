import numpy as np

from ._utils import move_axis_to_end, suppress_warnings


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
        Broadcast compatible arrays of forecasts (probabilities between 0 and
        1) and observations (0, 1 or NaN).

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
    if observations.ndim > 0:
        valid_obs = observations[~np.isnan(observations)]
    else:
        valid_obs = observations if not np.isnan(observations) else []
    if not set(np.unique(valid_obs)) <= {0, 1}:
        raise ValueError('observations can only contain 0, 1, or NaN')
    return (forecasts - observations) ** 2


def _threshold_brier_score_vectorized(observations, forecasts, thresholds):
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


try:
    from ._gufuncs import _threshold_brier_score_gufunc as \
        _threshold_brier_score_core
except ImportError:
    _threshold_brier_score_core = _threshold_brier_score_vectorized


def threshold_brier_score(observations, forecasts, threshold, issorted=False,
                          axis=-1):
    """
    Calculate the Brier scores of an ensemble for exceeding given thresholds.

    According to the threshold decomposition of CRPS, the resulting Brier
    scores can thus be summed along the last axis to calculate CRPS, as

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

    The Numba accelerated version of this function is much faster for
    calculating many thresholds simultaneously: it runs in time
    O(N * (E * log(E) + T)), where N is the number of observations, E is the
    ensemble size and T is the number of thresholds.

    The non-Numba accelerated version requires time and space O(N * E * T).

    Parameters
    ----------
    observations : float or array_like
        Observations float or array. Missing values (NaN) are given scores of
        NaN.
    forecasts : float or array_like
        Array of forecasts ensemble members, of the same shape as observations
        except for the extra axis corresponding to the ensemble. If forecasts
        has the same shape as observations, the forecasts are treated as
        deterministic. Missing values (NaN) are ignored.
    threshold : scalar or 1d array_like
        Threshold value(s) at which to calculate exceedence Brier scores.
    issorted : bool, optional
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
    axis : int, optional
        Axis in forecasts which corresponds to different ensemble members,
        along which to calculate the threshold decomposition.

    Returns
    -------
    out : np.ndarray
        Brier scores at each thresold for each ensemble forecast against the
        observations. If ``threshold`` is a scalar, the result will have the
        same shape as observations. Otherwise, it will have an additional final
        dimension corresponding to the threshold levels.

    References
    ----------
    Gneiting, T. and Ranjan, R. Comparing density forecasts using threshold-
       and quantile-weighted scoring rules. J. Bus. Econ. Stat. 29, 411-422
       (2011). http://www.stat.washington.edu/research/reports/2008/tr533.pdf

    See also
    --------
    crps_ensemble, brier_score
    """
    observations = np.asarray(observations)
    threshold = np.asarray(threshold)
    forecasts = np.asarray(forecasts)

    if axis != -1:
        forecasts = move_axis_to_end(forecasts, axis)

    if forecasts.shape == observations.shape:
        forecasts = forecasts[..., np.newaxis]

    if observations.shape != forecasts.shape[:-1]:
        raise ValueError('observations and forecasts must have matching '
                         'shapes or matching shapes except along `axis=%s`'
                         % axis)

    scalar_threshold = threshold.ndim == 0

    if threshold.ndim > 1:
        raise ValueError('threshold must be scalar or 1-dimensional')
    if threshold.ndim == 1 and not (np.sort(threshold) == threshold).all():
        raise ValueError('1D thresholds must be sorted')

    threshold = threshold.reshape((1,) * observations.ndim + (-1,))

    if not issorted:
        forecasts = np.sort(forecasts, axis=-1)

    result = _threshold_brier_score_core(observations, forecasts, threshold)

    if scalar_threshold:
        result = result.squeeze(axis=-1)

    return result
