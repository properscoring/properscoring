import warnings

import numba
import numpy as np


@numba.jit(nopython=True)
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


def threshold_decomposition(observations, forecasts, thresholds, axis=-1):
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

    def threshold_exceedances(x):
        # NaN safe calculation of threshold exceedances
        # TODO: expose a version of this function in the public API?
        # add an extra dimension to `x` and broadcast `thresholds` so that it
        # varies along that new dimension
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', 'invalid value encountered in greater')
            exceeds = (x[..., np.newaxis] >
                       thresholds.reshape(*([1] * x.ndim + [-1]))
                       ).astype(float)
        if x.ndim == 0 and np.isnan(x):
            exceeds[:] = np.nan
        else:
            exceeds[np.where(np.isnan(x))] = np.nan
        return exceeds

    binary_obs = threshold_exceedances(observations)
    if observations.shape == forecasts.shape:
        prob_forecast = threshold_exceedances(forecasts)
    elif (observations.shape ==
          tuple(s for n, s in enumerate(forecasts.shape)
                if n != axis % forecasts.ndim)):
        forecasts = np.swapaxes(forecasts, axis, -1)
        # axis=-2 should be the 'realization' axis, after swapping that axes
        # to the end of forecasts and inserting one extra axis
        prob_forecast = np.nanmean(threshold_exceedances(forecasts), axis=-2)
    else:
        raise ValueError('observations and forecasts must have matching '
                         'shapes or matching shapes except along `axis=%s`'
                         % axis)
    return brier_score(binary_obs, prob_forecast)
