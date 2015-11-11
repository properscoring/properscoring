import numpy as np
from numba import guvectorize


@guvectorize(["void(float64[:], float64[:], float64[:], float64[:])"],
             "(),(n),(n)->()", nopython=True)
def _crps_ensemble_gufunc(observation, forecasts, weights, result):
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


@guvectorize(["void(float64[:], float64[:], float64[:], float64[:])"],
             "(),(n),(m)->(m)", nopython=True)
def _threshold_brier_score_gufunc(observation, forecasts, thresholds, result):
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
