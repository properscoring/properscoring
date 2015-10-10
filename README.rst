properscoring
=============

Proper scoring rules in Python. Proper scoring rules are useful for evaluating
probability forecasts, and are widely used for evaluating weather forecasts.

Contents:
    * Continuous Ranked Probability Score (CRPS):

      - for an forecast ensemble: ``crps_ensemble``
      - for a Gaussian distribution: ``crps_gaussian``
      - for an arbitrary cummulative distribution function: ``crps_cdf``
    * Brier score: ``brier_score``
    * Threshold decomposition of CRPS: ``threshold_decomposition``

    To usefully interpret these scoring rules, you should average them across many
    observations.

    These functions have been optimized and extensively tested.

Install:
    Requires NumPy, SciPy and Numba. Then: ``pip install properscoring``.

History:
    This library was written by researchers at The Climate Corporation. The
    original authors include Leon Barrett, Stephan Hoyer, Alex Kleeman and
    Drew O'Kane.

License:
    Apache 2.0.
