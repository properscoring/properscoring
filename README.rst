properscoring
=============

Proper scoring rules in Python. `Proper scoring rules`_ are useful for evaluating
probability forecasts, and are widely used for evaluating weather forecasts.

.. _Proper scoring rules: https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

Contents:
    * Continuous Ranked Probability Score (CRPS):

      - for an ensemble forecast: ``crps_ensemble``
      - for a Gaussian distribution: ``crps_gaussian``
      - for an arbitrary cummulative distribution function: ``crps_quadrature``
    * Brier score:
      - for binary probability forecasts: ``brier_score``
      - for threshold exceedances for an ensemble forecast: ``threshold_brier_score``

    To usefully interpret these scoring rules, you should average them across many
    observations.

    These functions have been optimized and extensively tested.

Install:
    Requires NumPy and SciPy. Numba is optional, but highly encouraged: we use it
    for significant speedups with ``crps_ensemble`` and ``threshold_brier_score``.
    To install, use pip: ``pip install properscoring``.

History:
    This library was written by researchers at The Climate Corporation. The
    original authors include Leon Barrett, Stephan Hoyer, Alex Kleeman and
    Drew O'Kane.

License:
    Apache 2.0.
