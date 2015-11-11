properscoring
=============

`Proper scoring rules`_ for evaluating probabilistic forecasts in Python.
Evaluation methods that are "strictly proper" cannot be artificially improved
through hedging, which makes them fair methods for accessing the accuracy of
probabilistic forecasts. In particular, such rules are widely used for
evaluating weather forecasts.

.. _Proper scoring rules: https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

This library focuses on the closely related Continuous Ranked Probability
Score (CRPS) and Brier Score. We like these scores because they are both
interpretable (e.g., CRPS is a generalization of mean absolute error) and
easily calculated from a finite number of samples of a probability
distribution.

Contents:
    * Continuous Ranked Probability Score (CRPS):

      - for an ensemble forecast: ``crps_ensemble``
      - for a Gaussian distribution: ``crps_gaussian``
      - for an arbitrary cummulative distribution function: ``crps_quadrature``

    * Brier score:

      - for binary probability forecasts: ``brier_score``
      - for threshold exceedances with an ensemble forecast: ``threshold_brier_score``

    These functions have been optimized and extensively tested.

Install:
    Requires NumPy and SciPy. Numba is optional, but highly encouraged: we use it
    for significant speedups with ``crps_ensemble`` and ``threshold_brier_score``.
    To install, use pip: ``pip install properscoring``.

Guidelines:
    To usefully interpret these scoring rules, you should average them across many
    observations.

    It is also often useful to normalize such scores relative to a baseline
    forecast to calculate a so-called "skill score", defined such that 0
    indicates no improvement over the baseline and 1 indicates a perfect
    forecast.

History:
    This library was written by researchers at The Climate Corporation. The
    original authors include Leon Barrett, Stephan Hoyer, Alex Kleeman and
    Drew O'Kane.

License:
    Apache 2.0.

Contributions:
    Outside contributions would be very welcome! Please open a GitHub issue to
    discuss your plans.
