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
    Copyright 2015 The Climate Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Contributions:
    Outside contributions (bug fixes or new features related to proper scoring
    rules) would be very welcome! Please open a GitHub issue to discuss your
    plans.
