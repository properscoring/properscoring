properscoring
=============

.. image:: https://travis-ci.org/TheClimateCorporation/properscoring.svg?branch=master
    :target: https://travis-ci.org/TheClimateCorporation/properscoring

.. highlight:: python

`Proper scoring rules`_ for evaluating probabilistic forecasts in Python.
Evaluation methods that are "strictly proper" cannot be artificially improved
through hedging, which makes them fair methods for accessing the accuracy of
probabilistic forecasts. These methods are useful for evaluating machine
learning or statistical models that produce probabilities instead of point
estimates. In particular, these rules are often used for evaluating weather
forecasts.

.. _Proper scoring rules: https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

properscoring runs on both Python 2 and 3. It requires NumPy (1.8 or
later) and SciPy (any recent version should be fine). Numba is optional,
but highly encouraged: it enables significant speedups (e.g., 20x faster)
for ``crps_ensemble`` and ``threshold_brier_score``.

To install, use pip: ``pip install properscoring``.

Example: five ways to calculate CRPS
------------------------------------

This library focuses on the closely related
`Continuous Ranked Probability Score`_ (CRPS) and `Brier Score`_. We like
these scores because they are both interpretable (e.g., CRPS is a
generalization of mean absolute error) and easily calculated from a finite
number of samples of a probability distribution.

.. _Continuous Ranked Probability Score: http://www.eumetcal.org/resources/ukmeteocal/verification/www/english/msg/ver_prob_forec/uos3b/uos3b_ko1.htm
.. _Brier score: https://en.wikipedia.org/wiki/Brier_score

We will illustrate how to calculate CRPS against a forecast given by a
Gaussian random variable. To begin, import properscoring::

    import numpy as np
    import properscoring as ps
    from scipy.stats import norm

Exact calculation using ``crps_gaussian`` (this is the fastest method)::

    >>>> ps.crps_gaussian(0, mu=0, sig=1)
    0.23369497725510913

Numerical integration with ``crps_quadrature``::

    >>> ps.crps_quadrature(0, norm)
    array(0.23369497725510724)

From a finite sample with ``crps_ensemble``::

    >>> ensemble = np.random.RandomState(0).randn(1000)
    >>> ps.crps_ensemble(0, ensemble)
    0.2297109370729622

Weighted by PDF values with ``crps_ensemble``::

    >>> x = np.linspace(-5, 5, num=1000)
    >>> ps.crps_ensemble(0, x, weights=norm.pdf(x))
    0.23370047937569616

Based on the `threshold decomposition`_ of CRPS with
``threshold_brier_score``::

    >>> threshold_scores = ps.threshold_brier_score(0, ensemble, threshold=x)
    >>> (x[1] - x[0]) * threshold_scores.sum(axis=-1)
    0.22973090090090081

.. _threshold decomposition: https://www.stat.washington.edu/research/reports/2008/tr533.pdf

In this example, we only scored a single observation/forecast pair. But
to reliably evaluate a forecast model, you need to average these scores across
many observations. Fortunately, all scoring rules in properscoring happily
accept and return observations as multi-dimensional arrays::

    >>> ps.crps_gaussian([-2, -1, 0, 1, 2], mu=0, sig=1)
    array([ 1.45279182,  0.60244136,  0.23369498,  0.60244136,  1.45279182])

Once you calculate an average score, is often useful to normalize them
relative to a baseline forecast to calculate a so-called "skill score",
defined such that 0 indicates no improvement over the baseline and 1
indicates a perfect forecast. For example, suppose that our baseline
forecast is to always predict 0::

    >>> obs = [-2, -1, 0, 1, 2]
    >>> baseline_score = ps.crps_ensemble(obs, [0, 0, 0, 0, 0]).mean()
    >>> forecast_score = ps.crps_gaussian(obs, mu=0, sig=1).mean()
    >>> skill = (baseline_score - forecast_score) / baseline_score
    >>> skill
    0.27597311068630859

A standard normal distribution was 28% better at predicting these five
observations.

API
---

properscoring contains optimized and extensively tested routines for
scoring probability forecasts. These functions currently fall into two
categories:

* Continuous Ranked Probability Score (CRPS):

  - for an ensemble forecast: ``crps_ensemble``
  - for a Gaussian distribution: ``crps_gaussian``
  - for an arbitrary cumulative distribution function: ``crps_quadrature``

* Brier score:

  - for binary probability forecasts: ``brier_score``
  - for threshold exceedances with an ensemble forecast: ``threshold_brier_score``

All functions are robust to missing values represented by the floating
point value ``NaN``.

History
-------

This library was written by researchers at The Climate Corporation. The
original authors include Leon Barrett, Stephan Hoyer, Alex Kleeman and
Drew O'Kane.

License
-------

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

Contributions
-------------

Outside contributions (bug fixes or new features related to proper scoring
rules) would be very welcome! Please open a GitHub issue to discuss your
plans.
