#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**probability_fit.py.** functions related to estimate probability distribution.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import numpy as np
from cmath import exp, pi


def sample_from_pdf(N, values):
    """
    Returns N samples from a given discrete distribution ``P``.

    Args:
     * ``N`` (*int*): number of samples to draw.
     * ``values`` (*list*): values taken by the discrete distribution (``k -> P(X = k)``)

    Returns:
     * ``samples`` (*list*): ``N`` samples drawn from the ``P`` distribution
    """
    from random import random as rand
    from bisect import bisect

    values = np.asarray(values)
    assert (np.sum(values) == 1), 'input values do not sum to 1'
    assert (len(np.where(values < 0)[0]) == 0), 'input values contain negative number'

    probs = np.cumsum(values)
    samples = np.zeros(N)
    for k in xrange(N):
        r = rand()
        j = bisect(probs, r)
        samples[k] = j - 1

    return samples


##################################################################### For binary similarity
def estimate_poisson_binomial(N, p_values):
    """
    Estimate the values of a Poisson Binomial distribution given its p-parameters.

    Args:
     * ``N`` (*int*): number of independant Bernouilli experiments.
     * ``p_values`` (*list*): Bernouilli parameter for each experiment.

    Returns:
     * ``values`` (*list*): values taken by the distribution (``k -> P(X = k)``)
    """
    C = exp(2j*pi / (N+1))
    s = [np.prod([ (1 + p*( C**l - 1)) for p in p_values]) for l in xrange(0, N+1)]
    return np.abs(np.fft.fft(s) / (N+1))
