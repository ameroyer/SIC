#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**em_analysis.py.** EM estimation of the mixture parameters for each iteration.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import os
import sys
import time
import numpy as np
from multiprocessing import Process, Queue, current_process


def em_step_threaded(pi0, p0, p1, obs, N, res_queue):
    """
    Function for parameter estimation on one thread.

    Args:
     * ``pi0`` (*float*): current estimate of the pi0 parameter ( P(x~y) ).
     * ``pi1`` (*float*): current estimate of the p1 parameter (P(c1(c)=c1(y)...cN(x)=cN(y) | x ~ y).
     * ``pi0`` (*float*): current estimate of the p0 parameter (P(c1(c)=c1(y)...cN(x)=cN(y) | x <> y).
     * ``obs`` (*list*): observations given to this thread (as dict mapping unique observation -> number of occurences).
     * ``N`` (*int*): dimension of the multivariate Bernoulli.
     * ``res_queue`` (*Queue*): output queue.
    """

    # Compute Expectation values and break if one parameter gets extreme value (0 or 1)
    print '%s - Expectation phase' %current_process().name
    z1 = {}
    for xi in obs.keys():
        q0, q1 = 0., 0.
        break0 = False
        break1 = False
        for p0k, p1k, str_xik in zip(p0, p1, xi):
            xik = int(str_xik)
            if p0k == 0. and xik == 1:
                q0 = 0.
                break0 = True

            elif p0k == 1. and xik == 0:
                q0 = 0.
                break0 = True

            if p1k == 0. and xik == 1:
                q1 = 0.
                break1 = True
            elif p1k == 1. and xik == 1:
                q1 = 0.
                break1 = True

            if not break0:
                q0 += np.log(p0k) if xik == 1 else np.log(1 - p0k)
            if not break1:
                q1 += np.log(p1k) if xik == 1 else np.log(1 - p1k)

        if not break0:
            q0 = np.exp(np.log(pi0) + q0)
        if not break1:
            q1 = np.exp(np.log(1 - pi0) + q1)

        z1[xi] = q1 / (q0 + q1)


    # Compute Maximization values
    print '%s - Maximization phase' %current_process().name

    #pi0
    N1 = np.sum([z1[xi] * wi for xi, wi in obs.iteritems()])
    #p1
    np1 = np.array([np.sum([wi * z1[xi] for xi, wi in obs.iteritems() if int(xi[k]) == 1]) for k in xrange(N)])
    #p0
    np0 = np.array([np.sum([wi * (1 - z1[xi]) for xi, wi in obs.iteritems()  if int(xi[k]) == 1]) for k in xrange(N)])

    # Put results into queue
    res_queue.put((z1, N1, np1, np0))
    res_queue.close()
    res_queue.join_thread()



def em_step(pi0, p0, p1, x, n_obs, N, cores):
    """
    One single expectation maximization step. Multiprocess execution (each process computes the part of the summations for one set of observations).

    Args:
     * ``pi0`` (*float*): current estimate of the pi0 parameter ( P(x~y) ).
     * ``p1`` (*float*): current estimate of the p1 parameter (P(c1(c)=c1(y)...cN(x)=cN(y) | x ~ y).
     * ``p0`` (*float*): current estimate of the p0 parameter (P(c1(c)=c1(y)...cN(x)=cN(y) | x <> y).
     * ``x`` (*list*): observation (as dict mapping unique observation -> number of occurences).
     * ``n_obs`` (*int*): total number of observations.
     * ``N`` (*int*): dimension of the multivariate Bernoulli.
     * ``cores`` (*int*): number of cores to use.

    Returns:
     * ``nz1`` (*float*):  estimate of the z1 hidden variables.
     * ``npi0`` (*float*):  estimate of the pi0 parameter ( P(x~y) ).
     * ``np0`` (*float*): estimate of the p0 parameter (P(c1(c)=c1(y)...cN(x)=cN(y) | x ~ y).
     * ``np1`` (*float*): estimate of the p1 parameter (P(c1(c)=c1(y)...cN(x)=cN(y) | x <> y).
    """

    # Format observations
    obs = {str(int(xi)).zfill(N): wi for xi, wi in x.iteritems()}
    length = len(obs) / cores
    obj = [(x, obs[x]) for x in obs]

    # Start Threads
    res_queue = Queue()
    cores = min(cores, N)
    threads = [0] * cores

    for k in xrange(cores):
        if k == cores - 1:
            t_obs = obj[k*length:len(obs)]
        else:
            t_obs = obj[k*length:(k+1)*length]

        t = Process(name = 'Worker %d'%k, target = em_step_threaded, args = (pi0, p0, p1, {x:y for (x,y) in t_obs}, N, res_queue))
        threads[k] = t
        t.start()

    # Retrieve results from Queue and merge
    nN1, np0, np1 = 0., np.zeros_like(p0), np.zeros_like(p1)
    nz1 = {}
    for k in xrange(cores):
        (z1, N1, pp1, pp0) = res_queue.get()
        nN1 += N1
        np1 += pp1
        np0 += pp0
        nz1.update(z1)

    # Join Threads
    for t in threads:
        t.join()

    # Final estimates of the parameters
    np1 = np1 / nN1
    np0 = np0 / (n_obs - nN1)
    npi0 = 1. - float(nN1) / n_obs

    return nz1, npi0, np0, np1



def estimate_parameters_em(co_occ, N, p1i=0.9, p0i=0.1, pi0i=0.8, n_iter=20, cores=20):
    """
    Expectation Maximization on 2-components Bernoulli mixture.

    Args:
     * ``co_occ`` (*list*): list of observations.
     * ``N`` (*int*): dimension of an observation (here, corresponds to the number of considered iterations).
     * ``pi1i`` (*float, optional*): initial estimate of the p1 parameter (P(c1(c)=c1(y)...cN(x)=cN(y) | x ~ y). Defaults to 0.9.
     * ``pi0i`` (*float, optional*): initial estimate of the p0 parameter (P(c1(c)=c1(y)...cN(x)=cN(y) | x <> y). Defaults to 0.1.
     * ``pi0i`` (*float, optional*): initial estimate of the pi0 parameter ( P(x~y) ). Defaults to 0.2.
     * ``n_iter`` (*int, optional*): number of iterations. Defaults to 20.
    """

    print 'Expectation Maximization analysis at step %d' %N
    from collections import Counter

    # Init
    p1 = np.ones(N) * p1i
    p0 = np.ones(N) * p0i
    pi0 = pi0i

    # Count occurrences of each observation to optimize the summation loop
    print "Counting"
    c = Counter(co_occ)

    # EM iterations
    for n in xrange(n_iter):
        z1, pi0, p0, p1= em_step(pi0, p0, p1, c, len(co_occ), N, cores)

        print '\n\nStep %d\n> P(x ~ y) = %.8f' %(n, 1.-pi0)
        print '\n> P(S(x,y) | x ~ y) = %s' %p1
        print '\n> P(S(x,y) | x <> y) = %s' %p0

        if pi0 == 1. or pi0 == 0.:
            print 'Maximal weight reached. End.'
            break

    return z1, pi0, p0, p1
