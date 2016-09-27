#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**opt.py.**: Functions designed for vectorial updates of the similarity matrix instead of cell by cell.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"

import numpy as np

def cartesian(arrays, out=None, numpy=True):
    """
    Generate a cartesian product of input arrays.

    Parameters:
     * arrays : list of array-like. 1-D arrays to form the cartesian product of.
     * out : ndarray. Array to place the cartesian product in.

    Returns:
     * out : ndarray. 2-D array of shape (M, len(arrays)) containing cartesian products formed of input arrays.
    See:
     * http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays/1235363#1235363

    """
    if not numpy:
        arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def cartesian_prod(arrays, full=None, out=None, numpy=True):
    """
    Generates the products for all possible cartesian combinations of the arrays components.

    Args:
     * ``arrays``: list of array-like.
     * ``full`` (*int*): length of the base array. Starting value should be None.
     * ``out`` (*ndarray*): array where to put the result.
     * ``numpy`` (*bool, optional*): If False, convert the arrays to numpy type. Defaults to True.

    Returns:
     * ``out`` (*ndarray*): array containing the products for all possible cartesian combinations.
    """
    if not numpy:
        arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.ones(n, dtype=dtype)
        full = n

    m = n / arrays[0].size
    fill = full / n
    out *= np.tile(np.repeat(arrays[0], m), fill)
    if arrays[1:]:
        cartesian_prod(arrays[1:], full = full, out = out)
    return out


def pairs_combination(a, numpy=True):
    """
    Returns all the possible pair combinations from an index array.

    Args:
     * ``a``: array-like.
     * ``numpy`` (*bool, optional*): If False, convert the arrays to numpy type. Defaults to True.

    Returns:
     * ``out``: array containing all the possible pair combinations from an index array.
    """
    return cartesian([a, a], numpy = numpy)


def pairs_combination_indices(a, n_samples, numpy=True):
    """
    Returns all the possible pairs combinations from an index array, under their index form (upper triangle matrix).

    Args:
     * ``a``: array-like.
     * ``n_samples``: size of the 2D matrix (ie max possible value of the index + 1)
     * ``numpy`` (*bool, optional*): If False, convert the arrays to numpy type. Defaults to True.

    Returns:
     * ``out``: array containing the indices of all  possible pairs combinations.
    """
    return [j + i*n_samples - (i+1)*(i+2)/2 if i < j else i + j*n_samples - (j+1)*(j+2)/2  for ind, i in enumerate(a) for j in a[ind+1:]]


def product_combination(a, numpy=True):
    """
    Returns all the possible pair product combinations from an index array.

    Args:
     * ``a``: array-like.
     * ``numpy`` (*bool, optional*): If False, convert the arrays to numpy type. Defaults to True.

    Returns:
     * ``out``: array containing all the possible pair product combinations from an index array.
    """
    return [i*j for ind, i in enumerate(a) for j in a[ind+1:]]
