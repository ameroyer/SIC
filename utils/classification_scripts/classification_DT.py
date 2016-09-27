#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**classify.py.** For training and applying a weka decision tree classifier on the artifically annotated data set.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"

import os
from subprocess import Popen, PIPE


def train_DT(n, train, temp_folder, classification_params, verbose=1):
    """
    Trains a  decision tree classifier with weka and returns the resulting model.

    Args:
     * ``n`` (*int*): step number.
     * ``train``: annotated training set (structure may depend on the classifier).
     * ``temp_folder`` (*str*): path to the directory for storing temporary files.
     * ``classification_params`` (*list*): additional classification parameters.
     * ``verbose`` (*int, optional*): controls verbosity level.
    Returns:
     * ``model``: model built by the classifier from the given training set.
     """
    weka = os.environ.copy()['CLASSIFIER']
    modelf = os.path.join(temp_folder, 'step%d_model.txt' % n)
    FNULL = open(os.devnull, 'w')
    err = None if verbose >=3 else FNULL

    p = Popen(('java -cp %s weka.classifiers.trees.J48 %s -t %s -d %s'%(weka, ' '.join('-%s %s' % (x.upper(), y) for x,y in classification_params['additional_params']), train, modelf)).split(), shell=False, stdout=FNULL, stderr=err, close_fds=True)
    p.communicate()

    return modelf



def label_DT(model, test, test_entities_indices, verbose=1):
    """
    Labels a testing set using a weka decision tree and returns the resulting entities.

    Args:
     * ``model``: model built from training the classifier
     * ``test``: formatted testing set.
     * ``test_entities_indices`` (*list*): location of the interesting entities in the test dataset.
     * ``verbose`` (*int, optional*): controls verbosity level. Defaults to 1.
    Returns:
     * ``result_iter``: a generator expression on the result
    """

    weka = os.environ.copy()['CLASSIFIER']
    FNULL = open(os.devnull, 'w')
    err = None if verbose >=3 else FNULL

    #Test
    p = Popen(('java -cp %s weka.classifiers.trees.J48 -T %s -l %s -p 0'%(weka, test, model)).split(), shell=False, stdout=PIPE, stderr=err, close_fds=True)
    result = p.communicate()[0]
    FNULL.close()

    # Filter the lines containing entities
    result_lines = result.splitlines()[5:-1]
    n_result_lines = len(result_lines)
    return ((x[1], line.split()[2].split(':')[1]) for x, line in zip(test_entities_indices, result_lines))
