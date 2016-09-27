#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**classify_CRF.py.** For training and applying a wapiti CRF.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import os
from subprocess import Popen, PIPE
from ..annotation import choose_features


def train_CRF(n, train, temp_folder, classification_params, verbose=1, debug=False):
    """
    Trains a CRF classifier with wapiti and returns the resulting model.

    Args:
     * ``n`` (*int*): step number.
     * ``train``: annotated training set (structure may depend on the classifier).
     * ``temp_folder`` (*str*): path to the directory for storing temporary files.
     * ``classification_params`` (*dict*): additional classification parameters.
     * ``verbose`` (*int, optional*): controls verbosity level.
     * ``clean`` (*bool, optional*): if False, removes the temporary files that were created.
    Returns:
     * ``model``: model built by the classifier from the given training set.
    """

    # Features Selection
    wapiti = os.environ.copy()['CLASSIFIER']
    patternf = os.path.join(temp_folder, 'step%d_pattern.txt' % n)
    modelf = os.path.join(temp_folder, 'step%d_model.txt' % n)
    choose_features(classification_params['crf_pattern'][0], classification_params['crf_pattern'][1], patternf)

    FNULL = open(os.devnull, 'w')
    err = None if verbose >=3 else FNULL

    # Train Command
    train_cmd = ('%s train --compact -p %s %s /dev/stdin %s'%(wapiti, patternf, ' '.join('--%s %s' % (x, y) for x,y in classification_params['additional_params']), modelf)).split()

    #Subprocesses
    p = Popen(train_cmd, shell=False, close_fds=True, stdout=FNULL, stderr=err, stdin=PIPE)
    p.communicate(input=train)

    if not debug:
        os.remove(patternf)

    return modelf





################################################################################## LABEL
def label_CRF(model, test, test_entities_indices, classification_params, verbose=1):
    """
    Labels a testing set using a wapiti CRF classifier with wapiti and returns the resulting entities.

    Args:
     * ``model``: model built from training the classifier
     * ``test``: formatted testing set.
     * ``test_entities_indices`` (*list*): location of the interesting entities in the test dataset.
     * ``classification_params`` (*dict*): additional classification parameters.
     * ``verbose`` (*int, optional*): controls verbosity level. Defaults to 1.
    Returns:
     * ``result_iter``: a generator expression on the result
    """

    similarity_type = classification_params['similarity_type']
    wapiti = os.environ.copy()['CLASSIFIER']

    # Test Command
    FNULL = open(os.devnull, 'w')
    err = None if verbose >=3 else FNULL
    test_cmd = ('%s label --force --label %s -m %s /dev/stdin'%(wapiti, '--post --score' if similarity_type.endswith('PROB') else '', model)).split()

    p = Popen(test_cmd, shell=False, close_fds=True, stdout=PIPE, stderr=err, stdin=PIPE)
    result = p.communicate(input=test)[0]
    FNULL.close()

    # Filter the lines containing interesting entities
    # Result is an iterator containing a list of (entity index - class)
    result_lines = [x for x in result.splitlines() if not x.startswith('#')]
    n_result_lines = len(result_lines)
    result_iter = ((x[1], result_lines[x[0]]) for x in test_entities_indices if x[0] < n_result_lines)

    return result_iter
