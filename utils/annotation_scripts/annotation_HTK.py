#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**annotation_HTK.py.** Generation of synthetic annotations for HTK HMM classifiers.
"""
import os
from random import randint
from collections import defaultdict


__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"



def annotate_HTK_train(n, train, distrib, n_labels, temp_folder, preclustering):
    """
    Returns a synthetic annotation of the data (train + test) for the (HTK) HMM classifier.

    Args:
     * ``n`` (*int*): step identifier.
     * ``train`` (*list*): training entities.
     * ``distrib`` (*str*): type of the synthetic annotation.
     * ``n_labels`` (*int*): random max number of synthetic labels for this step.
    Returns:
     * ``n_unique_labels_used`` (*int*): number of synthetic labels that were actually used.
     * ``n_entities_train`` (*int*): number of entities in the training database.
     * ``train_file`` (*str*): path to the  formatted train data.
     * ``mlf`` (*str*): HTK master label file.
    """
    # Parameters
    unique_labels_used = [0] * (n_labels+3)   #counts the number of unique labels (N synthetic + 'O' + 'I')
    n_entities_train = 0
    annotation_dict = defaultdict(lambda:[])
    seen_entities = {}

    # Annotation
    for i, features in train:
        try:
            cl = seen_entities[preclustering[i]]
        except (KeyError, IndexError):
            n_cl = randint(0, n_labels - 1)
            unique_labels_used[n_cl] = 1
            cl = 'class_%d'%n_cl
            if len(preclustering) > i:
                seen_entities[preclustering[i]] = cl
        annotation_dict[cl].append(features)

    # Delete classes with too few observation
    todel = []
    for cl, obs in annotation_dict.iteritems():
        if len(obs) < 29: #At least 10 samples per class #base=19
            todel.append(cl)

    for cl in todel:
        unique_labels_used[int(cl.split('_')[1])] = 0
        del annotation_dict[cl]

    # Create and write mlf file
    mlf = ['#!MLF!#']
    for cl, obs in annotation_dict.iteritems():
        for features in obs:
            n_entities_train += 1
            mlf.append('"*/%s.lab"\n%s\n.'%(features.rsplit('/', 1)[1].rsplit('.', 1)[0], cl))

    mlf_file = os.path.join(temp_folder, 'step%d_annot.mlf' % n)
    with open(mlf_file, 'w') as f:
        f.write('\n'.join(mlf))
        f.write('\n')

    # Create train scripts for each class
    scp_trains = {cl: '\n'.join(features) for cl, features in annotation_dict.iteritems()}

    return sum(unique_labels_used), n_entities_train, scp_trains, mlf_file





def annotate_HTK_test(n, test):
    """
    Returns a HTK-formatted version of the test entities for the HTK classifier.

    Args:
     * ``n`` (*int*): step identifier.
     * ``test`` (*list*): test entities.
     * ``features`` (*list*): pattern for the feature selection.
     * ``fake_class``: fake weka class to give to all test entities for the weka format.
    Returns:
     * ``n_sentences_test`` (*int*): number of sequences in the training database.
     * ``test_entities`` (*list*): indices and identifiers of the entities of interest in the test database.
     * ``test_file`` (*str*): path to the  formatted test data.
    """
    # Test
    test_length = 0
    test_entities_indices = []
    scp_test = []
    for i_test, (i, features) in enumerate(test):
        test_entities_indices.append((i_test, i))
        scp_test.append(features)
        test_length += 1

    scp_test = '\n'.join(scp_test)

    return test_length, scp_test, test_entities_indices
