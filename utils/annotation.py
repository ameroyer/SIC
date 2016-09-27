#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**annotation.py.** Generation of synthetic annotations for the data samples.
"""

import os
import heapq
from random import randint, random
from collections import defaultdict

from annotation_scripts.annotation_CRF import *
from annotation_scripts.annotation_DT import *
from annotation_scripts.annotation_HTK import *

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"



def annotate(n, temp_folder, classifier_type, (train, test), annotation_params, **kwargs):
    """
    Return a synthetic annotation of both the training and the testing set.

    Args:
     * ``n`` (*int*): step identifier
     * ``temp_folder``: directory for temporary files
     * ``classifier_type`` (*str*): type of the classifier that will take the annotation as input.
     * ``train`` (*list*): initial train data structure.
     * ``test`` (*list*): initial test data structure.
     * ``annotation_params`` (*dict*): additional annotation parameters.
     * ``with_common_label_wordform `` (*bool, optional*): if ``True``, each entity occurence wordform receives the same label. Defaults to False.
     * ``verbose`` (*int, optional*): controls verbosity level.
     * ``debug`` (*bool, optional*): enable/disable debug mode.

    Returns:
     * ``N`` (*int*): random max number of synthetic labels for this step.
     * ``n_unique_labels_used`` (*int*): number of synthetic labels that were actually used.
     * ``train_file`` (*str*): path to the  formatted train data.
     * ``test_file`` (*str*): path to the  formatted test data.
     * ``training_size`` (*int*): number of sequences in the training database.
     * ``testing_size`` (*int*): number of sequences in the testing database.
     * ``n_entities_train`` (*int*): number of entities in the training database.
     * ``entites_indices`` (*list*): indentify the entities of interest (tag B) in the testing set; this will be used to filter the classifier's output.
    """

    # Samples a random number of labels
    distrib = annotation_params['distrib']
    n_min = annotation_params['n_min']
    n_max = annotation_params['n_max']
    n_labels = randint(n_min, n_max)
    preclustering = kwargs.pop('preclustering', [])
    verbose = kwargs.pop('verbose', False)
    debug = kwargs.pop('debug', False)


    # If distrib == UNI use the ``n_labels`` most present entities to define the synthetic classes
    if distrib == 'UNI':
        entities_occurrences = defaultdict(lambda:0)
        train = list(train)
        for sentence in train:
            for word in sentence:
                if word[-1] == 'B':
                    entities_occurrences[word[0]] += 1
        to_annotate = heapq.nlargest(min(n_labels, len(entities_occurrences)) , entities_occurrences.keys(), key=lambda k: entities_occurrences[k])

    # If OVA, only one label given to the chosen entity
    elif distrib == 'OVA':
        to_annotate = [annotation_params['sample']]

    # Else, no restriction
    else:
        to_annotate = None


    ################################################################ CRF
    if classifier_type == 'CRF':
        n_unique_labels_used, n_sentences_train, n_entities_train, train_file = annotate_CRF_train(n, train, distrib, n_labels, to_annotate, temp_folder, preclustering)
        n_sentences_test, test_entities_indices, test_file = annotate_CRF_test(n, test, temp_folder)

        summary = 'Step %d: %d (%d) synthetic labels. %d training sequences, %d testing sequences. %d entities in train, %d in test.'%(n, n_labels, n_unique_labels_used, n_sentences_train, n_sentences_test, n_entities_train, len(test_entities_indices))

        if debug:
            with open(os.path.join(temp_folder, 'step%d_train'%n), 'w') as f:
                f.write(train_file)
            with open(os.path.join(temp_folder, 'step%d_test'%n), 'w') as f:
                f.write(test_file)

        return n_labels, n_unique_labels_used, train_file, test_file, test_entities_indices, summary



    ################################################################# DT
    elif classifier_type == 'DT':
        # Features selection
        import re
        pattern = annotation_params['dt_pattern']
        features = choose_features(pattern[0], pattern[1])
        features = [[tuple(map(int, re.split('\%x\[|\]|,', attribute)[1:3])) for j, attribute in enumerate(pattern.split(':')[1].split('/'))] for i, pattern in enumerate(features)]

        # Create train and test dataset
        n_unique_labels_used, train_length, n_attributes, seen_classes, wekadata = annotate_DT_train(n, train, distrib, n_labels, features, to_annotate, temp_folder, preclustering)
        test_length, test_entities_indices, wekatest = annotate_DT_test(n, test, features, seen_classes[0])

        # Weka Header
        header = "@RELATION train\n\n%s"%('\n'.join(['@ATTRIBUTE att%s string'%i for i in xrange(n_attributes)]))
        wekadata = '%s\n%s\n\n@DATA\n%s\n%s'%(header, '@ATTRIBUTE class {%s}'%(','.join(list(set(seen_classes)))), '\n'.join(wekadata), '\n'.join(wekatest))

        if debug:
            with open(os.path.join(temp_folder, 'step%d_wekadata'%n), 'w') as f:
                f.write(wekadata)

        # Weka formatting (string attributes)
        train_file, test_file = weka_format(n, wekadata, train_length, test_length, temp_folder, verbose)
        summary = 'Step %d: %d (%d) synthetic labels. %d training entities, %d testing entities.'%(n, n_labels, n_unique_labels_used, train_length, test_length)

        return n_labels, n_unique_labels_used, train_file, test_file, test_entities_indices, summary


    ################################################################# HTK
    elif classifier_type == 'HTK':
        n_unique_labels_used, n_entities_train, scp_trains, mlf = annotate_HTK_train(n, train, distrib, n_labels, temp_folder, preclustering)
        n_entities_test, scp_test, test_entities_indices = annotate_HTK_test(n, test)

        summary = 'Step %d: %d (%d) synthetic labels. %d training entities, %d testing entities.'%(n, n_labels, n_unique_labels_used, n_entities_train, n_entities_test)

        return n_labels, n_unique_labels_used, (scp_trains, mlf), scp_test, test_entities_indices, summary





def choose_features(pattern, distrib, output_pattern_file=None):
    """
    Given a set of features and their probability of occurrence (pattern), choose features at random for the current training step.

    Args:
     * ``pattern`` (*list*): features organized by importance category.
     * ``distrib`` (*list*): probability of sampling a feature for each category.
     * ``output_pattern_file`` (*str*): testing set.

    Returns:
     * ``to_keep`` (*list*): the selected features.
    """

    # Sample  random features
    to_keep = []
    for feat, chance in zip(pattern, distrib):
        for x in feat:
            r = random()
            if r < chance:
                to_keep.append(x)

    # Write Pattern
    if output_pattern_file:
        with open(output_pattern_file, 'w') as f:
            f.write('\n'.join(to_keep))
    return to_keep
