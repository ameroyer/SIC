#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**classify.py.** For training and applying a classifier on the artifically annotated data set.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"

import os
import gc
import numpy as np
import random
from opt import product_combination, pairs_combination_indices
from collections import defaultdict

from classification_scripts.classification_CRF import *
from classification_scripts.classification_DT import *
from classification_scripts.classification_HTK import *


def do_classify_step(n, temp_folder, train, test, test_entities_indices, coocc_matrix, unit_length, count_lab, classification_params, **kwargs):
    """
    Builds the similarity for the current step given a classifier and synthetic annotation.

    Args:
     * ``n`` (*int*): step number.
     * ``temp_folder`` (*str*): path to the directory for storing temporary files.
     * ``train``: annotated training set (structure may depend on the classifier).
     * ``test``: formatted testing set.
     * ``test_entities_indices`` (*list*): indices of test entities and their position in the classifier output.
     * ``cocc_matrix`` (*ndarray*): shared similarity matrix.
     * ``unit_length`` (*int*): length of a locked cell in the matrix.
     * ``count_lab`` (*ndarray*): count how many times an entity has been classified as non-null in the test set.
     * ``classification_params`` (*dict*): additional classification parameters.
     * ``clean`` (*bool, optional*): if ``True``, removes the temporary files that were created. Defaults to ``True``.
     * ``with_unique_occurrences`` (*bool, optional*): if ``True``, each entity occurence is considered unique and can receive a different label. Defaults to ``True``.
     * ``verbose`` (*int, optional*): controls verbosity level. Defaults to 1.
     * ``debug`` (*bool, optional*): runs in debug mode.
     * ``pretrained`` (*optional*): pretrained model.
    """

    verbose = kwargs.pop('verbose', 1)
    debug = kwargs.pop('debug', False)
    with_unique_occurrences = kwargs.pop('with_unique_occurrences', True)
    model = kwargs.pop('pretrained', None)
    classifier_type = classification_params['classifier_type']
    similarity_type = classification_params['similarity_type']
    gc.collect()


    #-------------------- CRF
    if classifier_type == 'CRF':
        #Train
        if model is None:
            model = train_CRF(n, train, temp_folder, classification_params, verbose=verbose, debug=debug)
        del train

        #Label
        result_iter = label_CRF(model, test, test_entities_indices, classification_params, verbose=verbose)
        del test

        if not debug:
            os.remove(model)


    #----------------------- DT
    elif classifier_type == 'DT':
        if model is None:
            model = train_DT(n, train, temp_folder, classification_params, verbose=verbose)
            os.remove(train)

        #Label
        result_iter = label_DT(model, test, test_entities_indices, verbose)
        os.remove(test)

        if not debug:
            os.remove(model)


    #----------------------- HTK
    elif classifier_type == 'HTK':
        if model is None:
            hmmdef_file, hmmlist_file, wnet_file, dic_file = train_HTK(n, train, temp_folder, classification_params, verbose=verbose, debug=debug)
            del train

            result_iter = label_HTK(n, hmmdef_file, hmmlist_file, wnet_file, dic_file, test, test_entities_indices, temp_folder, verbose=verbose, debug=debug)
            del test


    #------------------- Update similarity matrix
    gc.collect()
    if with_unique_occurrences:
        return update_similarity_unique(n, result_iter, coocc_matrix, unit_length, count_lab, similarity_type, verbose)
    else:
        return update_similarity(n, result_iter, coocc_matrix, unit_length, count_lab, similarity_type, verbose)



################################################################################ UPDATE SIMILARITY MATRIX
def update_similarity_unique(n, result_iter, coocc_matrix, unit_cell_length, label_occurrences, similarity_type, verbose):
    """
    Updates the similarity matrix in the case where each occurence is an unique named entity (i.e. default case).

    Args:
     * ``n`` (*int*): step number.
     * ``result_iter`` (*str*): generator expression on the output of the classifier.
     * ``coocc_matrix`` (*ndarray*): similarity matrix of the current thread.
     * ``unit_cell_length`` (*int*): length of one locked cell in the matrix.
     * ``label_occurrences`` (*ndarray*): count how many times an entity has been classified as non-null in the test set.
     * ``similarity_type`` (*str*): type of the similarity to use.
     * ``verbose`` (*int, optional*): controls verbosity level. Defaults to 1.

    Returns:
     * ``n_test_labels``: number of distinct annotated labels in the testing base.
     * ``weights``: repartition of the samples in the test classifications (only with weighted similarity).
     * ``b`` (*float*): penalty to be added at the end of the building.
    """

    #------------------------------------ First parse
    n_samples = len(label_occurrences)
    with_weight = (similarity_type in ['WBIN', 'WPROB'])
    with_uncertainty = (similarity_type in ['UWBIN', 'UWPROB'])
    with_prob = (similarity_type in ['PROB', 'WPROB'])

    # Build auxillary structure (dict class -> samples)
    classes = defaultdict(lambda:[])
    if with_prob:
        scores = defaultdict(lambda:[])

    for id, label in result_iter:
        (_, ind) = id.split('-')
        ind = int(ind)
        if with_prob:
            typ, score = label.split('\t')[1].split('/')
            score = float(score)
            typ = typ.split('-')[1]
        else:
            typ, score = label.split('-')[1], 1

        classes[typ].append(ind)
        if with_prob:
            scores[typ].append(score)
        label_occurrences[ind] += 1

    if verbose >= 2:
        weights = {x : len(y) for x,y in classes.iteritems()}
        print 'Step %d\n --> %d test labels. Repartition: %s' % (n, len(weights), weights)

    # Weighted similarity
    if with_weight:
        weights = [len(y) for x, y in classes.iteritems() if not x in ['null', 'in', 'fake']]
        total = sum([len(y) for y in classes.itervalues()])
        p = float(sum([x**2 for x in weights])) / total**2
        if p == 0 or p == 1:
            a, b = 0., 0.
        else:
            b = - np.sqrt(p / (1-p))
            a =  np.sqrt((1-p) / p)

    # Weighted 2
    elif with_uncertainty:
        weights = [len(y) for x, y in classes.iteritems() if not x in ['null', 'in', 'fake']]
        total = sum([len(y) for y in classes.itervalues()])
        p = float(sum([x**2 for x in weights])) / total**2
        sigma = 1.
        q = 0.18
        b = - np.sqrt((p + (1-p)*q) / ((1-p)*(1-q))) * sigma
        a = - 1.0/b * sigma**2

    # Binary sim
    else:
        b = 0
        a = 1
    s = a - b

    #----------------------------------------------------------------- Second: update similarity

    ################################################ FIRST CASE. 1 lock: Use numpy vectorization (e.g. for OVA).
    if similarity_type != 'WEM' and len(coocc_matrix) == 1:
        with_ova = (len(coocc_matrix[0]) == n_samples)
        if with_uncertainty:
            pos_indices = []

        # Positive update
        for typ, samples in classes.iteritems():
            if not typ in ['null', 'in', 'fake']:
                # Indices to positively update
                if with_ova:
                    indx = samples
                else:
                    indx = pairs_combination_indices(samples, n_samples, numpy=False)

                if with_uncertainty:
                    pos_indices.extend(indx)

                # Probabilistic scores
                if with_prob:
                    if with_ova:
                        scores = scores[typ]
                    else:
                        scores = product_combination(scores[typ], numpy=False)

                # Positive update
                with coocc_matrix[0].get_lock():
                    arr = np.frombuffer(coocc_matrix[0].get_obj())
                    if not with_prob:
                        arr[indx] = arr[indx] + s
                    else:
                        arr[indx] = arr[indx] + s * scores

        # Negative Update with uncertainty
        # Sample a proportion ``p`` of cells to positively update among all the pairs not classified together
        if with_uncertainty:
            indx = np.random.choice(np.setdiff1d(np.arange(len(coocc_matrix[0])), pos_indices), q * len(coocc_matrix[0]))
            with coocc_matrix[0].get_lock():
                arr = np.frombuffer(coocc_matrix[0].get_obj())
                arr[indx] = arr[indx] +  s


    ######################################################## SECOND CASE. Multiple locks and basic loops.
    elif similarity_type != 'WEM':
        for typ, samples in classes.iteritems():
            if with_prob:
                scores_array = scores[typ]
            null_class = (typ in ['null', 'in', 'fake'])

            #------------------------------- Positive Update
            for i, x in enumerate(samples):

                if not null_class:
                    for j, y in enumerate(samples[i+1:]):
                        score = scores_array[i] * scores_array[i+1+j] * s if with_prob else s
                        index = y + x*n_samples - (x+1)*(x+2)/2 if x < y else x + y*n_samples - (y+1)*(y+2)/2
                        cell, offset = coocc_matrix[index/ unit_cell_length], index%unit_cell_length
                        with cell.get_lock():
                            arr = np.frombuffer(cell.get_obj())
                            arr[offset] = arr[offset] + score


                #------------------------- Negative update with uncertainty
                if with_uncertainty:
                    for y in (z for k,l in classes.iteritems()  for z in l if k < typ):
                        pi = random.random()
                        if pi <= q:
                            index = y + x*n_samples - (x+1)*(x+2)/2 if x < y else x + y*n_samples - (y+1)*(y+2)/2
                            cell, offset = coocc_matrix[index/ unit_cell_length], index%unit_cell_length
                            with cell.get_lock():
                                arr = np.frombuffer(cell.get_obj())
                                arr[offset] = arr[offset] + s


    ######################################################## THIRD CASE. EM
    else:
        # Samples in the same class -> 1
        for typ, samples in classes.iteritems():
            if with_prob:
                scores_array = scores[typ]
            null_class = (typ in ['null', 'in', 'fake'])
            for i, x in enumerate(samples):
                if not null_class:
                    for j, y in enumerate(samples[i+1:]):
                        index = y + x*n_samples - (x+1)*(x+2)/2 if x < y else x + y*n_samples - (y+1)*(y+2)/2
                        cell, offset = index / unit_cell_length, index % unit_cell_length
                        with coocc_matrix[cell].get_lock():
                            coocc_matrix[cell][offset][n] = 1

        # Samples not in test set -> 0
        test_samples = np.asarray([x for l in classes.itervalues() for x in l])
        train_samples = np.setdiff1d(range(n_samples), test_samples)
        for x in train_samples:
            for y in xrange(i+1, n_samples):
                index = y + x*n_samples - (x+1)*(x+2) / 2 if x < y else x + y*n_samples - (y+1)*(y+2)/2
                cell, offset = index / unit_cell_length, index % unit_cell_length
                with coocc_matrix[cell].get_lock():
                    coocc_matrix[cell][offset][n] = 0

    return len(classes), [len(x) for x in classes.itervalues()], b






def update_similarity(n, result_iter, coocc_matrix, unit_cell_length, label_occurrences, similarity_type, verbose):
    """
    Otpimizes the similarity matrix update in the case where multiple occurences correspond to the same entity(e.g. Aquaint2 case).

    Args:
     * ``n`` (*int*): step number.
     * ``result_iter`` (*str*): output of the classification algorithm (Wapiti CRF).
     * ``coocc_matrix`` (*ndarray*): similarity matrix of the current thread.
     * ``count_lab`` (*ndarray*): count how many times an entity has been classified as non-null in the test set.
     * ``similarity_type`` (*str*): type of the similarity to use.
     * ``verbose`` (*int, optional*): controls verbosity level. Defaults to 1.

    Returns:
     * ``n_test_labels``: number of distinct annotated labels in the testing base
     * ``weights``: repartition of the samples in the test classifications (only with weighted similarity).
     * ``b`` (*float*): penalty to be added at the end of construction.
    """

    #------------------------ First loop: keep interesting entities + number of occurences for the second loop
    n_samples = len(label_occurrences)
    classes = defaultdict(lambda: defaultdict(lambda: []))
    total_occurrences = np.zeros(n_samples)
    alpha = 0.5
    total_occurrences += alpha #smooth
    with_weight = (similarity_type in ['WBIN', 'WPROB', 'UWBIN', 'UWPROB'])
    with_uncertainty = (similarity_type in ['UBIN', 'UPROB', 'UWBIN', 'UWPROB'])
    with_prob = (similarity_type in ['PROB', 'WPROB', 'UPROB', 'UWPROB'])

    #Build dict structure
    for id, label in result_iter:
        (_, ind) = id.split('-')

        # Counts
        if with_prob:
            typ, score = label.split('/')
            score = float(score)
            typ = typ.split('-')[1]
        else:
            typ, score = label.split('-')[1], 1.

        classes[typ][int(ind)].append(score)
        total_occurrences[int(ind)] += 1

    #Lighter dict structure
    for typ, l in classes.iteritems():
        classes[typ] = [(x, len(scores), float(sum(scores)) / len(scores)) for x, scores in l.iteritems()]

    if verbose >= 2:
        print 'Step %d\n --> %d test labels. Repartition: {%s}'%(n, len(classes), ', '.join(['%s: %d (%d)'%(cl, len(sim), sum([x[1] for x in sim])) for cl, sim in classes.iteritems()]))

    # Weighted sim
    if with_weight:
        weights = [np.sum(t[1] for t in sim) for x, y in classes.iteritems() if not x in ['null', 'in', 'fake']]
        p = float(sum([x**2 for x in weights])) / (weights ** 2)
        if p == 0 or p == 1: #Only one class, or 0
            a, b = 0., 0.
        else:
            b = - np.sqrt(p / (1-p))
            a = np.sqrt((1-p) / p)

    # Binary sim
    else:
        a = 1.0
        b = 0.0
    s = a - b



    ################################################ FIRST CASE. 1 lock: Use numpy vectorization (e.g. for OVA).
    if similarity_type != 'WEM' and len(coocc_matrix) == 1:
        with_ova = (len(coocc_matrix[0]) == n_samples)
        if with_uncertainty:
            pos_indices = []

        #------------------------------- Positive update
        for typ, aux in classes.iteritems():
            if not typ in ['null', 'in', 'fake']:
                samples = [x[0] for x in aux]
                #samples_scores = np.asarray([x[2] / total_occurrences[x[0]] for x in aux])
                samples_scores = np.asarray([1.0 / len(samples) for _ in aux])

                if with_ova:
                    train_sample = int(typ)
                    indx = samples
                else:
                    indx = pairs_combination_indices(samples, n_samples, numpy = False)

                if with_uncertainty:
                    pos_indices.extend(indx)


                if with_ova:
                    with coocc_matrix[0].get_lock():
                        arr = np.frombuffer(coocc_matrix[0].get_obj())
                        arr[indx] = arr[indx] + s * samples_scores
                else:
                    with coocc_matrix[0].get_lock():
                        arr = np.frombuffer(coocc_matrix[0].get_obj())
                        scores = product_combination(samples_scores, numpy = False)
                        arr[indx] = arr[indx] + s * samples_scores

        #--------------------------------- (reverse) Negative Update
        # Sample ``p`` case to positively update with uncertainty
        if with_uncertainty:
            indx = np.random.choice(np.setdiff1d(np.arange(len(coocc_matrix[0])), pos_indices), p * len(coocc_matrix[0]))
            with coocc_matrix[0].get_lock():
                arr = np.frombuffer(coocc_matrix[0].get_obj())
                arr[indx] = arr[indx] +  s



    ######################################################## SECOND CASE. Multiple locks and basic loops.
    #------------------------------------- Second loop: Build similarity from classifier outputs
    elif similarity_type != 'WEM':
        #Loop over each label
        for k, (typ, neighbours) in enumerate(classes.iteritems()):
            null_class = (typ in ['null', 'in', 'fake'])
            sum_scores = np.sum(np.asarray([x[2] for x in neighbours]))


            # Loop over each sample in the class
            for j, (indj, occj, scorej) in enumerate(neighbours):
                #----------------------- Pairs of samples classified together (if type != O, I or fake)
                # Do not count fake O and I
                if not null_class:
                    for (indi, occi, scorei) in neighbours[:j]:
                        index = indj + indi*n_samples - (indi+1)*(indi+2)/2 if indi < indj else indi + indj*n_samples - (indj+1)*(indj+2)/2
                        cell, offset = coocc_matrix[index/ unit_cell_length], index%unit_cell_length
                        with cell.get_lock():
                            arr = np.frombuffer(cell.get_obj())
                            arr[offset] = arr[offset] + s * (scorej / total_occurrences[indj]) * (scorei / total_occurrences[indi])

                # -------------------------- Pairs of samples not classified together (for uncertainty)
                if with_uncertainty:
                    for (indi, occi, scorei) in [obj for z, l in classes.iteritems() for obj in l if z < typ]:
                        pi = random.random()
                        if pi <= p:
                            index = indj + indi*n_samples - (indi+1)*(indi+2)/2 if indi < indj else indi + indj*n_samples - (indj+1)*(indj+2)/2
                            cell, offset = coocc_matrix[index/ unit_cell_length], index%unit_cell_length
                            with cell.get_lock():
                                arr = np.frombuffer(cell.get_obj())
                                arr[offset] = arr[offset] + s * float(occj) * scorej / total_occurrences[indj] * float(occi) * scorei / total_occurrences[indi]




    ######################################################### THIRD CASE: EM SIMILARITY
    else:
        # Samples in the same class -> 1
        for k, (typ, neighbours) in enumerate(classes.iteritems()):
            null_class = (typ in ['null', 'in', 'fake'])
            sum_scores = np.sum(np.asarray([x[2] for x in neighbours]))
            # Loop over each sample in the class
            for j, (indj, occj, scorej) in enumerate(neighbours):
                if not null_class:
                    for (indi, occi, scorei) in neighbours[:j]:
                        index = indj + indi*n_samples - (indi+1)*(indi+2)/2 if indi < indj else indi + indj*n_samples - (indj+1)*(indj+2)/2
                        cell, offset = index / unit_cell_length, index % unit_cell_length
                        with coocc_matrix[cell].get_lock():
                            coocc_matrix[cell][offset][n] = 1


        # Samples not in test set -> 0
        test_samples = list(set([x[0] for x in l for l in classes.iteritems()]))
        train_samples = np.setdiff1d(range(n_samples), test_samples)
        for x in train_samples:
            for y in xrange(i+1, n-samples):
                index = y + x*n_samples - (x+1)*(x+2)/2 if x < y else x + y*n_samples - (y+1)*(y+2)/2
                cell, offset = index / unit_cell_length, index % unit_cell_length
                with coocc_matrix[cell].get_lock():
                    coocc_matrix[cell][offset][n] = 0


    #Update number of label occurrences
    total_occurrences -= alpha
    total_occurrences[total_occurrences != 0] = 1
    label_occurrences += total_occurrences

    return len(classes), [len(x) for x in classes.itervalues()], b
