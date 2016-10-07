#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**one_step.py.** one classification step for building the similarity. Designed for a threaded execution.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import os
import sys
import warnings
import traceback
import numpy as np
from collections import defaultdict
from random import seed, shuffle, choice
from multiprocessing import current_process
from annotation import annotate
from classify import do_classify_step
from parse import parse_AQUA_single_partial

sys.path.append('../Data/Audio/Scripts')
from extract_features import extract_features



def thread_step(n, coocc_matrix, unit_cell_length, n_samples, sim_queue, data, temp_folder, data_type, annotation_params, classification_params, verbose=1, debug=False, with_unique_occurrences=False, preclustering=[]):
    """
    One classification iteration. Results are output in a queue.

    Args:
     * ``n`` (*int*): index of current iteration.
     * ``coocc_matrix`` (*array*): similarity matrix (shared in memory).
     * ``unit_cell_length`` (*int*): length of one locked cell in the matrix.
     * ``n_samples`` (*int*): number of samples in the data set.
     * ``sim_queue`` (*Queue*): output queue.
     * ``data`` (*list*): initial data structure.
     * ``temp_folder`` (*str*): path to temporary folder.
     * ``data_type`` (*str*): data set used.
     * ``annotation_params`` (*dict*): parameters for the synthetic annotation.
     * ``classification_params`` (*dict*): parameters for the supervised classification algorithm.
     * ``verbose`` (*int, optional*): controls the verbosity level. Defaults to 1.
     * ``debug`` (*bool, optional*): runs in debugging mode. Defaults to False.
     * ``with_unique_occurrences`` (*bool, optional*): ``True`` when occurrences of a same entity are distinct items in the database (e.g. NER). Defaults to ``False``.
     * ``preclustering`` (*list, optional*): Entity index to class mapping. This clustring is used to given the same annotation to entities in the same class.
    """

    seed()
    thread_id = current_process().name
    label_occurrences = np.zeros(n_samples)
    if verbose >= 2:
        print 'Start %s' % thread_id

    try:
        # Split data
        train, test, test_indices = split_data(n, data, data_type, classification_params['training_size'], classification_params['classifier_type'], annotation_params, temp_folder, with_full_test=False)

        # Annotate data
        n_labels, n_trainlabels, train_file, test_file, test_entities_indices, summary = annotate(n, temp_folder, classification_params['classifier_type'], (train, test), annotation_params, verbose=verbose, debug=debug, preclustering=preclustering)

        if verbose >= 1:
            print >> sys.stderr, summary
            if not with_unique_occurrences:
                print ' > %d unique entities in test' % (len(np.where(test_indices != 0)[0]))

        # Classify
        n_testlabels, repartition, b = do_classify_step(n, temp_folder, train_file, test_file, test_entities_indices, coocc_matrix, unit_cell_length, label_occurrences, classification_params, verbose=verbose, debug=debug, with_unique_occurrences=with_unique_occurrences)
        synthetic_labels = (n, n_labels, n_trainlabels, n_testlabels, repartition)

        sim_queue.put((thread_id, n, label_occurrences, synthetic_labels, None if classification_params['similarity_type'] == 'WEM' else test_indices, b))

    except Exception as e:
         if verbose >= 1:
             print >> sys.stderr, '\n>WARNING: %s, step %d: aborted' % (thread_id, n)
             traceback.print_exc()
         sim_queue.put((thread_id, n, label_occurrences, (n, 0, 0, 0, {}), None, 0))

    if verbose >= 1:
        print >> sys.stderr, '> %s, step %d: done' % (thread_id, n)

    sim_queue.close()
    sim_queue.join_thread()




def split_data(n, data, data_type, train_frac, classifier_type, annotation_params, temp_folder, with_full_test=False):
    """
    Splits the given database into a training and testing set.

    Args:
     * ``n`` (*int*): iteration identifier.
     * ``data`` (*list*): initial data structure.
     * ``data_type`` (*str*): data set used for the experiments.
     * ``train_frac`` (*float*): proportion of the database to keep for training.
     * ``classifier_type`` (*str*): type of classifier (for on-the-fly parsing format in AQUAINT).
     * ``annotation_params`` (*str*): annotation parameter for OVA.
     * ``temp_folder`` (*str*): path to temporary folder.
     * ``with_full_test`` (*bool, optional*): if ``True``, use the whole dataset (including training) for testing.

    Returns:
     * ``train`` (*list*): the data kept for training (generator).
     * ``test`` (*list*): the data kept for testing (generator).
     * ``test_indices`` (*list*): indices of the test samples in the whole data; used to compute the number of test occurrences afterwards. (Except for AQUAINT where test_indices directly returns the occurrences of each samples in the test set).
    """

    ####################################################################### AQUAINT
    if data_type == 'AQUA':
        # Load data files
        data_folder, label_to_index, docs_scores = data
        data_files = [f for f in os.listdir(data_folder) if f.endswith('.xml.u8')]
        shuffle(data_files)
        with_ova_annotation = (annotation_params['distrib'] == 'OVA')
        train, test = (), ()
        test_indices = np.zeros(len(label_to_index))

        # Traing and Testing sizes in basic setting
        n_doc_train = int(train_frac * len(data_files))
        n_doc_test = 5 * n_doc_train
        train_files = data_files[:n_doc_train]
        test_files = data_files[:n_doc_test]
        docs_per_test_files = int(500. / n_doc_test) # nbr of documents loaded per test files
        docs_per_train_files = int(50.0 / n_doc_train) # nbr of documents loaded per train files

        # Traing and Testing sizes in OVA setting
        if with_ova_annotation:
            #If OVA, constrain the training set
            train_sentences, train_files = annotation_params['ova_occurrences']
            n_train_sentences = choice([40, 100, 200, 600, 1000])
            shuffle(train_sentences)
            train = (x for x in train_sentences[3:n_train_sentences])

            # Choose test files
            shuffle(train_files)
            test_files = train_files[:n_doc_test]

            # Add additional (negative) training samples (~same number of sentences as the original training set)
            train_files = train_files[:n_doc_train]
            docs_per_train_files = 3 * int((float(n_train_sentences) / 20) / n_doc_train)

            # Add some train sentences in the testing database to check that s(x, x) = 1
            test = (x for x in train_sentences[:3])
            for s in train_sentences[:3]:
                for _, lem, _, _, bio in s:
                    if bio == 'B':
                        test_indices[label_to_index[lem]] += 1

        # Full test (not recommanded)
        if with_full_test:
            test_files = datat_files
            docs_per_test_files = -1 # retrieve all documents in the file

        # Number of files to parse before breaking the loop
        seen_test_files = len(test_files)
        seen_train_files = len(train_files)


        # Parsing
        print 'Step %d: Parsing...' % n
        for i, file in enumerate(data_files):
            if (seen_test_files == 0) and (seen_train_files == 0):
                break

            # Choose documents for train
            if (seen_train_files > 0) and (file in train_files):
                training_docs_indices = docs_per_train_files
                seen_train_files -= 1
            else:
                training_docs_indices = 0

            # Choose documents for test
            if seen_test_files > 0 and  file in test_files:
                testing_docs_indices = int(docs_scores[file] * docs_per_test_files)
                seen_test_files -= 1
            else:
                testing_docs_indices = 0

            # Parse
            if training_docs_indices == 0 and testing_docs_indices == 0:
                continue
            else:
                train, test = parse_AQUA_single_partial(classifier_type, os.path.join(data_folder, file), label_to_index, training_docs_indices, testing_docs_indices, train, test, test_indices)


    ######################################################### NER
    elif data_type == 'NER':
        data_indices = range(len(data))
        shuffle(data_indices)
        training_size = int(train_frac * len(data_indices))

        train = (data[x] for x in data_indices[:training_size])
        if with_full_test:
            test = (data[x] for x in data_indices)
            test_indices = data_indices
        else:
            test = (data[x] for x in data_indices[training_size:])
            test_indices = data_indices[training_size:]


    ####################################################### AUDIO
    elif data_type == 'AUDIO':
        # Select features
        features_type = choice([x for x in data.keys() if x != 'Samples'])
        print 'Features Choice: %s' % features_type
        selected_data = data[features_type]

        # On the fly features generation if nto precomputed
        if 'Samples' in data:
            print 'Generating Features in %s'%temp_folder
            features_folder = extract_features(data['Samples'], temp_folder, features_type, temp_folder, binary_path=os.environ.copy()['CLASSIFIER'], n=n)
            selected_data = [(i, os.path.join(features_folder, x)) for i, x in enumerate(sorted(f for f in os.listdir(features_folder) if f.endswith('.mfc')))]

        # Split features into train and test
        data_indices = range(len(selected_data))
        shuffle(data_indices)
        training_size = int(train_frac * len(data_indices))
        train = (selected_data[x] for x in data_indices[:training_size])
        if with_full_test:
            test = (selected_data[x] for x in data_indices)
            test_indices = data_indices
        else:
            test = (selected_data[x] for x in data_indices[training_size:])
            test_indices = data_indices[training_size:]

    return train, test, test_indices
