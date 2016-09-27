#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**convergence_analysis.py.** Study of the convergence through the evolution of the Spearman and Pearson correlation + measure of the mAP value every 5 iteration.


Convergence analysis
^^^^^^^^^^^^^^^^^^^^

**Usage:** ::

	python convergence_analysis.py -N [1] -t [2] -d [3] -c [4] -ts [5] -s [6] -nmin [7] -nmax [8] -di [9] -o [10] -te [11] -in [12] -g [13] -cfg [14] -v [15] --debug --help

where:
 * Default options are found in the ``configuration.ini`` file.
 * [1] ``-i, --iter``: number of classification iterations.
 * [2] ``-t, --threads``: number of cores to use.
 * [3] ``-d, --dataset``: dataset to use.
 * [4] ``-c, --classifier``: classifier to use.
 * [5] ``-ts, --trainsize``: proportion of dataset to use for training.
 * [6] ``-s, --sim``: similarity type to use (EM not supported).
 * [7] ``-nmin``: minimum number of synthetic labels.
 * [8] ``-nmax``: maximum number of synthetic labels.
 * [9] ``-di, --distrib``: synthetic annotation mode (RND, UNI, OVA).
 * [10] ``-o, --output``: output folder.
 * [11] ``-te, --temp``: temporary folder.
 * [12] ``-in``: input data file.
 * [13] ``-g, --ground``: ground-truth file.
 * [14] ``-cfg, --config_file``: provide a custom configuration file.
 * [15] ``-v, --verbose``: controls verbosity level (0 to 4).
 * ``-db, --debug``: debug mode (save temporary files).
 * ``-h, --help``


Computes ``N`` iterations of SIC and compares the final similarity matrix to partial matrices in past iterations (see ``steps`` in ``convergence_analysis.py``).
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import os
import sys
import time
import signal
import numpy as np
from scipy.stats import pearsonr, spearmanr
from multiprocessing import sharedctypes, Process, Queue

from utils.read_config import read_config_file, parse_cmd_line
from utils.error import signal_handler
from utils.parse import parse_data
from utils.output_format import log, Tee, init_folder
from utils.annotation import annotate, choose_features
from utils.classify import do_classify_step
from utils.one_step import split_data
from evaluation_retrieval import evaluate


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)               # Hide Keyboard Interrupt trace

    #----------------------------------------------------------- PARAMETERS
    # Read configuration file for default options + set cmd line options
    main_dir = os.path.dirname(os.path.realpath(__file__))
    custom, cfg, verbose, debug = parse_cmd_line()
    config_file = os.path.join(main_dir, 'configuration.ini')
    if cfg is not None:
        config_file = cfg
    n_iter, n_cores, _, data_type, input_file, ground_truth_file, temp_folder, output_folder, annotation_params, classification_params, classifier_binary, _, cvg_step, cvg_criterion, cfg = read_config_file(config_file, *custom)
    base_name = time.strftime("%Y-%m-%d_%H-%M")
    output_folder =  init_folder(os.path.join(output_folder, 'Outputs_cvg_%s' % base_name))
    temp_folder = init_folder(temp_folder)
    eval_folder = init_folder(os.path.join(output_folder, 'MapEval'))

    # Save configuration for the experiment in the output folder
    with open(os.path.join(output_folder, 'exp_configuration.ini'), 'w') as f:
        cfg.write(f)
    os.environ['CLASSIFIER'] = classifier_binary

    # Read and parse data file
    n_samples, data, _, index_to_label, label_to_index = parse_data(data_type, classification_params, input_file)

    # Redirect output to log file + print summary
    sys.stdout, logf = log(output_folder)
    print '%d sequential iterations' % n_iter
    print '%d samples in %s' % (n_samples, input_file)
    print 'Output Folder: %s\n' % output_folder

    # Enforced parameters
    classification_params['similarity_type'] = 'BIN'
    map_save_step = 5 # save mAP values every ``step`` iteration
    steps = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 210, 230, 250, 270] # iterations to consider for the correlation comparisons
    with_unique_occurrences = (data_type in ['NER', 'AUDIO', 'AUDIOTINY'])
    with_common_label_wordform = (data_type == 'NER' or data_type == 'AQUA')
    preclustering = index_to_label if (data_type in ['NER', 'AQUA']) else []

    #----------------------------------------------------------- RUN THREADS
    # Initialize shared array
    total_length = n_samples * (n_samples - 1) / 2
    co_occ = [sharedctypes.Array(np.ctypeslib.ctypes.c_long, np.zeros(total_length, dtype='int32'), lock=True)]
    count_lab = np.zeros(n_samples)
    synthetic_labels = [0] * n_iter
    compares = {}

    # Sequential loop to build the similarity
    for n in xrange(n_iter):
        # Split + Annotation
        train, test, _ = split_data(n, data, data_type, classification_params['training_size'], classification_params['classifier_type'], annotation_params, temp_folder, with_full_test=False)

        N, n_trainlabels, train_data, test_data, test_entities_indices, summary = annotate(n, temp_folder, classification_params['classifier_type'], (train, test), annotation_params, with_common_label_wordform=with_common_label_wordform, verbose=verbose, debug=debug, preclustering=preclustering)
        if verbose >= 1:
            print >> sys.stderr, summary

        # Classification
        n_testlabels, repartition, _ = do_classify_step(n, temp_folder, train_data, test_data, test_entities_indices, co_occ, total_length, count_lab, classification_params, verbose = verbose, debug=debug, with_unique_occurrences = with_unique_occurrences)

        # Map evaluation
        if (n+1) % 5 == 0:
            truc = np.zeros((n_samples, n_samples))
            with co_occ[0].get_lock():
                truc[np.triu_indices(n_samples, k = 1)] = np.array(np.frombuffer(co_occ[0].get_obj())) / (n+1)
                evaluate(data_type, truc, label_to_index, index_to_label, ground_truth_file, eval_folder, suffix='step%d' % (n+1))

        # Save matrix for correlation comparison
        if (n+1) in steps:
            print 'Save step %d' %(n+1)
            with co_occ[0].get_lock():
                compares[n+1] = np.array(np.frombuffer(co_occ[0].get_obj())) / (n+1)

    # Format and save last (optimal) similarity matrix
    limit = np.array(np.frombuffer(co_occ[0].get_obj())) / n_iter
    np.save(os.path.join(output_folder, 'sim_matrix_1D'), limit)

    # Compare
    for n, mat in compares.iteritems():
        print 'Iteration %d' % n
        print '> Pearson Coefficient:', pearsonr(mat, limit)
        print '> Spearman Coefficient:', spearmanr(mat, limit)
        print '> Frobenius Norm:', np.linalg.norm(limit - mat)

    logf.close()
