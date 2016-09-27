#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running the experiments: ``python main.py --help`` displays tips about the command line options. Default options are found in the ``configuration.ini`` file.


**Usage:** ::

	python main.py -N [1] -t [2] -d [3] -c [4] -ts [5] -s [6] -nmin [7] -nmax [8] -di [9] -p [10] -cs [11] -cc [12] -o [13] -te [14] -in [15] -g [16] -cfg [17] -v [18] --debug --help

where:
 * Default options are found in the ``configuration.ini`` file.
 * [1] ``-i, --iter``: number of classification iterations.
 * [2] ``-t, --threads``: number of cores to use.
 * [3] ``-d, --dataset``: dataset to use.
 * [4] ``-c, --classifier``: classifier to use.
 * [5] ``-ts, --trainsize``: proportion of dataset to use for training.
 * [6] ``-s, --sim``: similarity type to use.
 * [7] ``-nmin``: minimum number of synthetic labels.
 * [8] ``-nmax``: maximum number of synthetic labels.
 * [9] ``-di, --distrib``: synthetic annotation mode (RND, UNI, OVA).
 * [10] ``-p, --post``: post-processing task/algorithm.
 * [11] ``-cs, --cvg_step``: check convergence criterion every ``cs`` step.
 * [12] ``-cc, --cvg_criterion``: convergence criterion threshold.
 * [13] ``-o, --output``: output folder.
 * [14] ``-te, --temp``: temporary folder.
 * [15] ``-in``: input data file.
 * [16] ``-g, --ground``: ground-truth file.
 * [17] ``-cfg, --config_file``: provide a custom configuration file.
 * [18] ``-v, --verbose``: controls verbosity level (0 to 4).
 * ``-db, --debug``: debug mode (save temporary files).
 * ``-h, --help``


 **Main outputs:**

 * *output.log*: log file
 * *sim_matrix_final.npy*: similarity matrix.


 **Verbosity levels:**

 * ``-v 0``: minimal verbose level; almost no printed trace.
 * ``-v 1``: Default.
 * ``-v 2``: Additional print trace.
 * ``-v 3``: Prints out the classifier's traces.
 * ``-v 4``: Outputs additional result (distributions plots, number of occurences in test for each entity ...) + save similarity matrix regularly.

"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import os
import sys
import time
import signal
import resource
import numpy as np

from utils.read_config import read_config_file, parse_cmd_line
from utils.error import signal_handler
from utils.parse import parse_data, parse_ground_truth
from evaluation_clustering import evaluate as eval_cluster
from evaluation_retrieval import evaluate as eval_retrieval
from utils.output_format import log, Tee, init_folder, save_coocc
from utils.matrix_op import normalize

from run_ova import compute_similarity_ova
from run_wem import compute_similarity_wem
from run_basic import compute_similarity_basic


if __name__ == "__main__":
    start_time, start_resources = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    signal.signal(signal.SIGINT, signal_handler)

    ####################################################################### PARAMETERS
    # Read configuration file for default options + set cmd line options
    main_dir = os.path.dirname(os.path.realpath(__file__))
    custom, cfg, verbose, debug = parse_cmd_line()
    config_file = cfg if (cfg is not None) else os.path.join(main_dir, 'configuration.ini')
    n_iter, n_cores, n_locks, data_type, input_file, ground_truth_file, temp_folder, output_folder, annotation_params, classification_params, classifier_binary, task_params, cvg_step, cvg_criterion, cfg = read_config_file(config_file, *custom)

    # Set temporary and output folders
    base_name = time.strftime("%Y-%m-%d_%H-%M")
    output_folder = init_folder(os.path.join(output_folder, 'Outputs_%s'%base_name))
    temp_folder = init_folder(temp_folder)

    # Save configuration for the experiment in the output folder
    with open(os.path.join(output_folder, 'exp_configuration.ini'), 'w') as f:
        cfg.write(f)

    # Read and parse data file
    n_samples, data, data_occurrences, index_to_label, label_to_index = parse_data(data_type, classification_params, input_file)
    os.environ['CLASSIFIER'] = classifier_binary

    # Additional parameters
    with_unique_occurrences = (data_type in ['NER', 'AUDIO', 'AUDIOTINY']) # For optimisation purpose: tasks with only one occurrence per entity
    preclustering = list(index_to_label) if (data_type in ['NER', 'AQUA']) else [] # Assign same training label to entities with the same class in preclustering (here: same wordform if NER or AQUAINT)
    writing_steps = [500, 1000, 2000, 3000, 4000] # Save similarity matrix at each step in writing_steps, if verbose >= 4


    ####################################################################### PROGRAM SUMMARY
    # Starting Information
    sys.stdout, logf = log(output_folder)
    print '>>>> Program Summary'
    print ' %d iterations on %d cores' %(n_iter, n_cores)
    print ' %s similarity type' %(classification_params['similarity_type'])
    print ' %d samples in %s' %(n_samples, input_file)
    print ' %.3f%% training set' %(classification_params['training_size'] * 100)
    print ' Output Folder: %s\n' %output_folder


    ####################################################################### SIMILARITY CONSTRUCTION
    # Launch SIC algorithm
    if annotation_params['distrib'] == 'OVA':
        if verbose >= 1:
            print 'Launching script in OVA mode'
        n_samples_occurrences, synthetic_labels, co_occ, test_occurrences  = compute_similarity_ova(n_iter, n_cores, n_locks, input_file, ground_truth_file, data_type, n_samples, data, data_occurrences, index_to_label, label_to_index, temp_folder, output_folder, annotation_params, classification_params, verbose, debug, with_unique_occurrences, preclustering)

    elif classification_params['similarity_type'] == 'WEM':
        if verbose >= 1:
            print '>>>> Launching script in WEM mode'
        n_samples_occurrences, synthetic_labels, co_occ, test_occurrences  = compute_similarity_wem(n_iter, n_cores, n_locks, data_type, n_samples, data, ground_truth_file, temp_folder, output_folder, annotation_params, classification_params, verbose, debug, with_unique_occurrences, preclustering)

    else:
        if verbose >= 1:
            print '>>>> Launching script in basic mode'
        n_samples_occurrences, synthetic_labels, co_occ, test_occurrences = compute_similarity_basic(n_iter, n_cores, n_locks, data_type, n_samples, data, data_occurrences, temp_folder, output_folder, annotation_params, classification_params, verbose, debug, with_unique_occurrences, preclustering, writing_steps, cvg_step, cvg_criterion)


    ####################################################################### WRITE SIC OUTPUTS
    # 1. Main matrix Output : square matrix n_samples x n_samples
    if verbose >= 1:
        print '> Saving final similarity matrix'
        save_coocc(output_folder, co_occ, suffix='final')
        print 'Done saving matrix'

    # 2. Number of times each pair of samples occur in the same test set: 1D matrix
    if verbose >= 4 and test_occurrences != None:
        print '> Saving test occurrences'
        save_coocc(output_folder, test_occurrences, suffix='test_occ')

    # 3. Additional Outputs
    # Save number of times each entity was seen in a test set classified as non-null
    if verbose >= 3 and n_samples_occurrences != None:
        print '> Saving counts of entities seen at each test step'
        with open(os.path.join(output_folder, 'test_entities_counts.txt'), 'w') as f:
            f.write('\n'.join(['%d-%s\t%d'%(i,l,v) for i, (l, v) in enumerate(zip(index_to_label, n_samples_occurrences))]))

    # Save synthetic labels distribution for each iteration
    if verbose >= 3 and synthetic_labels != None:
        print '> Saving synthetic labels repartition for each iteration'
        with open(os.path.join(output_folder, 'label_counts.txt'), 'w') as f:
            f.write('Iteration\tTraining(theo)\tTraining\tTest\tRepartition\n')
            f.write('\n'.join(['%d\t%d\t%d\t%d\t%s'%(n,a,b,c,'\t'.join([str(x) for x in l])) for n,a,b,c,l in sorted(synthetic_labels, key=lambda x: x[0])]))


    ####################################################################### TIME STATS
    end_time, end_resources = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    children_resources = resource.getrusage(resource.RUSAGE_CHILDREN)
    print 'Building SIC similarity matrix - time usage:'

    print 'real', end_time - start_time, 's'
    print 'sys', (end_resources.ru_stime - start_resources.ru_stime) + children_resources.ru_stime, 's'
    print 'user', (end_resources.ru_utime - start_resources.ru_utime) + children_resources.ru_utime, 's'
    if verbose < 1:
        logf.close()
        raise SystemExit


    ####################################################################### POST-PROCESSING (Clustering/mAP)
    # Clustering
    if task_params['task_type'] == 'MCL':
        print '> Normalizing matrix'
        co_occ = normalize(co_occ)
        ground_truth = parse_ground_truth(data_type, ground_truth_file, label_to_index)
        print '%d ground-truth clusters' %len(ground_truth)
        cluster_folder = init_folder(os.path.join(output_folder, 'Clustering'))
        eval_cluster(co_occ, cluster_folder, temp_folder, ground_truth, index_to_label, n_cores, task_params, verbose=verbose)

    # KNN search
    elif task_params['task_type'] == 'KNN':
        eval_retrieval(data_type, co_occ, label_to_index, index_to_label, ground_truth_file, output_folder, -1)


    ####################################################################### END
    if not debug:
        try:
            import shutil
            shutil.rmtree(temp_folder)
        except OSError as e:
            print 'Error while removing %s \n  > %s' %(temp_folder, sys.exc_info()[0])

    logf.close()
