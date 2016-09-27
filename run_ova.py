#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**run_ova.** Computing the similarity matrix in the basic setting (BIN, WBIN, WUBIN).
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import os
import gc
import sys
import time
import numpy as np
import pickle
import threading
from random import shuffle
from functools import partial
from itertools import chain
from Queue import Queue as Queue_thread
from multiprocessing import Process, Queue, sharedctypes, Array, Pool

from evaluation_retrieval import evaluate as eval_retrieval
from utils.parse import parse_ground_truth, ground_truth_indices, parse_AQUA_single_partial
from utils.one_step import thread_step


def load_file(file, input_file, train_docs, label_to_index, classification_params):
    """
    Parse an Aquaint file to retrieve all occurrences of a single words. Used for a threaded execution in a Pool.

    Args:
     * ``file`` (*str*): name the file to parse.
     * ``input_file`` (*str*): directory of ``file``.
     * ``train_docs`` (*dict*): dict mapping a file to the documents containing the considered word.
     * ``label_to_index`` (*dict*): maps a label to the corresponding entity index.
     * ``classification_params`` (*dict*): classification parameters.

    Returns:
     * ``samples`` (*list*): list of tagged sentences containing the considered word in the file.
    """
    training_docs_indices = train_docs[file]
    samples, _ = parse_AQUA_single_partial(classification_params['classifier_type'], os.path.join(input_file, file), label_to_index, training_docs_indices, 0, (), (), [], train_included_test=False)
    return list(samples)



def compute_similarity_ova(n_iter, n_cores, n_locks, input_file, ground_truth_file, data_type, n_samples, data, data_occurrences, index_to_label, label_to_index, temp_folder, output_folder, annotation_params, classification_params, verbose, debug, with_unique_occurrences, preclustering, gtonly=True):
    """
    Computes the simlarity matrix in the basic setting.

    Args:
     * ``n_iter`` (*int*): number of iterations.
     * ``n_cores`` (*int*): number of cores to use.
     * ``n_locks`` (*int*): number of locks to use on the full shared matrix.
     * ``data_type`` (*str*): data set to use.
     * ``n_samples`` (*int*): number of samples.
     * ``data`` (*struct*): initial data.
     * ``data_occurrences`` (*struct*): indicates where each of the samples occurs in the data base.
     * ``index_to_label`` (*list*): maps a sample's index to a string representation.
     * ``label_to_index`` (*dict*): reverse index_to_label mapping.
     * ``temp_folder`` (*str*): path to the temporary folder.
     * ``output_folder`` (*str*): path to the output folder.
     * ``annotation_params`` (*list*): parameters for synthetic annotation.
     * ``classification_params`` (*list*): parameters for classification.
     * ``verbose`` (*int*): sets the verbosity level.
     * ``with_unique_occurrences`` (*bool*): indicates wether samples occur only once in the data set or not.
     * ``with_common_label_wordform`` (*bool*): indicates wether to give the ame label to samples with identical wordform or not.
     * ``writing_steps`` (*list*): steps at which to save the partial matrix.
     * ``gt_only`` (*boolean, optional*): if True, experiments will only be conducted from query samples from the ground-truth. Defqults to True.

    Returns:
     * ``n_samples_occurrences`` (*list*): number of occurrences of each sample in a test set over all iterations.
     * ``synthetic_labels`` (*list*): synthetic labels repartition for each iteration.
     * ``co_occ`` (*ndarray*): full similarity matrix.
     """

    # Additional parameters
    full_matrix = np.zeros((n_samples, n_samples))
    n_locks, rst, cell_length = 1, 0, n_samples
    ground_truth = parse_ground_truth(data_type, ground_truth_file, label_to_index)
    sampleindx = ground_truth_indices(data_type, ground_truth_file, label_to_index) if gtonly else xrange(n_samples)
    shuffle(sampleindx)

    #################################### LOOP OVER EACH SAMPLE
    for train_sample in sampleindx:

        name = index_to_label[train_sample]
        print "Iteration for sample %d - %s"%(train_sample, name)
        iter_annotation_params = annotation_params.copy()
        iter_annotation_params['sample'] = 'B-%d'%train_sample

        print 'Pre-parsing train sentences'
        data_files = [f for f in os.listdir(input_file) if f.endswith('.xml.u8')]
        with open(os.path.join(iter_annotation_params['ova_occurrences'], '%s.pkl' %(name)), 'rb') as f:
            train_docs = pickle.load(f)
            train_files = train_docs.keys()

        partial_load_file = partial(load_file, input_file=input_file, train_docs=train_docs, label_to_index=label_to_index, classification_params=classification_params)
        pool = Pool(processes=n_cores)
        samples = pool.map(partial_load_file, train_files)
        train = ()
        for tl in samples:
            train = chain(train, tl)
        del samples
        iter_annotation_params['ova_occurrences'] = (list(train), train_files)

        # Shared Array
        print 'Creating Shared Array'
        shared_coocc = [sharedctypes.Array(np.ctypeslib.ctypes.c_double, np.zeros(n_samples), lock=True)]
        test_occurrences = np.zeros(n_samples, dtype=int)


        #----------------------------------------------------------- INITIAL WORKER PROCESSES
        threads = {}
        sim_queue = Queue()
        iterations = range(n_iter)
        n_samples_occurrences = np.zeros(n_samples)
        synthetic_labels = []
        n_iterations_done = 0
        score_penalty = 0.

        # Start initial processes
        for c in xrange(n_cores):
            t =  Process(name = 'Worker %d'%c,
                         target = thread_step,
                         args=(iterations.pop(0),
                               shared_coocc,
                               cell_length, n_samples,
                               sim_queue, data, temp_folder, data_type,
                               iter_annotation_params, classification_params,
                               verbose, debug, with_unique_occurrences, preclustering)
                               )
            threads[t.name] = t
            t.start()


        #----------------------------------------------------------- THREADED ADDITIONAL COMPUTATIONS
        ############################################################################################
        def aux_computation_OVA(incoming_queue, n_iter, test_occurrences, data_occurrences):
            """
            Additional computations for the OVA annotation
            """

            for k in xrange(n_iter):
                (_, _, _, _, _, mat) = incoming_queue.get()

                #Updating number of test occurences
                if mat != None:
                    if data_type == 'NER': #In NER #test ~ #entities + sparse matrices
                        mat = np.sum([data_occurrences[x] for x in mat])

                    mat[mat != 0] = 1
                    test_occurrences += mat


        ############################################################################################
        incoming_queue = Queue_thread()
        aux_thread = threading.Thread(target=aux_computation_OVA, args = (incoming_queue, n_iter, test_occurrences, data_occurrences,))
        aux_thread.start()


        #--------- Retrieve results from queue and restat threads if needed
        for k in xrange(n_iter):
            #---- Retrieve and restart
            (id, n_steps, counts, synth_lab, mat, b) = sim_queue.get()
            threads[id].join()

            #Launch Compute thread
            incoming_queue.put((k, id, n_steps, counts, synth_lab, mat))

            #Start next process
            if len(iterations) > 0:
                t =  Process(name = id, target = thread_step, args=(iterations.pop(0), shared_coocc, cell_length, n_samples, sim_queue, data, temp_folder, data_type, iter_annotation_params, classification_params, verbose, debug, with_unique_occurrences, preclustering))
                threads[id] = t
                t.start()
            else:
                del threads[id]

            #---- Additional Computations
            n_samples_occurrences += counts
            synthetic_labels.append(synth_lab)
            score_penalty += b

            #----- Verbose outputs
            del mat
            if verbose >= 1:
                n_iterations_done += 1
                print >> sys.stderr, '\n>>>> %d/%d iterations done\n' % (n_iterations_done, n_iter)



        #------------ Join remaining processes if any
        for i, t in enumerate(threads.values()):
            t.join()
            if verbose >= 1:
                print 'joined thread %d'%i
        aux_thread.join()
        gc.collect()

        with shared_coocc[0].get_lock():
            arr = np.frombuffer(shared_coocc[0].get_obj())

        # Normalizing
        test_occurrences[test_occurrences == 0] = 1.
        co_occ = arr / test_occurrences
        full_matrix[train_sample, :] = arr

        # Output sorted distribution for the current sample if verbosity level high enough
        if verbose >= 2:
            gt = ground_truth[name]
            sorted_ind = np.argsort(co_occ)
            with open(os.path.join(output_folder, '%s_sorted_sim.txt'%name), 'w') as f:
                f.write('\n'.join(['%s %s\t%d-%s\t%s\t%s\t%s'%('>' if index_to_label[i] in gt else '', name, i, index_to_label[i], co_occ[i], arr[i], test_occurrences[i]) for i in sorted_ind]))

        # Output mAP for the current sample
        if verbose >= 2:
            print "######## Result for iteration %d - %s"%(train_sample, name)
            eval_retrieval(data_type, co_occ, label_to_index, index_to_label, ground_truth_file, output_folder, ova=train_sample, writing=True)

    return n_samples_occurrences, synthetic_labels, arr, None
