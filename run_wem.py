#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**run_wem.** Computing the similarity matrix in the WEM setting.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import os
import gc
import sys
import numpy as np
import threading
from Queue import Queue as Queue_thread
from ctypes import Structure, c_int
from multiprocessing import Process, Queue, sharedctypes, Array

from utils.parse import ground_truth_pairs
from utils.one_step import thread_step
from utils.em_analysis import estimate_parameters_em
from utils.output_format import save_coocc



def compute_similarity_wem(n_iter, n_cores, n_locks, data_type, n_samples, data, ground_truth_file, temp_folder, output_folder, annotation_params, classification_params, verbose, debug, with_unique_occurrences, preclustering):
    """
    Computes the simlarity matrix in the basic setting.

    Args:
     * ``n_iter`` (*int*): number of iterations.
     * ``n_cores`` (*int*): number of cores to use.
     * ``n_locks`` (*int*): number of locks to use on the full shared matrix.
     * ``data_type`` (*str*): data set to use.
     * ``n_samples`` (*int*): number of samples.
     * ``data`` (*struct*): initial data.
     * ``ground_truth_file`` (*str*): path to ground-truth.
     * ``temp_folder`` (*str*): path to the temporary folder.
     * ``output_folder`` (*str*): path to the output folder.
     * ``annotation_params`` (*list*): parameters for synthetic annotation.
     * ``classification_params`` (*list*): parameters for classification.
     * ``verbose`` (*int*): sets the verbosity level.
     * ``with_unique_occurrences`` (*bool*): indicates wether samples occur only once in the data set or not.
     * ``with_common_label_wordform`` (*bool*): indicates wether to give the same label to samples with identical wordform or not.

    Returns:
     * ``n_samples_occurrences`` (*list*): number of occurrences of each sample in a test set over all iterations.
     * ``synthetic_labels`` (*list*): synthetic labels repartition for each iteration.
     * ``co_occ`` (*ndarray*): full similarity matrix.
    """

    #Additional variables for estimating the different parameters
    positive_pairs = ground_truth_pairs(data_type, ground_truth_file, n_samples)
    true_pi0 = 1.0 - float(2 * len(positive_pairs)) / (n_samples * (n_samples - 1))
    true_p1 = [0] * n_iter
    true_p0 = [0] * n_iter
    indep_p0 = [0] * n_iter

    #----------------------------------------------------------- SHARED MEMORY
    # Create shared array with ctype structure
    print '> Creating Shared Arrays'
    n_locks = n_samples * 2 if n_locks == -1 else n_locks
    total_length = n_samples * (n_samples - 1) / 2
    cell_length, rst = total_length / n_locks, total_length % n_locks

    if rst > 0:
        n_locks += rst / cell_length
        rst = rst % cell_length
        if rst > 0:
            n_locks += 1

    # An object of type Cell contains the sequence of iteration scores for one pair of samples
    UnitcellType = n_iter * c_int
    shared_coocc = [sharedctypes.Array(UnitcellType, [UnitcellType(*(2 for _ in xrange(n_iter))) for _ in xrange(cell_length)], lock=True) for x in xrange(n_locks)]

    #----------------------------------------------------------- INITIAL WORKER PROCESSES
    threads = {}
    sim_queue = Queue()
    iterations = range(n_iter)
    n_samples_occurrences = np.zeros(n_samples)
    synthetic_labels = []
    n_iterations_done = 0

    #Start initial processes
    for c in xrange(n_cores):
        t =  Process(name = 'Worker %d'%c,
                     target = thread_step,
                     args=(iterations.pop(0),
                           shared_coocc,
                           cell_length, n_samples,
                           sim_queue, data, temp_folder, data_type,
                           annotation_params, classification_params,
                           verbose, debug, with_unique_occurrences, preclustering)
                           )
        threads[t.name] = t
        t.start()

    #----------------------------------------------------------- THREADED ADDITIONAL COMPUTATIONS
    ############################################################################################
    def aux_computation_WEM(incoming_queue, n_iter, shared_coocc, total_length, positive_pairs, true_p1, true_p0, indep_p0):
        """
        Additional computations for the EM similarity
        """

        for k in xrange(n_iter):
            (iter, id, n_steps, counts, synth_lab) = incoming_queue.get()

            #Compute Indep p0
            repartition = synth_lab[-1]
            indep_p0[n_steps] = float(sum([x**2 for x in repartition])) / sum(repartition)**2

            print 'Computation %d done' %(iter+1)

    ############################################################################################

    incoming_queue = Queue_thread()
    aux_thread = threading.Thread(target=aux_computation_WEM, args = (incoming_queue, n_iter, shared_coocc, total_length, positive_pairs, true_p1, true_p0, indep_p0))
    aux_thread.start()

    # Retrieve results from queue and restat threads if needed
    for k in xrange(n_iter):
        #---- Retrieve and restart
        (id, n_steps, counts, synth_lab, _, b) = sim_queue.get()
        threads[id].join()

        #Launch Compute thread
        incoming_queue.put((k, id, n_steps, counts, synth_lab))

        #Start next process
        if len(iterations) > 0:
            t =  Process(name = id, target = thread_step, args=(iterations.pop(0), shared_coocc, cell_length, n_samples, sim_queue, data, temp_folder, data_type, annotation_params, classification_params, verbose, debug, with_unique_occurrences, preclustering))
            threads[id] = t
            t.start()
        else:
            del threads[id]

        #---- Additional Computations
        n_samples_occurrences += counts
        synthetic_labels.append(synth_lab)

        #----- Verbose outputs
        if verbose >= 1:
            n_iterations_done += 1
            print >> sys.stderr, '\n>>>> %d/%d iterations done\n' %(n_iterations_done, n_iter)




    # Join remaining processes (normally, they are processe that crashed)
    for i, t in enumerate(threads.values()):
        t.join()
        if verbose >= 1:
            print 'joined thread %d'%i

    aux_thread.join()
    gc.collect()

    #------------------------------------------------------------------------- POST PROCESS MATRIX
    ####### Reshapping the result into a true numpy matrix
    print '> Reshaping co_occurence matrix'
    #flat_cooc = np.hstack(shared_coocc)[:total_length]
    print 'Long format'
    flat_cooc = np.zeros(total_length, dtype='object')
    i = 0
    for x in shared_coocc:
        for y in x:
            if i >= total_length:
                break
            flat_cooc[i] = long(''.join([str(z) for z in y]))
            i += 1
    del shared_coocc


    # True parameters
    print 'Compute ground-truth paramters'
    for n_steps in xrange(n_iter):
        to_iter = np.vectorize(lambda x: int(str(x).zfill(n_iter)[n_steps]))
        mat = to_iter(flat_cooc)
        true_p1[n_steps] = float(len(np.where(mat[positive_pairs] == 1)[0])) / len(positive_pairs)
        true_p0[n_steps] = float(len(np.where(mat[np.setdiff1d(range(total_length), positive_pairs)] == 1)[0])) / (total_length - len(positive_pairs))
        del mat

    ###### EM Estimation
    print 'Running EM'
    z1, pi0, p0, p1 = estimate_parameters_em(flat_cooc, n_iter, p1i=0.9, p0i=0.1, pi0i=0.8, n_iter=25, cores=n_cores)

    # Estimates
    print 'Estimated parameters (pi0, p0, p1) %s \n %s \n %s' %(pi0, p0, p1)
    print 'True parameters (pi0, p0, p1) %s \n %s \n %s' %(true_pi0, true_p0, true_p1)
    print 'Independant parameters p0  \n  %s \n ' %(indep_p0)
    a = [0 if p == 0 else np.sqrt((1-p)/p) for p in p0]
    b = [0 if p == 0 else - np.sqrt(p/(1-p)) for p in p0]

    # Similarities
    if verbose >= 2:
        print 'Binary sim'
        to_bin_sim = np.vectorize(lambda x: float(str(long(x)).count(1)) / float(str(long(x)).count(2)))
        save_coocc(output_folder, to_bin_sim(flat_cooc).astype(np.float), suffix='binary_final_flat')

        print 'WBIN with EM weights'
        to_wbinem_sim = np.vectorize(lambda x: np.sum([a[i] if int(y) == 1 else b[i] if int(y) == 2 else 0 for y in str(long(x))]) / float((str(long(x)).count(2))))
        save_coocc(output_folder, to_wbinem_sim(flat_cooc).astype(np.float), suffix='weighted_final_flat')

    print 'EM similarities'
    to_em_sim = np.vectorize(lambda x: z1[str(long(x)).zfill(n_iter)])
    co_occ = np.zeros((n_samples, n_samples), dtype=float)
    co_occ[np.triu_indices(n_samples, k = 1)] = to_em_sim(flat_cooc)

    return n_samples_occurrences, synthetic_labels, co_occ, None
