#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**run_basic.** Computing the similarity matrix in the basic setting (BIN, WBIN, WUBIN).
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import os
import gc
import sys
import time
import numpy as np
import threading
from Queue import Queue as Queue_thread
from multiprocessing import Process, Queue, sharedctypes, Array
from utils.one_step import thread_step


def compute_similarity_basic(n_iter, n_cores, n_locks, data_type, n_samples, data, data_occurrences, temp_folder, output_folder, annotation_params, classification_params, verbose, debug, with_unique_occurrences, preclustering, writing_steps, convergence_step, convergence_criterion):
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
     * ``temp_folder`` (*str*): path to the temporary folder.
     * ``output_folder`` (*str*): path to the output folder.
     * ``annotation_params`` (*list*): parameters for synthetic annotation.
     * ``classification_params`` (*list*): parameters for classification.
     * ``verbose`` (*int*): sets the verbosity level.
     * ``with_unique_occurrences`` (*bool*): indicates wether samples occur only once in the data set or not.
     * ``with_common_label_wordform`` (*bool*): indicates wether to give the same label to samples with identical wordform or not.
     * ``writing_steps`` (*list*): steps at which to save the partial matrix.
     * ``convergence_step`` (*int*): every 'convergence_step', the script checks if convergence is reached.

    Returns:
     * ``n_samples_occurrences`` (*list*): number of occurrences of each sample in a test set over all iterations.
     * ``synthetic_labels`` (*list*): synthetic labels repartition for each iteration.
     * ``co_occ`` (*ndarray*): full similarity matrix.
     """

    #----------------------------------------------------------- SHARED MEMORY
    # Create shared array
    print '> Creating Shared Array'
    n_locks = n_samples * 2 if n_locks == -1 else n_locks
    total_length = n_samples * (n_samples - 1) / 2
    cell_length, rst = total_length / n_locks, total_length % n_locks

    if rst > 0:
        n_locks += rst / cell_length
        rst = rst % cell_length
        if rst > 0:
            n_locks += 1

    shared_coocc = [sharedctypes.Array(np.ctypeslib.ctypes.c_double, np.zeros(cell_length), lock=True) for x in xrange(n_locks)]
    test_occurrences = np.zeros(total_length, dtype = int)


    #----------------------------------------------------------- INITIAL WORKER PROCESSES
    threads = {}
    sim_queue = Queue()
    iterations = range(n_iter)
    n_samples_occurrences = np.zeros(n_samples)
    synthetic_labels = []
    n_iterations_done = 0
    score_penalty = 0.
    convergence_values = []

    # Start initial processes
    for c in xrange(n_cores):
        t =  Process(name = 'Worker %d'%c,
                     target = thread_step,
                     args=(iterations.pop(0), shared_coocc,
                           cell_length, n_samples,
                           sim_queue, data, temp_folder, data_type,
                           annotation_params, classification_params,
                           verbose, debug, with_unique_occurrences, preclustering)
                           )
        threads[t.name] = t
        t.start()

    running_threads = len(threads)

    # Compute entropy values
    last_mean_ent = 0.



    #----------------------------------------------------------- THREADED ADDITIONAL COMPUTATIONS
    ############################################################################################
    def aux_computation(incoming_queue, convergence_check_queue, n_iter, n_samples, cell_length, total_length, shared_coocc, data_occurrences, test_occurrences, verbose, convergence_step, writing_steps, last_mean_ent, convergence_criterion, output_folder):
        """
        Additional computations for the BIN and WEIGHT similarity
        """

        for k in xrange(n_iter):
            (iter, id, n_steps, counts, synth_lab, mat) = incoming_queue.get()

            #  Updating number of test occurences
            if mat is not None:
                if data_type == 'NER':
                    mat = np.sum([data_occurrences[x] for x in mat])
                    in_train = np.setdiff1d(range(n_samples), mat.nonzero()[1])
                elif data_type in ['AQUA', 'AUDIO', 'AUDIOTINY']:
                    in_train = np.where(mat == 0)[0]

                test_occurrences += 1
                for i in in_train:
                    test_occurrences[i - 1 + np.cumsum([n_samples - np.arange(2, i+1)])] -= 1 # Column
                    test_occurrences[(i+1+i*n_samples - (i+1)*(i+2)/2):(n_samples - 1 + i*n_samples - (i+1)*(i+2)/2)] -= 1 # Line

            # Convergence check if required
            if convergence_step > 1 and k % convergence_step == 1:
                print 'Checking Convergence'
                entropies = np.zeros(total_length)
                # Lock
                for cell in shared_coocc:
                    cell.get_lock().acquire()

                # Compute entropies
                for v, cell in enumerate(shared_coocc):
                    start_ind = v*cell_length
                    if v == len(shared_coocc) -1 and rst > 0:
                        sim = np.array(np.frombuffer(cell.get_obj()))[:rst] / np.clip(test_occurrences[start_ind:], 1, n_iter)
                    else:
                        sim = np.array(np.frombuffer(cell.get_obj())) / np.clip(test_occurrences[start_ind:(start_ind + cell_length)], 1, n_iter)
                    entropies[start_ind:(start_ind + cell_length)] = [ - np.sum(st * np.log(st) + ( 1. - st) * np.log(1. - st)) if (st > 0 and st < 1) else 0. for st in sim]
                    del sim

                # Release
                for cell in shared_coocc:
                    cell.get_lock().release()

                entropies /= np.log(2)
                mean_ent = np.mean(entropies)
                nonz_mean_ent = np.mean(entropies[entropies != 0])
                del entropies
                if verbose >= 2:
                    print 'Mean Shannon Entropy: %.4f' %mean_ent
                    print '(Non-zero) Mean Shannon Entropy: %.4f' %nonz_mean_ent

                # Compare to threshold
                    if np.abs(mean_ent - last_mean_ent) < convergence_criterion:
                        print 'convergence reached for criterion %s at step %d' %(convergence_criterion, iter+1)
                        # Reached convergence: stop execution
                        convergence_check_queue.put((k, mean_ent, nonz_mean_ent, True))
                        return
                last_mean_ent = mean_ent
                convergence_check_queue.put((k, mean_ent, nonz_mean_ent, False))

            # Write matrix if required
            if verbose >= 4 and (iter + 1) in  writing_steps:
                print 'Saving co-occurrence matrix at step %d' %(iter + 1)
                for cell in shared_coocc:
                    cell.get_lock().acquire()

                np.save(os.path.join(output_folder, 'sim_matrix_%d_partial'%(iter + 1)), (np.hstack(shared_coocc)[:total_length] + score_penalty) / np.clip(test_occurrences, 1, n_iter))

                for cell in shared_coocc:
                    cell.get_lock().release()

            if verbose >= 1:
                print 'Computation %d done' %(iter+1)



    ############################################################################################
    incoming_queue = Queue_thread()
    convergence_check_queue = Queue_thread()
    aux_thread = threading.Thread(target=aux_computation, args = (incoming_queue, convergence_check_queue, n_iter, n_samples, cell_length, total_length, shared_coocc, data_occurrences, test_occurrences, verbose, convergence_step, writing_steps, last_mean_ent, convergence_criterion, output_folder))
    aux_thread.start()
    waiting_for_cvg_check = []

    # Retrieve results from queue and restat threads if needed
    for k in xrange(n_iter):
        # Retrieve and restart
        (id, n_steps, counts, synth_lab, mat, b) = sim_queue.get()
        threads[id].join()
        running_threads -= 1

        # Launch Compute thread
        incoming_queue.put((k, id, n_steps, counts, synth_lab, mat))

        # Wait for convergence check before modifying the cooc matrix even further
        if convergence_step > 1 and k % convergence_step == 1:
            print 'Waiting for convergence check'
            a,b,c,d = convergence_check_queue.get()
            convergence_values.append((a,b,c))
            if d:
                print 'Convergence reached, ending program'
                n_samples_occurrences += counts
                synthetic_labels.append(synth_lab)
                score_penalty += b
                iterations[:] = []

                # Terminates other processes + get lock to avoid termination problems
                for cell in shared_coocc:
                    cell.get_lock().acquire()
                for t in threads.values():
                    t.terminate()
                for cell in shared_coocc:
                    cell.get_lock().release()
                break

        # Start next process
        if len(iterations) > 0:
            t =  Process(name = id, target = thread_step, args=(iterations.pop(0), shared_coocc, cell_length, n_samples, sim_queue, data, temp_folder, data_type, annotation_params, classification_params, verbose, debug, with_unique_occurrences, preclustering))
            threads[id] = t
            running_threads += 1
            t.start()
        else:
            del threads[id]

        # Additional Computations
        n_samples_occurrences += counts
        synthetic_labels.append(synth_lab)
        score_penalty += b

        # Verbose outputs
        del mat
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
    # Reshapping the result into a true numpy matrix
    print '> Reshaping co_occurence matrix'
    co_occ = np.zeros((n_samples, n_samples), dtype = float)
    co_occ[np.triu_indices(n_samples, k = 1)] = np.hstack(shared_coocc)[:total_length] + score_penalty
    del shared_coocc

    print 'Normalizing similarities...'
    test_occurrences[test_occurrences == 0] = 1.
    co_occ[np.triu_indices(n_samples, k = 1)] /= test_occurrences

    if convergence_step > 0:
        print 'Saving Convergence criterion values'
        with open(os.path.join(output_folder, 'convergence_check'), 'w') as f:
            f.write('Iteration\tMean\tNonZeroMean\n')
            f.write('\n'.join(['\t'.join(map(str, obj)) for obj in convergence_values]))

    return n_samples_occurrences, synthetic_labels, co_occ, test_occurrences
