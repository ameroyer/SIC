#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**evaluation_clustering.py.** Main script for running and evaluating result of the clustering process.

 **Usage**::

	python evaluation_clustering.py [1] -i [2] -p [3] -t [4] -cfg [5] --mcl --help

 where:

 * [1] : input similarity matrix (unnormalized similarities or pre-treated MCL format). The script expects a 'exp_configuration.ini' file in the same folder, usually generated when using ``main.py``.
 * [2] ``-i``: MCL inflation parameter. Defaults to 1.4.
 * [3] ``-p``: MCL pre-inflation parameter. Defaults to 1.0.
 * [4] ``-t``: number of cores to use for MCL.
 * [5] ``-cfg``: provide a custom configuration file to replace 'exp_configuration.ini'.
 * ``-m, --mcl``: if present, the script expects an input matrix in MCL label format.
 * ``-h, --help``

 This outputs the results of the MCL clustering with the given inflation and pre-inflation parameters.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"

import sys
import os
import subprocess
import numpy as np
from collections import defaultdict
from utils.plot import fraction_plot
from utils.output_format import readable_clustering, clustering_to_file, clustering_to_string, init_folder, save_coocc_mcl, load_cooc


def cluster(co_occ, output_folder, index_to_label, cores, task_params, **kwargs):
    """
    Returns the clustering obtained after applying the chosen algorithm on the co-occurence matrix.

    Args:
     * ``co_occ`` (*ndarray*): Co-occurence matrix.
     * ``output_folder`` (*str*): path to the output folder.
     * ``index_to_label`` (*list*): list mapping an index to the corresponding named entity.
     * ``cores`` (*int*): Number of cores to use for the clustering algorithm (if threading option available).
     * ``task_params`` (*dict*): additional clustering algorithms.
     * ``formated`` (*bool, optional*): if ``True`` the co-occurence matrix is expected to be already formatted for MCL input.
     * ``verbose`` (*int, optional*): controls verbosity level.

    Returns:
     * ``clustering`` (*list*): Resulting clustering (as a list mapping a sample's index to the index of its cluster).
     * ``n_clusters`` (*int*): number of retrieved clusters.
     * ``step_id`` (%str*): step identifier, optionnal, for output
     * ``summary`` (*str*): string representation of the execution (for displaying purpose).
    """

    cluster_type = task_params['task_type']

    ## MCL clustering
    if cluster_type == 'MCL':
        # --------------------- Parameters
        formated = kwargs.pop('formated', False)
        verbose = kwargs.pop('verbose', 1)

        mcl = task_params['binary']
        inflation = float(task_params['additional_params']['i'])
        pre_inflation = float(task_params['additional_params']['p'])
        suffix = 'p%s_i%s'%(pre_inflation, inflation)

        # ---------------------- Write matrix in MCL format if needed
        if formated:
            inp = co_occ
        else:
            print 'Saving matrix in MCL format'
            inp = save_coocc_mcl(output_folder, co_occ, index_to_label)

        # ----------------------- Apply MCL algorithm
        print 'Call to MCL with inflation parameter %f and pre-inflation %f' % (inflation, pre_inflation)
        FNULL = open(os.devnull, 'w')
        err = None if verbose >= 2 else FNULL
        p = subprocess.Popen([mcl, inp,  '--abc', '-scheme', '7', '-pi', str(pre_inflation), '-I', str(inflation), '-te', str(cores), '-o', '-'], stdout=subprocess.PIPE, stderr=err)
        output = p.communicate()[0]
        FNULL.close()

        # ------------------------ Write the clustering
        clustering = defaultdict(lambda: [])
        n_clusters = 0
        for line in output.splitlines():
            if line.strip():
                clustering[n_clusters] = [int(v.strip().split('-')[0]) for v in line.split('\t')]
                n_clusters += 1

        return clustering, n_clusters, inp, suffix

    else:
        print >> sys.stderr, 'Unknown option %s' %cluster_type
        raise SystemExit



def evaluate(co_occ, output_folder, temp_folder, ground_truth, index_to_label, cores, task_params, **kwargs):
    """
    Evaluate a clustering method given a similarity matrix and various clustering parameters.

    Args:
     * ``co_occ`` (*ndarray*): co-occurence matrix.
     * ``output_folder`` (*str*): path to the output folder.
     * ``temp_folder`` (*str*): path to temporary folder.
     * ``ground_truth`` (*dict*): ground truth clustering to compare against.
     * ``index_to_label`` (*list*): list mapping an index to the corresponding named entity. used to generate a readable clustering.
     * ``cores`` (*int*): number of cores to use.
     * ``task_params`` (*list*): additional clustering parameters.
     * ``formated`` (*bool, optional*): if ``True`` the co-occurence matrix is expected to be already formatted for MCL input.
    """

    print "> Clustering and Evaluation Step"
    formated = kwargs.pop('formated', False) # if input is a matrix already in MCL format file
    verbose = kwargs.pop('verbose', 1)

    # Run MCL
    clustering, n_clusters, outp, suffix = cluster(co_occ, output_folder, index_to_label, cores, task_params, formated=formated, verbose=verbose)

    #Evaluate
    print '%d Found clusters'%(n_clusters)
    base_dir = os.path.dirname(os.path.realpath(__file__))

    gtf = clustering_to_file(temp_folder, ground_truth, 'ground_truth')
    p = subprocess.Popen(['python', os.path.join(base_dir, 'utils', 'eval.py'), '-in', '/dev/fd/0', '-ref', gtf, '-o', os.path.join(output_folder, 'eval_%s'%suffix)], stdin=subprocess.PIPE)
    p.communicate(input=clustering_to_string(clustering))
    os.remove(gtf)

    readable_clustering(output_folder, clustering, index_to_label, suffix)
    fraction_plot(clustering, ground_truth, os.path.join(output_folder, 'histo_%s'%suffix))





if __name__ == '__main__':
    import argparse
    import ConfigParser as cfg                         # 'configparser' for Python 3+
    from utils.read_config import read_config_file
    from utils.parse import parse_ground_truth
    from utils.matrix_op import normalize

        #--------------------------------------------------------------------------------- PARAMS
    parser = argparse.ArgumentParser(description='Clustering evaluation.')
    parser.add_argument(dest='input_matrix', type=str, help='path to the file containing the similarity matrix.')
    parser.add_argument('-i', dest='inflation', default=1.4, type=float, help='MCL inflation parameter.')
    parser.add_argument('-p', dest='pre_inflation', default=1.0, type=float, help='MCL pre-inflation parameter.')
    parser.add_argument('-t', dest='cores', type=int, help='MCL threads parameter.')
    parser.add_argument('-m', '--mcl', help="if present, the expected input is a MCL label formatted matrix", action="store_true")
    parser.add_argument('-cfg', dest='cfg_file', type=str, help='Input a custom config file for default option values.')
    parser.add_argument('-v', '--verbose', dest='verbose', type=int, help='Controls verbosity level.')
    args = parser.parse_args()

    # Read config file given with the similarity matrix
    base_folder = os.path.dirname(os.path.realpath(args.input_matrix))
    cfg_file = args.cfg_file if args.cfg_file is not None else os.path.join(base_folder, 'exp_configuration.ini')
    config = cfg.ConfigParser()
    config.read(cfg_file)

    cores = args.cores if args.cores is not None else config.getint('General', 'cores')
    data_type = config.get('General', 'data')
    idtf = config.get(data_type, 'index_to_label')
    ground_truth_file = config.get(data_type, 'ground_truth')
    temp_folder = config.get('General', 'temp')
    task_type = config.get('General', 'task')
    task_binary = config.get(task_type, 'binary')
    ground_truth = parse_ground_truth(data_type, ground_truth_file, None)

    # Create setting for MCL
    output_folder = init_folder(os.path.join(os.path.dirname(os.path.realpath(args.input_matrix)), 'Clustering'))
    temp_folder = init_folder(temp_folder)
    if task_type == 'MCL':
        settings = {'task_type': 'MCL', 'binary': task_binary, 'additional_params': {'i': str(args.inflation), 'p': args.pre_inflation}}
    else:
        print >> sys.stderr, 'Unknown clustering algorithm'
        raise SystemExit


    #----------------------------------------------------------------------------- LOAD INPUTS
    # Import index to label
    try:
        index_to_label = []
        with open(idtf, 'r') as f:
            for line in f:
                index_to_label.append(line.split('\t')[1].replace('\n', ''))
    except IOError:
        print >> sys.stderr, 'Error: Index to label file %s not found' %idtf
        raise SystemExit


    # Load co_occ matrix then run and evaluate the clustering
    if args.mcl:
        co_occ = args.input_matrix
    else:
        co_occ = load_cooc(args.input_matrix)
        print '> Normalizing matrix'
        co_occ = normalize(co_occ)

    evaluate(co_occ, output_folder, temp_folder, ground_truth,  index_to_label, cores, settings, formated=args.mcl, verbose=args.verbose)
