#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**similarity_analysis.py.** Main script for running experiments on the distribution of the similarities and the prunning processes.

 **Usage**::

	python similarity_analysis.py [1] -n [2] -cfg [3] --mean --theo --help

 where:

 * [1] : input similarity matrix (unnormalized). The script expects a 'exp_configuration.ini' file in the same folder, usually generated when using ``main.py``.
 * [2] ``-n``: number of samples to plot for each class. Defaults to 5.
 * [3] ``-cfg``: provide a custom configuration file to replace 'exp_configuration.ini'.
 * ``--mean``: if given, plot an average ROC curve for each ground-truth class.
 * ``--theo``: if given, plot the comparison of the distribution against the theoretical model of the corresponding SIC variant.
 * ``-h, --help``

 This outputs pdf histograms of the distribution of similarities for several samples across the matrix and for several normalization parameters.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import os
import argparse
import ConfigParser as cfg                         # 'configparser' for Python 3+
from utils.read_config import read_config_file
from utils.parse import split_on_ground_truth_no_indices, parse_data, ground_truth_pairs
from utils.matrix_op import *
from utils.output_format import init_folder, Tee, load_cooc


if __name__ == '__main__':
    #----------------------------------------------------------------------------- INIT
    # Parse command line
    parser = argparse.ArgumentParser(description='Clustering by Diverting supervised classification techniques.')
    parser.add_argument(dest='input_matrix', type=str, help='path to the file containing the similarity matrix.')
    parser.add_argument('-n', dest='n', type=int, default=5, help='number of entities to plot per class.')
    parser.add_argument('-cfg', dest='cfg_file', type=str, help='Input a custom config file for default option values.')
    parser.add_argument('--theo', help="If given, plot a comparison graph against the theoretical case", action="store_true")
    parser.add_argument('--mean', help="If given, plot the average ROC curves for each class in the ground-truth", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.input_matrix):
        print >> sys.stderr, 'input %s is not a file' %args.input_matrix
        raise SystemExit

    # Import index to label
    config = cfg.ConfigParser()
    base_folder = os.path.join(os.path.dirname(os.path.realpath(args.input_matrix)))
    config_file = os.path.join(base_folder, 'exp_configuration.ini') if (args.cfg_file is None) else args.cfg_file
    config.read(config_file)
    datatype = config.get('General', 'data')
    similarity_type = config.get('General', 'similarity')
    N = config.getint('General', 'N')

    # Index to label
    idtf = config.get(datatype, 'index_to_label')
    index_to_label = []
    with open(idtf, 'r') as f:
        for line in f.read().splitlines():
            index_to_label.append(line.split('\t')[1])

    # Import ground truth file
    gtf = config.get(datatype, 'ground_truth')
    ground_truth, entities_to_plot = split_on_ground_truth_no_indices(datatype, gtf, numb=args.n)

    # Import similarity matrix
    print 'Loading full similarity matrix'
    co_occ = load_cooc(args.input_matrix)
    co_occ = co_occ + co_occ.T
    temp_folder = init_folder(config.get('General', 'temp'))
    output_folder = init_folder(os.path.dirname(os.path.realpath(args.input_matrix)))
    distrib_folder = init_folder(os.path.join(output_folder, 'Distrib'))

    # Load Poisson Biomial parameters if needed
    if similarity_type == 'BIN' and args.theo:
        try:
            label_counts_file = os.path.join(output_folder, 'label_counts.txt')
            with open(label_counts_file, 'r') as f:
                lines = f.read().split('\n')[1:]
                p_values_train = [0] * N
                p_values_test = [0] * N
                for i, l in enumerate(lines):
                    tr, te = l.split('\t')[2:4]
                    p_values_train[i] = 1.0 / int(tr)
                    p_values_test[i] = 1.0 / int(te)
        except IOError, ValueError:
            print 'Wrong input file %s' %label_counts_file
        rnd_prob_train = np.abs(estimate_poisson_binomial(N, p_values_train))
        rnd_prob_test = np.abs(estimate_poisson_binomial(N, p_values_test))


    ############################################## PLOT MEAN ROC CURVES IF REQUIRED
    if datatype in  ['NER', 'AUDIO', 'AUDIOTINY'] and args.mean:
        distrib_mean_folder = init_folder(os.path.join(distrib_folder, 'Means'))
        for key, gt in ground_truth.iteritems():
            print 'Mean ROC - %s'%key
            ROC_mean_analysis(co_occ[gt, :], key, distrib_mean_folder, np.asarray(gt))

    entities_to_plot = [(2610, 'fonc')] + entities_to_plot
    ############################################# MAIN PLOT
    for x, key in entities_to_plot:
        name = str(x) + '-' + index_to_label[x] + '-' + key
        line = co_occ[x,:]
        gt = ground_truth[key]

        # Joint sort similarities and ground-truth
        sorted_ind = np.argsort(line)
        sorted_gt = np.zeros(len(line))
        sorted_gt[gt] = 1.0
        sorted_gt = sorted_gt[sorted_ind]
        sorted_gt = np.nonzero(sorted_gt)[0]

        # Histogram
        print '--->\nPlotting', name
        distribution_analysis(line[sorted_ind], name, distrib_folder, temp_folder, sorted_gt)

        # Sorted similarities
        with open(os.path.join(distrib_folder, '%s_sorted_sim.txt'%name), 'w') as f:
            f.write('\n'.join(['> %s\t%d-%s\t%s'%(name, i, index_to_label[i], line[i]) if i in gt else '%s\t%d-%s\t%s'%(name, i, index_to_label[i], line[i]) for i in sorted_ind]))

        # ROC Curve
        print '> ROC curve'
        ROC_analysis(line[sorted_ind], name, distrib_folder, sorted_gt)

        # Theoretical comparison
        if args.theo:
            # Weighted SIC - Gaussian
            if similarity_type in ['WBIN', 'UWBIN']:
                statistical_analysis_weighted(line, 1./N, gt, temp_folder, distrib_folder, name, step=0.025)

            # Binary SIC - Poisson Binomial
            elif similarity_type == 'BIN':
                statistical_analysis_binary(line, rnd_prob_train, gt, temp_folder, distrib_folder, name, suffix='trainlabels')
                statistical_analysis_binary(line, rnd_prob_test, gt, temp_folder, distrib_folder, name, suffix='testlabels')
