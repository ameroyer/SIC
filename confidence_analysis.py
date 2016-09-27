#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**confidence_analysis.** Computes several confidence scores and compare them to corresponding mAPs values.

**Usage:** ::

	python confidence_analysis.py  [1] -cfg [2] --mean --theo --help

where:

 * [1] : input similarity matrix. The script expects a 'exp_configuration.ini' file in the same folder and a ``eval_*.log`` file, containing the mAP results both usually generated when using ``main.py``.
 * [2] ``-cfg``: provide a custom configuration file to replace 'exp_configuration.ini'.
 * ``-h, --help``


Computes confidence scores for the input matrix and compares them to the corresponding mAP results.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


if __name__ == '__main__':
    import ConfigParser as cfg
    import numpy as np
    import argparse
    import os
    from scipy.stats import skew, kurtosis, tvar, spearmanr, pearsonr
    from utils.output_format import init_folder, load_cooc

    # Parse commad line
    parser = argparse.ArgumentParser(description='Clustering by Diverting supervised classification techniques.')
    parser.add_argument(dest='input_matrix', type=str, help='path to the file containing the similarity matrix.')
    parser.add_argument('-cfg', dest='cfg_file', type=str, help='Input a custom config file for default option values.')
    args = parser.parse_args()

    if not os.path.isfile(args.input_matrix):
        print >> sys.stderr, 'input %s is not a file' %args.input_matrix
        raise SystemExit

    base_folder = os.path.join(os.path.dirname(os.path.realpath(args.input_matrix)))
    config_file = os.path.join(base_folder, 'exp_configuration.ini') if (args.cfg_file is None) else args.cfg_file
    coocc_matrix = load_cooc(args.input_matrix)
    coocc_matrix += coocc_matrix.T
    map_file = os.path.join(base_folder, 'eval_%d.log'%(coocc_matrix.shape[0]))

    # Load index to label
    config = cfg.ConfigParser()
    config.read(config_file)
    datatype = config.get('General', 'data')
    with open(config.get(datatype, 'index_to_label'), 'r') as f:
        index_to_labels = [l.split('\t')[1].replace('\n', '') for l in f.readlines()]

    # Load precomputed mAPs for correlation
    print 'Load maps'
    maps = [0] * coocc_matrix.shape[0]
    with open(map_file, 'r') as f:
        for line in f:
            vals = line.split('\t')
            id = vals[0].split()
            try:
                maps[int(id[0])] = float(vals[1])
            except (ValueError, IndexError):
                continue

    # Compute confidence scores
    print 'Compute entropies'
    from collections import defaultdict
    entropies = defaultdict(lambda: np.zeros(coocc_matrix.shape[0]))
    for i, line in enumerate(coocc_matrix):
        entropies['max'][i] = max(line)
        entropies['var'][i] = np.var(line)
        entropies['skewtrimmed'][i] = skew(line[line > np.mean(line)])
        entropies['skew'][i] = skew(line)
        entropies['bimodal'][i] = float(skew(line)**2 + 1) / kurtosis(line)

    # Write Outputs
    print 'Output'
    output_folder = init_folder(os.path.join(base_folder, 'Confidence'))
    for k, la in entropies.iteritems():
        la[np.where(np.isnan(la))] = 0
        entropies[k] = la
        with open(os.path.join(output_folder, 'maps_with_confidences_%s.log' % k), 'w') as f:
            f.write('Entity\tmAP\tConfidence score\n')
            f.write('\n'.join('%d %s\t%s\t%s'%(i, l, m, e) for i, (l, m, e) in enumerate(zip(index_to_labels, maps, la))))
            f.write('\n\n\nPearson Correlation: %s\nSpearman Correlation: %s'%(pearsonr(la, maps), spearmanr(la, maps)))
            print '%s\nPearson Correlation: %s\nSpearman Correlation: %s\nMean: %s'%(k,pearsonr(la, maps), spearmanr(la, maps), np.mean(la))
