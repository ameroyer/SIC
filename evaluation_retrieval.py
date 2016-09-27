#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**evaluation_retrieval.py**: Evaluation for the KNN task using Information retrieval measures.

 **Usage**::

	python evaluation_retrieval.py [1] -s [2] -ov [3] -cfg [4] --help

 where:

 * [1] : input similarity matrix (unnormalized similarities or pre-treated MCL format). The script expects a 'exp_configuration.ini' file in the same folder, usually generated when using ``main.py``.
 * [2] ``-s``: number of samples to evaluate (``s`` first samples of the ground-truth). If -1, the use the whole set. Defaults to -1

 * [3] ``-ov``: If positive, assume the resulting script was obtained in OVA mode for the sample of index ``ov``. Defaults to -1.
 * [4] ``-cfg``: provide a custom configuration file to replace 'exp_configuration.ini'.
 * ``-h, --help``

 This outputs the results of the neighbour retrieval evaluation on the given matrix.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"

import os
import sys
import argparse
import subprocess
import numpy as np
from utils.parse import ground_truth_indices, parse_ground_truth
import ConfigParser as cfg


def evaluate(data_type, co_occ, label_to_index, index_to_label, ground_truth, output_folder, samples=-1, ova=-1, writing=True, idtf=None, suffix=None):
    """
    Evaluates the KNN task on the given similarity matrix and ground-truth

    Args:
     * ``datat_type`` (*str*): Dataset used in the experiments (for ground-truth parsing).
     * ``co_occ`` (*ndarray*): co-occurrence matrix.
     * ``label_to_index`` (*dictt*): Reverse mapping of index_to_label.
     * ``index_to_label`` (*list*): list mapping an index to the corresponding named entity; used to generate a readable clustering.
     * ``ground_truth`` (*dict*): ground truth clustering to compare against.
     * ``output_folder`` (*str*): path to the output folder.
     * ``samples`` (*int, optional*): evaluation on the first ``samples`` samples (if -1, evaluation of all queries). Default to -1.
     * ``ova`` (*int, optional*): if non negative, evaluation on the sample ``ova`` only. Defaults to -1.
     * ``writing`` (*bool, optional*): if ``True``, outputs the resulting measures in a file. Defaults to True.
    """

    # Ground truth qrel
    qrel_gtf = ground_truth + '.qrel'
    gtindx = sorted(ground_truth_indices(data_type, ground_truth, label_to_index))

    # Generate neighbours list from similarity matrix
    qrel_sim = os.path.join(output_folder, 'similarities.res')
    f = open(qrel_sim, 'w')

    # if OVA, evaluate against one sample
    if ova >= 0:
        print 'Evaluation of %d-%s sample' %(ova, index_to_label[ova])
        samples = 1
        line = np.argsort( -co_occ)
        pos_i = np.where(line == ova)[0][0]
        f.write('\n'.join(['%d Q0 %s %d %s %s'%(ova, index_to_label[j], r+1, co_occ[j], index_to_label[ova]) if r < pos_i else '%d Q0 %s %d %s %s'%(ova, index_to_label[j], r, co_occ[j], index_to_label[ova])  for r, j in enumerate(line) if r != pos_i]))
        f.write('\n')
        qid_opt = '%d-%d' %(ova, ova)


    # else evaluate all ground-truth
    else:
        if samples == -1:
            samples = len(gtindx)
            gtindx = np.asarray(gtindx)

        print 'Evaluation on the %d first ground-truth samples' %samples
        print 'Formatting input'
        for i in gtindx[:samples]:
            full = co_occ[i,:] + co_occ[:, i]
            line = np.argsort( - full)
            pos_i = np.where(line == i)[0][0]
            # For NER, add the index to the label to prevent confusion
            if data_type == 'NER':
                f.write('\n'.join(['%d Q0 %d-%s %d %s %d-%s'%(i, j, index_to_label[j].replace(' ', '_'), r+1, full[j], i, index_to_label[i].replace(' ', '_')) if r < pos_i else '%d Q0 %d-%s %d %s %d-%s'%(i, j, index_to_label[j].replace(' ', '_'), r, full[j], i,  index_to_label[i].replace(' ', '_'))  for r, j in enumerate(line) if r != pos_i]))
            # For other dataset, label is unique
            else:
                f.write('\n'.join(['%d Q0 %s %d %s %s'%(i, index_to_label[j], r+1, full[j], index_to_label[i]) if r < pos_i else '%d Q0 %s %d %s %s'%(i, index_to_label[j], r, full[j], index_to_label[i])  for r, j in enumerate(line) if r != pos_i]))
            f.write('\n')
        qid_opt = '0-%d' %i
    f.close()

    # If index to label not given, create it
    del_idtf = (idtf is None)
    if del_idtf:
        idtf = os.path.join(output_folder, 'aux_idtf')
        with open(idtf, 'w') as f:
            f.write('\n'.join('%d\t%s'%(i,l) for i,l in enumerate(index_to_label)))

    # Call the evaluation script
    main_dir = os.path.dirname(os.path.realpath(__file__))
    if writing:
        output_file = os.path.join(output_folder, 'eval_%s.log'%index_to_label[ova]) if ova >= 0 else os.path.join(output_folder, 'eval_%d.log'%samples if suffix is None else 'eval_%d_%s.log'%(samples,suffix))

        with open(output_file, 'w') as f:
            p = subprocess.Popen(('%s -out titi  -qid %s -qrel %s -nostat -run %s -idtf %s -notrec'%(os.path.join(main_dir, 'utils/evaluation_RI.prl'), qid_opt, qrel_gtf, qrel_sim, idtf)).split(), stdout = f)
            p.communicate()
    else:
            p = subprocess.Popen(('%s -out titi  -qid %s -qrel %s -nostat -run %s -idtf %s -notrec'%(os.path.join(main_dir, 'utils/evaluation_RI.prl'), qid_opt, qrel_gtf, qrel_sim, idtf)).split())
            p.communicate()

    if del_idtf:
        os.remove(idtf)

    # If AQUA, outputs an additional list with mAP sorted according to the number of occurrences of each sample
    if ova < 0 and data_type =='AQUA':
        # Importing number of occurrences
        words_occ = {}
        with open(os.path.join(main_dir, 'Precomputed/aquaint_entities_list'), 'r') as f:
            for line in f:
                w, o = line.split('\t')
                words_occ[w] = int(o)
        # Ordering
        ordered_keys = sorted(gtindx[:samples], key=lambda x: - words_occ[index_to_label[x]])
        ordered_keys = {x : i for i, x in enumerate(ordered_keys)}

        # Write
        output_file_ordered = os.path.join(output_folder, 'eval_%d_ordered.log'%samples)
        ordered_scores = [0] * samples
        seen = []
        with open(output_file, 'r') as f:
            begin = False
            for line in f:
                if line.startswith(str(gtindx[0])) and not begin: # First ground-truth sample
                    begin = True

                if line.startswith('*') and begin: # End of MAP list
                    break

                if begin and line.strip():
                    w = line.split('\t')[0]
                    indx = int(w.split()[0])
                    seen.append(ordered_keys[indx])
                    ordered_scores[ordered_keys[indx]] = '(%d)\t%s' % (words_occ[index_to_label[indx]], line)

        with open(output_file_ordered, 'w') as f:
            f.write(''.join(ordered_scores))


    # If NER or AUDIO, outputs the mean mAP over each class of the ground-truth
    elif ova < 0 and (data_type == 'NER' or data_type == 'AUDIO'):
        # Ground-truth
        gt = parse_ground_truth(data_type, ground_truth)
        rev_gt = [0] * sum([len(y) for y in gt.values()])
        for k, y in gt.iteritems():
            for x in y:
                rev_gt[x] = k
        gt_map = {k : [] for k in gt.keys()}

        # Write
        with open(output_file, 'r') as f:
            begin = False
            for line in f:
                if line.startswith(str(gtindx[0])) and not begin:
                    begin = True

                if line.startswith('*') and begin:
                    break

                if begin and line.strip():
                    w = line.split('\t')[0]
                    indx = int(w.split()[0])
                    ap = float(line.split('\t')[2])
                    gt_map[rev_gt[indx]].append(ap)
        output_file_ordered = os.path.join(output_folder, 'eval_%d_perclass.log' % samples)

        with open(output_file_ordered, 'w') as f:
            f.write('\n'.join('MaP\t%s (%d)\t%s'%(k, len(y), float(sum(y)) / len(y)) for k, y in gt_map.iteritems() if len(y) > 0))

    # Clean
    os.remove(qrel_sim)



if __name__ == '__main__':
    from utils.output_format import load_cooc
    parser = argparse.ArgumentParser(description='Nearest neighbour retrieval evaluation.')
    parser.add_argument(dest='input_matrix', type=str, help='path to the file containing the similarity matrix.')
    parser.add_argument('-cfg', dest='cfg_file', type=str, help='Input a custom config file.')
    parser.add_argument('-s', dest='samples', default=-1, type=int, help='Number of samples to use for evaluation. Defaults to -1, ie all samples.')
    parser.add_argument('-ov', dest='ova', default=-1, type=int, help='OVA sample.')
    args = parser.parse_args()

    #----------------------------------------------------------------------------- LOAD INPUTS

    # Read config file given with the similarity matrix
    base_folder = os.path.dirname(os.path.realpath(args.input_matrix))
    cfg_file = args.cfg_file if args.cfg_file is not None else os.path.join(base_folder, 'exp_configuration.ini')
    config = cfg.ConfigParser()
    config.read(cfg_file)
    data_type = config.get('General', 'data')
    idtf = config.get(data_type, 'index_to_label')
    gtf = config.get(data_type, 'ground_truth')
    output_folder = os.path.dirname(os.path.realpath(args.input_matrix))

    # Import index to label
    print 'Loading Index to Label for dataset %s' %data_type
    try:
        index_to_label = []
        label_to_index = {}
        with open(idtf, 'r') as f:
            for line in f:
                word = line.split('\t')[1].replace('\n', '')
                label_to_index[word] = len(index_to_label)
                index_to_label.append(word)
        n_samples = len(index_to_label)
    except IOError:
        print >> sys.stderr, 'Error: Index to label file %s not found' %idtf
        raise SystemExit

    print 'Loading Matrix from %s' %args.input_matrix
    co_occ = load_cooc(args.input_matrix)

    evaluate(data_type, co_occ, label_to_index, index_to_label,  gtf, output_folder, args.samples, ova = args.ova, idtf=idtf)
