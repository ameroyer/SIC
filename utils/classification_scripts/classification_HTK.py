#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**classify.py.** For training and applying a classifier on the artifically annotated data set.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"

import os
import shutil
from random import choice
from subprocess import Popen, PIPE
from ..output_format import init_folder
from basic_hmm import generate_basic_hmm


def train_HTK(n, train, temp_folder, classification_params, verbose=1, debug=False):
    """
    Trains a HMM classifier with HTK and returns the resulting model.

    Args:
     * ``n`` (*int*): step number.
     * ``train``: annotated training set (structure may depend on the classifier).
     * ``temp_folder`` (*str*): path to the directory for storing temporary files.
     * ``classification_params`` (*list*): additional classification parameters.
     * ``verbose`` (*int, optional*): controls verbosity level. Defaults to 1.
     * ``debug`` (*bool, optional*): if True, some outputs are kept in the temporary directory.
    Returns:
     * ``hmmdef``: HMMs master file.
     * ``hmmlist``: list of HMMs in the model.
     """

    htk_path = os.environ.copy()['CLASSIFIER']
    FNULL = open(os.devnull, 'w')
    err = None if verbose >= 3 else FNULL

    #Build hmm master file iteratively
    hmm_folder = init_folder(os.path.join(temp_folder, 'step_%d' % n))
    hmmdef_file = os.path.join(temp_folder, 'step%d_hmmdefs' % n)
    hmmlist_file = os.path.join(temp_folder, 'step%d_hmmlist' % n)
    hmmdef = open(hmmdef_file, 'w')
    hmmlist = open(hmmlist_file, 'w')
    scp_trains, mlf_file = train

    # Dummy hmm
    base_name = os.path.dirname(scp_trains.values()[0]).rsplit('/', 1)[1]
    aux = base_name.split('STEP')
    if len(aux) > 1:
        base_name = aux[0]
    features_type, components = base_name.rsplit('_', 1)
    basic_hmm = generate_basic_hmm(features_type, int(components), 'step%d_hmm' % n, temp_folder, n_state=12, hmm_type=choice(classification_params['hmm_topo']))
    result_hmm = os.path.join(hmm_folder, 'step%d_hmm' % n)

    # For each training class
    for cl, scp_train in scp_trains.iteritems():
        # Init HMM
        train_file = os.path.join(temp_folder, 'step%d_train.scp' % n)
        with open(train_file, 'w') as g:
            g.write(scp_train)
        p = Popen(('%s -I %s -M %s -S %s %s' % (os.path.join(htk_path, './HInit'), mlf_file, hmm_folder,  train_file, basic_hmm)).split(), shell=False, stdout=err, stderr=err, close_fds=True)
        p.communicate()

        # Train
        p = Popen(('%s -v 0.000001 -I %s -M %s -S %s %s' % (os.path.join(htk_path, './HRest'), mlf_file, temp_folder, train_file, result_hmm)).split(), shell=False, stdout=FNULL, stderr=err, close_fds=True)
        p.communicate()

        # Rename HMM and append to mmf
        hmmlist.write('%s\n' % cl)
        with open(basic_hmm, 'r') as f:
            lines = f.read().split('\n')
            for i, line in enumerate(lines):
                if line.startswith('~h'):
                    break
            lines[i] = '~h %s' % cl
            hmmdef.write('\n'.join(lines))
            hmmdef.write('\n\n')

    # Clean
    hmmdef.close()
    hmmlist.close()
    os.remove(basic_hmm)
    shutil.rmtree(hmm_folder)
    os.remove(train_file)
    if not debug:
        os.remove(mlf_file)

    # Create vocabulary and dictionnary for test step
    # 1.Wordnet
    voc_file = os.path.join(temp_folder, 'step%d_voc' % n)
    wnet_file = os.path.join(temp_folder, 'step%d_wordnet' % n)
    with open(voc_file, 'w') as f:
        f.write('(%s)' % ('|'.join(scp_trains.keys())))
    p = Popen([os.path.join(htk_path, './HParse'), voc_file, wnet_file], shell=False, stdout=FNULL, stderr=err, close_fds=True)
    p.communicate()
    os.remove(voc_file)
    FNULL.close()

    # 2. Dictionnary
    dic_file = os.path.join(temp_folder, 'step%d_dict'%n)
    with open(dic_file, 'w') as f:
        f.write('\n'.join(['%s %s'%(c,c) for c in sorted(scp_trains.keys())]))
        f.write('\n')

    return hmmdef_file, hmmlist_file, wnet_file, dic_file



def label_HTK(n, hmmdef_file, hmmlist_file, wnet_file, dic_file, test, test_entities_indices, temp_folder, verbose=1, debug=False):
    """
    Labels a testing set using a HTK HMM classifier and returns the resulting entities.

    Args:
     * ``model``: model built from training the classifier
     * ``test``: formatted testing set.
     * ``test_entities_indices`` (*list*): location of the interesting entities in the test dataset.
     * ``verbose`` (*int, optional*): controls verbosity level. Defaults to 1.
     * ``debug`` (*bool, optional*): if True, some outputs are kept in the temporary directory.
    Returns:
     * ``result_iter``: a generator expression on the result
    """

    htk_path = os.environ.copy()['CLASSIFIER']
    FNULL = open(os.devnull, 'w')
    err = None if verbose >= 3 else FNULL

    # Estimate
    test_file = os.path.join(temp_folder, 'step%d_test.scp' % n)
    with open(test_file, 'w') as f:
        f.write(test)

    p = Popen(('%s -T 1 -t 250.0 -S %s -H %s -w %s %s %s'%(os.path.join(htk_path, './HVite'), test_file, hmmdef_file, wnet_file, dic_file, hmmlist_file)).split(), shell=False, stdout=PIPE, stderr=err, close_fds=True)
    out = p.communicate()[0]
    FNULL.close()

    # Clean
    os.remove(test_file)
    os.remove(hmmlist_file)
    os.remove(dic_file)
    os.remove(wnet_file)
    if not debug:
        os.remove(hmmdef_file)

    # Remove test features
    features_folder = os.path.dirname(test.split('\n')[0])
    print features_folder
    if len(features_folder.split('STEP')) > 1:
        shutil.rmtree(features_folder)

    # Result
    result_iter = []
    indx = -1
    for line in out.splitlines():
        if line.strip().endswith('.mfc'):
            indx += 1
        else:
            attr = line.split()
            if len(attr) > 1 and attr[1] == '==':
                classe = attr[0].strip()
                result_iter.append(('B-%d'%test_entities_indices[indx][1], 'B-%s'%classe))

    return result_iter
