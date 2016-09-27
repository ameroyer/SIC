#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**output_format.py.** functions linked to format of the program outputs.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import os
import sys
import errno
import time


def init_folder(path):
    """
    Create directory in ``path`` if not already existing.

    Args:
     * ``path`` (*str*): path of the directory to initialize.
    """
    # Normalize path
    npath = os.path.join(os.path.normpath(path), '')

    # Create directory if needed
    try:
        os.makedirs(npath)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return npath



class Tee(object):
    """
    Tee object used for linking several files (used for linking stdout to log file).
    """
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()



def log(output_folder):
    """
    Create file for redirecting output

    Args:
     * ``output_folder`` (*str*): path to root of directory for outputs.
    """
    init_folder(output_folder)
    f = open(os.path.join(output_folder, "output.log"), 'w')
    return Tee(sys.stdout, f), f



def save_coocc(output_folder, coocc, suffix=None, type='binary'):
    """
    Saves the current co-occurence matrix.

    Args:
     * ``output_folder`` (*str*): path to root of output folder.
     * ``n`` (*int*): step of the matrix.
     * ``coocc`` (*ndarray*): co-occurence matrix.
     * ``type`` (*str*): output file format (``text``, ``binary`` or ``pickle``).

    Returns:
     * ``output_file`` (*str*): path to the file now containing the matrix.
    """
    import numpy as np
    output_file = os.path.join(output_folder, 'sim_matrix')
    if suffix != None:
        output_file += '_' + suffix

    # Save in pickle file
    if type == 'pickle':
        import pickle
        output_file += '.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(coocc, f)

    # Save in text file
    elif type == 'text':
        output_file += '.txt'
        np.savetxt(output_file, coocc)

    # Save in text file
    elif type == 'binary':
        np.save(output_file, coocc)

    return output_file


def load_cooc(mat):
    """
    Load and format, if needed, the given similarity matrix.

    Args:
     * ``mat`` (*str*): path to the similarity matrix (txt, npy or pickle)
     """
    import numpy as np
    if mat.endswith('.txt'):
        co_occ = np.loadtxt(mat)
    elif mat.endswith('.pkl'):
        import pickle
        with open(mat, 'rb') as f:
            co_occ = pickle.load(f)
    else:
        co_occ = np.load(mat)

    if (len(co_occ.shape) != 2) or (co_occ.shape[0] != co_occ.shape[1]):
        n_samples = int((1 + np.sqrt(1 + 8* len(co_occ))) / 2)
        mat = np.zeros((n_samples, n_samples))
        mat[np.triu_indices(n_samples, k = 1)] = co_occ
        co_occ = np.array(mat)

    return co_occ



def save_coocc_mcl(output_folder, coocc, index_to_label):
    """
    Save current co-occurence matrix (only non-zero entries) in the label format of MCL.

    Args:
     * ``output_folder`` (*str*): path to root of output folder.
     * ``coocc`` (*ndarray*): co-occurence matrix.
     * ``index_to_label`` (*list*): mapping from an entity index to a readable label.

    Returns:
     * ``output_file`` (*str*): path to the file now containing the matrix.
    """

    import numpy as np

    output_file = os.path.join(output_folder, 'sim_matrix_MCL.txt')
    indx, indy = np.nonzero(coocc)

    with open(output_file, 'w') as f:
        f.write('\n'.join(['%d-%s\t%d-%s\t%.5f' %(i, index_to_label[i], j, index_to_label[j], coocc[i, j]) for i,j in zip(indx, indy)]))

    return output_file



def readable_clustering(output_folder, clustering, index_to_label, suffix=None):
    """
    Outputs a clustering in a more readable format.

    Args:
     * ``output_folder`` (*str*): Path to the output folder.
     * ``clustering`` (*dict*): clustering represented as a dictionnary mapping a cluster to the lsit of its elements.
     * ``index_to_label`` (*list*): mapping from an entity index to a string label.
     * ``suffix`` (*str, optional*): if given, this is added as a suffix to the name of the output file. Defaults to ``None``.

    Returns:
     * ``output_file`` (*str*): path to the file now containing the clustering under readable format.
    """
    of = os.path.join(output_folder, 'readable_clustering%s.txt'%('_%s'%suffix if suffix is not None else ''))
    with open(of, 'w') as f:
        f.write('\n\n\n'.join(['=================%s\n%s'%(x,'\n'.join(['%d-%s'%(t, index_to_label[t]) for t in y])) for (x,y) in clustering.iteritems()]))

    return of



def clustering_to_string(clustering):
    """
    Return a clustering as a string with entities separated by tabulations and clusters by newlines '\n'.

    Args:
     * ``clustering`` (*dict*): clustering represented as a dictionnary mapping a cluster to its elements.

    Returns:
     * ``clustering_to_string`` (*str*): string representation of the clustering.
    """
    return '\n'.join(['\t'.join([str(y) for y in x]) for x in clustering.values()])



def clustering_to_file(output_folder, clustering, suffix=None):
    """
    Writes a clustering in a file.

    Args:
     * ``output_folder`` (*str*): Path to the output folder.
     * ``clustering`` (*dict*): clustering represented as a dictionnary mapping a cluster to its elements.
     * ``suffix`` (*str, optional*): if given, this is added as a suffix to the name of the output file. Defaults to ``None``.

    Returns:
     * ``output_path`` (*str*): Path to the file in which the clustering was written.
    """
    of = os.path.join(output_folder, 'clustering%s.txt'%('_%s'%suffix if suffix is not None else ''))
    with open(of, 'w') as f:
        f.write(clustering_to_string(clustering))
    return of
