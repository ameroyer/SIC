#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**annotation_DT.py.** Generation of synthetic annotations for weka Decision Tree J48 classifiers.
"""

import os
from subprocess import Popen, PIPE
from random import randint


__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


def weka_compatible_string(s):
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("%", r"\\%")



def annotate_DT_train(n, train, distrib, n_labels, features, to_annotate, temp_folder, preclustering):
    """
    Returns a synthetic annotation of the data (train + test) for the (weka) DT classifier.

    Args:
     * ``n`` (*int*): step identifier.
     * ``train`` (*list*): training entities.
     * ``distrib`` (*str*): type of the synthetic annotation.
     * ``n_min`` (*int*): minimum number of synthetic labels to use.
     * ``n_max`` (*int*): maximum number of synthetic labels to use.
     * ``features`` (*list*): pattern for the feature selection.
     * ``to_annotate`` (*list*): in case of UNI annotation, list of indices of the entities to have their own class.
     * ``with_common_label_wordform`` (*bool, optional*): if ``True``, each entity occurence wordform receives the same label. Defaults to False.
     * ``temp_folder``: directory for temporary files.
    Returns:
     * ``N`` (*int*): random max number of synthetic labels for this step.
     * ``n_unique_labels_used`` (*int*): number of synthetic labels that were actually used.
     * ``n_sentences_train`` (*int*): number of sequences in the training database.
     * ``n_entities_train`` (*int*): number of entities in the training database.
     * ``train_file`` (*str*): path to the  formatted train data.
    """
    # Parameters
    unique_labels_used = [0] * (n_labels+3)   #counts the number of unique labels (N synthetic + 'O' + 'I')
    n_entities_train = 0
    seen_entities = {}

    # ------------- Build the train part of the data file
    wekadata = []
    train_length = 0
    seen_classes = []

    if to_annotate != None:
        unique_labels_used[:len(to_annotate)] = [1] * len(to_annotate)

    #Train
    for sentence in train:
        words_acc = []
        for i, word in enumerate(sentence):

            # Only keep interesting entities
            bio = word[-1]
            if bio != 'B':
                continue

            # Features selection
            word_attributes = []
            for k, pattern in enumerate(features):
                aux_pattern = []
                for x,y in pattern:
                    line = i + x
                    if line < 0 or line >= len(sentence):
                        aux_pattern.append('U') #Missing info
                    else:
                        try:
                            att = sentence[line][y]
                            if att.strip():
                                aux_pattern.append(sentence[line][y])
                            else:
                                aux_pattern.append('U')
                        except IndexError:
                            raise ParsingError('Bad pattern input')
                word_attributes.append("'%s'"%(weka_compatible_string('-'.join(aux_pattern))))

            # Annotation
            current_entity = int(word[0].split('-')[1])
            try:
                annotation = seen_entities[preclustering[current_entity]]
            except (KeyError, IndexError):
                if (distrib == 'UNI' or distrib == 'OVA') and word[0] in to_annotate:
                    annotation = word[0]

                elif distrib == 'RND':
                    new_lab = randint(0, n_labels-1)
                    annotation = 'B-fake%d' %new_lab
                    unique_labels_used[new_lab] = 1

                else:
                    annotation = 'B-fake'
                    unique_labels_used[-3] = 1

                if len(preclustering) > current_entity:
                    seen_entities[preclustering[current_entity]] = annotation

            seen_classes.append(annotation)
            word_attributes.append(annotation)
            words_acc.append(','.join(word_attributes))
            train_length += 1

        if len(words_acc) > 0:
            wekadata.append('\n'.join(words_acc))

    return sum(unique_labels_used), train_length,  len(word_attributes) - 1, seen_classes, wekadata




def annotate_DT_test(n, test, features, fake_class):
    """
    Returns a weka-formatted version of the test entities for the DT classifier.

    Args:
     * ``n`` (*int*): step identifier.
     * ``test`` (*list*): test entities.
     * ``features`` (*list*): pattern for the feature selection.
     * ``fake_class``: fake weka class to give to all test entities for the weka format.
    Returns:
     * ``n_sentences_test`` (*int*): number of sequences in the training database.
     * ``test_entities`` (*list*): indices and identifiers of the entities of interest in the test database.
     * ``test_file`` (*str*): path to the  formatted test data.
    """
    # Test
    test_length = 0
    test_entities_indices = []
    wekatest = []

    for sentence in test:
        words_acc = []
        for i, word in enumerate(sentence):
            bio = word[-1]  #BIO
            if bio != 'B':
                continue

            test_entities_indices.append((len(test_entities_indices), word[0]))

            # Features selction
            word_attributes = []
            for k, pattern in enumerate(features):
                aux_pattern = []
                for x,y in pattern:
                    line = i + x
                    if line < 0 or line >= len(sentence):
                        aux_pattern.append('U') #Missing info
                    else:
                        try:
                            aux_pattern.append(sentence[line][y])
                        except IndexError:
                            raise ParsingError('Bad pattern input')
                word_attributes.append("'%s'"%(weka_compatible_string('-'.join(aux_pattern))))

            word_attributes.append(fake_class)
            words_acc.append(','.join(word_attributes))
            test_length += 1

        if len(words_acc) > 0:
            wekatest.append('\n'.join(words_acc))

    return test_length, test_entities_indices, wekatest



def weka_format(n, wekadata, train_length, test_length, temp_folder, verbose):
    """
    Formats the inout weka file to be compatible with J48 tree and splits it as a train and test file

    Args:
     * ``n`` (*int*): iteration identifier.
     * ``wekadata`` (*str*): input weka file.
     * ``train_length`` (*int*): number of attributes in train.
     * ``test_length`` (*int*): number of attributes in test.
     * ``temp_folder`` (*str*): path to temporary folder.
     * ``verbose`` (*int*): verbosity level.
    """

    full_length = train_length + test_length
    train_percentage = 100 * float(train_length) / full_length

    #---------- Weka formatting (string attributes)
    FNULL = open(os.devnull, 'w')
    classifier = os.environ.copy()['CLASSIFIER']
    err = None if verbose >=2 else FNULL

    p = Popen(('java -cp %s weka.filters.unsupervised.attribute.StringToWordVector'%(classifier)).split(), shell = False, stdout = PIPE, stdin = PIPE, stderr = err, close_fds = True)
    aux1 = p.communicate(input = str(wekadata))[0]

    p = Popen(('java -cp %s weka.filters.unsupervised.attribute.NumericToBinary'%(classifier)).split(), shell = False, stdout = PIPE, stdin = PIPE, stderr = err, close_fds = True)
    aux1 = p.communicate(input = aux1)[0]

    p = Popen(('java -cp %s weka.filters.unsupervised.attribute.Reorder -R 2-last,first'%(classifier)).split(), shell = False, stdout = PIPE, stdin = PIPE, stderr = err, close_fds = True)
    aux1 = p.communicate(input = aux1)[0]

    #------------- Train and Test splitting
    p = Popen(('java -cp %s weka.filters.unsupervised.instance.RemovePercentage -P %.2f -V'%(classifier, train_percentage)).split(), shell = False, stdout = PIPE, stdin = PIPE, stderr = err, close_fds = True)
    train = p.communicate(input = aux1)[0]

    p = Popen(('java -cp %s weka.filters.unsupervised.instance.RemovePercentage -P %.2f'%(classifier, train_percentage)).split(), shell = False, stdout = PIPE, stdin = PIPE, stderr = err, close_fds = True)
    test = p.communicate(input = aux1)[0]
    FNULL.close()

    #Note: Weka does not manage stdin input
    train_file = os.path.join(temp_folder, 'step%d_train.arff'%n)
    with open(train_file, 'w') as f:
        f.write(train)

    test_file = os.path.join(temp_folder, 'step%d_test.arff'%n)
    with open(test_file, 'w') as f:
        f.write(test)

    return train_file, test_file
