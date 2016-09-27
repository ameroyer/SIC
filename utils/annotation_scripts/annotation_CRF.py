#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**annotation_CRF.py.** Generation of synthetic annotations for wapiti CRF classifier.
"""

from random import randint
from itertools import takewhile


__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"



def annotate_CRF_train(n, train, distrib, n_labels, to_annotate, temp_folder, preclustering):
    """
    Returns a synthetic annotation of the data for the (wapiti) CRF classifier.

    Args:
     * ``n`` (*int*): iteration identifier.
     * ``train`` (*list*): training entities.
     * ``distrib`` (*str*): type of the synthetic annotation.
     * ``n_min`` (*int*): minimum number of synthetic labels to use.
     * ``n_max`` (*int*): maximum number of synthetic labels to use.
     * ``to_annotate`` (*list*): in case of UNI annotation, list of indices of the entities to have their own class.
     * ``with_common_label_wordform`` (*bool, optional*): if ``True``, each entity occurence wordform receives the same label. Defaults to ``False``.
     * ``temp_folder``: path to the directory for temporary files.
    Returns:
     * ``n_unique_labels_used`` (*int*): number of synthetic labels that were actually used.
     * ``n_sentences_train`` (*int*): number of sequences in the training database.
     * ``n_entities_train`` (*int*): number of entities in the training database.
     * ``train_file`` (*str*): path to the  formatted train data.
    """
    # Parameters
    unique_labels_used = [0] * (n_labels+3)   #counts the number of unique labels (N synthetic + 'O' + 'I')
    n_entities_train = 0
    seen_entities = {}
    if to_annotate is not None:
        unique_labels_used[:len(to_annotate)] = [1] * len(to_annotate)

    # Annotation
    sentences_acc = []
    for sentence in train:
        words_acc = []
        for i, word in enumerate(sentence):
            bio = word[-1]

            # Common word
            if bio == 'O':
                annotation = 'O-null'
                unique_labels_used[-1] = 1

            elif bio == 'I':
                annotation = 'I-in'
                unique_labels_used[-2] = 1

            # Interesting entity
            elif bio == 'B':
                n_entities_train += 1
                current_entity = int(word[0].split('-')[1])

                # Sample a synthetic label
                try:
                    annotation = seen_entities[preclustering[current_entity]]
                except (KeyError, IndexError):
                    if (distrib == 'UNI' or distrib == 'OVA') and (word[0] in to_annotate):
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

            words_acc.append('%s\t%s\t%s\t%s\t%s'%(word[0], word[1], '\t'.join(word[2:-1]), bio, annotation))
        sentences_acc.append('\n'.join(words_acc))

    return sum(unique_labels_used), len(sentences_acc), n_entities_train, '\n\n'.join(sentences_acc)




def annotate_CRF_test(n, test, temp_folder):
    """
    Returns a wapiti-formatted version of the test entities for the CRF classifier.

    Args:
     * ``n`` (*int*): step identifier.
     * ``test`` (*list*): test entities.
     * ``temp_folder``: directory for temporary files.
    Returns:
     * ``n_sentences_test`` (*int*): number of sequences in the training database.
     * ``test_entities_indices`` (*list*): indices and identifiers of the entities of interest in the test database.
     * ``test_file`` (*str*): path to the  formatted test data.
    """
    sentences_acc = []
    test_entities_indices = []
    n_line = 0

    # Write test data in the same format
    for sentence in test:
        words_acc = []
        for word in sentence:
            bio = word[-1]
            annotation = 'O-null' if bio == 'O' else 'doe'
            words_acc.append('%s\t%s\t%s\t%s\t%s'%(word[0], word[1], '\t'.join(word[2:-1]), bio, annotation))

            if bio == 'B':
                test_entities_indices.append((n_line, word[0]))
            n_line += 1

        sentences_acc.append('\n'.join(words_acc))
        n_line += 1

    test_file = '\n\n'.join(sentences_acc)

    return len(sentences_acc), test_entities_indices, '\n\n'.join(sentences_acc)
