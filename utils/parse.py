#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**parse.py.** Functions for parsing data files and ground-truth file for a given data set. The module can also be used as a script to generate ``.qrel`` version of the ground-truth for the information retrieval evaluation procedure.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"

import os
from error import *
from itertools import chain
from collections import defaultdict, Counter
from random import shuffle


###################################################################### PARSING INPUT DATA FILE
def parse_data(data_type, classification_params, data_file):
    """
    Parse data file(s) depending on the chosen options.

    Args:
     * ``data_type`` (*str*): dataset identifier.
     * ``classification_params`` (*str*): classifier parameters.
     * ``data_file`` (*str*): path to data file.

    Returns:
     * ``n_samples`` (*int*): number of samples in the database.
     * ``data`` (*list*): structure containing the data.
     * ``data_occurrences`` (*list*): number of entity occurrences in each sentence/docs of the data (only for AQUAINT and NER).
     * ``index_to_label`` (*list*): list associating a sample's index with its string representation.
     * ``label_to_index`` (*list*): reversed mapping of index_to_label.
    """

    print >> sys.stderr, "Parsing Data"
    classifier_type = classification_params['classifier_type']

    # (ESTER2 dataset)
    if data_type == 'NER':
        n_samples, data, data_occurrences, index_to_label = parse_NER(classifier_type, data_file)
        label_to_index = {('%d-%s'%(i,s)) : i for i, s in enumerate(index_to_label)}
        assert(len(data) == len(data_occurrences)), "Error in data_occurrences structure"

    # (AQUAINT2 dataset)
    elif data_type == 'AQUA':
        precomputed_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'Precomputed/')
        # Get list of entities
        aqua_entities = os.path.join(precomputed_folder, 'aquaint_entities_list')
        n_samples, index_to_label, label_to_index = parse_AQUA_entities(aqua_entities)

        # Get a "relevance" score for each file (high score <-> rare words occuring)
        docs_scores_file = os.path.join(precomputed_folder, 'aquaint_docs_scores')
        docs_scores = defaultdict(lambda:0)
        docs_number = defaultdict(lambda:0)
        with open(docs_scores_file, 'r') as f:
            for line in f:
                doc, _, file_name, doc_in_file, score = line.split()
                docs_scores[file_name] += float(score)
                docs_number[file_name] += 1
        docs_scores = {x : y / docs_number[x] for x,y in docs_scores.iteritems()}

        # Store everything into the data variable
        data, data_occurrences = (data_file, label_to_index, docs_scores), None

    # (AUDIO datasets)
    elif data_type in ['AUDIO', 'AUDIOTINY']:
        n_samples, data, index_to_label = parse_audio(data_file, classification_params['features'])
        label_to_index = {s : i for i, s in enumerate(index_to_label)}
        data_occurrences = None

    return n_samples, data, data_occurrences, index_to_label, label_to_index





################################################################################# PARSING NER
def parse_NER(classifier_type, data_file):
    """
    Reads and parses the given data file for the Named Entity Recognition (NER) task.

    Args:
     * ``classifier_type`` (*str*): type of the classifier that will be used for the experiments.
     * ``data_file`` (*str*): path to data file.

    Returns:
     * ``n_samples`` (*int*): number of samples in the database.
     * ``data`` (*list*): structure containing the data. A word is usually represented by a tuple (index of the sample if interesting entity, word representant of the sample, additional tags, B(I)O tag).
     * ``data_occurrences`` (*list*): number of occurences of each word in each sentence.
     * ``index_to_label`` (*list*): list assocating a sample's index with its string representation.
     * ``summary`` (*str*): additional information on the dataset.
    """

    data = []
    data_occurrences = []
    index_to_label = []         # Named entity index to string representation
    samples = 0                 # Index of current entity

    if classifier_type in ['CRF', 'DT']:
        index = "O-None"
        acc_entity = ""
        acc_sentence = []
        acc_occurences = defaultdict(lambda: 0)

        with open(data_file, 'r') as f:
            for line in f:
                # Parse line
                if line.strip():

                    # Ignore missing entries
                    attr = line.split()
                    if sum([bool(x.strip()) for x in attr]) < len(attr):
                        continue

                    # Attributes
                    word = attr[1]
                    aux = attr[-1].split('-')
                    tags = attr[2:-1]

                    # Build index_to_label list
                    if aux[0] == 'null': # End of current entity
                        if acc_entity:
                            index_to_label.append(acc_entity)
                            samples += 1
                            acc_entity = ''
                        biotag, index = "O", "O-None"

                    elif aux[1] == 'B': # End of current entity and beginning of a new one
                        if acc_entity:
                            index_to_label.append(acc_entity)
                            samples += 1
                        acc_entity =  word
                        acc_occurences[samples] = 1
                        biotag, index = "B", 'B-%d'%samples

                    elif aux[1] == 'I': # Still in current entity
                        acc_entity += ' %s'%word
                        biotag, index = "I", "I-%d"%samples

                    # Add data entry: Entity index, Word, additional tags, BIO
                    acc_sentence.append(tuple([index, word] + tags + [biotag]))

                # Empty line = end of sentence
                else:
                    if acc_sentence:
                        data.append(acc_sentence)
                        data_occurrences.append(Counter(acc_occurences))
                        acc_sentence = []
                        acc_occurences = defaultdict(lambda:0)

        # End of file: flush current named entity if needed
        if acc_entity:
            index_to_label.append(acc_entity)
            samples += 1

    # Lighter data_occurrences structure
    import numpy as np
    from scipy.sparse import csr_matrix
    for i, sent_acc in enumerate(data_occurrences):
        sent_acc_arr = np.zeros(samples, dtype = int)
        for k, o in sent_acc.iteritems():
            sent_acc_arr[k] = o
        data_occurrences[i] = csr_matrix(sent_acc_arr)

    return samples, data, data_occurrences, index_to_label




################################################################################# PARSING AQUAINT
def parse_AQUA_entities(entities_file):
    """
    Parse the file containing all entities of the AQUAINT2 dataset to build the index_to_label and label_to_index mappings.

    Args:
     * ``entities_file`` (*str*): file containing the retrieved entities and number of occurences.

    Returns:
     * ``index_to_label`` (*list*): list assocating a sample's index with its string representation.
     * ``label_to_index`` (*list*): reversed mapping of index_to_label.
    """
    index_to_label = []
    label_to_index = {}
    with open(entities_file, 'r') as f:
        for i, line in enumerate(f):
            word = line.split('\t')[0].strip()
            index_to_label.append(word)
            label_to_index[word] = i

    return len(index_to_label), index_to_label, label_to_index



def parse_AQUA_single_partial(classifier_type, data_file, label_to_index, training_size, testing_size, train_acc, test_acc, test_occurences, train_included_test=False):
    """
    Read and parse partially the given Aquaint2 data file. Contrary to ``parse_AQUA_single``, this function does not load the whole data, but directly builds the required training and testing sets.

    Args:
     * ``classifier_type`` (*str*): type of the classifier that will be used for the experiment.
     * ``data_file`` (*str*): path tothe data file.
     * ``label_to_index`` (*str*): maps a word to an integer index (alphabetical order). Used to maps multiple occurences of a same word to the same index.
     * ``training_size`` (*int*): number of docs for training from this document, or a list of docs indices + sentences to keep.
     * ``testing_size`` (*int*): number of docs for testing from this document, or a list of docs indices + sentences to keep.
     * ``train_acc`` (*iterator*): accumulator for training sentences.
     * ``test_acc`` (*iterator*): accumulator for testing sentences.
     * ``test_indices`` (*array*): accumulator for the number of occurrences of each word in the test database.
     * ``train_included_test`` (*bool, optional*): If ``True``, retrieved training sentences will also be included in the testing set.

    Returns:
     * ``train_acc`` (*iterator*): updated training sentences accumulator.
     * ``test_acc`` (*iterator*): updated testing sentences accumulator.
    """

    # Parameters
    n_samples = len(label_to_index)
    cstr_tr = (not (type(training_size) is int)) # Constrained training set
    cstr_te = (not (type(testing_size) is int)) # Constrained testing set

    # Indicate where to break the parsing of a documents, depending on wether the list of sentences was constrained or not
    if cstr_tr and cstr_te:
        def break_sentence(sentence, max_sentence, train_doc, test_doc):
            return sentence > max_sentence
    elif cstr_tr:
        def break_sentence(sentence, max_sentence, train_doc, test_doc):
            return train_doc and (not test_doc) and (sentence > max_sentence)
    elif cstr_te:
        def break_sentence(sentence, max_sentence, train_doc, test_doc):
            return test_doc and (not train_doc) and (sentence > max_sentence)
    else:
        def break_sentence(sentence, max_sentence, train_doc, test_doc):
            return False

    # Compute number of documents
    with open(data_file, 'r') as f:
        n_docs = 0
        for line in f:
            if line.strip().startswith('<DOC '):
                n_docs += 1
        limit_doc = n_docs

    # If taking all documents
    if testing_size == -1:
        testing_size = n_docs
        training_size = 0
    if training_size == -1:
        testing_size = 0
        training_size = n_docs

    # ------------------ Select train and test documents
    indx = range(n_docs)
    shuffle(indx)
    if cstr_tr and cstr_te:
        training_indices = training_size.keys()
        testing_indices = testing_size.keys()
    elif cstr_tr:
        training_indices = training_size.keys()
        testing_indices = indx[:testing_size]
    elif cstr_te:
        testing_indices = testing_size.keys()
        training_indices = indx[:training_size]
    else:
        training_indices = indx[:training_size]
        testing_indices = indx[:testing_size+training_size] if train_included_test else indx[training_size:(testing_size+training_size)]
    del indx

    # -------------------- Determines index of last document required to be parsed
    if len(training_indices) > 0 and len(testing_indices) > 0:
        limit_doc = max(max(training_indices), max(testing_indices))
    elif len(training_indices) == 0  and len(testing_indices) > 0:
        limit_doc = max(testing_indices)
    elif len(testing_indices) == 0  and len(training_indices) > 0:
        limit_doc = max(training_indices)
    else:
        limit_doc = 0

    # --------------------- Browse the given file
    with open(data_file, 'r') as f:
        train_doc, test_doc = False, False # is current doc a test or train doc
        train_sentence, test_sentence = True, True # is current sentence a train or test sentence
        training_sentences, testing_sentences = [], []
        words_acc = []
        doc_id = -1

        for line in f:
            l = line.strip()

            # New document
            if l.startswith('<DOC '):
                doc_id += 1
                if doc_id > limit_doc:
                    break
                train_doc = (doc_id in training_indices)
                test_doc = (doc_id in testing_indices)

                # Determines which sentences to parse if constrained
                max_train_sentence = -1
                max_test_sentence = -1
                if train_doc and cstr_tr:
                    training_sentences = training_size[doc_id]
                    max_train_sentence = max(training_sentences)
                if test_doc and cstr_te:
                    testing_sentences = testing_size[doc_id]
                    max_test_sentence = max(testing_sentences)
                max_sentence = max(max_train_sentence, max_test_sentence)

                #Accumulator for setence
                train_sentences_acc = []
                test_sentences_acc = []
                sentence_id = -1

            elif (train_doc or test_doc):
                # End document
                if l.startswith('</DOC>'):
                    if train_doc:
                        train_acc = chain(train_acc, train_sentences_acc)
                    if test_doc:
                        test_acc = chain(test_acc, test_sentences_acc)

                elif not break_sentence(sentence_id, max_sentence, train_doc, test_doc):
                    # Start sentence
                    if l.startswith('<S>'):
                        words_acc = []
                        sentence_id += 1
                        train_sentence = train_doc and ((not cstr_tr) or (sentence_id in training_sentences))
                        test_sentence = test_doc and ((not cstr_te) or (sentence_id in testing_sentences))

                    # End sentence
                    elif l.startswith('</S>'):
                        if train_sentence:
                            train_sentences_acc.append(words_acc)
                        if test_sentence:
                            test_sentences_acc.append(words_acc)


                    # Parse word + ignore if missing value
                    elif l.startswith('<w ') and (train_sentence or test_sentence):
                        try:
                            _,word,_,pos,_,lem,_ = l.split('"') # word occurence, POS-tag, lemmatized word

                            if not (word.strip() and pos.strip() and lem.strip()):
                                raise ValueError
                        except ValueError:
                            continue

                        # Check if word is an Aquaint entity
                        try:
                            ind = str(label_to_index[lem])
                            if not pos in ['NN', 'NNS']: # Not-nouns Homonyms
                                raise KeyError
                            biotag = 'B'
                            if test_sentence:
                                test_occurences[int(ind)] += 1
                        except KeyError:
                            ind = 'None'
                            biotag = 'O'

                        #Add data entry: Entity index, word, tags, BIO
                        words_acc.append((biotag + '-' + ind, lem, pos, word, biotag))


    return train_acc, test_acc



################################################################################### AUDIO
def parse_audio(data_folder, selected_features):
    """
    Parse features selection for the Audio task.

    Args:
     * ``data_folder`` (*str*): If the features are precomputed, then ``data_folder`` is the path to the directory containing the features. Otherwise, it is a file containing the samples folder as its first line and all the possible HTK features markers (one per line) to consider.
     * ``selected_features`` (*list*): list of the features type to use in the experiments (given in the configuration file).

    Returns:
     * ``n_samples`` (*int*): number of samples in the dataset.
     * ``data`` (*list*): maps a feature identifier to the corresponding HTK generated features.
     * ``index_to_label`` (*lsit*): maps an entity index to a string label.
    """
    data = {}
    if os.path.isfile(data_folder):
        with open(data_folder, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    data['Samples'] = line.strip()
                    samples = [x.rsplit('.', 1)[0] for x in os.listdir(line.strip()) if x.endswith('.wav')]
                    index_to_label = sorted(samples)
                    n_samples = len(samples)
                elif not line.startswith('#') and line.split('_', 1)[0].strip() in selected_features:
                    data[line.strip()] = []

    elif os.path.isdir(data_folder):
        features_folder = [(d.rsplit('_', 1)[0], os.path.join(data_folder, d)) for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]

        data = {}
        if len(features_folder) > 0:
            for key, folder in features_folder:
                if key.split('_', 1)[0] in selected_features:
                    features = [f for f in os.listdir(folder) if f.endswith('.mfc')]
                    features = sorted(features)
                    index_to_label = [x.rsplit('.', 1)[0] for x in features]

                    data[key] = [(i, os.path.join(folder, x)) for i, x in enumerate(features)]

            n_samples = len(features)
        else:
            print >> sys.stderr, 'Error, no  features directory found at %s' %data_folder
            raise SystemExit
    else:
        print >> sys.stderr, 'Error, %s not found' %data_folder
        raise SystemExit

    return n_samples, data, index_to_label







################################################################################### PARSING GROUND-TRUTH
def parse_ground_truth(data_type, ground_truth_file, label_to_index=None):
    """
    Reads and parses the given ground-truth file.

    Args:
     * ``data`` (*list*): structure containing the data.
     * ``ground_truth_file`` (*str*): path to the ground truth file.
     * ``label_to_index`` (*dict, optional*): maps an AQUAINT word to its index. Required when outputting  the AQUAINT groundtruth with the entity indices rather than string representation.

    Returns:
     * ``ground_truth`` (*list*): list associating a sample with its ground-truth cluster:

       * for NER, ``ground_truth``: cluster (*str*) -> entity indices (*int list*)
       * for AQUA, ``ground_truth``: entity (*str*) -> entity indices (*int list*) if label_to_index, else str list
    """
    ground_truth = defaultdict(lambda: [])

    # ESTER2 dataset
    if data_type in ['NER', 'AUDIO', 'AUDIOTINY']:
        n_entity = 0
        with open(ground_truth_file, 'r') as f:
            for line in f:
                if line.strip():
                    cl, _ = line.split('\t')
                    ground_truth[cl].append(n_entity)
                    n_entity += 1

    #  AQUAINT2 dataset
    elif data_type == 'AQUA':
        with open(ground_truth_file, 'r') as f:
            if label_to_index is not None:
                for line in f:
                    if line.strip():
                        aux = line.replace('#n', '').split()
                        word = aux[0].strip()
                        syn = [y.strip() for y in aux[1:] if y in label_to_index]
                        if word in label_to_index and len(syn) > 0:
                            ground_truth[word] = syn
            else:
                for line in f:
                    if line.strip():
                        syn = [y.strip() for y in line.replace('#n', '').split()]
                        ground_truth[syn[0]] = syn[1:]

    return ground_truth





################################################################## ADDITIONAL PARSING OF GROUND-TRUTH

def split_on_ground_truth_no_indices(data_type, ground_truth_file, numb=6, keys=None):
    """
    Given the ground_truth data file, returns a random set number of entities for each class (usually used to visualize similarity distributions)

    Args:
     * ``data`` (*list*): structure containing the data.
     * ``ground_truth_file`` (*str*): path to the ground truth file.
     * ``numb`` (*int*): number to plot for each entity
     * ``keys`` (*list, optional*): if given, the algorithm returns a list of samples whose ground-truth classes form the ``keys`` list.

    Returns:
     * ``ground_truth`` (*list*): list associating a sample with its ground-truth cluster.
     * ``selected_entities`` (*list*): selected entities to be plotted.
    """
    from random import choice, shuffle
    ground_truth = defaultdict(lambda: [])

    # --------------------------- NER
    if data_type in ['NER', 'AUDIO', 'AUDIOTINY']:
        # Ground-truth
        n_entity = 0
        with open(ground_truth_file, 'r') as f:
            for line in f:
                if line.strip():
                    cl, _ = line.split('\t')

                    ground_truth[cl].append(n_entity)
                    n_entity += 1

        # Select a subset of entities
        selected_entities = []
        if keys:
            for w in range(numb):
                selected_entities += [(choice(ground_truth[key]), key) for key in keys]
        else:
            for w in range(numb):
                selected_entities += [(choice(x), key) for key, x in ground_truth.iteritems()]


    #--------------------------- AQUA
    elif data_type == 'AQUA':
        precomputed_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'Precomputed/')
        aqua_entities = os.path.join(precomputed_folder, 'aquaint_entities_list')
        _, index_to_label, label_to_index = parse_AQUA_entities(aqua_entities)
        selected_entities = []
        with open(ground_truth_file, 'r') as f:
            lines = f.read().split('\n')
            if keys:
                shuffle(keys)
                lines = [lines[x] for x in keys[:numb]]
            else:
                shuffle(lines)
                lines = lines[:numb]
            for line in lines:
                if line.strip():
                    syn = [y.strip() for y in line.replace('#n', '').split()]
                    ground_truth[syn[0]] = [label_to_index[y] for y in syn[1:] if y in label_to_index]
                    selected_entities.append((label_to_index[syn[0]], syn[0]))

    return ground_truth, selected_entities




def ground_truth_indices(data_type, ground_truth_file, label_to_index=None):
    """
    Return indices of the entities that are in the ground truth. Only used for AQUAINT as for NER, samples = ground_truth

    Args:
     * ``data_type`` (*str*): dataset identifier
     * ``ground_truth_file`` (*str*): path to the ground_truth_file
     * ``label_to_index`` (*list*): maps a label to the corresponding entity index. Required for Aquaint.

    Returns:
     * ``indices`` (*list*): indices of samples in ground-truth.
    """

    indices = []
    if data_type == 'AQUA':
        if label_to_index is None:
            print >> sys.stderr, 'Missing label_to_index in parse.ground_truth_indices'
            raise SystemExit

        with open(ground_truth_file, 'r') as f:
            for line in f:
                if line.strip():
                    aux = line.replace('#n', '').split()
                    word = aux[0].strip()
                    syn = [y.strip() for y in aux[1:] if y in label_to_index]
                    if word in label_to_index and len(syn) > 0:
                        indices.append(label_to_index[word])
        return indices

    elif data_type in ['NER', 'AUDIO', 'AUDIOTINY']:
        with open(ground_truth_file) as f:
            n_entities = len([l for l in f if l.strip()])
        return range(n_entities)




def ground_truth_pairs(data_type, ground_truth_file, n_samples):
    """
    Retrieves indices of samples'pairs that have the same ground-truth class. Indices are computed as if using the upper triangle of a squqre n_samples x n_samples matrix.

    Args:
     * ``data_type`` (*str*): data set.
     * ``ground_truth_file`` (*str*): ground-truth file for the given data set.
     * ``n_samples`` (*str*): number of samples for the given data set.

    Return:
     * ``indices`` (*int ist*): list of indices of samples pairs with same ground-truth class.
    """
    ground_truth = defaultdict(lambda: [])
    indices = []

    # --------------------------- NER
    if data_type in ['NER', 'AUDIO', 'AUDIOTINY']:
        with open(ground_truth_file, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    cl, _ = line.split('\t')

                    for j in ground_truth[cl]:
                        if i < j:
                            index = j + n_samples * i - (i+1)*(i+2)/2
                        else:
                            index = i + n_samples * j - (j+1)*(j+2)/2
                        indices.append(index)

                    ground_truth[cl].append(i)

    #--------------------------- AQUA
    elif data_type == 'AQUA':
        print 'Not (yet ?) implemented for AQUAINT'
        indices = None

    return indices




########################################## QREL CONSTRUCTION FOR KNN RETRIEVAL
def ground_truth_AQUA_qrel(ground_truth_file, output_file, aqua_entities_file):
    """
    Builds the qrel file for the AQUAINT ground-truth to be used for the nearest neighbour evaluation.

    Args:
     * ``ground_truth_file`` (*str*): path to the AQUAINT ground-truth file.
     * ``output_file`` (*str*): path to the qrel output file.
     * ``aqua_entities_file`` (*str*): path to the file listing all AQUAINT entities.
    """
    _, index_to_label, label_to_index = parse_AQUA_entities(aqua_entities_file)
    out = open(output_file, 'w')
    with open(ground_truth_file, 'r') as f:
        for line in f:
            if line.strip():
                syn = [y.strip() for y in line.replace('#n', '').split()]
                if syn[0] in label_to_index:
                    indx = label_to_index[syn[0]]
                    out.write('\n'.join(['%d 0 %s 1'%(indx, s) for s in syn[1:] if s in label_to_index]) + '\n')
    out.close()



def ground_truth_qrel(ground_truth_file, output_file, index_to_label):
    """
    Builds the qrel file for NER and AUDIO ground-truth to be used for the nearest neighbour evaluation.

    Args:
     * ``ground_truth_file`` (*str*): path to the AQUAINT ground-truth file.
     * ``output_file`` (*str*): path to the qrel output file.
     * ``index_to_label`` (list*): maps an entity index to the corresponding string label.
    """
    out = open(output_file, 'w')
    ground_truth = defaultdict(lambda: [])
    n_entity = 0
    with open(ground_truth_file, 'r') as f:
        for line in f:
            if line.strip():
                    cl, label = line.split('\t')
                    ground_truth[cl].append(n_entity)
                    n_entity += 1

    for classe, entities in ground_truth.iteritems():
        for x in entities:
            out.write('\n'.join(['%d 0 %s 1'%(x,index_to_label[y]) for y in entities if x != y]) + '\n')
    out.close()
    out.close()



################################################################################ PARSING CRF/DT PATTERN
def parse_pattern(classifier_type, pattern_file):
    """
    Reads and parses the given pattern file (Wapiti/CRF++ expected format).

    Args:
     * ``classifier_type`` (*str*): type of the classifier that will be used for the experiment.
     * ``pattern_file`` (*str*): path to the pattern file.

    Returns:
     * ``features`` (*list*): features organized by category.
     * ``distrib`` (*list*): probability of sampling a feature for each category.
    """

    features = [] # Manage features by category
    distrib = []

    if classifier_type in ['CRF', 'DT']:
        acc = []
        with open(pattern_file, 'r') as f:
            for line in f:
                l = line.strip()

                if not l and acc: #Empty line: new category
                    features.append(acc)
                    acc = []

                elif l and not l.startswith('#'): #Add pattern
                    acc.append(l)

                elif l.startswith('##P='): #Add distrib
                    distrib.append(float(l.split()[0].split('=', 1)[1]))
        if acc:
            features.append(acc)

        print 'Pattern distribution:', distrib
        return features, distrib




################################################################################ AS MAIN: GENERATE QREL
if __name__ == '__main__':
    root_dir = '../'

    ## Qrel for AUDIO
    gt_audio = os.path.join(root_dir, 'Data/Audio/ESTER2/audio_ester2.cluster')
    gt_audio_homo = os.path.join(root_dir, 'Data/Audio/ESTER2/audio_ester2_homonym.cluster')
    index_to_label_audio =  os.path.join(root_dir, 'Data/Audio/ESTER2/index_to_labels_audio_ester2')

    with open(index_to_label_audio, 'r') as f:
        index_to_label = ['%s'%(l.split('\t')[1].strip()) for l in f if l.strip()]
    ground_truth_qrel(gt_audio, '%s.qrel'%gt_audio, index_to_label)
    ground_truth_qrel(gt_audio_homo, '%s.qrel'%gt_audio_homo, index_to_label)

    ## Qrel for NER
    gt_ner = os.path.join(root_dir, 'Data/NER/ester2_dev.cluster')
    index_to_label_ner = os.path.join(root_dir, 'Data/NER/index_to_labels_NER')
    with open(index_to_label_ner, 'r') as f:
        index_to_label = ['%d-%s'%(i, l.split('\t')[1].strip().replace(' ', '_')) for i, l in enumerate(f) if l.strip()]
    ground_truth_qrel(gt_ner, '%s.qrel'%gt_ner, index_to_label)
