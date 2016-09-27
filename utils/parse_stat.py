#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**parse.py_stat.** Additional functions for parsing some statistics and precomputing information on the data sets (mostly Aquaint).
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"

import os
from error import *
from collections import defaultdict, Counter
import numpy as np
import pickle

#Notes on the Aquaint2 dataset
  #Bug1: Some XML failures (some lines are badly formatted)
  #Bug2: Some empty/missing tags
  #Bug3: Some words in the ground-truth not appearing > 10 (at least: Selenium - Se ?)


################################################################################# FULL AQUA PARSING

def parse_AQUA_single(classifier_type, data_file, label_to_index):
    """
    Read and parse the full given AQUAINT data file.

    Args:
     * ``classifier_type`` (*str*): type of the classifier that will be used for the experiment.
     * ``data_file`` (*str*): path tothe data file.
     * ``label_to_index`` (*str*): maps a word to an integer index (alphabetical order). Used to maps multiple occurences of a same word to the same index.

    Returns:
     * ``data`` (*list*): structure containing the data (file -> docs -> sentence -> words).
     * ``data_occurrences`` (*list*): number of entity occurrences in each sentence/docs of the data.
    """

    # Direct manual parsing (instead of using etree or lxml); seems to be faster + tolerates bad format in input files
    n_samples = len(label_to_index)

    with open(data_file, 'r') as f:
        if classifier_type in ['CRF', 'DT']:
            docs_acc = []
            data_occurences = []

            for line in f:
                l = line.strip()

                # New document
                if l.startswith('<DOC'):
                    sentences_acc = []

                # End document
                elif l.startswith('</DOC>'):
                    docs_acc.append(sentences_acc)
                    data_occurences.append(data_occurences_acc)
                    data_occurences_acc = np.zeros(n_samples, dtype = int)

                # Start sentence
                elif l.startswith('<S>'):
                    words_acc = []

                # End sentence
                elif l.startswith('</S>'):
                    sentences_acc.append(words_acc)


                elif l.startswith('<w '):
                    # Parse word + ignore if missing value
                    try:
                        _,word,_,pos,_,lem,_ = l.split('"') # word occurence, POS-tag, lemmatized word

                        if not (word.strip() and pos.strip() and lem.strip()):
                            raise ValueError
                    except ValueError:
                        continue

                    # Check if word is an interesting entity
                    try:
                        ind = str(label_to_index[lem])
                        if not pos in ['NN', 'NNS']:
                            raise KeyError
                        biotag = 'B'
                        data_occurences_acc[int(ind)] += 1
                    except KeyError:
                        ind = 'None'
                        biotag = 'O'

                    #Add data entry: Entity index, word, tags, BIO
                    words_acc.append((biotag + '-' + ind, lem, pos, word, biotag))

    return docs_acc, data_occurences




def parse_AQUA(classifier_type, data_folder, label_to_index):
    """
    Read and parse all data files from the AQUAINT folder.

    Args:
     * ``classifier_type`` (*str*): type of the classifier that will be used for the experiment.
     * ``data_folder`` (*str*): path to folder containing all data files.
     * ``label_to_index`` (*str*): maps a word to an integer index (alphabetical order). Used to maps multiple occurences of a same word to the same index.

    Returns:
     * ``data`` (*list*): structure containing the data (file -> docs -> sentence -> words).
     * ``data_occurrences`` (*list*): number of entity occurrences in each sentence/docs of the data.
     * ``summary`` (*str*): additional information on the dataset.
    """
    data_files = [f for f in os.listdir(data_folder) if f.endswith('.xml.u8')]
    files = []
    occurrences = []

    # Parsing
    for i, d in enumerate(data_files):
        print >> sys.stderr, '\n -------> Parsing data file %s: %d/%d' %(d, i+1, len(data_files))
        file, occurence = parse_AQUA_single(classifier_type, os.path.join(data_folder, d), label_to_index)
        files[i] = file
        occurrences[i] = occurrence

    summary = '%d documents in %d files' %(sum([len(x) for x in files]), len(files))
    return files, occurrences, summary




###################################################################### ADDITIONAL PARSING FOR AQUA FILES
def retrieve_aqua_entities(directory, output_file):
    """
    Retrieves all interesting entities for the AQUAINT2 dataset (common names with strictly more than 10 occurences).

    Args:
     * ``directory`` (*str*): directory containing all xml documents of the dataset.
     * ``output_file`` (*str*): path where to output the retrieved entities and number of occurences.
    """
    counts = defaultdict(lambda: 0)    # count occurences
    documents = [f for f in os.listdir(directory) if f.endswith('.xml.u8')]

    # Parsing all documents
    for k, d in enumerate(documents):
        print '\n -------> Parsing document %s: %d/%d' %(d, k+1, len(documents))
        with open(os.path.join(directory, d), 'r') as f:
            for i, line in enumerate(f):
                l = line.strip()
                if l.startswith('<w '):
                    try:
                        _,word,_,pos,_,lem,_ = l.split('"') # word occurence, POS-tag, lemmatized word

                        if not (word.strip() and pos.strip() and lem.strip()):
                            raise ValueError

                        if pos in ['NN', 'NNS'] and lem.strip() != 'xxxxx':
                            counts[lem.strip()] += 1

                    except ValueError:
                        print 'Warning; Parsing Error: line %d -  %s' %(i,l)

    # Writing output
    to_write = [(x, counts[x]) for x in sorted(counts.keys()) if counts[x] > 10]
    print len(to_write), 'retrieved names with at least 10 occurences'
    with open(output_file, 'w') as f:
        f.write('\n'.join(['%s\t%s'%(a,b) for a,b in to_write]))



def stat_aqua(directory):
    """
    Computes some statistics about the AQUAINT dataset.

    Args:
     * ``directory`` (*str*): directory containing all xml documents of the dataset.
    """
    words = defaultdict(lambda:0)
    names = defaultdict(lambda:0)
    docs = 0
    sentences = 0
    files = 0
    documents = [f for f in os.listdir(directory) if f.endswith('.xml.u8')]

    # Parsing all documents
    for k, d in enumerate(documents):
        print '\n -------> Parsing document %s: %d/%d' %(d, k+1, len(documents))
        with open(os.path.join(directory, d), 'r') as f:
            for i, line in enumerate(f):
                l = line.strip()
                if l.startswith('<DOC '):
                    docs += 1

                elif l.startswith('<S>'):
                    sentences += 1

                elif l.startswith('<w '):
                    try:
                        _,word,_,pos,_,lem,_ = l.split('"') # word occurence, POS-tag, lemmatized word

                        if not (word.strip() and pos.strip() and lem.strip()):
                            raise ValueError

                        words[lem.strip()] += 1
                        if pos in ['NN', 'NNS'] and not lem.strip() == 'xxxxx':
                            names[lem.strip()] += 1

                    except ValueError:
                        print 'Warning; Parsing Error: line %d -  %s' %(i,l)


    # Writing output
    print 'AQUA Summary \n  > %d files\n  > %d documents\n  > %d sentences\n\n%d unique words\n%d unique names including %d interesting entities'%(len(documents), docs, sentences, len(words), len(names), len([x for x in names.values() if x > 10]))




def count_aqua_docs(directory, output_folder, aqua_entities_file):
    """
    Counts the number occurrences of each word in each document as well as for the whole data set

    Args:
     * ``directory`` (*str*): directory containing all xml documents of the dataset.
     * ``output_folder`` (*str*): path to output directory.
     * ``aqua_entities_file`` (*str*): path to the file listing all AQUAINT entities.
    """
    all_docs_occ = Counter({})
    n_samples, index_to_label, label_to_index = parse_AQUA_entities(aqua_entities_file)
    documents = [f for f in os.listdir(directory) if f.endswith('.xml.u8')]

    # Parsing all documents
    for i, d in enumerate(documents):
        output = os.path.join(output_folder, d) + '.pkl'
        print '\n -------> Parsing document %s: %d/%d' %(d, i+1, len(documents))
        with open(os.path.join(directory, d), 'r') as f:
            total_occ = []

            for i, line in enumerate(f):
                l = line.strip()

                if l.startswith('<DOC '):
                    occ = defaultdict(lambda:0)

                elif l.startswith('</DOC>'):
                    total_occ.append(dict(occ))

                if l.startswith('<w '):
                    try:
                        _,word,_,pos,_,lem,_ = l.split('"') # word occurence, POS-tag, lemmatized word

                        if not (word.strip() and pos.strip() and lem.strip()):
                            raise ValueError

                        if lem.strip() in label_to_index:
                            occ[label_to_index[lem.strip()]] += 1

                    except ValueError:
                        print 'Warning; Parsing Error: line %d -  %s' %(i,l)
        all_docs_occ += Counter(dict(occ))
        with open(output, 'wb') as o:
            pickle.dump(total_occ, o)

    #Save all occs
    output = os.path.join(output_folder, 'all_documents_occurrences.pkl')
    with open(output, 'wb') as o:
        pickle.dump(dict(all_docs_occ), o)



def count_aqua_docs_score(directory, output_file, aqua_entities_file):
    """
    Computes a score for each document based on how rare are the words occuring in the document.

    Args:
     * ``directory`` (*str*): path to the directory containing the Aquaint data files.
     * ``output_file`` (*str*): path to the file to write the output scores.
     * ``aqua_entities_file`` (*str*): path to the file containing all aquaint entities and their number of occurrences.
    """
    n_samples, index_to_label, label_to_index = parse_AQUA_entities(aqua_entities_file)

    all_docs_occ = defaultdict(lambda:0)
    with open(aqua_entities, 'r') as f:
        for line in f:
            noun, occ = line.split('\t')
            all_docs_occ[label_to_index[noun.strip()]] = int(occ)

    documents = [f for f in os.listdir(directory) if f.endswith('.xml.u8')]
    out = open(output_file, 'w')
    doc_id = -1

    # Parsing all documents
    for k, d in enumerate(documents):
        print '\n -------> Parsing document %s: %d/%d' %(d, k+1, len(documents))
        with open(os.path.join(directory, d), 'r') as f:
            doc_id_in_file = -1

            for i, line in enumerate(f):
                l = line.strip()

                if l.startswith('<DOC '):
                    occ = defaultdict(lambda:0)
                    doc_id_in_file += 1
                    doc_id += 1

                elif l.startswith('</DOC>'):
                    score = np.sum([ float(occ[x]) / all_docs_occ[x] for x in xrange(len(index_to_label))])
                    out.write('%d\t%d\t%s\t%d\t%s\n'%(doc_id, k, d, doc_id_in_file, score))

                if l.startswith('<w '):
                    try:
                        _,word,_,pos,_,lem,_ = l.split('"') # word occurence, POS-tag, lemmatized word

                        if not (word.strip() and pos.strip() and lem.strip()):
                            raise ValueError

                        if lem.strip() in label_to_index and pos in ['NN', 'NNS']:
                            occ[label_to_index[lem.strip()]] += 1

                    except ValueError:
                        print 'Warning; Parsing Error: line %d -  %s' %(i,l)

    out.close()





def retrieve_aqua_occurrences(directory, output_file, aqua_entities_file):
    """
    Retrieve all occurrences of each word in the dataset (position of their occurrences given as a tupple file -> doc -> sentence)

    Args:
     * ``directory`` (*str*): path to the directory containing the Aquaint data files.
     * ``output_file`` (*str*): path to the directory to write the output files (1 file = 1 word).
     * ``aqua_entities_file`` (*str*): path to the file containing all aquaint entities and their number of occurrences.
    """
    n_samples, index_to_label, label_to_index = parse_AQUA_entities(aqua_entities_file)
    all_docs_occ = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:[])))
    documents = [f for f in os.listdir(directory) if f.endswith('.xml.u8')]


    # Parsing all documents
    for k, d in enumerate(documents):
        print '\n -------> Parsing document %s: %d/%d' %(d, k+1, len(documents))
        with open(os.path.join(directory, d), 'r') as f:
            doc_id= -1

            for i, line in enumerate(f):
                l = line.strip()

                if l.startswith('<DOC '):
                    doc_id += 1
                    sentence_id = -1

                elif l.startswith('<S>'):
                    sentence_id += 1

                elif l.startswith('<w '):
                    try:
                        _,word,_,pos,_,lem,_ = l.split('"') # word occurence, POS-tag, lemmatized word

                        if not (word.strip() and pos.strip() and lem.strip()):
                            raise ValueError

                        if lem.strip() in label_to_index and pos in ['NN', 'NNS']:
                            all_docs_occ[lem][d][doc_id].append(sentence_id)

                    except ValueError:
                        print 'Warning; Parsing Error: line %d -  %s' %(i,l)

    for key in all_docs_occ:
        for doc in all_docs_occ[key]:
            all_docs_occ[key][doc] = dict(all_docs_occ[key][doc])
        all_docs_occ[key] = dict(all_docs_occ[key])

        f = open(os.path.join(output_file, '%s.pkl'%key), 'wb')
        pickle.dump(all_docs_occ[key], f)
        f.close()





def retrieve_aqua_occurrences_sentences(directory, output_file, aqua_entities_file):
    """
    Same as ``parse_stat.retrieve_aqua_occurrences``, but outputs the sentence of the occurrences rather than its position.

    Args:
     * ``directory`` (*str*): path to the directory containing the Aquaint data files.
     * ``output_file`` (*str*): path to the file to write the output scores.
     * ``aqua_entities_file`` (*str*): path to the file containing all aquaint entities and their number of occurrences.
    """
    n_samples, index_to_label, label_to_index = parse_AQUA_entities(aqua_entities_file)
    all_docs_occ = defaultdict(lambda:[])
    documents = [f for f in os.listdir(directory) if f.endswith('.xml.u8')]

    # Parsing all documents
    for k, d in enumerate(documents):
        print '\n -------> Parsing document %s: %d/%d' %(d, k+1, len(documents))
        with open(os.path.join(directory, d), 'r') as f:
            doc_id= -1

            for i, line in enumerate(f):
                l = line.strip()

                if l.startswith('<DOC '):
                    doc_id += 1
                    sentence_id = -1

                elif l.startswith('<S>'):
                    sentence_id += 1

                elif l.startswith('<w '):
                    try:
                        _,word,_,pos,_,lem,_ = l.split('"') # word occurence, POS-tag, lemmatized word

                        if not (word.strip() and pos.strip() and lem.strip()):
                            raise ValueError

                        if lem.strip() in label_to_index and pos in ['NN', 'NNS']:
                            all_docs_occ[lem].append(l)

                    except ValueError:
                        print 'Warning; Parsing Error: line %d -  %s' %(i,l)

    for key in all_docs_occ:
        sentences = all_docs_occ[key]
        f = open(os.path.join(output_file, '%s.pkl'%key), 'wb')
        pickle.dump(sentences, f)
        f.close()




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





################################################################################ XP
if __name__ == '__main__':

    aqua_dir = '/nfs/titanex/vol/text1/tmx-text/corpus/Aquaint2/'
    aqua_entities_file = '/udd/aroyer/Stage/Code/src/Precomputed/aquaint_entities_list'

    stat_aqua(aqua_dir)
    #retrieve_aqua_entities(aqua_dir, aqua_entities_file)
    #retrieve_aqua_occurrences(aqua_dir, '/udd/aroyer/Stage/Code/src/Precomputed/AQUA_words_occurrences', aqua_entities_file)
    #retrieve_aqua_occurrences_sentences(aqua_dir, '/udd/aroyer/Stage/Code/src/Precomputed/AQUA_words_sentences', aqua_entities_file)
    #count_aqua_docs(aqua_dir, '/udd/aroyer/Stage/Code/Data/AQUAINT/Data_Occ', aqua_entities_file)
    #count_aqua_docs_score(aqua_dir, '/udd/aroyer/Stage/Code/src/Precomputed/doc_scores_AQUA', aqua_entities_file)
