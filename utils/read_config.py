#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**read_config.py.** Functions for setting the main parameters. Reads the configuration file in ``configuration.ini`` and the command line options.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import ConfigParser as cfg
import argparse
from error import *
from parse import parse_pattern

# Allowed options values
data_types = ['NER', 'AQUA', 'AUDIO'] # Ester2 txt, Aquaint2, Ester2 audio, Ester2 audio tiny
annotation_types = ['RND', 'UNI', 'OVA'] # RaNDom, UNIque , One Versus All
classifier_types = ['CRF', 'DT', 'HTK']
task_types = ['MCL', 'KNN']
similarity_types = ['WPROB', 'WBIN', 'PROB', 'BIN', 'WEM', 'UWBIN', 'UWPROB']


def get_config_option(config, section, name, arg, type='str'):
    """
    Returns the default configuration if ``arg`` (command line argument) is ``None``, else returns ``arg``.

    Args:
     * ``config``: current configuration object returned by ConfigParser.
     * ``section`` (*str*): section of the considered argument.
     * ``name`` (*str*): name of the considered argument.
     * ``arg``: value passed through command line for the considered argument.
     * ``type`` (*str, optional*): type of the considered argument (str, int or float). Defaults to *str*.
    """
    if arg is None:
        if type == 'str':
            return config.get(section, name)
        elif type == 'int':
            return config.getint(section, name)
        elif type == 'float':
            return config.getfloat(section, name)
    else:
        config.set(section, name, arg)
        return arg



def read_config_file(f, N=None, cores=None, data_type=None, input_file=None, ground_truth_file=None, n_distrib=None, training_size=None, nmin=None, nmax=None, classifier_type=None, similarity_type=None, task_type=None, cvg_step=None, cvg_criterion=None, output_folder=None, temp_folder=None, oar=False):
    """
    Reads and parses the default arguments in the configuration file.

    Args:
     * ``f`` (*str*): path to the configuration file.
     * ``opt`` (*list*): specified command line options (see result of ``read_config.parse_cmd_line``).

    Returns:
     * ``N`` (*int*): number of iterations.
     * ``cores`` (*int*): number of cores to use.
     * ``locks`` (*int*): number of independant locks to add on the similarity matrix.
     * ``data_type`` (*str*): name of the dataset chosen for the experiments.
     * ``input_file`` (*str*): path to default input file.
     * ``ground_truth_file`` (*str*): path to default configuration file.
     * ``temp_folder`` (*str*): path to the folder for temporary files (e.g. MCL input format file).
     * ``output_folder`` (*str*): path to the folder for output files.
     * ``annotation_params`` (*dict*): parameters for the synthetic annotation.
     * ``classification_params`` (*dict*): parameters for the supervised classification algorithm. (classifier type, training percentage, similarity type, additional parameters).
     * ``classifier_binary`` (*str*): path to the binary for the classifier.
     * ``task_params`` (*list*): parameters for the post-processing task. (task type, algorithm binary, additional parameters).
     * ``cvg_step`` (*int*): If ``step`` > 2, the convergence criterion will be evaluated every ``step`` iteration.
     * ``cvg_criterion`` (*float*): value of the criterion on the mean entropies to stop the algorithm.
     * ``config`` (*ConfigParser*): configuration object updated with the values in command line.
    """

    global data_types, annotation_types, classifier_types, task_types, similarity_types

    config = cfg.ConfigParser()
    config.read(f)

    # Basic
    N = get_config_option(config, 'General', 'N', N, type='int')
    locks = config.getint('General', 'locks')
    temp_folder = get_config_option(config, 'General', 'temp', temp_folder)
    output_folder = get_config_option(config, 'General', 'output', output_folder)
    cvg_step = get_config_option(config, 'General', 'cvg_step', cvg_step, type='int')
    cvg_criterion = get_config_option(config, 'General', 'cvg_criterion', cvg_criterion, type='float')

    # Data set
    if data_type is None:
        data_type = config.get('General', 'data')
        if not data_type in data_types:
            raise ConfigError('data option not in %s'%data_types)
    else:
        config.set('General', 'data', data_type)

    input_file = get_config_option(config, data_type, 'input', input_file)
    ground_truth_file = get_config_option(config, data_type, 'ground_truth', ground_truth_file)


    # ----------------------------------------------- Synthetic annotation parameters
    if n_distrib is None:
        n_distrib = config.get('General', 'n_distrib')
        if not n_distrib in annotation_types:
            raise ConfigError('n_distrib option not in %s'%annotation_types)
    else:
        config.set('General', 'n_distrib', n_distrib)

    nmin = get_config_option(config, 'General', 'n_min', nmin, type='int')
    nmax = get_config_option(config, 'General', 'n_max', nmax, type='int')
    annotation_params = {'distrib': n_distrib, 'n_min': nmin, 'n_max': nmax}


    # ---------------------------------------------- Classifier parameters
    training_size = get_config_option(config, 'General', 'training_size', training_size, type='float')

    if similarity_type is None:
        similarity_type = config.get('General', 'similarity')
        if not similarity_type in similarity_types:
            raise ConfigError('similarity option not in %s'%similarity_types)
    else:
        config.set('General', 'similarity', similarity_type)

    if classifier_type is None:
        classifier_type = config.get('General', 'classifier')
        if not classifier_type in classifier_types:
            raise ConfigError('classifier option not in %s'%classifier_types)
    else:
        config.set('General', 'classifier', classifier_type)
    classification_params = {'classifier_type': classifier_type, 'training_size': training_size, 'similarity_type': similarity_type}

    # Replace binary path if OAR/GRID
    classifier_binary = config.get(classifier_type, 'binary')
    if oar:
        try:
            classifier_binary = config.get(classifier_type, 'oar_binary')
        except KeyError:
            print 'Warning: No OAR binary found for classifier %d' %classifier_type

    # Pattern for CRF and Decision Trees
    if classifier_type == 'CRF':
        pattern = parse_pattern(classifier_type, config.get(data_type, '%s_pattern'%classifier_type.lower()))
        classification_params['crf_pattern'] = pattern
    elif classifier_type == 'DT':
        pattern = parse_pattern(classifier_type, config.get(data_type, '%s_pattern'%classifier_type.lower()))
        annotation_params['dt_pattern'] = pattern

    # Other Parameters
    if classifier_type == 'HTK':
        classification_params['hmm_topo'] = [int(x) for x in config.get('HTK', 'hmm_topo').split(',')]
        classification_params['features'] = [x.strip() for x in config.get('HTK', 'features').split(',')]
    else:
        classification_params['additional_params'] = [(x,y) for (x,y) in config.items(classifier_type) if not x in ['binary', 'oar_binary', 'root_dir']]

    # OVA
    if n_distrib == 'OVA': # Store structure mapping a word to all of its occurrences in the dataset
        annotation_params['ova_occurrences'] = config.get(data_type, 'words_occurrences')

    # Post-processing task parameters
    if task_type is None:
        task_type = config.get('General', 'task')
        if not task_type in task_types:
            raise ConfigError('task option not in %s'%task_types)
    else:
        config.set('General', 'task', task_type)
    task_params = {'task_type': task_type, 'binary': config.get(task_type, 'binary'), 'additional_params': {x: y for (x, y) in config.items(task_type) if not x in ['binary', 'oar_binary', 'root_dir']}}
    if oar:
        try:
            task_params['binary'] = config.get(task_type, 'oar_binary')
        except KeyError:
            print 'Warning: No OAR binary found for task %d' %task_type

    # Update number of cores if needed
    cores = get_config_option(config, 'General', 'cores', cores, type='int')
    from multiprocessing import cpu_count
    cpuc = cpu_count()
    if cores < 1:
        cores = 1
    elif (cores + 1) > cpuc:
        cores = cpuc - 1
        print >> sys.stderr, "Warning: Core parameter exceeds cpu limit. Setting it to %d" %cores

    if N < cores:
        cores = N
    config.set('General', 'cores', cores)

    # Check options compatibility
    if (classifier_type == 'CRF' and data_type in ['AUDIO', 'AUDIOTINY']) or (classifier_type == 'HTK' and data_type in ['AQUA', 'NER']):
        print >> sys.stderr, 'Data type %s is incompatible with %s classifier option'%(data_type, classifier_type)
        raise SystemExit

    if (n_distrib == 'OVA' and similarity_type == 'WEM'):
        print >> sys.stderr, 'WEM similarity is incompatible with OVA scheme'


    return N, cores, locks, data_type, input_file, ground_truth_file, temp_folder, output_folder, annotation_params, classification_params, classifier_binary, task_params, cvg_step, cvg_criterion, config




def parse_cmd_line():
    """
    Parses the command line for options to override the default configuration parameters.

    Returns:
     * ``opt`` (*list*):  specified command line options to override the default config options:

       * ``N`` (*int*): number of iterations (-N, --iter).
       * ``cores`` (*int*): number of cores (-t, --threads).
       * ``data_type`` (*str*): chosen dataset (-d, --dataset).
       * ``input`` (*str*): chosen data file (-in, --input).
       * ``groundtruth`` (*str*): chosen data file (-g, --groundtruth).
       * ``n_distrib`` (*str*): type of annotation (-di, --distrib).
       * ``training_size`` (*float*): training percentage, strictly between 0 and 1 (-ts, --trainsize).
       * ``nmin`` (*int*): minimum number of synthetic labels (-nmin).
       * ``nmax`` (*int*): minimum number of synthetic labels (-nmax).

     * ``cfg`` (*str*): path to default configuration file.
     * ``verbose`` (*int*): controls verbosity level (0 to 4).
     * ``debug`` (*bool*): runs in debugging mode.
    """

    global data_types, classifier_types, task_types

    # Args Parser
    parser = argparse.ArgumentParser(description='Clustering by Diverting supervised classification techniques.')
    parser.add_argument('-N', '--iter', dest='N', type=int, help='number of classification iterations.')
    parser.add_argument('-t', '--threads', dest='cores', type=int, help='number of cores to use.')

    parser.add_argument('-d', '--dataset', dest='data_type', type=str, help='dataset identifier.')
    parser.add_argument('-in', dest='input', type=str, help='path to the file containing the data.')
    parser.add_argument('-g', '--groundtruth', type=str, help='path to the file containing the ground_truth.')

    parser.add_argument('-di', '--distrib', dest='n_distrib', type=str, help='annotation type.')
    parser.add_argument('-ts', '--trainsize', dest='training_size', type=float, help='training percentage (no effect on Aquaint for now).')
    parser.add_argument('-nmin',  dest='nmin', type=int, help='minimum number of synthetic labels.')
    parser.add_argument('-nmax',  dest='nmax', type=int, help='maximum number of synthetic labels.')

    parser.add_argument('-c', '--classifier', dest='classifier_type', type=str, help='classifier to use.')
    parser.add_argument('-s', '--sim', dest='similarity_type', type= str, help='similarity identifier.')

    parser.add_argument('-p', '--post', dest='task_type', type=str, help='post-processing algorithm.')

    parser.add_argument('-cs', '--cvg_step', dest='cvg_step', type=int, help='Convergence step.')
    parser.add_argument('-cc', '--cvg_criterion', dest='cvg_criterion', type=float, help='Convergence stopping ciretion.')

    parser.add_argument('-o', '--out', dest='output_folder', type=str, help='output folder.')
    parser.add_argument('-te', '--temp', dest='temp_folder', type=str, help='temporary files folder.')
    parser.add_argument('-cfg', dest='cfg_file', type=str, help='custom configuration file for default options.')

    parser.add_argument('-v', '--verbose', dest='verbosity', type=int, help='controls verbosity level.')
    parser.add_argument('-db', '--debug', help='debugging mode.', action="store_true")
    parser.add_argument('--oar', action="store_true", help="if present, set correct binary pathes for the cluster")

    args = parser.parse_args()

    # Checking args values
    if (args.data_type is not None) and not (args.data_type in data_types):
        raise argparse.ArgumentTypeError('Data option not in %s'%data_types)

    if (args.similarity_type is not None) and not (args.similarity_type in similarity_types):
        raise argparse.ArgumentTypeError('Similarity option not in %s'%similarity_types)

    if (args.classifier_type is not None) and not (args.classifier_type in classifier_types):
        raise argparse.ArgumentTypeError('Classifier option not in %s'%classifier_types)

    if (args.task_type is not None) and not(args.task_type in task_types):
        raise argparse.ArgumentTypeError('Task option not in %s'%task_types)

    if (args.n_distrib is not None) and not (args.n_distrib in annotation_types):
        raise argparse.ArgumentTypeError('Annotation option not in %s'%annotation_types)

    if (args.training_size is not None) and (args.training_size <= 0 or args.training_size >= 1):
        raise argparse.ArgumentTypeError('Training percentage not in [0,1]')

    if args.verbosity is None:
        verbose = 1
    else:
        verbose = args.verbosity

    return (args.N, args.cores, args.data_type, args.input, args.groundtruth, args.n_distrib,  args.training_size, args.nmin, args.nmax, args.classifier_type, args.similarity_type, args.task_type, args.cvg_step, args.cvg_criterion, args.output_folder, args.temp_folder, args.oar), args.cfg_file, verbose, args.debug
