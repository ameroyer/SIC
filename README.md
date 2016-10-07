# [ S I C ]  r e a d   m e

Implementation of **SIC** (Similarity by Iterative Classification). See the following technical report for more details about the method and implementation:

> **Similarity by diverting supervised machine learning — Application to knowledge discovery in multimedia content.**
> [Technical Report Inria Rennes](https://hal.inria.fr/hal-01285965v1).
> Amélie Royer, Vincent Claveau, Guillaume Gravier,  Teddy Furon


# Requirements

#### Python dependencies
 * Tested with Python 2.7
 * Numpy
 * Scipy
 * Matplotlib

#### External libraries
 * [MCL](http://micans.org/mcl/), Markov clustering algorithm
 * [Wapiti](http://wapiti.limsi.fr/), CRF implementation (text data)
 * [HTK](http://htk.eng.cam.ac.uk/), HMM implementation (audio data)

# Usage

#### main script

	python main.py -N [1] -t [2] -d [3] -c [4] -ts [5] -s [6] -nmin [7] -nmax [8] -di [9] -p [10] -cs [11] -cc [12] -o [13] -te [14] -in [15] -g [16] -cfg [17] -v [18] --debug --oar --help

where:
 * Default options are found in the ``configuration.ini`` file.
 * ``[1]`` *-i, --iter*: number of classification iterations.
 * ``[2]`` *-t, --threads*: number of cores to use.
 * ``[3]`` *-d, --dataset*: dataset to use.
 * ``[4]`` *-c, --classifier*: classifier to use.
 * ``[5]`` *-ts, --trainsize*: proportion of dataset to use for training.
 * ``[6]`` *-s, --sim*: similarity type to use. Defaults to BIN.
 * ``[7]`` *-nmin*: minimum number of synthetic labels.
 * ``[8]`` *-nmax*: maximum number of synthetic labels.
 * ``[9]`` *-di, --distrib*: synthetic annotation mode (RND, UNI, OVA). Defaults to RND.
 * ``[10]`` *-p, --post*: post-processing task/algorithm (MCL or KNN, which requires a .qrel version of the ground-truth, see ``parse.py``).
 * ``[11]`` *-cs, --cvg_step*: check convergence criterion every ``cs`` step.
 * ``[12]`` *-cc, --cvg_criterion*: convergence criterion threshold. (Note that the current implementation of the convergence criterion with the concurrency of processes is far from perfect).
 * ``[13]`` *-o, --output*: output folder.
 * ``[14]`` *-te, --temp*: temporary folder.
 * ``[15]`` *-in*: input data file.
 * ``[16]`` *-g, --ground*: ground-truth file.
 * ``[17]`` *-cfg, --cfg_file*: provide a custom configuration file.
 * ``[18]`` *-v, --verbose*: controls verbosity level (0 - low to 4 - high).
 * ``[-db, --debug]``: debug mode (save temporary files).
 * ``[--oar]``: for usage on the cluster.
 * ``[-h, --help]``

This create log file **output.log** and a similarity matrix in the numpy format **sim_matrix_final.npy** (read with  ``numpy.load``) in the output directory.

#### verbosity levels
 * ``-v 0``: minimal verbose level; almost no printed trace.
 * ``-v 1``: Default.
 * ``-v 2``: Additional print trace.
 * ``-v 3``: Prints out the classifier's traces.
 * ``-v 4``: Outputs additional result (distributions plots, number of occurences in test for each entity ...) + save similarity matrix regularly.

 #### examples
 
 ** a typical run on NER **
 ```
 	python main.py -d NER -N 150 -c CRF -nmin 300 -nmax 300
 ```
 
 ** a typical run on AUDIOTINY ** 
 ```
 	python main.py -d AUDIO -N 300 -c HTK -nmin 20 -nmax 40
 ```

# Running on the cluster
To run the experiments on a cluster (OAR scheduler), you can use the scripts located in the OAR folder. Each script configures the options for the cluster and then call the ``main.py`` script on one node. Once the computation is done, the corresponding similarity matrix is output in the folder ``/temp_dd/igrida-fs1/$USER/Outputs/Outputs_JobID``. These matrices can also be combined with the results of previous interations with a weighted average for instance.

#### usage on the cluster frontend (batch mode)

	oarsub -S ./oar_aqua.sh


**configuration options in the .sh file**
 * ``OAR -n [1]``: name of the job.
 * ``OAR -l nodes=[2], walltime=[3]``: [2] is the number of nodes to use. It should always be 1 so that ``main.py`` can use all the processes on the node, and no more than 1 because it can not manage several nodes (only processes). [3] is the limit of running time for the experiment with format hh:mm:ss.
 * ``OAR -p [4]``: condition on the resources to use (for instance for Aquaint we request a node with at least 45GB memory).
 * ``OAR -O [5]``: output log file.
 * ``OAR -E [6]``: error log file.
 * ``EXECUTABLE=[7]``: path to ``main.py``.

The other options (DATA, ITER...) in the script are those of the ``main.py`` programm introduced previously.

#### examples

 * a typical run on *Aquaint*: ``oarsub -S ./oar_aqua.sh`` (Default is OVA mode, 150 iterations per sample)
 * a typical run on *AUDIO* ``oarsub -S ./oar_aqua.sh`` (Default is 2000 iterations, HMM mixed type 1 and 2, 14 states total)


# General structure of the implementation
 * ``main.py`` is a wrapper for the ``run_*`` functions. It sets up the correct parameters for the run, apply SIC and then stores and evaluates the matrix.
 * ``run_basic.py``, ``run_ova.py`` and ``run_wem.py`` takes care of running SIC (respectively normal SIC, SIC with OVA, SIC with EM similarity).
 * ``utils/one_step.py`` contains the code for a SIC iteration one one process/thread.
 * ``utils/annotate.py`` and ``utils/annotation_scripts`` contains all scripts relevant to synthetic annotation.
 * ``utils/classify.py`` and ``utils/classificationon_scripts`` contains all scripts relevant to training and application of the classifiers.
 * ``utils/eval.py`` contains the functions for evaluation of a clustering and ``utils/evaluation_retrieval.prl`` deals with mAP evaluation. 
 * ``evaluation_clustering`` and ``evaluation_retrieval.py`` are wrappers for the previous evaluation scripts.




# Options and Configuration files
The following is the list of the customizable options found in the configuration file and their roles:

General options
^^^^^^^^^^^^^^^
The options are:

 * ``root_dir``: path to the folder extracted from the original archive.
 * ``N``: number of iterations. Defaults to 50.
 * ``cores``: number of cores to use for the iterations (not counting the main process). Defaults to 20.
 * ``locks``: number of locks/cells in the similarity matrix shared in memory. Default value computed at runtime.
 * ``n_min``: minimum number of synthetic labels at each iteration. Defaults to 300.
 * ``n_max``: maximum number of synthetic labels at each iteration. Defaults to 300.
 * ``n_distrib``: type of synthetic annotation. RND is random annotation, OVA is the one-versus-all setting, and UNI is the combination of the two (n between ``n_min`` and ``n_max`` classes are used, and each class characterizes only one sample). Defaults to RND.
 * ``training_size``: proportion of the dataset to use for training. Defaults to 5%.
 * ``cvg_step``: check convergence criterion every ``cvg_step`` step if positive. Defaults to -1.
 * ``cvg_criterion``: convergence criterion. Defaults to 0.001.
 * ``similarity``: type of similarity. BIN is the default SIC, WBIN is the first weighted scores variant, UWBIN the second one. Similarly, PROB, WPROB and UWPROB are the same but using probablistic scores instead of the basic binary scores (only coded for wapiti CRF which output probability of membership to a class for each sample). Finall WEM is for EM similarity. Defaults to BIN.
 * ``data``: dataset (NER, AUDIO, AUDIOTINY or AQUA). Defaults to NER.
 * ``classifier``: type of classifier to use (CRF or HTK or DT). Defaults to CRF.
 * ``task``: type of evaluation task (MCL or KNN). Defaults to MCL.
 * ``temp``: temporary folder.
 * ``output``: output folder.
 * ``root_dir``

Classifier options (CRF, HTK, DT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The common options are:

 * ``binary``: path to local binary of the classifier (or in case of HTK, folder containing the binaries) if not installed globally.
 * ``oar_binary``: path to binary compiled for Igrida machines.

For CRF and DT, you can then simply add a list of options as taken by the original classifier. For instance, the line ``algo=rprop`` in the configuration file will be translated as a call to wapiti with option ``--algo rprop``.

For HTK, the following specific options are available:

 * ``hmm_topo``: topology of HMM. Defaults to 1,2.
 * ``features``: list of the type of features to use, separated by commas (MFCC, LPC, PLP, LPCEPSTRA). Defaults to MFCC alone.


Evaluation options
^^^^^^^^^^^^^^^^^^
The common options are:

 * ``binary``: path to local binary of the evaluation utilitary.
 * ``oar_binary``: path to binary compiled for Igrida machines.


For MCL, the following specific options are available:

 * ``i``: inflation parameter. Defaults to 1.4.
 * ``p``: pre-inflation parameter. Defaults to 1.0.



Dataset options
^^^^^^^^^^^^^^^
The common options are:

 * ``input``: input data (for NER, the text file containing the dataset; for AQUA, the folder containing the dataset; for AUDIO and AUDIOTINY the input can either be a folder containing precomputed folder (each set of features in a different subfolder named as 'featureHTKidentifier_numberofcomponents'. Or it can either be a text file containing on its first line a path to the audio samples of the dataset and on the following lines, the list of HTK features to consider).
 * ``ground_truth``: path to ground-truth.
 * ``index_to_label``: path to file containing a entity index to label mapping (use the result of ``parse_data`` in utils/parse.py for its generation).

Additionally:

 * ``crf_pattern``: wapiti pattern for a CRF classifier (NER, AQUA).
 * ``dt_pattern``: pattern to select features for a weka decision tree (NER, AQUA).
 * ``words_occurrences``: Structure to store the position of every occurrences of each sample for parsing (AQUA for OVA only).


Additional scripts
------------------

Plotting similarity distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 **Usage**::

	python similarity_analysis.py [1] -n [2] -cfg [3] --mean --theo --help

 where:

 * [1] : input similarity matrix (unnormalized). The script expects a 'exp_configuration.ini' file in the same folder, usually generated when using ``main.py``.
 * [2] ``-n``: number of samples to plot for each class. Defaults to 5.
 * [3] ``-cfg``: provide a custom configuration file to replace 'exp_configuration.ini'.
 * ``--mean``: if given, plot an average ROC curve for each ground-truth class.
 * ``--theo``: if given, plot the comparison of the distribution against the theoretical model of the corresponding SIC variant.
 * ``-h, --help``

 This outputs pdf histograms plots of the distribution of similarities for several samples across the matrix and for several normalization parameters.


Convergence analysis
^^^^^^^^^^^^^^^^^^^^

**Usage:** ::

	python convergence_analysis.py -N [1] -t [2] -d [3] -c [4] -ts [5] -s [6] -nmin [7] -nmax [8] -di [9] -o [10] -te [11] -in [12] -g [13] -cfg [14] -v [15] --debug --help

where:

 * Default options are found in the ``configuration.ini`` file.
 * [1] ``-i, --iter``: number of classification iterations.
 * [2] ``-t, --threads``: number of cores to use.
 * [3] ``-d, --dataset``: dataset to use.
 * [4] ``-c, --classifier``: classifier to use.
 * [5] ``-ts, --trainsize``: proportion of dataset to use for training.
 * [6] ``-s, --sim``: similarity type to use (EM not supported).
 * [7] ``-nmin``: minimum number of synthetic labels.
 * [8] ``-nmax``: maximum number of synthetic labels.
 * [9] ``-di, --distrib``: synthetic annotation mode (RND, UNI, OVA).
 * [10] ``-o, --output``: output folder.
 * [11] ``-te, --temp``: temporary folder.
 * [12] ``-in``: input data file.
 * [13] ``-g, --ground``: ground-truth file.
 * [14] ``-cfg, --cfg_file``: provide a custom configuration file.
 * [15] ``-v, --verbose``: controls verbosity level (0 to 4).
 * ``-db, --debug``: debug mode (save temporary files).
 * ``-h, --help``


Computes ``N`` iterations of SIC and compares the final similarity matrix to partial matrices in past iterations (see ``steps`` in ``convergence_analysis.py``).


Confidence analysis
^^^^^^^^^^^^^^^^^^^^

**Usage:** ::

	python confidence_analysis.py  [1] -cfg [2] --mean --theo --help

where:

 * [1] : input similarity matrix. The script expects a 'exp_configuration.ini' file in the same folder and a ``eval_*.log`` file, containing the mAP results both usually generated when using ``main.py``.
 * [2] ``-cfg``: provide a custom configuration file to replace 'exp_configuration.ini'.
 * ``-h, --help``


Computes confidence scores for the input matrix and compares them to the corresponding mAP results.



Clustering evaluation
^^^^^^^^^^^^^^^^^^^^^

 **Usage**::

	python evaluation_clustering.py [1] -i [2] -p [3] -t [4] -cfg [5] --mcl --help

 where:

 * [1] : input similarity matrix (unnormalized similarities or pre-treated MCL format). The script expects a 'exp_configuration.ini' file in the same folder, usually generated when using ``main.py``.
 * [2] ``-i``: MCL inflation parameter. Defaults to 1.4.
 * [3] ``-p``: MCL pre-inflation parameter. Defaults to 1.0.
 * [4] ``-t``: number of cores to use for MCL.
 * [5] ``-cfg``: provide a custom configuration file to replace 'exp_configuration.ini'.
 * ``-m, --mcl``: if present, the script expects an input matrix in MCL label format.
 * ``-h, --help``

 This outputs the results of the MCL clustering with the given inflation and pre-inflation parameters.


mAP evaluation
^^^^^^^^^^^^^^

 **Usage**::

	python evaluation_retrieval.py [1] -s [2] -ov [3] -cfg [4] --help

 where:

 * [1] : input similarity matrix (unnormalized similarities or pre-treated MCL format). The script expects a 'exp_configuration.ini' file in the same folder, usually generated when using ``main.py``.
 * [2] ``-s``: number of samples to evaluate (``s`` first samples of the ground-truth). If -1, the use the whole set. Defaults to -1

 * [3] ``-ov``: If positive, assume the resulting script was obtained in OVA mode for the sample of index ``ov``. Defaults to -1.
 * [4] ``-cfg``: provide a custom configuration file to replace 'exp_configuration.ini'.
 * ``-h, --help``

 This outputs the results of the neighbour retrieval evaluation on the given matrix.


Visualization
^^^^^^^^^^^^^

 **Usage**::

	python utils/plot.py [1] -cfg [2]

 where:

 * [1] : input similarity matrix (generally unnormalized).
 * [2] ``-cfg``: provide a custom configuration file to replace 'exp_configuration.ini'.

 By default this outputs a piechart representation of the ground-truth clustering and a heatmap and MDS/TSNE representation of the similarity matrix. See the ``plot.py`` script for more visualization tools.


The parse.py script
^^^^^^^^^^^^^^^^^^^
The parse.py and parse_stat.py script can be used to generate some information on the input data that may be required for some experiments.

 * ``utils/ground_truth_qrel`` can be used to generate .qrel files from a ground-truth as needed by the ``evaluation_retrieval.prl`` script for mAP evaluation.
 * ``utils/parse_data`` can be used to generate the ``index_to_label`` and ``label_to_index`` mapping needed for readable outputs.

Additionally for the AUDIO database: Scripts in ``./Data/AUDIO/Scripts`` and ``./Data/AUDIO`` can be used to extract features, compute a DTW (R dtw library) and prepare the AUDIO database.

Finally, for Aquaint: 
 * ``parse_stat/retrieve_aqua_entities`` can be used to count number of occurrences for each file to be stored into a file later used for parsing (this file was already computed and is ``aqua_entities_list`` in Data/AQUAINT/entity_occurrences_aqua and src/Precomputed).
 * ``parse_stat/retrieve_aqua_occurrences`` retrieves, for each sample, all its occurrences in the database and stores their position in a pickle file (1 file = 1 sample). The resulting files are used in OVA mode for Aquaint for parsing. The folder containing the pickle files should be given in the configuration file in section [AQUA], entry ``words_occurrences``. (note for the full Aquaint dataset this results in about 26000 files for 1.2 gigabuytes).
 * ``parse_stat/count_aqua_docs_scores``: computes a ``relevance`` score for each document in the Aquaint database. The higher the score, the more rare words the documents contains. This file is already computed and can be found in Data/AQUAINT/entity_occurrences_aqua and src/Precomputed.

