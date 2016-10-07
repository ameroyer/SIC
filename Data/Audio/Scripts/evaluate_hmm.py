import os
import errno
from collections import defaultdict
from basic_hmm import generate_basic_hmm
import shutil
import subprocess
from random import shuffle

def init_ground_truth(features_folder, temp_folder, training_size):
    """
    Compute basic information on the features (classes, ground truth, annotation, wordnet, dictionnary)
    """
    output_folder = os.path.dirname(os.path.dirname(features_folder))

    #Build annotation and ground_truth
    classes = defaultdict(lambda: [])
    for f in (f for f in os.listdir(features_folder) if f.endswith('.mfc')):
        base_name = f.split('.', 1)[0]
        classes[base_name].append(os.path.join(features_folder, f))

    ground_truth_file = os.path.join(output_folder, 'ground_truth')
    with open(ground_truth_file, 'w') as f:
        f.write('\n'.join(['%s %s'%(cl, feat.rsplit('/', 1)[1].rsplit('.', 1)[0]) for cl in sorted(classes.keys()) for feat in classes[cl] ]))

    mlf_file = os.path.join(temp_folder, 'annotation.mlf')
    with open(mlf_file, 'w') as f:
        f.write('#!MLF!#\n')
        f.write('\n'.join(['"*/%s.*.lab"\n%s\n.'%(cl, cl) for cl in sorted(classes.keys())]))
        
    #Build Dictionnary
    dict_file = os.path.join(temp_folder, 'dict')
    with open(dict_file, 'w') as f:
        f.write('\n'.join(['%s %s'%(x, x) for x in sorted(classes.keys())]))
        f.write('\n')

    #Build Wordnet
    voc_file = os.path.join(temp_folder, 'voc')
    with open(voc_file, 'w') as f:
        f.write('(%s)'%('|'.join(sorted(classes.keys()))))
    wnet_file = os.path.join(temp_folder, 'wnet')
    subprocess.call(('HParse %s %s'%(voc_file, wnet_file)).split(), stdout = FNULL)
    os.remove(voc_file)

    scp_trains =  {}
    scp_test = []
    for cl, x in classes.iteritems():
        train_p = int(len(x) * training_size)
        shuffle(x)
        scp_trains[cl] = x[:train_p]
        scp_test.extend(x[train_p:])
        

    return ground_truth_file, mlf_file, dict_file, wnet_file, scp_trains, scp_test



def init_folder(path):
    """
    Create directory in ``path`` if not already existing.

    Args:
     * ``path`` (*str*): path of the directory to initialize
    """
    npath = os.path.join(os.path.normpath(path), '')
    try:
        os.makedirs(npath)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return npath


if __name__ == "__main__":
    import ConfigParser as cfg   
    import argparse
    import subprocess
    import shutil
                  
    # Add own configuration (Argparser)
    parser = argparse.ArgumentParser(description='Extracting Features using HTK.')
    parser.add_argument('-d', '--dataset', dest = 'dataset', type = str, help='dataset identifier (i.e. folder name).')
    parser.add_argument('-r',dest = 'result', type = str, default = 'result', help='Training samples percentage.')
    args = parser.parse_args()

    # Parameters
    main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))                       

    #Ground truth
    classes = {}
    with open(os.path.join(main_dir, args.dataset, 'ground_truth'), 'r') as f:
        for line in f:
            cl, x = line.split()
            classes[x] = cl

    #Evaluate
    accuracies = defaultdict(lambda: 0)
    sizes = defaultdict(lambda: 0)
    cl = None
    with open(args.result, 'r') as f:
        for line in f.read().split('\n'):
            if line.endswith('.mfc'):
                sample = line.rsplit('/', 1)[1].rsplit('.', 1)[0]
                cl = classes[sample]
                sizes[cl] += 1
            elif cl != None:
                fcl = line.split()[0]
                if fcl == cl:
                    accuracies[cl] += 1
                cl = None

    print '----- Result\n'
    print '> Total accuracy: %f\n'%(float(sum(accuracies.values())) / sum(sizes.values()))
    print '> Detailed accuracies:'
    print '\n'.join(['  %s: %f'%(cl, float(accuracies[cl]) / sizes[cl]) for cl in accuracies.keys()])
