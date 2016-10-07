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
        f.write('\n'.join(['%s\t%s'%(cl, feat.rsplit('/', 1)[1].rsplit('.', 1)[0]) for cl in sorted(classes.keys()) for feat in classes[cl] ]))

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
    parser.add_argument('-ts',dest = 'training_size', type = float, default = 0.8, help='Training samples percentage.')
    parser.add_argument('-o',dest = 'output', type = str, default = 'result', help='Training samples percentage.')
    parser.add_argument('-f', '--features', dest = 'features', type = str, default = 'MFCC_0_D_A_Z', help='features type and qualifiers (see HTK Book for format). Allowed types are MFCC, LPCEPSTRA and PLP. Defaults to MFCC_0_D_A_Z')
    parser.add_argument('-c', '--cmpn', dest = 'cmpn', type = int, default = 12, help='number of base spectral components. Defaults to 12.')
    args = parser.parse_args()

    # Parameters
    main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))                            
    temp_folder = init_folder(os.path.join(main_dir, 'temp'))

    # Check if features exist. If not, generate them
    FNULL = open(os.devnull, 'w')
    features_folder = [os.path.join(main_dir, args.dataset, 'Features', f) for f in os.listdir(os.path.join(main_dir, args.dataset, 'Features')) if f.startswith(args.features)]
    
    if len(features_folder) == 0:   
        print 'Extracting Features'
        subprocess.call(('%s -d %s -f %s'%(os.path.join(main_dir, 'Scripts', 'extract_features.py'), args.dataset, args.features)).split(), stdout = FNULL)
        
    features_folder = [os.path.join(main_dir, args.dataset, 'Features', f) for f in os.listdir(os.path.join(main_dir, args.dataset, 'Features')) if f.startswith(args.features)][0]
    args.cmpn = int(features_folder.rsplit('_', 1)[1])

    # Build annotation, train and test database
    ground_truth_file, mlf_file, dict_file, wnet_file, scp_trains, scp_test = init_ground_truth(features_folder, temp_folder, args.training_size)
    
    hmmdefs = os.path.join(temp_folder, 'hmmdefs')
    hmmlist = os.path.join(temp_folder, 'hmmlist')
    hd = open(hmmdefs, 'w')
    hl = open(hmmlist, 'w')
    
    # Train HMMs
    n_state = defaultdict(lambda:10)
    n_state['autres'] = 5
    n_state['ville'] = 9
    n_state['contre'] = 8
    for cl in scp_trains:
        print 'Training %s'%cl
        train_file = os.path.join(temp_folder, 'train.scp')
        with open(train_file, 'w') as f:
            f.write('\n'.join(scp_trains[cl]))

        # Restimation
        basic_hmm = generate_basic_hmm(args.features, args.cmpn, cl, temp_folder, n_state = n_state[cl])
        subprocess.call(('HInit -I %s -S %s -M %s %s'%(mlf_file, train_file, temp_folder, basic_hmm)).split(), stdout = FNULL)
        subprocess.call(('HRest -I %s -S %s -M %s %s'%(mlf_file, train_file, temp_folder, basic_hmm)).split(), stdout = FNULL)
        hl.write('%s\n'%cl)
        with open(basic_hmm, 'r') as f:
            hd.write(f.read())
            hd.write('\n\n')
        os.remove(basic_hmm)

    hl.close()
    hd.close()
    del scp_trains
    os.remove(mlf_file)
    os.remove(train_file)

    # Test HMM
    test_file = os.path.join(temp_folder, 'test.scp')
    with open(test_file, 'w') as f:
        f.write('\n'.join(scp_test))        
    p = subprocess.Popen(('HVite -T 1 -S %s -t 250.0 -H %s -w %s %s %s'%(test_file, hmmdefs, wnet_file, dict_file, hmmlist)).split(), stdout = subprocess.PIPE)
    out = p.communicate()[0]

    # Clean
    os.remove(test_file)
    os.remove(hmmdefs)
    os.remove(hmmlist)
    os.remove(wnet_file)
    os.remove(dict_file)

    with open(args.output, 'w') as f:
        f.write(out)

    shutil.rmtree(temp_folder) 
