import os
import argparse
from collections import defaultdict

if __name__ == '__main__':                
    #Add own configuration (Argparser)
    parser = argparse.ArgumentParser(description='Extracting Features using HTK.')
    parser.add_argument(dest = 'dataset', type = str, help='dataset identifier (i.e. folder name).')
    parser.add_argument('-ho', dest = 'homonyms', type = str, help='homonyms file.')
    parser.add_argument('-o', dest = 'output_folder', default='./', type = str, help='homonyms file.')
    args = parser.parse_args()

    #Load homonys
    common_class = {}
    if args.homonyms is not None:
        with open(args.homonyms, 'r') as f:
            for line in f:
                homo = line.split()
                for h in homo:
                    common_class[h] = homo[0]
                
    
    #Write index_to_label and ground_truth
    main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))         
    files = [f for f in os.listdir(os.path.join(args.dataset, 'Samples')) if f.endswith('.wav')]
    files = sorted(files)
    
    gtf = open(os.path.join(args.output_folder, 'audio_ester2.cluster'), 'w') if args.homonyms is None  else  open(os.path.join(args.output_folder, 'audio_ester2_homonym.cluster'), 'w')
    idf = open(os.path.join(args.output_folder, 'index_to_labels_audio_ester2'), 'w')
    ground_truth = defaultdict(lambda:[])
    
    for i, f in enumerate(files):
        sample = f.rsplit('.', 1)[0]
        word = f.split('.', 1)[0]
        try:
            classe = common_class[word]
        except KeyError:
            classe = word
        ground_truth[classe].append(sample)
        gtf.write('%s\t%s\n'%(classe, sample))
        idf.write('%d\t%s\n'%(i, sample))

    #Print
    print "%d ground-truth clusters and %d samples" %(len(ground_truth), sum([len(x) for x in ground_truth.values()]))
    print '\n'.join('%s (%d)' %(k, len(y)) for k,y in ground_truth.iteritems())

    gtf.close()
    idf.close()
