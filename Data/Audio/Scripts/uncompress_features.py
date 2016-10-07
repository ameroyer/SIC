#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**extract_features.** Transforms HTK features into a readable ASCII format.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__copyright__ = "Copyright 2015, Amélie Royer"
__date__ = "2015"

if __name__ == "__main__":
    import argparse
    import subprocess
    import shutil
    import os
    from extract_features import init_folder, n_coeff

    # ========== Custom parameters
    parser = argparse.ArgumentParser(description='Uncompress HTK Features.')
    parser.add_argument('-i', '--input', dest = 'samples_folder', type = str, help='Folder containing the audio samples.')
    parser.add_argument('-o', '--output', dest = 'output_folder', type = str, help='Output folder for features (does not need to exist beforehand).')
    parser.add_argument('-cf', '--compressed', dest = 'compressedfeatures_folder', default='', type = str, help='Compressed features folder.')
    parser.add_argument('-f', '--features', dest = 'features', type = str, default = 'MFCC_0_D_A_Z', help='features type and qualifiers (see HTK Book for format). Allowed types are MFCC, LPCEPSTRA, LPC and PLP. Defaults to MFCC_0_D_A_Z')
    parser.add_argument('-c', '--cmpn', dest = 'cmpn', type = int, default = 12, help='number of base spectral components. Defaults to 12.')
    args = parser.parse_args()

    # Init folders
    main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    temp_folder = init_folder(os.path.join(main_dir, 'temp'))
    n_comp = n_coeff(args.cmpn, args.features)

    # Check if features exist. If not, generate them
    print 'Generate Features'
    features_folder = os.path.join(args.compressedfeatures_folder, '%s_%d'%(args.features, n_comp))
    if not os.path.isdir(features_folder):
        FNULL = open(os.devnull, 'w')
        print 'Extracting Features'
        subprocess.call(('python %s -i %s -o %s -f %s -c %s'%(os.path.join(main_dir, 'Scripts', 'extract_features.py'), args.samples_folder, args.compressedfeatures_folder, args.features, args.cmpn)).split(), stdout = FNULL)
        FNULL.close()


    # Init new folder
    # Resulting text features. 1 line = 1 time unit
    print 'Uncompress Features'
    unc_features_folder = init_folder(os.path.join(args.output_folder, '%s_%d'%(args.features, n_comp)))
    print 'Writing uncompressed features in %s' %unc_features_folder
    feat_list = [x for x in os.listdir(features_folder) if x.endswith('.mfc')]
    for i, feature in enumerate(feat_list):
        print '%s: %d/%d'%(feature, i + 1, len(feat_list))        
        p = subprocess.Popen(['/udd/aroyer/Stage/Code/Lib/HTK/htk/HTKTools/HList', os.path.join(features_folder, feature)], stdout = subprocess.PIPE)
        #p = subprocess.Popen(['HList', os.path.join(features_folder, feature)], stdout = subprocess.PIPE)
        out = p.communicate()[0]

        with open(os.path.join(unc_features_folder, '%s.txt'%feature.rsplit('.', 1)[0]), 'w') as f:
            acc = []
            for line in out.splitlines():
                if not line.startswith('-') and line.strip():
                    aux = line.split(':')
                    # If new time, append current accumualator
                    if len(aux) > 1 and len(acc) > 0:
                        f.write(' '.join(acc))
                        f.write('\n')
                        acc = []
                    # Else, accumulate
                    cpn = aux[1] if len(aux) > 1 else line
                    acc.extend(cpn.split())
                elif len(acc) > 0:
                    f.write(' '.join(acc))
