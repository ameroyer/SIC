#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**compute_dtw.** Computes a DTW similarity matrix using the R dtw library.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__copyright__ = "Copyright 2015, Amélie Royer"
__date__ = "2015"

import sys
sys.path.append('/udd/aroyer/.local/lib/python2.7/rpy2/')
import time
import numpy as np
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from rpy2.robjects.packages import importr
from rpy2.rinterface import RRuntimeError

from extract_features import init_folder
from random import choice
import os
from multiprocessing import Process, Queue, sharedctypes, Array, Pool

# Set up R namespaces
R = rpy2.robjects.r
#_ = importr("proxy")
#DTW = importr('dtw')
main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
R.source(os.path.join(main_dir, 'Scripts', 'compute_dtw.R'))


def R_dtw(query_path, ref_path, unconstrained=False, asFile=False):
    """
    Try to compute a constrained DTW through R. If the call fails, drop the local constraint
    """
    if asFile:
        query = query_path
        ref = ref_path
    else:
        query = np.loadtxt(query_path)
        ref = np.loadtxt(ref_path)
    if unconstrained:
        return R.compute_dtw(query, ref, unconstrained=2)
    else:
        try:
            return R.compute_dtw(query, ref, unconstrained = 0)
        except RRuntimeError:
            #print >> sys.stderr, '%s x %s: Constraint Error lvl 1'%(query_path, ref_path)
            try:
                return  R.compute_dtw(query, ref, unconstrained=1)
            except RRuntimeError:
                #print >> sys.stderr, '%s x %s: Constraint Error lvl 2'%(query_path, ref_path)
                return R.compute_dtw(query, ref, unconstrained=2)
        


if __name__ == "__main__":
    import argparse
    import subprocess
    import shutil
                  
    # ---------------- Add own configuration (Argparser)
    parser = argparse.ArgumentParser(description='Extracting Features using HTK.')
    parser.add_argument('-d', '--dataset', dest = 'dataset', type = str, default = './', help='dataset identifier (i.e. folder name).')
    parser.add_argument('-f', '--features', dest = 'features', type = str, default = 'MFCC_0_D_A_T_Z', help='features type and qualifiers (see HTK Book for format). Allowed types are MFCC, LPCEPSTRA and PLP. Defaults to MFCC_0_D_A_T_Z')
    parser.add_argument('-c', '--cmpn', dest = 'cmpn', type = int, default = 12, help='number of base spectral components. Defaults to 12.')
    args = parser.parse_args()

    
    # Check if (uncompressed) features exist. If not, generate them    
    FNULL = open(os.devnull, 'w')
    output_folder = init_folder('/temp_dd/igrida-fs1/aroyer/Temp/')
    #output_folder = os.path.join(main_dir, args.dataset)
    print 'Searching for uncompressed features in %s' %os.path.join(output_folder, 'Uncompressed_Features')
    try:
        features_folder = [os.path.join(output_folder, 'Uncompressed_Features', f) for f in os.listdir(os.path.join(output_folder, 'Uncompressed_Features')) if f.startswith(args.features)]
    
        if len(features_folder) == 0:
            raise OSError
    except OSError:
        print 'Extracting Features'
        subprocess.call(('python %s -i %s -f %s -cf %s -o %s'%(os.path.join(main_dir, 'Scripts', 'uncompress_features.py'), args.dataset, args.features, os.path.join(output_folder, 'Features'), os.path.join(output_folder, 'Uncompressed_Features'))).split(), stdout = FNULL)
        
    features_folder = [os.path.join(output_folder, 'Uncompressed_Features', f) for f in os.listdir(os.path.join(output_folder, 'Uncompressed_Features')) if f.startswith(args.features)][0]
    args.cmpn = int(features_folder.rsplit('_', 1)[1])
    FNULL.close()
    asFile = True

    
    # ----------------- Init DTW similarity matrix
    cores = 20
    if asFile:
        samples_list = [os.path.join(features_folder, x) for x in os.listdir(features_folder) if x.endswith('.txt')]
        samples_list = sorted(samples_list) #meme ordre alphabetique normalement
        pool = Pool(processes = cores)
        samples = pool.map(np.loadtxt, samples_list)
        #samples = [np.loadtxt(os.path.join(features_folder, x)) for x in os.listdir(features_folder) if x.endswith('.txt')]
    else:
        samples = [os.path.join(features_folder, x) for x in os.listdir(features_folder) if x.endswith('.txt')]
        samples = sorted(samples) #meme ordre alphabetique normalement
    n_samples = len(samples)
    print '%d Samples' %n_samples
    
    #samples = samples[:2]
    #n_samples = 2
    #print samples
    
    dtw_matrix = np.zeros((n_samples, n_samples))

    
    # ------------------------- Plot some DTWs alignemnt
    ## plot_folder = init_folder(os.path.join(output_folder, 'DTW_%s_%d_Plots'%(args.features, args.cmpn)))
    ## print len(samples)
    ## toplot = [choice(samples[i*50:(i+1)*50]) for i in xrange(0,19)]
    ## for x in toplot:
    ##     nx = os.path.basename(x).rsplit('.', 1)[0]
    ##     print 'Plotting', nx
    ##     for y in toplot:
    ##         ny = os.path.basename(y).rsplit('.', 1)[0]
    ##         R.plot_dtw(nx, ny, R_dtw(x, y), os.path.join(plot_folder, 'Q%s_R%s.pdf'%(nx, ny)))
    #raise SystemExit
    



    # ----------------------- Build the (non-symmetric) DTW  matrix
    print 'Building DTW'
    ## for i, feati in enumerate(samples):
    ##     start = time.time()
    ##     print '\n  > Line %d/%d'%(i+1, n_samples)
    ##     for j, featj in enumerate(samples):
    ##         alignment = R_dtw(feati, featj)                
    ##         dtw_matrix[i, j] = alignment.rx('normalizedDistance')[0][0]
    ##     print '     %s seconds' %(time.time() - start)

    def dtw_threaded(tid, interval, samples, result_queue):
        print 'Starting thread %d/20'%tid
        for i in interval:
            feati = samples[i]
            start = time.time()
            line = np.zeros(len(samples))
            for j, featj in enumerate(samples):
                alignment = R_dtw(feati, featj, asFile=asFile)                
                dtw_res = alignment.rx('normalizedDistance')[0][0]
                line[j] = dtw_res 
            result_queue.put((i, line))
            print '> Line %d/%d done: %s seconds' %(i+1, len(samples), time.time() - start)
        result_queue.close()
        result_queue.join_thread()
            

    #-----------Parallel
    result_queue = Queue()
    steps = len(samples) / cores
    #run processes
    pis = []
    for k in xrange(cores):
        if k < cores - 1:
            interval = range(k*steps, (k+1)*steps)
        else:
            interval = range(k*steps, len(samples))
        t = Process(target = dtw_threaded, args = (k, interval, samples, result_queue))
        pis.append(t)
        t.start()


    #get from queue
    for k in xrange(len(samples)):
        (i, dtw_res) = result_queue.get()
        dtw_matrix[i, :] = dtw_res
        print 'Line %d set in matrix' %(i+1)
        print '%d left' %(len(samples) - k - 1)

    #join
    for t in pis:
        t.join()
        

            
    # Symmetrization
    print 'Symmetrizing DTW'
    dtw_matrix[np.triu_indices(n_samples, k = 1)] += dtw_matrix[np.tril_indices(n_samples, k = -1)]
    dtw_matrix = np.amax(dtw_matrix) - dtw_matrix
    dtw_matrix = np.triu(dtw_matrix, k = 1)
    dtw_matrix /= 2

    #print >> sys.stderr, dtw_matrix

    np.save(os.path.join(output_folder, 'dtw_matrix_%s_%s'%(args.features, args.cmpn)), dtw_matrix)
        
