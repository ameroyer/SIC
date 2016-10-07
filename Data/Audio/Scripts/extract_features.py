#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**extract_features.** Extract Features from audio samples using HTK given a base configuration file, a target kind parameter (MFCC, LPCEPSTRA, PLP, LPC) and a number of components.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__copyright__ = "Copyright 2015, Amélie Royer"
__date__ = "2015"


import os
import sys
import errno
import ConfigParser as cfg
import argparse
import subprocess
import shutil

def n_coeff_qualifiers(nceps, qualif):
    """
    Compute the additional number of coefficients brought by a given qualifier.

    Args:
     * ``n`` (*int*): number of base coefficients.
     * ``qualif`` (*str*): features qualifier.
    """
    nbase = nceps
    n = nbase
    if ('E' in qualif): # Log - Energy
        nbase += 1
        n += 1
    if ('0' in qualif): # C0 Energy (only for MFCC)
        nbase += 1
        n += 1
    if ('D' in qualif): # Delta coefficients
        n += nbase
    if ('A' in qualif): # Acceleration coefficients
        n += nbase
    if ('T' in qualif): # Third order coefficients
        n += nbase
    if ('N' in qualif): # Suppress absolute energy
        n -= 1
    return n


def n_coeff (nceps, target_form):
    """
    Returns the number of features components brought by a given HTK features target kind

    Args:
     * ``n_ceps`` (*int*): number of base components.
     * ``target_form`` (*str*) : target kind.
    """
    n = nceps
    try:
        qualifiers = target_form.split('_', 1)[1].split('_')
        n = n_coeff_qualifiers(nceps, qualifiers)
    except IndexError: # No qualifiers
        pass
    print '%d components'%n
    return n


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


def create_HTK_config(base_config, target_kind, nceps, temp_folder, n=0):
    """
    Creates a HTK configuration file for features extraction.

    Args:
     * ``base_config`` (*str*): path to a basic HTK configuration file.
     * ``target_kind`` (*str*): HTK target kind for the features.
     * ``nceps`` (*int*): base components number for the features.
     * ``temp_folder`` (*str*): path to temp folder.
    Returns:
     * ``final_config`` (*str*): path to updated configuration file.
    """
    config = cfg.ConfigParser()
    config.read(base_config)
    params = config._sections['Main']

    # Check target kind argument
    aux = target_kind.split('_', 1)
    if len(aux) >= 1:
        if not aux[0] in ['MFCC', 'LPCEPSTRA', 'PLP', 'LPC']:
            print >> sys.stderr, 'Unknown features kind: %s'%aux[0]
            raise SystemExit
        if len(aux) > 1:
            for q in aux[1].split('_'):
                if not q.strip() in ['0', 'E', 'D', 'A', 'T', 'N', 'Z']:
                    print >> sys.stderr, 'Unkown qualifier option: %s'%q
                    raise SystemExit

    # Write configuration file
    params['targetkind'] = target_kind
    if target_kind.startswith(('MFCC', 'LPCEPSTRA', 'PLP')):
        params['numceps'] = nceps
    else:
        params['lpcorder'] = nceps

    final_config = os.path.join(temp_folder, 'config_step%d.cfg'%n)
    with open(final_config, 'w') as f:
        f.write('\n'.join('%s = %s'%(x.upper(),y) for (x,y) in params.iteritems() if x != '__name__'))

    return final_config


def create_HTK_scp(input_folder, output_folder, input_extension, output_extension, temp_folder, n=0):
    """
    Creates a simple HTK script file for features extraction.

    Args:
     * ``input_folder`` (*str*): path to input folder.
     * ``output_folder`` (*str*): path to output folder.
     * ``input_extension`` (*str*): extension of input files.
     * ``output_extension`` (*str*): extension of output files (same basename).
     * ``temp_folder`` (*str*): path to temp folder.

    Returns:
     * ``scp_file``: HTK script path file.
    """
    scp_file = os.path.join(temp_folder, 'script_path_step%d.scp'%n)
    base_files = [d.rsplit('.', 1)[0] for d in os.listdir(input_folder) if d.endswith('.%s'%input_extension)]
    with open(scp_file, 'w') as f:
        f.write('\n'.join('%s.%s %s.%s'%(os.path.join(input_folder, x), input_extension, os.path.join(output_folder, x), output_extension) for x in base_files))

    return scp_file


def extract_features(samples_folder, output_folder, target_kind, temp_folder, binary_path='/udd/aroyer/Stage/Code/Lib/HTK/htk/HTKTools/', components=12, n=-1):

    if n < 0:
        features_folder = init_folder(os.path.join(output_folder, '%s_%d'%(target_kind, n_coeff(components, target_kind))))
    else:
        features_folder = init_folder(os.path.join(output_folder, '%s_%dSTEP%d'%(target_kind, n_coeff(components, target_kind), n)))
        

    # Update configuration file
    main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    final_config = create_HTK_config(os.path.join(main_dir, 'base.cfg'), target_kind, components, temp_folder, n=n)

    # Create SCP and Target folder
    scp_file = create_HTK_scp(samples_folder, features_folder, 'wav', 'mfc', temp_folder, n=n)

    #Call to HTK
    FNULL = open(os.devnull, 'w')
    try:
        subprocess.call(('HCopy -T 1 -C %s -S %s'%(final_config, scp_file)).split(), stdout = FNULL)
    except OSError:
        subprocess.call(('%s -T 1 -C %s -S %s'%(os.path.join(binary_path, './HCopy'), final_config, scp_file)).split(), stdout = FNULL)
    FNULL.close()

    return features_folder
    



if __name__ == "__main__":

    # ========== Custom parameters
    parser = argparse.ArgumentParser(description='Extracting Features using HTK.')
    parser.add_argument('-i', '--input', dest = 'samples_folder', type = str, help='Folder containing the audio samples.')
    parser.add_argument('-o', '--output', dest = 'output_folder', type = str, help='Output folder for features (does not need to exist beforehand).')
    parser.add_argument('-f', '--features', dest = 'features', type = str, default = 'MFCC_0_D_A_Z', help='features type and qualifiers (see HTK Book for format). Allowed types are MFCC, LPCEPSTRA, LPC and PLP. Defaults to MFCC_0_D_A_Z')
    parser.add_argument('-c', '--cmpn', dest = 'cmpn', type = int, default = 12, help='number of base spectral components. Defaults to 12.')
    args = parser.parse_args()
    
    main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    temp_folder = init_folder(os.path.join(main_dir, 'Temp'))
    print 'Writing HTK features in %s' %args.output_folder
    _ = extract_features(args.samples_folder, args.output_folder, args.features, temp_folder, components=args.cmpn)
    shutil.rmtree(temp_folder)
