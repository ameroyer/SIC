#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**matrix_op.py.** Functions operating on the full similarity matrix (normalization and distribution analysis).
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


import os
import sys
import math
sys.path.append('/udd/aroyer/local/lib64/python2.7/site-packages') # bottleneck for matrix_op.keep_k_best
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from output_format import init_folder
from probability_fit import estimate_poisson_binomial, sample_from_pdf

################################################################################### Normalization
def normalize_gauss_global(co_occ):
    """
    Normalize the full matrix with respect to its global standard deviation and mean (X <- (X - mean) / std).

    Args:

     * ``co_occ`` (*ndarray*): input matrix.

    Returns:

     * ``normalized`` (*ndarray*): normalized matrix.
    """
    mu = np.mean(co_occ)
    sigma = np.std(co_occ)
    if sigma != 0:
        return (co_occ - mu)/sigma
    else:
        return co_occ - mu



def normalize_gauss_local(co_occ):
    """
    Normalize the full matrix line by line with respect to their global standard deviation and mean (X <- (X - mean) / std).

    Args:

     * ``co_occ`` (*ndarray*): input matrix.

    Returns:

     * ``normalized`` (*ndarray*): normalized matrix.
    """
    mu = np.mean(co_occ, axis = 1)
    sigma = np.std(co_occ, axis = 1)
    sigma[sigma == 0] = 1.0
    return (co_occ - mu) / sigma



def normalize_min_max(co_occ):
    """
    Normalize the full matrix globally with respect to its minimum and maximum value (X <- (X - min) / (max - min)).

    Args:

     * ``co_occ`` (*ndarray*): input matrix.

    Returns:

     * ``normalized`` (*ndarray*): normalized matrix.
    """
    mini = np.amin(co_occ)
    maxi = mp.amax(co_occ)
    if mini != maxi:
        return (co_occ - mini) / (maxi - mini)
    else:
        return co_occ



def keep_k_best(co_occ, k=200):
    """
    Keep the ``k`` best values in the matrix and set the rest to 0. Relies on the bottleneck library for fast sort.

    Args:

     * ``co_occ`` (*ndarray*): input matrix.
     * ``k`` (*int, optional*): number of values to keep. Defaults to 200.

    Returns:

     * ``normalized`` (*ndarray*): normalized matrix.
    """
    import bottleneck as bn
    part = bn.argpartsort(-co_occ, k, axis = 1)[:, :k]
    for line in xrange(co_occ.shape[0]):
        c = co_occ[line,:]
        kbest = c[part[line, -1]]
        c[c < kbest] = 0.
        co_occ[line, :] = c
    return co_occ



def normalize(co_occ):
    """
    Returns a normalized version of the input matrix.

    Args:

     * ``co_occ`` (*ndarray*): co-occurence matrix.

    Returns:

     * ``normalized`` (*ndarray*): a normalized version of the co-occurence matrix.
    """
    #Std and Mean
    full = co_occ + co_occ.T
    full = normalize_gauss_local(full)
    full[full < 0.] = 0.
    #full = keep_k_best(full, k = 10)
    return full



############################################################################# SIMILARITY DISTRIBUTION
def distribution_analysis(line, name, output_folder, temp_folder, ground_truth, kbest= [2000, 1000, 500, 200], mode='matlab'):
    """
    Plots the similarities histogram and densities (+ ground-truth display) for a line (sample) of the similarity matrix, at various scales.

    Args:
     * ``line`` (*ndarray*): **sorted** similarities in increasing order for one sample.
     * ``name`` (*str*): prefix for naming the plots.
     * ``output_folder`` (*str*): path to directory to output the plots.
     * ``temp_folder`` (*str*): path to directory to output temporary plots (before concatenation).
     * ``ground_truth``: indices of samples belonging to the same class as current sample.
     * ``kbest`` (*list, optional*): indices for zoom ins (keep the ``k`` best values, for all ``k`` in ``kbest``).
     * ``mode`` (*str*): if 'matlab', then plots the distribution on the whole interval using matplotlib. If 'R', plots the distribution for all zoom-values in ``kbest`` using ``R`` (requires ``ggplot2`` library).
    """

    if mode == 'R':
        length = len(line)
        kbest = [length] + kbest
        files = [0] * len(kbest)
        temp_all = os.path.join(temp_folder, 'simall')
        temp_gt = os.path.join(temp_folder, 'simgt')
        FNULL = open(os.devnull, 'w')

        # Plotting with R
        for i, k in enumerate(kbest):
            np.savetxt(temp_all, line[(length - k):])
            np.savetxt(temp_gt, line[np.intersect1d(range(length - k, length), ground_truth)])
            files[i] = os.path.join(temp_folder, '%s_%d_best_distribution.pdf'%(name,k))
            subprocess.call(['Rscript', os.path.join(utils_dir, 'density_plot.R'), os.path.abspath(temp_all), os.path.abspath(temp_gt), files[i], '(%d best similarities)' % k], stdout=FNULL, stderr=FNULL)

        # Clean up and concatenate
        os.remove(temp_gt)
        os.remove(temp_all)
        if len(files) == 1:
            os.rename(files[0], os.path.join(output_folder, '%s_distribution.pdf'%name))
        elif len(files) > 1:
            subprocess.call(['pdfunite'] + files + [os.path.join(output_folder, '%s_distribution.pdf' % name)], stdout=FNULL, stderr=FNULL)
        for f in files:
            os.remove(f)
        FNULL.close()

    elif mode == 'matlab':
        # Plotting with Matplotlib
        minv = int(math.floor(min(line)))
        negative = np.setdiff1d(range(len(line)), ground_truth)
        indx = np.arange(minv, int(math.ceil(max(line))), 0.05)

        fig = plt.figure()
        ax = fig.add_subplot(111, axisbg=(0.96, 0.96, 0.96))
        ax.hist(line[negative], normed=True, color='#7A68A6', alpha=0.4, hatch='/', bins=indx, label='$S_{\perp}$')
        ax.hist(line[ground_truth], normed=True, color='#348ABD', alpha=0.4, bins=indx, label='$S_{\sim}$')
        ax.legend(loc=1)

        plt.savefig(os.path.join(output_folder,'%s_distribution.pdf' % name), bbox_inches='tight')
        plt.close(fig)



############################################################################# ROC CURVES
def ROC_analysis(line, name, output_folder, ground_truth):
    """
    Plots a ROC curve for one given sample (one line of the matrix).

    Args:
     * ``line`` (*ndarray*): similarities for one sample.
     * ``name`` (*str*): prefix for naming the plots.
     * ``output_folder`` (*str*): path to directory to output the plots.
     * ``ground_truth`` (*str*): indices of the samples belonging to the same class as the current sample.
    """
    n_samples = len(line)
    n_pos = len(ground_truth)
    tpr = [0] * n_samples
    fpr = [0] * n_samples

    # ROC Curve
    for i in xrange(n_samples):
        nline = line[i:]
        tpr[i] = float(len(ground_truth[ground_truth >= i])) / n_pos
        fpr[i] = float(len(nline) -  n_pos * tpr[i]) / (n_samples - n_pos)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg=(0.96, 0.96, 0.96))
    ax.plot(fpr, tpr, color='#7A68A6', linestyle='-', label='ROC curve')
    ax.plot([0,1], [0,1], color='#348ABD', linestyle='--', alpha=0.5, label='Random case')

    # Figure parameters
    plt.xlim([0,1])
    plt.ylim([0,1])
    for k in ['top', 'bottom', 'right', 'left']:
        ax.spines[k].set_visible(False)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('%s - ROC curve' % name)
    ax.legend(loc=4)

    plt.savefig(os.path.join(output_folder, '%s_ROC.pdf' % name), bbox_inches='tight')
    plt.close(fig)



def ROC_mean_analysis(lines, key, output_folder, gt):
    """
    Plots all ROC curve and their horizontal/vertical means for several samples of the same class (lines of the matrix).

    Args:
     * ``line`` (*ndarray*): similarities for one sample.
     * ``name`` (*str*): prefix for naming the plots.
     * ``output_folder`` (*str*): path to directory to output the plots.
     * ``gt`` (*list*): indices of the samples belonging to the currently considered class.
    """
    from collections import defaultdict

    n_samples = len(lines[0])
    n_pos = len(gt)
    tpr = defaultdict(lambda:[]) #False positive -> associated true positive
    fpr = defaultdict(lambda:[]) #True positive -> associated false positive

    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg=(0.96, 0.96, 0.96))

    # For each sample sort line and ground-truth
    for line in lines:
        #Sort
        sorted_ind = np.argsort(line)
        sorted_gt = np.zeros(len(line))
        sorted_gt[gt] = 1.0
        sorted_gt = sorted_gt[sorted_ind]
        sorted_gt = np.nonzero(sorted_gt)[0]
        line = line[sorted_ind]

        # Compute points of the ROC curve
        tpr_local = []
        fpr_local = []

        for i in xrange(n_samples):
            nline = line[i:]
            tp = float(len(sorted_gt[sorted_gt >= i])) / n_pos
            fp = float(len(nline) -  n_pos * tp) / (n_samples - n_pos)
            fpr[round(tp, 2)].append(fp)
            tpr[round(fp, 2)].append(tp)
            tpr_local.append(tp)
            fpr_local.append(fp)

        # Plot current ROC
        ax.plot(fpr_local, tpr_local, color='green', alpha=0.1, linewidth=0.2)

    # Plot mean ROC curves
    ax.scatter([float(sum(x)) / len(x) for x in fpr.values()], fpr.keys(), color='#7A68A6', marker='o', label='Average (horizontal) ROC curve')
    ax.scatter(tpr.keys(), [float(sum(x)) / len(x) for x in tpr.values()], color='#348ABD', marker='o', label='Average (vertical) ROC curve')
    ax.plot([0,1], [0,1], color='red', linestyle='--', alpha=0.5, label='Random case')

    # Figure paramenters
    plt.xlim([0,1])
    plt.ylim([0,1])
    for k in ['top', 'bottom', 'right', 'left']:
        ax.spines[k].set_visible(False)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('%s - Average ROC curves' % key)
    ax.legend(loc=4)

    plt.savefig(os.path.join(output_folder, '%s_average_ROC.png' % key), bbox_inches='tight')
    plt.close(fig)



########################################################### SIMILARITY DISTRIBUTION AGAINST THEORETICAL ONES
def statistical_analysis_binary(line, rnd_distrib, ground_truth, temp_folder, output_folder, name, suffix=None):
    """
    For binary SIC: plots the similarity distribution and its given theoretical model (Poisson Binomial).

    Args:
     * ``line`` (*list*): similarities for the considered sample x.
     * ``rnd_distrib`` (*list*): values of the discrete random case distribution (``k -> P(X = k)``)
     * ``ground_truth`` (*list*): indices of the samples in the same class as x.
     * ``temp_folder`` (*str*): path to the folder containing the temporary files.
     * ``output_folder`` (*str*): path to the output folder.
     * ``name`` (*str*): string representation of the considered sample.
     * ``suffix`` (*str, optional*): additional suffix for the output file. Defaults to None (no suffix).
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    k = len(rnd_distrib)
    bin_width = 40 # histogram width
    negative = np.setdiff1d(range(len(line)), ground_truth) # indices not in ground-truth

    # Plotting
    bins = range(0, k + bin_width, bin_width)
    fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    #Histogram Counts
    neg = np.intersect1d(negative, np.where(line <= k)[0])
    rnd_line = sample_from_pdf(len(neg), rnd_distrib)
    ax1.hist(line[negative], normed=True, color='#7A68A6', alpha=0.4, hatch='/', bins=bins, label='$S_{\perp}$')
    ax1.hist(line[ground_truth], normed=True, color='#348ABD', alpha=0.4, bins=bins, label='$S_{\sim}$')
    ax1.hist(rnd_line, alpha=0.4, bins=bins, color='red', label='Theory (sampled histogram)')

    #Densities
    density = gaussian_kde(line)
    ax2.plot(range(k+1), rnd_distrib[:k+1], 'bo-', label='Theorical density')
    ax2.plot(range(k+1), density(range(k+1)), 'go-', label='Practical density')

    # Parameters
    ax1.set_ylabel('Counts')
    ax1.set_xlabel('Similarity values')
    ax1.set_title('Distributions for samples with similarity below %d' %k)
    ax1.legend()
    ax2.set_ylabel('P(s(x,y) = p)')
    ax2.legend()

    fi = os.path.join(temp_folder, '%s_binary_statistical_analysis' % name  if suffix is None else '%s_theo_binary_%s' % (name, suffix))
    plt.savefig(fi, bbox_inches='tight')
    plt.close(fig)



def statistical_analysis_weighted(line, N, ground_truth, temp_folder, output_folder, name, step=1.):
    """
    Plots the similarity distribution and gaussian theoretical distribution in the case of a weighted similarity (negative samples).

    Args:
     * ``line`` (*list*): similarities for the considered sample x.
     * ``N`` (*int*): standard deviation of the gaussian.
     * ``ground_truth`` (*list*): indices of the samples in the same class as x.
     * ``temp_folder`` (*str*): path to the folder containing the temporary files.
     * ``output_folder`` (*str*): path to the output folder.
     * ``name`` (*str*): string representation of the considered sample.
     * ``step`` (*float, optional*): step between x-axis' ticks.
    """
    from scipy.stats import norm, gaussian_kde

    minv = int(math.floor(min(line)))
    negative = np.setdiff1d(range(len(line)), ground_truth)
    indx = np.arange(minv, int(math.ceil(max(line))), step)
    density = gaussian_kde(line[negative])

    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg=(0.96, 0.96, 0.96))

    ax.hist(line[negative], normed=True, color='#7A68A6', alpha=0.4, hatch='/', bins=indx, label='$S_{\perp}$')
    ax.hist(line[ground_truth], normed=True, color='#348ABD', alpha=0.4, bins=indx, label='$S_{\sim}$')
    ax.plot(indx, density(indx), color='green', label='$S_{\perp}$ density')
    ax.plot(indx, norm.pdf(indx, loc=0., scale=np.sqrt(N)), color='red', label='Asymptotical Gaussian distribution')
    ax.legend(loc=1)

    plt.savefig(os.path.join(output_folder, '%s_theo_weighted.pdf' % name), bbox_inches='tight')
    plt.close(fig)
