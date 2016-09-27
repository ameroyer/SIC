#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**plot.py.** Functions related to plotting and data visualization.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"

import os
import matplotlib
matplotlib.use('Agg') # No graphics device (e.g. for use through ssh)
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cmx
import matplotlib.ticker as ticker


##################################################################### PLOTTING A CLUSTERING
def histo_cluster(clustering, output_name):
    """
    Plots an histogram representation of a clustering (number of samples per cluster).

    Args:
     * ``clustering`` (*(cluster -> values) dict*): a dict representation of a clustering.
     * ``output_name`` (*str*): prefix of the file in which to output the figure.
    """

    # With distinction on occurences
    fig = plt.figure()
    ax = fig.add_subplot(111)
    clusters = [len(x)  for x in clustering.values()]
    indx = range(len(clusters))
    ax.bar(indx, clusters, 0.4, color='red')

    # Axes and labels
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Population')
    ax.set_title('Number of samples by cluster')
    ax.set_xticks(indx)
    xtickNames = ax.set_xticklabels([str(x) for x in clustering.keys()])
    plt.setp(xtickNames, rotation=45, fontsize=10)

    plt.savefig('%s.pdf' % output_name, bbox_inches='tight')



def pie_chart(clusters, output_folder):
    """
    Plots a pie-chart representation of a clustering.

    Args:
     * ``clustering`` (*(cluster -> values) dict*): a dict representation of a clustering.
     * ``output_folder`` (*str*): path to output folder.
    """
    matplotlib.rcParams['font.size'] = 22
    n_samples = sum([len(x) for x in clusters.values()])
    sizes = [len(x) for x in clusters.values()]
    labels = ['%s - %d'%(k,s) for k,s in zip(clusters.keys(), sizes)]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'darkseagreen', 'plum', 'cornflowerblue', 'indianred']
    explode = tuple([0.1] * 8)

    plt.pie([float(x)/n_samples for x in sizes], explode = explode, labels=labels, colors=colors, shadow=True, startangle=90)
    plt.axis('equal')

    plt.savefig(os.path.join(output_folder, 'cluster_piechart.pdf'))




def fraction_plot(clustering, ground_truth, output_name):
    """
    Plots an histogram representation of a clustering with colors representing the ground-truth clustering.

    Args:
     * ``clustering`` (*(cluster -> values) dict*): a dict representation of a clustering.
     * ``ground_truth`` (*(cluster -> values) dict*): a dict representation of the ground-truth clustering.
     * ``output_name`` (*str*): prefix of the file in which to output the figure.
    """

    found_c = clustering.keys()
    true_c = ground_truth.keys()

    #Build ground-truth auxillary representation
    value_to_gt = {}
    for k, l in ground_truth.iteritems():
        for x in l:
            value_to_gt[x] = k

    # Initializing colors
    color_norm  = clr.Normalize(vmin= 0, vmax=len(true_c))
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    # For each class of the clustering, count the number of times it occurs in each cluster of the ground-truth
    key_to_index = {x : i for i, x in enumerate(found_c)}
    fractions = {x : ([0] * len(found_c)) for x in true_c}
    for k, l in clustering.iteritems():
        keyind = key_to_index[k]
        for x in l:
            fractions[value_to_gt[x]][keyind] += 1

    # Plot stacked histograms
    fig = plt.figure()
    ax = fig.add_subplot(111)
    indx = range(len(found_c))
    btm = [0] * len(found_c)
    for i, (l, f) in enumerate(zip(true_c, fractions.values())):
        plt.bar(indx, f, color = scalar_map.to_rgba(i), bottom=btm, label=l)
        for x in xrange(len(found_c)):
            btm[x] += f[x]

    # Axes and labels
    ax.set_title('Retrieved clusters and confusion with ground-truth')
    ax.set_xlabel('Retrieved Clusters')
    ax.set_ylabel('Population')
    ax.legend()

    plt.savefig('%s.pdf' % output_name, bbox_inches='tight')


##################################################################### VISUALIZING THE SIMILARITY MATRIX
def mds_representation(sim_matrix, ground_truth, index_to_label, colors, output_folder, dim=2, cores=20, mode='mds'):
    """
    Compute euclidean distances (MDS) from the computed similarity matrix

    Args:
     * ``sim_matrix`` (*ndarray*): similarity matrix.
     * ``ground_truth`` (*dict*): ground-truth clustering.
     * ``colors`` (*dict*): mapping from a class to a color.
     * ``output_folder`` (*str*): path to the output folder.
     * ``dim`` (*int, optional*): number of dimensions in the metric space. Defaults to 2.
     * ``cores`` (*int, optional*): number of cores to use (threaded MDS).
     * ``mode`` (*str, optional*): Projection algorithm (``'mds'`` or ``'tsne'``).
    """

    from sklearn.manifold import MDS, TSNE
    print 'Running sklearn'
    try:
        Y = np.load('%scoor.npy' % mode)
    except:
        if mode == 'mds':
            from sklearn.manifold import MDS
            mds = MDS(n_components=dim, n_jobs=cores, dissimilarity='precomputed')#, metric=False )
            Y = mds.fit(sim_matrix).embedding_
        elif mode == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components = dim, metric='precomputed' )
            Y = tsne.fit_transform(sim_matrix)
        np.save('%scoor' % mode, Y)

    Y = Y * 10000000000000 # For readable annotation
    if dim == 2:
        print 'Plotting'
        fig = plt.figure()
        ax = fig.add_subplot(111, axisbg=(0.96, 0.96, 0.96))

        for l, pts in ground_truth.iteritems():
            ax.scatter(Y[pts, 0], Y[pts, 1], color=colors[l], label=l, alpha=0.3)
            ax.annotate(l, (Y[pts[2], 0], Y[pts[2], 1]), fontsize=8, horizontalalignment='center', verticalalignment='center')

        plt.xlim([min(Y[:,0]) - 1000, max(Y[:,0]) + 1000])
        plt.ylim([min(Y[:,1]) - 1000, max(Y[:,1]) + 1000] )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.axis('off')
        ax.legend(loc=4, bbox_to_anchor=(1.45, 0.))
        plt.savefig(os.path.join(output_folder, '%s_%ddim.png' % (mode, dim)), bbox_inches="tight", dpi=900)




def heatmap(similarity_matrix, ground_truth, output_folder):
    """
    Plots a heat map of the similarity matrix.

    Args:
     * ``sim_matrix`` (*ndarray*): similarity matrix.
     * ``ground_truth`` (*dict*): ground-truth clustering.
     * ``output_folder`` (*str*): path to the output folder.
    """

    print 'Reordering matrix'
    ordered = [x for cluster in ground_truth.values() for x in cluster]
    similarity_matrix = similarity_matrix[ordered, :]
    similarity_matrix = similarity_matrix[:, ordered]

    # heatmap
    print 'Drawing heatmap'
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg=(0.96, 0.96, 0.96))
    #ax.pcolor(similarity_matrix, cmap=plt.cm.Blues, alpha=0.8)
    #ax.pcolormesh(similarity_matrix, cmap=plt.cm.Blues, alpha=0.8)
    ax.imshow(similarity_matrix, cmap=plt.cm.Blues, alpha=0.8)

    # Major ticks
    ticks = np.cumsum([0] + [len([x for x in cluster]) for cluster in ground_truth.values()])
    plt.xticks(ticks, [])
    plt.yticks(ticks, [])
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    # Minor ticks
    minor_locs = [((ticks[k] + ticks[k+1]) / 2) for k in xrange(len(ticks) - 1)]
    ax.set_xticks(minor_locs,minor=True)
    ax.set_xticklabels(ground_truth.keys(),minor=True,rotation='vertical')
    ax.set_yticks(minor_locs,minor=True)
    ax.set_yticklabels(ground_truth.keys(),minor=True)

    print 'Saving Figure'
    #plt.savefig(os.path.join(output_folder, 'sim_heatmap.pdf'), bbox_inches="tight")
    plt.savefig(os.path.join(output_folder, 'sim_heatmap.png'), bbox_inches="tight")
    plt.close(fig)




################################################################ PARAMETERS ESTIMATION
def compare_params(true_params, em_params, ip0, output_folder):
    """
    Plots several visual comparisons of the parameters estimation.

    Args:
     * ``true_params`` (*list*): ground-truth p0 (true_params[0]) and p1 (true_params[1]) parameters.
     * ``em_params`` (*dict*): EM estimates of p0 (em_params[0]) and p1 (em_params[1]) parameters.
     * ``ip0`` (*list*): p0 parameters estimates with independance assumption.
     * ``output_folder`` (*str*): path to the output folder.
    """

    import numpy as np
    ind = range(len(true_params[0]))

    # Mean and Standard deviations
    print 'True P0 Mean %.5f and std %.5f' %(np.mean(true_params[0]), np.std(true_params[0]))
    print 'True P1 Mean %.5f and std %.5f' %(np.mean(true_params[1]), np.std(true_params[1]))
    print 'EM P0 Mean %.5f and std %.5f' %(np.mean(em_params[0]), np.std(em_params[0]))
    print 'EM P1 Mean %.5f and std %.5f' %(np.mean(em_params[1]), np.std(em_params[1]))
    print 'Indep P0 Mean %.5f and std %.5f' %(np.mean(ip0), np.std(ip0))

    # Comparison for each iteration
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_axis_bgcolor((0.96, 0.96, 0.96))
    ax2.set_axis_bgcolor((0.96, 0.96, 0.96))

    ax1.plot(ind, true_params[0], color='#7A68A6', linestyle='-', linewidth=4, marker='o', label='p0 (true)')
    ax1.plot(ind, em_params[0], color='#348ABD', linestyle='-', linewidth=4, marker='o', label='p0 (EM)')
    ax1.plot(ind, ip0, color='red', linestyle='-', linewidth=4, marker='o', label='p0 (indep)')
    ax2.plot(ind, true_params[1], color='#7A68A6', linestyle='-', linewidth=4, marker='o', label='p1 (true)')
    ax2.plot(ind, em_params[1], color='#348ABD', linestyle='-', linewidth=4, marker='o', label='p1 (EM)')

    ax1.set_ylabel('Estimates')
    ax2.set_ylabel('Estimates')
    ax2.set_xlabel('Iterations')
    ax2.set_ylim([0,0.6])
    ax1.set_ylim([0,0.16])
    plt.suptitle('Estimated parameters at each iteration')
    ax1.legend(loc=2, ncol=3)
    ax2.legend(loc=2, ncol=2)

    plt.savefig(os.path.join(output_folder, 'params_estimates_curves.pdf'))
    plt.close(fig)


    # Whisker plots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.set_axis_bgcolor((0.96, 0.96, 0.96))
    ax2.set_axis_bgcolor((0.96, 0.96, 0.96))

    ax1.boxplot([true_params[0], em_params[0], ip0])
    ax2.boxplot([true_params[1], em_params[1]])

    ax1.set_xticklabels(('p0 (true)','p0 (EM)','p0 (indep)'))
    ax2.set_xticklabels(('p1 (true)', 'p1 (EM)'))
    ax1.set_ylabel('Estimates')
    plt.suptitle('Parameters estimates - Boxplot representation')

    plt.savefig(os.path.join(output_folder, 'params_estimates_boxplot.pdf'))
    plt.close(fig)





def expected_binary(true_params, em_params, tpi0, epi0, output_folder):
    """
    Plot the 2-components Poisson Binomial mixture model given its parameters.

    Args:
     * ``true_params`` (*list*): ground-truth p0 (true_params[0]) and p1 (true_params[1]) parameters.
     * ``em_params`` (*dict*): EM estimates of p0 (em_params[0]) and p1 (em_params[1]) parameters.
     * ``tpi0`` (*float*): ground-truth pi0 estimate.
     * ``epi0`` (*float*): EM pi0 estimate.
     * ``output_folder`` (*str*): path to the output folder.
    """
    from probability_fit import estimate_poisson_binomial
    N = len(true_params[0])

    f, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = True)
    ax1.set_axis_bgcolor((0.96, 0.96, 0.96))
    ax2.set_axis_bgcolor((0.96, 0.96, 0.96))

    #GROUND-TURH PARAMETER
    p1 = estimate_poisson_binomial(N, true_params[0])
    p2 = estimate_poisson_binomial(N, true_params[1])
    # Plot
    ax1.plot(range(N+1), p1, color='#7A68A6', label='Components')
    ax1.plot(range(N+1), p2, color='#7A68A6')
    # Fill  the components
    ax1.fill_between(range(N+1), 0, p1, color='#7A68A6', alpha=0.5)
    ax1.fill_between(range(N+1), 0, p2, color='#7A68A6', alpha=0.5)
    ax1.plot(range(N+1), tpi0 * p1 + (1. - tpi0) * p2, color='red', linewidth=2, label='Mixture distribution')

    #EM
    p1 = estimate_poisson_binomial(N, em_params[0])
    p2 = estimate_poisson_binomial(N, em_params[1])
    ax2.plot(range(N+1), p1, color='#348ABD', label='Components')
    ax2.plot(range(N+1), p2, color='#348ABD')
    ax2.fill_between(range(N+1), 0, p1, color='#348ABD', alpha=0.5)
    ax2.fill_between(range(N+1), 0, p2, color='#348ABD', alpha=0.5)
    ax2.plot(range(N+1), epi0 * p1 + (1. - epi0) * p2, color='red', linewidth=2, label='Mixture distribution')

    ax1.legend(loc=1)
    ax2.legend(loc=1)
    plt.xlim([0,N])
    ax2.set_xlabel('Number of co-classifications, k')
    ax1.set_ylabel('Distribution: P(S = k/N)')
    ax2.set_ylabel('Distribution: P(S = k/N)')
    ax1.set_title('Similarity distribution using the true parameters')
    ax2.set_title('Similarity distribution using the EM estimates')
    plt.savefig(os.path.join(output_folder, 'params_estimates_distrib.pdf'))


########################################################################### CONVERGENCE ANALYSIS
def convergence_curve(output_folder, log_file):
    """
    Plots the evolution of the correlation coefficients for the convergence experiments.

    Args:
     * ``output_folder`` (*str*): path to the directory for the outputs.
     * ``log_file`` (*str*): path to the log file output during the experiments (convergence_analysis.py script)
    """
    import re
    points = {}
    current_iter = -1
    current_values = [0] * 3

    # Read and parse
    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                n_iter = int(line.split()[0])

            if line.startswith('Iteration'):
                current_iter = int(line.split()[1])

            elif line.startswith('> Pearson Coefficient:'):
                aux = re.split(r'[(),]+', line)
                current_values[0] = float(aux[-3])

            elif line.startswith('> Spearman Coefficient:'):
                aux = re.split(r'[(),]+', line)
                current_values[1] = float(aux[-3])

            elif line.startswith('> Frobenius'):
                current_values[2] = float(line.split(':')[-1])
                points[current_iter] = current_values
                current_values = [0] * 3
        iters = i

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_axis_bgcolor((0.96, 0.96, 0.96))
    ax2.set_axis_bgcolor((0.96, 0.96, 0.96))
    points[n_iter] = [1, 1, 0]
    keys = sorted(points.keys())

    #Plot
    ax1.plot(keys, [points[x][0] for x in keys], color='red', linestyle='-', marker='o', label='Pearson Coefficient')
    ax1.plot(keys, [points[x][1] for x in keys], color='#7A68A6', linestyle='-', marker='o', label='Spearman Coefficient')
    ax2.plot(keys, [points[x][2] for x in keys], color='#348ABD', linestyle='-', marker='o', label='Frobenius distance')

    ax1.set_xlabel('Iteration (X)')
    ax1.set_ylabel('Correlation')
    ax2.set_xlabel('Iteration (X)')
    ax2.set_ylabel('Frobenius')
    ax1.legend(loc=4)
    ax2.legend()
    plt.suptitle('Matrix variation against %d iterations similarity' % iters)

    plt.savefig(os.path.join(output_folder, 'convergence_windows_%d.pdf' % iters))


def on_the_fly_cvg(file):
    """
    Plot some statistics on the on-the-fly convergence criterion.

    Args:
     * ``file`` (*str*): path to the file containing the measurements of the on-the-fly criterion
     """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.set_axis_bgcolor((0.96, 0.96, 0.96))
    ax2.set_axis_bgcolor((0.96, 0.96, 0.96))
    ax3.set_axis_bgcolor((0.96, 0.96, 0.96))
    ax4.set_axis_bgcolor((0.96, 0.96, 0.96))

    # Read and Parse
    from collections import defaultdict
    plots = defaultdict(lambda: [])
    labels = {}
    indx = []
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            aux = line.split()
            if i == 0:
                labels = aux[1:]
                plots = [[] for x in xrange(len(labels))]
            else:
                for j, x in enumerate(aux[1:]):
                    plots[j].append(x)
                indx.append(int(aux[0]))

    # Mean
    ax1.plot(indx, plots[0], label=labels[0], color='#7A68A6')
    ax1.plot(indx, plots[1], label=labels[1], color='#348ABD')
    ax1.legend()

    # Std
    ax2.plot(indx, plots[6], label=labels[6], color='#7A68A6')
    ax2.plot(indx, plots[7], label=labels[7], color='#348ABD')
    ax2.legend(loc=4)

    # Min/Max
    ax3.plot(indx, plots[2], label=labels[2], color='#7A68A6')
    ax3.plot(indx, plots[3], label=labels[3], color='#348ABD')
    ax3.legend()

    # Median
    ax4.plot(indx, plots[4], label=labels[4])
    ax4.plot(indx, plots[5], label=labels[5])
    ax4.legend()

    plt.savefig(os.path.join(os.path.dirname(file), 'plot_convergence_check.pdf'))





############################################################ PLOT THE AQUAINT SYNONYMY GRAPH
def is_in_upper_level(nodes, word, level):
    """
    Determines wether a node has already been seen as a closer neighbour (level).

    Args:
     * ``nodes`` (*dict*): dict mapping a level to the nodes it contains.
     * ``word`` (*str*): label of the node to consider.
     * ``level`` (*int*): current level.
    """

    for k, n in nodes.iteritems():
        if k <= level:
            if word in n:
                return True
        else:
            if word in n:
                nodes[k] = [x for x in n if x != word]
                return False
    return False


def get_Aquaint_graph(start, synonyms, nodes, edges, level, maxlevel):
    """
    Returns an excerpt networkx graph from the Aquaint ground-truth. Recursive function.

    Args:
     * ``start`` (*list*): list of nodes to build edge from.
     * ``synonyms`` (*dict*): Aquaint ground-truth neighbours relations.
     * ``nodes`` (*dict*): list of nodes already built, organized by depth (minimum depth relatively to one of the starting nodes).
     * ``edges`` (*list*): list of edges already built.
     * ``level`` (*int*): current depth (starting at 0).
     * ``maxlevel`` (*int*): max depth to consider.
    """
    for word in start:
        if not is_in_upper_level(nodes, word, level):
            nodes[level].append(word)
        if word in synonyms and level <= maxlevel:
            for x in synonyms[word]:
                edges.append((word, x))
            get_Aquaint_graph(synonyms[word], synonyms, nodes, edges, level + 1, maxlevel)


def plot_Aquaint_graph(words, aqua_gt, level=1):
    """
    Plot an excerpt  graph from the Aquaint ground-truth using the networkx library.

    Args:
     * ``words`` (*list*): nodes to consider as origin.
     * ``aqua_gt`` (*str*): path to the Aquaint ground-truth.
     * ``level`` (*int*): max depth to consider (starting at 0).
    """
    import networkx as nx
    from collections import defaultdict

    # Load synonyms
    synonyms = {}
    with open(aqua_gt, 'r') as f:
        for line in f.readlines():
            if line.strip():
                aux = line.replace('#n', '').split()
                word = aux[0].strip()
                syn = [y.strip() for y in aux[1:]][:8]
                synonyms[word] = syn

    # Graph
    nodes = defaultdict(lambda:[])
    edges = []
    get_Aquaint_graph(words, synonyms, nodes, edges, 0, level)
    edges = list(set(edges))
    #test
    double_edges = []
    for (a,b) in edges:
        if (b,a) in edges:
            double_edges.append((a,b))
    for k, l in nodes.iteritems():
        nodes[k] = list(set(l))

    # Networkx Graph
    G = nx.DiGraph()
    G.add_nodes_from([x for l in nodes.values() for x in l ])
    G.add_edges_from(edges)
    try:
        pos=nx.graphviz_layout(G)
    except:
        pos=nx.spring_layout(G)

    # Plot
    colors = ['Crimson', 'AliceBlue', 'Gold']
    sizes = [3.7, 3.5, 3.5]
    sizes = [x * 200 for x in sizes]
    for k, l in nodes.iteritems():
        print k, l
        nx.draw_networkx_nodes(G, pos, nodelist=l, node_size=sizes[k], node_color=colors[k], node_shape='o', linewidths=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=double_edges, width=3.2, arrows=False)
    nx.draw_networkx_labels(G,pos,font_size=4.8,font_family='sans-serif')

    plt.gca().axison = False
    plt.savefig('aquaint_graph.pdf')




if __name__ == '__main__':
    import os
    import argparse
    import ConfigParser as cfg                         # 'configparser' for Python 3+'
    from parse import parse_ground_truth
    from matrix_op import normalize
    from output_format import init_folder, load_cooc
    import numpy as np

    #----------------------------------------------------------------------------- INIT
    # Parse command line
    parser = argparse.ArgumentParser(description='Clustering by Diverting supervised classification techniques.')
    parser.add_argument(dest='input_matrix', type=str, help='path to the file containing the similarity matrix.')
    parser.add_argument('-cfg', dest='cfg_file', type=str, help='Input a custom config file for default option values.')
    args = parser.parse_args()
    output_folder = init_folder(os.path.join(os.path.dirname(os.path.realpath(args.input_matrix)), 'Plot'))

    # Import ground-truth for coloring
    config = cfg.ConfigParser()
    config_file = os.path.join(output_folder, 'exp_configuration.ini') if (args.cfg_file is None) else args.cfg_file
    config.read(config_file)
    args.dataset = config.get('General', 'data')
    gtf = config.get(args.dataset, 'ground_truth')
    ground_truth = parse_ground_truth(args.dataset, gtf, [])
    colours = ['#FFD860', '#999877', '#CCC638', '#B2D3FF', '#389ECC', '#FF7266', '#54CC3D', '#813399', '#663BFF', '#CC961B', 'DarkOrange', 'CornFlowerBlue', 'Crimson', 'Coral', 'Olive', 'Navy', 'LightSeaGreen', 'Indigo', 'MediumSlateBlue']
    colours_dict = {x: colours.pop() for x in ground_truth.keys()}

    # Import index to label
    idtf = config.get(args.dataset, 'index_to_label')
    try:
        index_to_label = []
        label_to_index = {}
        with open(idtf, 'r') as f:
            for line in f:
                word = line.split('\t')[1].replace('\n', '')
                label_to_index[word] = len(index_to_label)
                index_to_label.append(word)
        n_samples = len(index_to_label)
    except IOError:
        print >> sys.stderr, 'Error: Index to label file %s not found' %idtf
        raise SystemExit

    # Import similarity matrix
    print 'Loading full similarity matrix'
    co_occ = load_cooc(args.input_matrix)
    co_occ = co_occ + co_occ.T
    co_occ = np.amax(co_occ) - co_occ
    np.fill_diagonal(co_occ, 0)

    pie_chart(ground_truth, output_folder)
    heatmap(np.amax(co_occ) - co_occ, ground_truth, output_folder)
    mds_representation(co_occ, ground_truth, index_to_label, colours_dict, output_folder, dim=2, cores=1)
