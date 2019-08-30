import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import jit


logger = logging.getLogger(__name__)

"""deprecated
# classical sparse vector to compare with
def sparse_vector(q, epsilon, k, threshold, middle_prng=np.random):
    indices = []
    i, count = 0, 0
    noisy_threshold = threshold + np.random.laplace(scale=2.0 / epsilon)
    while i < len(q) and count < k:
        if q[i] + middle_prng.laplace(scale=4.0 * k / epsilon) >= noisy_threshold:
            indices.append(i)
            count += 1
        i += 1
    return np.asarray(indices), i
"""

_INVALID_ARRAY = np.array([-1])


# this is a combination of classical and adaptive svt
@jit(nopython=True)
def adaptive_sparse_vector(q, epsilon, k, threshold):
    top_indices, middle_indices = [], []
    classical_indices, count, classical_i = [], 0, 0
    epsilon_0, epsilon_1, epsilon_2 = epsilon / 2.0, epsilon / (8.0 * k), epsilon / (4.0 * k)
    sigma = 2 * np.sqrt(2) / epsilon_1
    i, priv = 0, epsilon_0
    noisy_threshold = threshold + np.random.laplace(0, 1.0 / epsilon_0)
    while i < len(q) and priv <= epsilon - 2 * epsilon_2:
        eta_i = np.random.laplace(0, 1.0 / epsilon_1)
        xi_i = np.random.laplace(0, 1.0 / epsilon_2)
        if q[i] + eta_i - noisy_threshold >= sigma:
            top_indices.append(i)
            priv += 2 * epsilon_1
        elif q[i] + xi_i - noisy_threshold >= 0:
            middle_indices.append(i)
            priv += 2 * epsilon_2

        # classical svt
        if count < k:
            if q[i] + xi_i - noisy_threshold >= 0:
                classical_indices.append(i)
                count += 1
            classical_i += 1
        i += 1

    indices = np.asarray(top_indices + middle_indices)
    indices.sort()
    classical_indices = np.asarray(classical_indices)
    return indices, i, top_indices, middle_indices, \
           classical_indices, classical_i, classical_indices, _INVALID_ARRAY


def above_threshold_answers(indices, total, top_indices, middle_indices, truth_indices):
    return len(indices)


def f_measure(indices, total, top_indices, middle_indices, truth_indices):
    precision_val = len(np.intersect1d(indices, truth_indices)) / float(len(indices))
    # generate truth_indices based on total returned indices
    recall_val = len(np.intersect1d(indices, truth_indices)) / float(len(truth_indices))
    if precision_val == 0:
        return 0
    else:
        return 2 * precision_val * recall_val / (precision_val + recall_val)


def top_branch(indices, total, top_indices, middle_indices, truth_indices):
    return len(top_indices)


def middle_branch(indices, total, top_indices, middle_indices, truth_indices):
    return len(middle_indices)


def precision(indices, total, top_indices, middle_indices, truth_indices):
    return len(np.intersect1d(indices, truth_indices)) / float(len(indices))


def top_branch_precision(indices, total, top_indices, middle_indices, truth_indices):
    if len(top_indices) == 0:
        return 1.0
    else:
        return len(np.intersect1d(top_indices, truth_indices)) / float(len(top_indices))


def middle_branch_precision(indices, total, top_indices, middle_indices, truth_indices):
    if len(middle_indices) == 0:
        return 1.0
    else:
        return len(np.intersect1d(middle_indices, truth_indices)) / float(len(middle_indices))


def plot(k_array, dataset_name, data, output_prefix):
    generated_files = []
    epsilon = '0.7'
    """ code to find the best quantile
    quantiles = tuple(data[epsilon]['top_branch']['adaptive_sparse_vector'].keys())
    quantile_scores = []
    for quantile in quantiles:
        top = np.asarray(data[epsilon]['top_branch']['adaptive_sparse_vector'][quantile], dtype=np.int)
        middle = np.asarray(data[epsilon]['middle_branch']['adaptive_sparse_vector'][quantile], dtype=np.int)
        quantile_scores.append((top + middle).sum())
    quantile_scores = np.asarray(quantile_scores)
    quantile = quantiles[quantile_scores.argmax()]
    """
    quantile = '0.05'
    logger.info('best quantile is {}'.format(quantile))

    # plot number of above threshold answers
    baseline_top_branch = np.asarray(data[epsilon]['top_branch']['sparse_vector'])
    algorithm_top_branch = np.asarray(data[epsilon]['top_branch']['adaptive_sparse_vector'])
    algorithm_middle_branch = np.asarray(data[epsilon]['middle_branch']['adaptive_sparse_vector'])
    width = 0.6
    plt.ylim(0, 50)
    sub_k_array = np.arange(2, 24, 2)
    colormap = plt.get_cmap('tab10')
    plt.bar(sub_k_array - width, baseline_top_branch[sub_k_array - 1], width, align='edge',
            label=r'\huge Sparse Vector', facecolor=colormap.colors[0] + (0.8,), edgecolor='black', hatch='/')
    #plt.bar(sub_k_array - width, algorithm_total[sub_k_array - 1], width, align='edge',
            #label='\\huge Adaptive SVT w/ Gap  (Total)', facecolor=colormap.colors[1] + (0.8,), hatch='O')
    plt.bar(sub_k_array, algorithm_middle_branch[sub_k_array - 1], width, align='edge', facecolor=colormap.colors[1] + (0.8,),
            edgecolor='black',
            label=r'\huge Adaptive SVT w/ Gap (Middle)', hatch='.')
    plt.bar(sub_k_array, algorithm_top_branch[sub_k_array - 1], width, bottom=algorithm_middle_branch[sub_k_array - 1], align='edge', facecolor=colormap.colors[3] + (0.8,),
            edgecolor='black',
            label=r'\huge Adaptive SVT w/ Gap (Top)', hatch='*')
    plt.ylabel(r'\huge {}'.format(r'\# of Above-Threshold Answers'))
    plt.xlabel(r'\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(sub_k_array)
    legend = plt.legend(framealpha=0, loc=2)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Figures saved to {}'.format(output_prefix))
    filename = os.path.join(output_prefix,
                            '{}-{}-{}.pdf'.format(dataset_name, 'above_threshold_answers',
                                                  str(epsilon).replace('.', '-')))
    plt.savefig(filename)
    generated_files.append(filename)
    plt.clf()

    # plot the f-measure
    adaptive_f_measure = np.asarray(data[epsilon]['f_measure']['adaptive_sparse_vector'])
    sparse_vector_f_measure = np.asarray(data[epsilon]['f_measure']['sparse_vector'])
    plt.plot(k_array, sparse_vector_f_measure,
             label=r'\huge {}'.format('Sparse Vector'),
             linewidth=3, markersize=10, marker='P', zorder=5)
    plt.plot(k_array, adaptive_f_measure,
             label=r'\huge {}'.format('Adaptive SVT w/ Gap'),
             linewidth=3, markersize=10, marker='P', zorder=5)
    plt.ylim(0, 1.0)
    plt.ylabel(r'\huge {}'.format('F-Measure'))
    plt.xlabel(r'\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend(loc=3)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Figures saved to {}'.format(output_prefix))
    filename = '{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'fmeasure',
                                         str(epsilon).replace('.', '-'))
    plt.savefig(filename)
    generated_files.append(filename)
    plt.clf()

    # plot the precision
    adaptive_f_measure = np.asarray(data[epsilon]['precision']['adaptive_sparse_vector'])
    sparse_vector_f_measure = np.asarray(data[epsilon]['precision']['sparse_vector'])
    plt.plot(k_array, sparse_vector_f_measure,
             label=r'\huge {}'.format('Sparse Vector'),
             linewidth=3, markersize=10, marker='P', zorder=5)
    plt.plot(k_array, adaptive_f_measure,
             label=r'\huge {}'.format('Adaptive SVT w/ Gap'),
             linewidth=3, markersize=10, marker='P', zorder=5)
    plt.ylim(0, 1.0)
    plt.ylabel(r'\huge {}'.format('Precision'))
    plt.xlabel(r'\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend(loc=3)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Figures saved to {}'.format(output_prefix))
    filename = '{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'precision',
                                        str(epsilon).replace('.', '-'))
    plt.savefig(filename)
    generated_files.append(filename)
    plt.clf()

    # plot remaining epsilons
    """
    epsilons = np.asarray(tuple(data.keys()), dtype=np.float)
    left_budget = np.asarray(remaining_epsilons) * 100
    plt.plot(epsilons, left_budget,
             label=r'\huge {}'.format('Adaptive Sparse Vector with Gap'),
             linewidth=3, markersize=10, marker='o')
    plt.ylim(0, 25)
    plt.ylabel(r'\huge \% Remaining Privacy Budget')
    plt.xlabel(r'\huge $\epsilon$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend(loc=3)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Figures saved to {}'.format(output_prefix))
    filename = '{}/{}-{}.pdf'.format(output_prefix, dataset_name, 'left-epsilon')
    plt.savefig(filename)
    generated_files.append(filename)
    plt.clf()
    """
    return generated_files
