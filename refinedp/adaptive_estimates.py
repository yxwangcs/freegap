import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import numba

from refinedp.gapestimates import laplace_mechanism


logger = logging.getLogger(__name__)


# this is a combination of classical and adaptive svt
@numba.njit(fastmath=True)
def adaptive_sparse_vector(q, epsilon, k, threshold, allocate_x=0.5, allocate_y=0.5, counting_queries=False):
    #threshold_allocation, query_allocation = allocation
    #assert abs(threshold_allocation + query_allocation - 1.0) < 1e-05
    indices, variances, gaps = [], [], []
    classical_indices, classical_variances, classical_gaps = [], [], []
    count, classical_count = 0, 0
    epsilon_0, epsilon_1, epsilon_2 = allocate_x * epsilon, allocate_y * epsilon / (2.0 * k), allocate_y * epsilon / k
    sigma = 2 * np.sqrt(2) / epsilon_1 if counting_queries else 4 * np.sqrt(2) / epsilon_1
    i = 0
    noisy_threshold = threshold + np.random.laplace(0, 1.0 / epsilon_0)
    cost = epsilon_0
    while i < len(q):
        top_scale, middle_scale = (1.0 / epsilon_1, 1.0 / epsilon_2) if counting_queries else \
            (2.0 / epsilon_1, 2.0 / epsilon_2)
        eta_i, xi_i = np.random.laplace(0, top_scale), np.random.laplace(0, middle_scale)
        if count < k:
            if q[i] + eta_i - noisy_threshold >= sigma:
                indices.append(i)
                variances.append(2 * np.square(top_scale) + 2 * np.square(1.0 / epsilon_0))
                count += 1
                cost += epsilon_1
                gaps.append(q[i] + eta_i - noisy_threshold)
            elif q[i] + xi_i - noisy_threshold >= 0:
                indices.append(i)
                variances.append(2 * np.square(middle_scale) + 2 * np.square(1.0 / epsilon_0))
                count += 1
                cost += epsilon_2
                gaps.append(q[i] + xi_i - noisy_threshold)

        # classical svt
        if classical_count < k:
            if q[i] + xi_i - noisy_threshold >= 0:
                classical_indices.append(i)
                classical_count += 1
                classical_variances.append(2 * np.square(middle_scale) + 2 * np.square(1.0 / epsilon_0))
                classical_gaps.append(q[i] + xi_i - noisy_threshold)

        if classical_count >= k and count >= k:
            break
        i += 1

    classical_indices = np.asarray(classical_indices, dtype=np.int64)
    classical_variances = np.asarray(classical_variances, dtype=np.float64)
    classical_gaps = np.asarray(classical_gaps, dtype=np.float64)
    gaps = np.asarray(gaps, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    return indices, variances, gaps, epsilon - cost, \
           classical_indices, classical_variances, classical_gaps, 0


@numba.njit(fastmath=True)
def adaptive_estimates(q, epsilon, k, threshold, counting_queries=False):
    x, y = (1, np.power(k, 2.0 / 3.0)) if counting_queries else (1, np.power(2 * k, 2.0 / 3.0))
    gap_x, gap_y = x / (x + y), y / (x + y)
    assert abs(gap_x + gap_y - 1.0) < 1e-5

    indices, variances, gaps, remaining_budget, \
    classical_indices, classical_variances, classical_gaps, _ = \
        adaptive_sparse_vector(q, 0.5 * epsilon, k, threshold, gap_x, gap_y, counting_queries)

    assert len(indices) == len(variances) == len(gaps) <= k
    assert len(classical_indices) == len(classical_variances) == len(classical_gaps) <= k
    initial_estimates = np.asarray(gaps + threshold)
    direct_estimates = np.asarray(laplace_mechanism(q, 0.5 * epsilon + remaining_budget, indices))
    variance_lap = np.full(len(variances), 2 * np.square(k / (0.5 * epsilon + remaining_budget)))

    # do weighted average
    refined_estimates = \
        (initial_estimates / variances + direct_estimates / variance_lap) / (1.0 / variances + 1.0 / variance_lap)
    classical_direct_estimates = np.asarray(laplace_mechanism(q, 0.5 * epsilon, classical_indices))
    classical_initial_estimates = np.asarray(classical_gaps + threshold)
    classical_variance_lap = np.full(len(classical_variances), 2 * np.square(k / (0.5 * epsilon)))
    classical_refined_estimates = \
        (classical_initial_estimates / classical_variances + classical_direct_estimates / classical_variance_lap) / \
        (1.0 / classical_variances + 1.0 / classical_variance_lap)

    return (indices, refined_estimates), \
           (classical_indices, classical_refined_estimates), \
           (classical_indices, classical_direct_estimates)


# metric functions
@numba.njit(fastmath=True)
def mean_square_error(indices, estimates, truth_indices, truth_estimates):
    return np.square(truth_estimates - estimates).sum() / float(len(truth_estimates))


def plot(k_array, dataset_name, data, output_prefix):
    generated_files = []
    improves_for_epsilons = []
    ADAPTIVE_INDEX, GAP_INDEX, BASELINE_INDEX = 0, 1, -1
    for epsilon, epsilon_dict in data.items():
        assert len(epsilon_dict) == 1 and 'mean_square_error' in epsilon_dict
        metric_dict = epsilon_dict['mean_square_error']
        baseline = np.asarray(metric_dict[BASELINE_INDEX])
        algorithm_data = np.asarray(metric_dict[ADAPTIVE_INDEX])
        gap_data = np.asarray(metric_dict[GAP_INDEX])
        adaptive_improvements = 100 * (baseline - algorithm_data) / baseline
        gap_improvements = 100 * (baseline - gap_data) / baseline
        improves_for_epsilons.append(adaptive_improvements[8])
        plt.plot(k_array, adaptive_improvements, label=r'\huge {}'.format('Adaptive SVT with Gap'), linewidth=3, markersize=12,
                 marker='o')
        plt.plot(k_array, gap_improvements, label=r'\huge {}'.format('Sparse Vector with Gap'), linewidth=3,
                 markersize=12,
                 marker='o')
        plt.ylim(0, 80)
        plt.ylabel(r'\huge \% Improvement in MSE')
        plt.xlabel(r'\huge $k$')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        legend = plt.legend(loc=3)
        legend.get_frame().set_linewidth(0.0)
        plt.gcf().set_tight_layout(True)
        if abs(float(epsilon) - 0.3) < 1e-5:
            logger.info('Fix-epsilon Figures saved to {}'.format(output_prefix))
            filename = '{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'Mean_Square_Error',
                                                str(epsilon).replace('.', '-'))
            plt.savefig(filename)
            generated_files.append(filename)
        plt.clf()

    epsilons = np.asarray(tuple(data.keys()), dtype=np.float)
    plt.plot(epsilons, improves_for_epsilons, label=r'\huge {}'.format('Adaptive Estimates'), linewidth=3,
             markersize=10, marker='o')
    plt.ylabel(r'\huge \% Improvement in MSE')
    plt.ylim(0, 80)
    plt.xlabel(r'\huge $\epsilon$')
    plt.xticks(np.arange(epsilons.min(), epsilons.max() + 0.1, 0.2))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend(loc=3)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Fix-k Figures saved to {}'.format(output_prefix))
    filename = '{}/{}-{}-epsilons.pdf'.format(output_prefix, dataset_name, 'Mean_Square_Error',)
    plt.savefig(filename)
    generated_files.append(filename)
    plt.clf()
    return generated_files
