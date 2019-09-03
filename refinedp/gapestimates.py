import logging
import numpy as np
import numba
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


@numba.njit(fastmath=True)
def laplace_mechanism(q, epsilon, indices):
    request_q = q[indices]
    return request_q + np.random.laplace(0, float(len(request_q)) / epsilon, len(request_q))


# we implement baseline algorithm into the algorithm itself, i.e., the algorithm returns the result together
# with the un-refined result which would be returned by the baseline algorithm. Both for sake of time for experiment and
# for the requirement that the noise added to the algorithm and baseline algorithm should be the same.
def gap_noisy_topk(q, epsilon, k, counting_queries=False):
    assert k <= len(q), 'k must be less or equal to the length of q'
    scale = k / epsilon if counting_queries else 2 * k / epsilon
    noisy_q = q + np.random.laplace(scale=scale, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    gaps = np.fromiter((noisy_q[first] - noisy_q[second] for first, second in zip(indices[:-1], indices[1:])),
                       dtype=np.float)
    # baseline algorithm would just return (indices)
    return indices, gaps


# Noisy Top-K with Measures (together with baseline)
def gap_topk_estimates(q, epsilon, k, counting_queries=False):
    # allocate the privacy budget 1:1 to noisy k max and laplace mechanism
    indices, gaps = gap_noisy_topk(q, 0.5 * epsilon, k)
    direct_estimates = laplace_mechanism(q, 0.5 * epsilon, indices)
    p_total = (np.fromiter((k - i for i in range(1, k)), dtype=np.int, count=k - 1) * gaps).sum()
    p = np.empty(k, dtype=np.float)
    np.cumsum(gaps, out=p[1:])
    p[0] = 0
    if counting_queries:
        refined_estimates = (direct_estimates.sum() + k * direct_estimates + p_total - k * p) / (2 * k)
    else:
        refined_estimates = (direct_estimates.sum() + 4 * k * direct_estimates + p_total - k * p) / (5 * k)

    # baseline algorithm would just return (indices, direct_estimates)
    return indices, refined_estimates, indices, direct_estimates


# Sparse Vector (with Gap)
@numba.njit(fastmath=True)
def gap_sparse_vector(q, epsilon, k, threshold, allocation=(0.5, 0.5), counting_queries=False):
    threshold_allocation, query_allocation = allocation
    assert abs(threshold_allocation + query_allocation - 1.0) < 1e-05
    epsilon_1, epsilon_2 = threshold_allocation * epsilon, query_allocation * epsilon
    indices, gaps = [], []
    i, count = 0, 0
    noisy_threshold = threshold + np.random.laplace(0, 1.0 / epsilon_1)
    scale = k / epsilon_2 if counting_queries else 2 * k / epsilon_2
    while i < len(q) and count < k:
        noisy_q_i = q[i] + np.random.laplace(0, scale)
        if noisy_q_i >= noisy_threshold:
            indices.append(i)
            gaps.append(noisy_q_i - noisy_threshold)
            count += 1
        i += 1
    # baseline algorithm would just return (np.asarray(indices))
    return np.asarray(indices), np.asarray(gaps)


# Sparse Vector with Measures (together with baseline algorithm)
@numba.njit(fastmath=True)
def gap_svt_estimates(q, epsilon, k, threshold, counting_queries=False):
    # budget allocation for gap svt
    x, y = (1, np.power(k, 2.0 / 3.0)) if counting_queries else (1, np.power(2 * k, 2.0 / 3.0))
    gap_x, gap_y = x / (x + y), y / (x + y)

    # budget allocation between gap / laplace
    indices, gaps = gap_sparse_vector(q, 0.5 * epsilon, k, threshold, allocation=(gap_x, gap_y),
                                      counting_queries=counting_queries)
    assert len(indices) == len(gaps)
    direct_estimates = np.asarray(laplace_mechanism(q, 0.5 * epsilon, indices))

    variance_gap = 8 * np.power((1 + np.power(k, 2.0 / 3)), 3) / np.square(epsilon) if counting_queries else \
        8 * np.power((1 + np.power(2 * k, 2.0 / 3)), 3) / np.square(epsilon)

    variance_lap = 8 * np.square(k) / np.square(epsilon)

    # do weighted average
    initial_estimates = np.asarray(gaps + threshold)
    refined_estimates = \
        (initial_estimates / variance_gap + direct_estimates / variance_lap) / (1.0 / variance_gap + 1.0 / variance_lap)

    # baseline algorithm would simply return (indices, direct_estimates)
    return indices, refined_estimates, indices, direct_estimates


# metric functions
@numba.njit(fastmath=True)
def mean_square_error(indices, estimates, truth_indices, truth_estimates):
    return np.square(truth_estimates - estimates).sum() / float(len(truth_estimates))


def plot(k_array, dataset_name, data, output_prefix, theoretical, algorithm_name):
    generated_files = []
    theoretical_x = np.arange(k_array.min(), k_array.max())
    theoretical_y = theoretical(theoretical_x)
    improves_for_epsilons = []
    for epsilon, epsilon_dict in data.items():
        assert len(epsilon_dict) == 1 and 'mean_square_error' in epsilon_dict
        metric_dict = epsilon_dict['mean_square_error']
        baseline = np.asarray(metric_dict['baseline'])
        algorithm_data = np.asarray(metric_dict['algorithm'])
        improvements = 100 * (baseline - algorithm_data) / baseline
        improves_for_epsilons.append(improvements[8])
        plt.plot(k_array, improvements, label=r'\huge {}'.format(algorithm_name), linewidth=3, markersize=10,
                 marker='o')
        plt.ylim(0, 50)
        plt.ylabel(r'\huge \% Improvement in MSE')
        plt.plot(theoretical_x, 100 * theoretical_y, linewidth=5,
                 linestyle='--', label=r'\huge Theoretical Expected Improvement')
        plt.xlabel(r'\huge $k$')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        legend = plt.legend(loc=3)
        legend.get_frame().set_linewidth(0.0)
        plt.gcf().set_tight_layout(True)
        if abs(float(epsilon) - 0.7) < 1e-5:
            logger.info('Fix-epsilon Figures saved to {}'.format(output_prefix))
            filename = '{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'Mean_Square_Error',
                                                str(epsilon).replace('.', '-'))
            plt.savefig(filename)
            generated_files.append(filename)
        plt.clf()

    epsilons = np.asarray(tuple(data.keys()), dtype=np.float)
    plt.plot(epsilons, improves_for_epsilons, label=r'\huge {}'.format(algorithm_name), linewidth=3,
             markersize=10, marker='o')
    plt.plot(epsilons, [100 * theoretical(10) for _ in range(len(epsilons))], linewidth=5,
             linestyle='--', label=r'\huge Theoretical Expected Improvement')
    plt.ylabel(r'\huge \% Improvement in MSE')
    plt.ylim(0, 50)
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
